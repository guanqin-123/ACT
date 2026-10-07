# ===- act/back_end/bab/linear_refresh.py - Split-aware linear refresh -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   G2 (D23): per-lane, split-aware CROWN-style backward refresh of the
#   intermediate pre-activation bounds, the implied phase literals it exposes,
#   and conservative E_i supports for CLIMB literal costs.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import torch

from act.back_end.bab.support_tracker import LayerBoundSupport, SupportRefreshResult
from act.back_end.core import Bounds, Net
from act.back_end.layer_schema import LayerKind

LINEAR_REFRESH_MODE: str = "split_refresh_linear"
"""``root_bounds_reuse`` mode: ``split_refresh`` with the linear refresh below."""

_SOURCE_KINDS = (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value)
_IDENTITY_KINDS = (LayerKind.FLATTEN.value, LayerKind.RESHAPE.value)
_CHAIN_KINDS = _SOURCE_KINDS + _IDENTITY_KINDS + (LayerKind.DENSE.value, LayerKind.RELU.value)
_MAX_BACKWARD_ELEMENTS = 1 << 26


def _kind(layer: Any) -> str:
    return layer.kind.upper() if isinstance(layer.kind, str) else layer.kind


def _chain_steps(net: Net) -> Optional[List[Tuple[int, str]]]:
    """Forward ``(layer id, kind)`` steps of a single-path INPUT/DENSE/RELU chain, else None."""
    steps: List[Tuple[int, str]] = []
    for layer in net.layers:
        kind = _kind(layer)
        if kind == LayerKind.ASSERT.value:
            continue
        if kind not in _CHAIN_KINDS:
            return None
        if steps and list(net.preds.get(layer.id, [])) != [steps[-1][0]]:
            return None
        if kind == LayerKind.DENSE.value and not isinstance(layer.params.get("weight"), torch.Tensor):
            return None
        steps.append((layer.id, kind))
    return steps


def _relu_relaxation(lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Lower slope, upper slope, upper intercept of a ReLU over ``[lb, ub]``."""
    active, inactive = lb >= 0, ub <= 0
    unstable = ~(active | inactive)
    width = (ub - lb).clamp(min=torch.finfo(lb.dtype).tiny)
    upper_slope = torch.where(unstable, ub / width, active.to(lb))
    upper_bias = torch.where(unstable, -upper_slope * lb, torch.zeros_like(lb))
    lower_slope = torch.where(unstable, (ub >= -lb).to(lb), active.to(lb))
    return lower_slope, upper_slope, upper_bias


def _backward_bounds(
    ops: Sequence[Tuple[Any, ...]], source: Tuple[torch.Tensor, torch.Tensor], width: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sound ``[lb, ub]`` of the last dense output via one backward pass over ``ops``.

    Rows ``[I; -I]`` are pushed back through dense maps and ReLU relaxations
    (positive coefficients take the lower line, negative ones the upper line)
    and concretized on the lane's input box.
    """
    lower_in, upper_in = source
    eye = torch.eye(width, dtype=lower_in.dtype, device=lower_in.device)
    coeffs = torch.cat((eye, -eye)).unsqueeze(0).expand(lower_in.shape[0], -1, -1)
    const = torch.zeros(coeffs.shape[:2], dtype=lower_in.dtype, device=lower_in.device)
    for op in reversed(ops):
        if op[0] == "dense":
            _, weight, bias = op
            if bias is not None:
                const = const + coeffs @ bias
            coeffs = coeffs @ weight
        else:
            _, lower_slope, upper_slope, upper_bias = op
            positive, negative = coeffs.clamp(min=0), coeffs.clamp(max=0)
            const = const + (negative * upper_bias.unsqueeze(1)).sum(-1)
            coeffs = positive * lower_slope.unsqueeze(1) + negative * upper_slope.unsqueeze(1)
    lower = (
        const
        + (coeffs.clamp(min=0) * lower_in.unsqueeze(1)).sum(-1)
        + (coeffs.clamp(max=0) * upper_in.unsqueeze(1)).sum(-1)
    )
    return lower[:, :width], -lower[:, width:]


def _halfspace_bounds(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    source: Tuple[torch.Tensor, torch.Tensor],
    phase: torch.Tensor,
    iters: int = 30,
    support_out: Optional[List[torch.Tensor]] = None,
    beta_out: Optional[List[torch.Tensor]] = None,
    warm_phase: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bounds of ``z = W x + b`` on ``box ∩ {s_i z_i >= 0}`` by weak duality.

    For any ``beta >= 0``, ``min_box (W_j x + b_j - sum_i beta_i s_i z_i(x))``
    lower-bounds ``z_j`` on the split region; ``beta`` takes normalized
    supergradient steps and the best value seen is kept, so every iterate is sound.
    """
    lower_in, upper_in = source
    width = weight.shape[0]
    offset = bias if bias is not None else torch.zeros(width, dtype=weight.dtype, device=weight.device)
    targets = torch.cat((weight, -weight)).unsqueeze(0)
    target_offset = torch.cat((offset, -offset)).unsqueeze(0)
    constraint = phase.unsqueeze(-1) * weight
    constraint_offset = phase * offset
    active = (phase != 0).unsqueeze(1).to(weight)
    beta = torch.zeros((phase.shape[0], 2 * width, width), dtype=weight.dtype, device=weight.device)
    best = torch.full((phase.shape[0], 2 * width), -torch.inf, dtype=weight.dtype, device=weight.device)
    support = torch.zeros_like(beta, dtype=torch.bool) if support_out is not None else None
    best_beta = torch.zeros_like(beta) if beta_out is not None else None
    step = 1.0
    for _ in range(iters):
        coeffs = targets - beta @ constraint
        point = torch.where(coeffs > 0, lower_in.unsqueeze(1), upper_in.unsqueeze(1))
        value = (
            target_offset
            - (beta * constraint_offset.unsqueeze(1)).sum(-1)
            + (coeffs * point).sum(-1)
        )
        if best_beta is not None:
            best_beta = torch.where((value > best).unsqueeze(-1), beta, best_beta)
        if support is not None:
            support = torch.where((value > best).unsqueeze(-1), beta > 0, support)
        best = torch.maximum(best, value)
        slack = point @ constraint.transpose(1, 2) + constraint_offset.unsqueeze(1)
        grad = -slack * active
        scale = grad.abs().amax(-1, keepdim=True).clamp(min=torch.finfo(weight.dtype).tiny)
        beta = (beta + step * grad / scale).clamp(min=0.0) * active
        step *= 0.85
    if warm_phase is not None:
        # The source dual witness is only a hint. Project it onto the candidate's
        # constraints and evaluate weak duality again on the candidate region.
        # No source bound or implied phase is assumed here.
        witnesses: List[torch.Tensor] = []
        _halfspace_bounds(weight, bias, source, warm_phase, iters, beta_out=witnesses)
        warm_beta = witnesses[0] * active
        coeffs = targets - warm_beta @ constraint
        point = torch.where(coeffs > 0, lower_in.unsqueeze(1), upper_in.unsqueeze(1))
        value = target_offset - (warm_beta * constraint_offset.unsqueeze(1)).sum(-1) + (coeffs * point).sum(-1)
        if support is not None:
            support = torch.where((value > best).unsqueeze(-1), warm_beta > 0, support)
        if best_beta is not None:
            best_beta = torch.where((value > best).unsqueeze(-1), warm_beta, best_beta)
        best = torch.maximum(best, value)
    if beta_out is not None and best_beta is not None:
        beta_out.append(best_beta)
    if support_out is not None and support is not None:
        support_out.extend((support[:, :width], support[:, width:]))
    return best[:, :width], -best[:, width:]


def _refresh_lanes(
    net: Net,
    steps: Sequence[Tuple[int, str]],
    base: Dict[int, Bounds],
    split_signs: Dict[int, torch.Tensor],
    warm_signs: Optional[Dict[int, torch.Tensor]] = None,
) -> Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
    refreshed: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    next_relu = {
        lid: steps[index + 1][0]
        for index, (lid, kind) in enumerate(steps[:-1])
        if kind == LayerKind.DENSE.value and steps[index + 1][1] == LayerKind.RELU.value
    }
    ops: List[Tuple[Any, ...]] = []
    source: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    current: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    for lid, kind in steps:
        known = base.get(lid)
        if kind in _SOURCE_KINDS:
            if known is None:
                return None
            source = current = (known.lb.flatten(1), known.ub.flatten(1))
            ops = []
            continue
        if current is None or source is None:
            return None
        if kind in _IDENTITY_KINDS:
            continue
        lb, ub = current
        if kind == LayerKind.DENSE.value:
            params = net.by_id[lid].params
            weight = cast(torch.Tensor, params["weight"]).to(lb)
            bias = params.get("bias")
            bias = bias.to(lb) if isinstance(bias, torch.Tensor) else None
            center, radius = (lb + ub) * 0.5, (ub - lb) * 0.5
            mid, spread = center @ weight.T, radius @ weight.abs().T
            lb, ub = mid - spread, mid + spread
            if bias is not None:
                lb, ub = lb + bias, ub + bias
            follower = next_relu.get(lid)
            signs = split_signs.get(follower) if follower is not None else None
            if not ops and signs is not None:
                phase = (signs[:, 0, :] if signs.dim() == 3 else signs).to(lb)
                if bool((phase != 0).any()):
                    warm = warm_signs.get(follower) if warm_signs is not None and follower is not None else None
                    warm_phase = (warm[:, 0, :] if warm.dim() == 3 else warm).to(lb) if warm is not None else None
                    half_lb, half_ub = _halfspace_bounds(weight, bias, source, phase, warm_phase=warm_phase)
                    lb, ub = torch.maximum(lb, half_lb), torch.minimum(ub, half_ub)
            ops.append(("dense", weight, bias))
            if any(op[0] == "relu" for op in ops):
                linear_lb, linear_ub = _backward_bounds(ops, source, weight.shape[0])
                lb, ub = torch.maximum(lb, linear_lb), torch.minimum(ub, linear_ub)
        if known is not None:
            lb = torch.maximum(lb, known.lb.flatten(1))
            ub = torch.minimum(ub, known.ub.flatten(1))
        if kind == LayerKind.RELU.value:
            signs = split_signs.get(lid)
            if signs is not None:
                phase = (signs[:, 0, :] if signs.dim() == 3 else signs).to(lb.device)
                lb = torch.where(phase > 0, lb.clamp(min=0.0), lb)
                ub = torch.where(phase < 0, ub.clamp(max=0.0), ub)
        # An empty region (split contradicting a bound) keeps a degenerate box.
        ub = torch.maximum(ub, lb)
        refreshed[lid] = (lb, ub)
        if kind == LayerKind.RELU.value:
            ops.append(("relu", *_relu_relaxation(lb, ub)))
            current = (lb.clamp(min=0.0), ub.clamp(min=0.0))
        else:
            current = (lb, ub)
    return refreshed


@torch.no_grad()
def linear_refresh_bounds(
    net: Net, base: Dict[int, Bounds], split_signs: Dict[int, torch.Tensor],
    warm_signs: Optional[Dict[int, torch.Tensor]] = None,
) -> Optional[Dict[int, Bounds]]:
    """Split-free ``base`` tightened per lane by its split literals (sound, batched).

    Each DENSE output is bounded by a backward linear pass through the
    lane's ReLU relaxations, whose split neurons are clamped to their phase,
    intersected with the interval step and ``base``. Returns None for nets
    that are not single-path INPUT/DENSE/RELU chains.
    """
    steps = _chain_steps(net)
    if not steps or not base:
        return None
    lanes = int(next(iter(base.values())).lb.shape[0])
    widths = [
        max(cast(torch.Tensor, net.by_id[lid].params["weight"]).shape) for lid, kind in steps
        if kind == LayerKind.DENSE.value
    ]
    per_lane = 2 * max(widths, default=1) ** 2 * 2
    chunk = max(1, min(lanes, _MAX_BACKWARD_ELEMENTS // per_lane))
    parts: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = []
    try:
        for start in range(0, lanes, chunk):
            rows = slice(start, min(start + chunk, lanes))
            part = _refresh_lanes(
                net,
                steps,
                {lid: Bounds(b.lb[rows], b.ub[rows]) for lid, b in base.items()},
                {lid: s[rows] for lid, s in (split_signs or {}).items()},
                {lid: s[rows] for lid, s in warm_signs.items()} if warm_signs is not None else None,
            )
            if part is None:
                return None
            parts.append(part)
    except (KeyError, IndexError, RuntimeError, ValueError):
        return None
    out = dict(base)
    for lid in parts[0]:
        if lid not in base:
            continue
        lb = torch.cat([part[lid][0] for part in parts])
        ub = torch.cat([part[lid][1] for part in parts])
        out[lid] = Bounds(lb.view_as(base[lid].lb).clone(), ub.view_as(base[lid].ub).clone())
    return out


def refresh_split_bounds(
    solver: Any,
    net: Net,
    base: Dict[int, Bounds],
    split_signs: Dict[int, torch.Tensor],
    mode: str,
) -> Optional[Dict[int, Bounds]]:
    """Split refresh for ``mode``: linear under ``LINEAR_REFRESH_MODE``, else interval.

    The linear mode falls back to the interval refresh for unsupported nets,
    so a lane solve and its core-local replay always use the same refresh.
    """
    if mode == LINEAR_REFRESH_MODE:
        refreshed = linear_refresh_bounds(net, base, split_signs)
        if refreshed is not None:
            return refreshed
    return solver._interval_refresh_bounds(net, base, split_signs)


def implied_phases(
    layer_ids: Sequence[int],
    base: Dict[int, Bounds],
    refreshed: Dict[int, Bounds],
    split_signs: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """Implied literals ``[K, n]`` in {-1, 0, +1}: unstable in ``base``, unsplit, stable after refresh."""
    implied: Dict[int, torch.Tensor] = {}
    for lid in layer_ids:
        before, after = base.get(lid), refreshed.get(lid)
        if before is None or after is None:
            continue
        lb0, ub0 = before.lb.flatten(1), before.ub.flatten(1)
        lb, ub = after.lb.flatten(1), after.ub.flatten(1)
        free = (lb0 < 0) & (ub0 > 0)
        signs = split_signs.get(lid)
        if signs is not None:
            free &= (signs[:, 0, :] if signs.dim() == 3 else signs).to(lb.device) == 0
        phase = torch.where(lb >= 0, 1, torch.where(ub <= 0, -1, 0))
        implied[lid] = torch.where(free, phase, 0).to(torch.int8)
    return implied


@torch.no_grad()
def conservative_supports(
    net: Net,
    refreshed: Dict[int, Bounds],
    literals: torch.Tensor,
    codec: Any,
    same_layer_supports: bool = True,
) -> SupportRefreshResult:
    """E_i supports of the linear refresh: every decision literal of an earlier layer.

    First-layer halfspace endpoints use the nonzero multipliers of their
    best dual witness, plus the neuron's own clamp. Later layers conservatively
    depend on all earlier decisions and their own clamp. The candidate must
    still pass replay: recomputing the refresh can find a weaker witness.
    ``same_layer_supports=False`` restores the pre-N3 supports (earlier-layer
    decisions plus the own clamp only).
    """
    lanes, slots = literals.shape
    device = literals.device
    valid = literals != 0
    layer_ids, neurons, _ = codec.decode_tensor(literals.masked_fill(~valid, 1))
    table = torch.tensor(codec.layer_ids, dtype=torch.long, device=device)
    position = torch.searchsorted(table, layer_ids)
    supports: Dict[int, LayerBoundSupport] = {}
    first_support: List[torch.Tensor] = []
    steps = _chain_steps(net)
    if same_layer_supports and steps and codec.layer_ids and lanes and slots:
        source = None
        for step, (lid, kind) in enumerate(steps):
            if kind in _SOURCE_KINDS:
                source = refreshed.get(lid)
                continue
            if kind in _IDENTITY_KINDS:
                continue
            if (kind == LayerKind.DENSE.value and source is not None
                    and step + 1 < len(steps) and steps[step + 1][0] == codec.layer_ids[0]):
                params = net.by_id[lid].params
                weight = params.get("weight")
                if isinstance(weight, torch.Tensor):
                    phase = torch.zeros((lanes, weight.shape[0]), device=device, dtype=source.lb.dtype)
                    rows, columns = torch.where(valid & (position == 0))
                    phase[rows, neurons[rows, columns]] = literals[rows, columns].sign().to(phase)
                    bias = params.get("bias")
                    width = weight.shape[0]
                    chunk = max(1, min(lanes, _MAX_BACKWARD_ELEMENTS // (4 * max(weight.shape) ** 2)))
                    endpoint_parts: List[List[torch.Tensor]] = [[], []]
                    for start in range(0, lanes, chunk):
                        row_slice = slice(start, min(start + chunk, lanes))
                        endpoint_support: List[torch.Tensor] = []
                        _halfspace_bounds(
                            weight.to(source.lb), bias.to(source.lb) if isinstance(bias, torch.Tensor) else None,
                            (source.lb[row_slice].flatten(1), source.ub[row_slice].flatten(1)),
                            phase[row_slice], support_out=endpoint_support,
                        )
                        columns = neurons[row_slice].clamp(max=width - 1).unsqueeze(1).expand(-1, width, -1)
                        for endpoint, mask in enumerate(endpoint_support):
                            endpoint_parts[endpoint].append(mask.gather(2, columns))
                    first_support = [torch.cat(parts) for parts in endpoint_parts]

            break
    for index, (lid, width) in enumerate(zip(codec.layer_ids, codec.widths)):
        earlier = (valid & (position < index)).unsqueeze(1)
        own = (valid & (position == index)).unsqueeze(1) & (
            neurons.unsqueeze(1) == torch.arange(width, device=device).view(1, -1, 1)
        )
        same_first = (valid & (position == index)).unsqueeze(1) if index == 0 and same_layer_supports else False
        lower = upper = earlier | own | same_first
        if index == 0 and first_support:
            lower = own | (first_support[0] & same_first)
            upper = own | (first_support[1] & same_first)
        supports[lid] = LayerBoundSupport(
            lower=lower,
            upper=upper.clone(),
            known=torch.ones((lanes, width), dtype=torch.bool, device=device),
        )
    return SupportRefreshResult(
        bounds=refreshed,
        supports=supports,
        protected_literals=torch.zeros((lanes, slots), dtype=torch.bool, device=device),
    )
