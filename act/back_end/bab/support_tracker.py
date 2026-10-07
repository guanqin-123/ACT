"""Dependency supports for split-refreshed intervals and CLIMB costs.

Support masks use the columns of CLIMB's packed literal tensor.  A true entry
``support[b, i, k]`` means that endpoint ``i`` in lane ``b`` may cease to be
valid if literal slot ``k`` is removed.  The masks are sufficient supports,
not necessarily minimal ones, as required by the paper's :math:`E_i` sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

import torch

from act.back_end.core import Bounds, Net
from act.back_end.layer_schema import LayerKind


_REFRESHED_KINDS = {
    LayerKind.CONV2D.value,
    LayerKind.DENSE.value,
    LayerKind.ADD.value,
    LayerKind.FLATTEN.value,
    LayerKind.RESHAPE.value,
    LayerKind.RELU.value,
}


class LiteralDecoder(Protocol):
    """Structural type implemented by ``climb.LiteralCodec``."""

    def decode_tensor(
        self, literals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class LayerBoundSupport:
    """Lower/upper endpoint supports for one layer.

    ``lower`` and ``upper`` have shape ``[batch, neurons, literal_slots]``.
    ``known`` has shape ``[batch, neurons]`` and is false when a sound support
    could not be reconstructed.  Consumers must treat such entries as
    undeletable rather than silently assuming empty support.
    """

    lower: torch.Tensor
    upper: torch.Tensor
    known: torch.Tensor

    def __post_init__(self) -> None:
        if self.lower.dtype != torch.bool or self.upper.dtype != torch.bool:
            raise TypeError("support masks must have dtype bool")
        if self.known.dtype != torch.bool:
            raise TypeError("known mask must have dtype bool")
        if self.lower.shape != self.upper.shape or self.lower.dim() != 3:
            raise ValueError("endpoint support shapes must be [batch, neurons, slots]")
        if self.known.shape != self.lower.shape[:2]:
            raise ValueError("known shape must match support batch and neurons")
        if self.upper.device != self.lower.device or self.known.device != self.lower.device:
            raise ValueError("support tensors must share a device")

    @property
    def combined(self) -> torch.Tensor:
        """Union of both endpoint supports, suitable for an interval chord."""

        return self.lower | self.upper


@dataclass(frozen=True)
class SupportRefreshResult:
    """Split-refreshed bounds and their per-layer dependency masks."""

    bounds: Dict[int, Bounds]
    supports: Dict[int, LayerBoundSupport]
    protected_literals: torch.Tensor


def _flat_bounds(bounds: Bounds) -> tuple[torch.Tensor, torch.Tensor]:
    return bounds.lb.flatten(start_dim=1), bounds.ub.flatten(start_dim=1)


def _empty_support(
    batch: int, neurons: int, slots: int, device: torch.device
) -> LayerBoundSupport:
    shape = (batch, neurons, slots)
    return LayerBoundSupport(
        torch.zeros(shape, dtype=torch.bool, device=device),
        torch.zeros(shape, dtype=torch.bool, device=device),
        torch.ones((batch, neurons), dtype=torch.bool, device=device),
    )


def _intersect_base(
    lower: torch.Tensor,
    upper: torch.Tensor,
    support: LayerBoundSupport,
    base: Optional[Bounds],
) -> tuple[torch.Tensor, torch.Tensor, LayerBoundSupport]:
    if base is None:
        return lower, upper, support
    base_lower, base_upper = _flat_bounds(base)
    if lower.shape != base_lower.shape or upper.shape != base_upper.shape:
        raise ValueError("refreshed and base interval shapes differ")
    lower_is_tighter = lower > base_lower
    upper_is_tighter = upper < base_upper
    lower = torch.maximum(lower, base_lower)
    upper = torch.minimum(upper, base_upper)
    lower_support = support.lower & lower_is_tighter.unsqueeze(-1)
    upper_support = support.upper & upper_is_tighter.unsqueeze(-1)
    known = support.known

    # Match DualSolver._interval_refresh_bounds on contradictory phase rows.
    repair = upper < lower
    upper = torch.maximum(upper, lower)
    upper_support = torch.where(
        repair.unsqueeze(-1), lower_support | upper_support, upper_support
    )
    return lower, upper, LayerBoundSupport(lower_support, upper_support, known)


def _union_all_neurons(mask: torch.Tensor, output_neurons: int) -> torch.Tensor:
    union = mask.any(dim=1, keepdim=True)
    return union.expand(-1, output_neurons, -1)


def _dense_endpoint_support(
    predecessor: LayerBoundSupport, weight: torch.Tensor
) -> LayerBoundSupport:
    positive = weight > 0
    negative = weight < 0

    def depends(mask: torch.Tensor, selector: torch.Tensor) -> torch.Tensor:
        # [B, I, W] -> [B, W, I] @ [I, O] -> [B, O, W]
        count = torch.matmul(
            mask.permute(0, 2, 1).to(torch.float32),
            selector.T.to(device=mask.device, dtype=torch.float32),
        )
        return count.permute(0, 2, 1) > 0

    lower = depends(predecessor.lower, positive) | depends(
        predecessor.upper, negative
    )
    upper = depends(predecessor.upper, positive) | depends(
        predecessor.lower, negative
    )
    required = weight != 0
    unknown = torch.matmul(
        (~predecessor.known).to(torch.float32),
        required.T.to(device=predecessor.known.device, dtype=torch.float32),
    ) > 0
    return LayerBoundSupport(lower, upper, ~unknown)


def _literal_slot_map(
    literals: torch.Tensor,
    codec: LiteralDecoder,
    layer_id: int,
    signs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each split neuron's packed slot and whether that slot is known."""

    decoded_layers, decoded_neurons, decoded_signs = codec.decode_tensor(literals)
    batch, neurons = signs.shape
    slots = torch.full(
        (batch, neurons), -1, dtype=torch.long, device=literals.device
    )
    lane, slot = torch.where((literals != 0) & (decoded_layers == layer_id))
    if lane.numel():
        neuron = decoded_neurons[lane, slot].to(torch.long)
        valid = (neuron >= 0) & (neuron < neurons)
        lane, slot, neuron = lane[valid], slot[valid], neuron[valid]
        sign_matches = decoded_signs[lane, slot].to(signs) == signs[lane, neuron].sign()
        lane, slot, neuron = lane[sign_matches], slot[sign_matches], neuron[sign_matches]
        slots[lane, neuron] = slot
    required = signs != 0
    return slots, (~required) | (slots >= 0)


def _slot_support(slots: torch.Tensor, width: int) -> torch.Tensor:
    if width == 0:
        return torch.zeros(
            (*slots.shape, 0), dtype=torch.bool, device=slots.device
        )
    valid = slots >= 0
    return torch.nn.functional.one_hot(slots.clamp(min=0), width).to(torch.bool) & valid.unsqueeze(-1)


@torch.no_grad()
def interval_refresh_with_supports(
    net: Net,
    base: Dict[int, Bounds],
    split_signs: Dict[int, torch.Tensor],
    literals: torch.Tensor,
    codec: LiteralDecoder,
) -> Optional[SupportRefreshResult]:
    """Repeat split interval refresh while recording sufficient supports.

    The numeric path intentionally mirrors ``DualSolver._interval_refresh_bounds``
    for INPUT, CONV2D, DENSE, ADD, FLATTEN, RESHAPE, and RELU.  Other layers
    retain their supplied base interval.  If such a non-ReLU operation receives
    a split-dependent interval, every source literal is recorded as support;
    this is the paper's conservative policy for additional nonlinear operators.
    ``None`` is returned when no sound numeric interval is available, matching
    the existing optional refresh contract.
    """

    if literals.dim() != 2:
        raise ValueError("literals must have shape [batch, slots]")
    batch, literal_width = literals.shape
    source_mask = literals != 0
    out = dict(base)
    values: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    value_supports: Dict[int, LayerBoundSupport] = {}
    recorded_supports: Dict[int, LayerBoundSupport] = {}
    opaque_outputs: set[int] = set()
    protected_literals = torch.zeros_like(source_mask)

    for layer in net.layers:
        kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
        layer_id = layer.id
        conservative_operator = False
        if kind == LayerKind.ASSERT.value:
            continue
        layer_base = out.get(layer_id)
        if kind in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value):
            if layer_base is None:
                return None
            lower, upper = _flat_bounds(layer_base)
            if lower.shape[0] != batch:
                raise ValueError("base bound batch does not match literals")
            support = _empty_support(
                batch, lower.shape[1], literal_width, literals.device
            )
            values[layer_id] = (lower, upper)
            value_supports[layer_id] = support
            recorded_supports[layer_id] = support
            continue

        predecessors = net.preds.get(layer_id, [])
        opaque_predecessor = any(pred in opaque_outputs for pred in predecessors)
        try:
            if opaque_predecessor and kind in _REFRESHED_KINDS:
                if layer_base is None:
                    return None
                conservative_operator = True
                lower, upper = _flat_bounds(layer_base)
                support = LayerBoundSupport(
                    source_mask[:, None, :].expand(-1, lower.shape[1], -1),
                    source_mask[:, None, :].expand(-1, lower.shape[1], -1),
                    torch.ones(
                        (batch, lower.shape[1]),
                        dtype=torch.bool,
                        device=literals.device,
                    ),
                )
            elif kind == LayerKind.CONV2D.value:
                from act.back_end.dual_tf.tf_forward import _fwd_conv2d_interval

                predecessor_id = predecessors[0]
                pred_lower, pred_upper = values[predecessor_id]
                lower, upper = _fwd_conv2d_interval(
                    layer, pred_lower, pred_upper
                )
                lower, upper = lower.flatten(start_dim=1), upper.flatten(start_dim=1)
                pred_support = value_supports[predecessor_id]
                support = LayerBoundSupport(
                    _union_all_neurons(pred_support.lower | pred_support.upper, lower.shape[1]),
                    _union_all_neurons(pred_support.lower | pred_support.upper, upper.shape[1]),
                    pred_support.known.all(dim=1, keepdim=True).expand(-1, lower.shape[1]),
                )
            elif kind == LayerKind.DENSE.value:
                predecessor_id = predecessors[0]
                pred_lower, pred_upper = values[predecessor_id]
                weight = layer.params["weight"]
                if not isinstance(weight, torch.Tensor):
                    return None
                weight = weight.to(device=pred_lower.device, dtype=pred_lower.dtype)
                center = (pred_lower + pred_upper) * 0.5
                radius = (pred_upper - pred_lower) * 0.5
                midpoint = center @ weight.T
                spread = radius @ weight.abs().T
                lower, upper = midpoint - spread, midpoint + spread
                bias = layer.params.get("bias")
                if isinstance(bias, torch.Tensor):
                    bias = bias.to(device=lower.device, dtype=lower.dtype)
                    lower, upper = lower + bias, upper + bias
                support = _dense_endpoint_support(
                    value_supports[predecessor_id], weight
                )
            elif kind == LayerKind.ADD.value:
                left_id, right_id = predecessors[0], predecessors[1]
                left_lower, left_upper = values[left_id]
                right_lower, right_upper = values[right_id]
                lower, upper = left_lower + right_lower, left_upper + right_upper
                left_support, right_support = value_supports[left_id], value_supports[right_id]
                support = LayerBoundSupport(
                    left_support.lower | right_support.lower,
                    left_support.upper | right_support.upper,
                    left_support.known & right_support.known,
                )
            elif kind in (LayerKind.FLATTEN.value, LayerKind.RESHAPE.value):
                predecessor_id = predecessors[0]
                lower, upper = values[predecessor_id]
                support = value_supports[predecessor_id]
            elif kind == LayerKind.RELU.value:
                predecessor_id = predecessors[0]
                lower, upper = values[predecessor_id]
                support = value_supports[predecessor_id]
            else:
                if layer_base is None or not predecessors:
                    return None
                conservative_operator = True
                lower, upper = _flat_bounds(layer_base)
                predecessor_supports = [value_supports[pred] for pred in predecessors]
                predecessor_known = torch.stack(
                    [item.known.all(dim=1) for item in predecessor_supports]
                ).all(dim=0)
                fed_by_refresh = torch.stack(
                    [item.combined.any(dim=(1, 2)) for item in predecessor_supports]
                ).any(dim=0) | ~predecessor_known
                conservative = source_mask[:, None, :].expand(
                    -1, lower.shape[1], -1
                ) & fed_by_refresh[:, None, None]
                protected_literals |= source_mask & fed_by_refresh[:, None]
                support = LayerBoundSupport(
                    conservative,
                    conservative.clone(),
                    predecessor_known[:, None].expand(-1, lower.shape[1]),
                )
                opaque_outputs.add(layer_id)
        except (KeyError, IndexError, RuntimeError, ValueError):
            return None

        if bool(torch.isnan(lower).any()) or bool(torch.isnan(upper).any()):
            return None

        conservative_support = support
        lower, upper, support = _intersect_base(
            lower, upper, support, layer_base
        )
        if bool(torch.isnan(lower).any()) or bool(torch.isnan(upper).any()):
            return None
        if conservative_operator:
            # This support protects the operator's local substitution, not
            # merely its output box, so an equal global output endpoint must
            # not erase the dependency on refreshed input intervals.
            support = conservative_support
        if kind == LayerKind.RELU.value:
            signs = split_signs.get(layer_id)
            if signs is not None:
                if signs.shape[0] != batch:
                    raise ValueError("split signs batch does not match literals")
                if signs.dim() == 2:
                    first = signs
                elif signs.dim() >= 3:
                    flattened = signs.flatten(start_dim=2)
                    first = flattened[:, 0, :]
                    if not torch.equal(
                        flattened, first[:, None, :].expand_as(flattened)
                    ):
                        raise ValueError("split signs must be spec-invariant")
                else:
                    raise ValueError("split signs must have at least two dimensions")
                count = min(lower.shape[1], first.shape[1])
                phase = first[:, :count].to(device=lower.device)
                slots, slot_known = _literal_slot_map(
                    literals, codec, layer_id, phase
                )
                direct = _slot_support(slots, literal_width)
                old_lower, old_upper = lower[:, :count], upper[:, :count]
                clamped_lower = torch.where(
                    phase > 0, old_lower.clamp(min=0.0), old_lower
                )
                clamped_upper = torch.where(
                    phase < 0, old_upper.clamp(max=0.0), old_upper
                )
                lower_changed = clamped_lower > old_lower
                upper_changed = clamped_upper < old_upper
                lower, upper = lower.clone(), upper.clone()
                lower[:, :count], upper[:, :count] = clamped_lower, clamped_upper
                lower_support, upper_support = support.lower.clone(), support.upper.clone()
                lower_support[:, :count] = torch.where(
                    lower_changed.unsqueeze(-1), direct, lower_support[:, :count]
                )
                upper_support[:, :count] = torch.where(
                    upper_changed.unsqueeze(-1), direct, upper_support[:, :count]
                )
                known = support.known.clone()
                changed = lower_changed | upper_changed
                if bool((changed & ~slot_known).any()):
                    return None
                known[:, :count] &= (~changed) | slot_known
                repair = upper < lower
                upper = torch.maximum(upper, lower)
                upper_support = torch.where(
                    repair.unsqueeze(-1), lower_support | upper_support, upper_support
                )
                support = LayerBoundSupport(lower_support, upper_support, known)

        if layer_base is not None:
            out[layer_id] = Bounds(
                lower.view_as(layer_base.lb).clone(),
                upper.view_as(layer_base.ub).clone(),
            )
        recorded_supports[layer_id] = support
        if kind == LayerKind.RELU.value:
            post_lower, post_upper = lower.clamp(min=0.0), upper.clamp(min=0.0)
            post_lower_support = support.lower & (lower > 0).unsqueeze(-1)
            # A local proof that ``upper <= 0`` is exactly what makes the
            # post-ReLU upper endpoint zero, so its support must not be lost.
            post_upper_support = support.upper
            values[layer_id] = (post_lower, post_upper)
            value_supports[layer_id] = LayerBoundSupport(
                post_lower_support, post_upper_support, support.known
            )
        else:
            values[layer_id] = (lower, upper)
            value_supports[layer_id] = support

    return SupportRefreshResult(out, recorded_supports, protected_literals)


@torch.no_grad()
def add_supported_chord_costs(
    own_costs: torch.Tensor,
    *,
    coefficients: Dict[int, torch.Tensor],
    reference_bounds: Dict[int, Bounds],
    supports: Dict[int, LayerBoundSupport],
    literal_mask: torch.Tensor,
) -> torch.Tensor:
    """Add supported unstable-ReLU chord residuals to own literal costs.

    ``own_costs`` is the existing ``eta`` plus split-phase term, shaped
    ``[batch, specs, literal_slots]``.  For every unstable neuron this adds
    ``max(0, -Lambda) * (-lower * upper) / (upper - lower)`` to each literal
    in that neuron's support.  Missing or explicitly unknown support makes all
    active literals in the affected lane/spec infinite, preventing deletion.
    """

    if own_costs.dim() != 3:
        raise ValueError("own_costs must have shape [batch, specs, slots]")
    batch, specs, slots = own_costs.shape
    if literal_mask.shape != (batch, slots) or literal_mask.dtype != torch.bool:
        raise ValueError("literal_mask must be bool with shape [batch, slots]")
    costs = own_costs.clone()
    active = literal_mask.to(costs.device).unsqueeze(1)

    for layer_id, coefficient in coefficients.items():
        if coefficient.dim() != 3 or coefficient.shape[:2] != (batch, specs):
            raise ValueError("coefficients must have shape [batch, specs, neurons]")
        coefficient = coefficient.to(device=costs.device, dtype=costs.dtype)
        nonfinite_coefficient = ~torch.isfinite(coefficient)
        if bool(nonfinite_coefficient.any()):
            affected = nonfinite_coefficient.any(dim=2, keepdim=True)
            costs.masked_fill_(affected & active, torch.inf)
            coefficient = coefficient.masked_fill(nonfinite_coefficient, 0)
        bounds = reference_bounds.get(layer_id)
        if bounds is None:
            missing_effect = (coefficient < 0).any(dim=2, keepdim=True)
            costs.masked_fill_(missing_effect & active, torch.inf)
            continue
        lower, upper = _flat_bounds(bounds)
        lower = lower.to(device=costs.device, dtype=costs.dtype)
        upper = upper.to(device=costs.device, dtype=costs.dtype)
        if lower.shape != (batch, coefficient.shape[2]) or upper.shape != lower.shape:
            raise ValueError("coefficient and reference-bound neuron shapes differ")
        nonfinite_bounds = ~torch.isfinite(lower) | ~torch.isfinite(upper)
        if bool(nonfinite_bounds.any()):
            affected = ((coefficient < 0) & nonfinite_bounds.unsqueeze(1)).any(
                dim=2, keepdim=True
            )
            costs.masked_fill_(affected & active, torch.inf)
            lower = lower.masked_fill(nonfinite_bounds, 0)
            upper = upper.masked_fill(nonfinite_bounds, 0)
        unstable = (lower < 0) & (upper > 0)
        denominator = (upper - lower).clamp(min=torch.finfo(costs.dtype).tiny)
        residual = torch.where(
            unstable, (upper * ((-lower) / denominator)).clamp(min=0), 0.0
        )
        correction = (-coefficient).clamp(min=0) * residual.unsqueeze(1)
        layer_support = supports.get(layer_id)
        if layer_support is None:
            missing_effect = (correction > 0).any(dim=2, keepdim=True)
            costs.masked_fill_(missing_effect & active, torch.inf)
            continue
        if layer_support.lower.shape != (batch, coefficient.shape[2], slots):
            raise ValueError("support and cost shapes differ")
        known = layer_support.known.to(costs.device)
        unknown_effect = ((correction > 0) & ~known.unsqueeze(1)).any(
            dim=2, keepdim=True
        )
        costs.masked_fill_(unknown_effect & active, torch.inf)
        finite_correction = correction * known.unsqueeze(1)
        costs += torch.einsum(
            "bmn,bnw->bmw",
            finite_correction,
            layer_support.combined.to(device=costs.device, dtype=costs.dtype),
        )
    return torch.nan_to_num(costs, nan=torch.inf, posinf=torch.inf, neginf=torch.inf)
