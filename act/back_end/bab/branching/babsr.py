# ===- act/back_end/bab/branching/babsr.py - BaBSR Branching -------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Canonical BaBSR (Branch-and-Bound with Score-based Refinement) neuron
#   selection for batched BaB. Implements the classical beta-CROWN scoring
#   heuristic: a single backward sweep collects per-layer lA coefficients
#   which are then scored against pre-activation bounds to rank splittable
#   neurons. Returns one (layer_id, neuron_idx, kind) per subproblem.
#
#   Phase 3 of .sisyphus/plans/batched-bab-eta.md.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

import logging
import atexit
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple, cast

import torch

from act.back_end.bounds_dispatch import get_conv_mode
from act.back_end.bab.branching.branching import BranchingStrategy
from act.back_end.bab.eta import get_pre_activation_layer_id
from act.back_end.bab.node import SubproblemBatch
from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf.tf_cnn_patches import (
    linear_form_patches_to_tensor,
    linear_form_tensor_to_patches,
)
from act.back_end.dual_tf.dual_tf import DualTF
from act.back_end.layer_schema import LayerKind
from act.back_end.patches import Patches

log = logging.getLogger(__name__)
_WARNED_SITES: set[tuple[str, int]] = set()
_MATERIALIZATION_COUNTS: Counter[int] = Counter()

def _warn_once(site: str, layer_id: int, message: str) -> None:
    key = (site, layer_id)
    if key in _WARNED_SITES:
        log.debug(message)
        return
    _WARNED_SITES.add(key)
    log.warning(message)


def _record_lA_materialization(layer_id: int) -> None:
    _MATERIALIZATION_COUNTS[layer_id] += 1
    _warn_once(
        "select_neurons_lA_materialize",
        layer_id,
        f"BaBSR select_neurons: materializing patches lA for layer {layer_id}",
    )


def emit_materialization_summary(logger: logging.Logger) -> None:
    if not _MATERIALIZATION_COUNTS:
        return
    logger.warning("[bab.branching.babsr] lA materializations during run:")
    for layer_id, count in sorted(_MATERIALIZATION_COUNTS.items()):
        logger.warning("  layer_id=%s count=%s", layer_id, count)


def reset_materialization_tracking() -> None:
    emit_materialization_summary(log)
    _WARNED_SITES.clear()
    _MATERIALIZATION_COUNTS.clear()


# ---------------------------------------------------------------------------
# Configuration / activation classification
# ---------------------------------------------------------------------------

# Smooth (S-shaped) activations handled in best-effort mode for v1.
_SMOOTH_KINDS = frozenset({
    LayerKind.SIGMOID.value,
    LayerKind.TANH.value,
})

# ReLU-family activations (canonical BaBSR target).
_RELU_KINDS = frozenset({
    LayerKind.RELU.value,
    LayerKind.LRELU.value,
    "LEAKY_RELU",  # alias present in DualTF registry
})


def _reverse_topological_sort(net: Net) -> List[int]:
    in_deg: Dict[int, int] = {
        layer.id: len(net.succs.get(layer.id, [])) for layer in net.layers
    }
    queue: List[int] = [lid for lid, degree in in_deg.items() if degree == 0]
    order: List[int] = []
    while queue:
        lid = queue.pop(0)
        order.append(lid)
        for pred in net.preds.get(lid, []):
            in_deg[pred] -= 1
            if in_deg[pred] == 0:
                queue.append(pred)
    if len(order) != len(net.layers):
        raise ValueError(
            f"BaBSRBranching: graph has cycle or disconnected layers "
            f"({len(order)}/{len(net.layers)} sorted)"
        )
    return order


# ---------------------------------------------------------------------------
# Bias extraction helper
# ---------------------------------------------------------------------------


def _get_pre_act_bias(
    pre_layer,
    D: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the bias vector ``b_layer`` of the pre-activation affine layer,
    broadcast/aligned to length ``D`` (the flattened layer output size).

    Shape rules
    -----------
    * DENSE(out_features=D): bias has shape [D] — returned as-is.
    * CONV2D: bias has shape [out_channels]; broadcast across spatial dims so
      that position ``c * H*W + i`` gets ``bias[c]``. Implemented via
      ``repeat_interleave`` when ``D % out_channels == 0``.
    * BIAS layer: ``c`` param is the bias vector.
    * BN layer: ``c`` param is the bias (offset) after scaling.
    * SCALE / everything else: no additive bias — returns zeros[D].

    This is a best-effort extractor. If the bias cannot be safely broadcast
    to ``D`` (e.g. unknown shape mismatch), falls back to zeros, so BaBSR
    scores degrade gracefully rather than crashing.
    """
    k = pre_layer.kind.upper() if isinstance(pre_layer.kind, str) else pre_layer.kind
    b: Optional[torch.Tensor] = None
    if k in (LayerKind.DENSE.value, LayerKind.CONV2D.value):
        val = pre_layer.params.get("bias")
        if isinstance(val, torch.Tensor):
            b = val
    elif k in (LayerKind.BIAS.value, LayerKind.BN.value):
        val = pre_layer.params.get("c")
        if isinstance(val, torch.Tensor):
            b = val
    # SCALE / ADD: no additive bias in our param convention.

    if b is None:
        return torch.zeros(D, device=device, dtype=dtype)

    b_flat = b.flatten().to(device=device, dtype=dtype)
    n = b_flat.numel()
    if n == D:
        return b_flat
    if n > D:
        return b_flat[:D]
    if D % n == 0:
        # Broadcast across trailing spatial dims (typical conv case).
        return b_flat.repeat_interleave(D // n)
    # Mismatch we can't safely broadcast — fall back to zeros.
    out = torch.zeros(D, device=device, dtype=dtype)
    out[:n] = b_flat
    return out


# ---------------------------------------------------------------------------
# Per-layer lA collector
# ---------------------------------------------------------------------------


def compute_lA_per_layer(
    net: Net,
    bounds_dict: Dict[int, Bounds],
    c: torch.Tensor,
    dual_tf: DualTF,
    target_layer_ids: Optional[List[int]] = None,
) -> Dict[int, torch.Tensor | Patches]:
    """Run ONE reverse-topo backward sweep through ``dual_tf._BACKWARD_REGISTRY``,
    collecting per-pre-activation-layer ``lA`` coefficients.

    Parameters
    ----------
    net
        ACT network.
    bounds_dict
        Forward-analysis bounds per layer id; must cover every layer the
        backward handlers touch.
    c
        Objective coefficients, shape ``[B, num_classes]``.
    dual_tf
        DualTF instance (supplies the backward handler registry).
    target_layer_ids
        If given, only snapshot ``lA`` at layer ids in this set — keeps
        memory bounded when the caller only needs a few pre-activation
        layers. If ``None``, snapshot at every visited layer.

    Returns
    -------
    dict[int, Tensor]
        ``{layer_id: lA}`` where ``lA`` is the ν coefficient accumulated on
        that layer's OUTPUT in the canonical Lagrangian lower bound
        ``f(x) >= lA^T z + ...``. Shape ``[B, *layer_shape]`` (native; no
        flattening here — caller reshapes as needed).

    Notes
    -----
    This is a *pure scoring pass* — no eta is applied and the input-box
    contribution step is skipped. Soundness is not required because BaBSR
    only needs relative scores to rank split candidates.
    """
    if c.dim() != 2:
        raise ValueError(
            f"compute_lA_per_layer: c must be 2-D [B, num_classes], got shape "
            f"{tuple(c.shape)}"
        )
    sample_bound = next(iter(bounds_dict.values()))
    if sample_bound.lb.dim() >= 3 and sample_bound.lb.stride(1) == 0:
        _warn_once(
            "compute_lA_per_layer_rank3_boundary",
            -1,
            "compute_lA_per_layer: materializing rank-3 bounds at BaBSR boundary",
        )
        bounds_dict = {
            lid: Bounds(
                lb=bounds.lb.contiguous().reshape(bounds.lb.shape[0] * bounds.lb.shape[1], *bounds.lb.shape[2:]),
                ub=bounds.ub.contiguous().reshape(bounds.ub.shape[0] * bounds.ub.shape[1], *bounds.ub.shape[2:]),
            )
            for lid, bounds in bounds_dict.items()
        }

    # Locate ASSERT layer → its unique predecessor is the output layer.
    output_lid: Optional[int] = None
    for layer in net.layers:
        k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
        if k == LayerKind.ASSERT.value:
            preds = net.preds.get(layer.id, [])
            if len(preds) != 1:
                raise ValueError(
                    f"compute_lA_per_layer: ASSERT layer {layer.id} must have "
                    f"exactly 1 predecessor, got {len(preds)}"
                )
            output_lid = preds[0]
            break
    if output_lid is None:
        raise ValueError("compute_lA_per_layer: net has no ASSERT layer")

    nu_accum: Dict[int, torch.Tensor | Patches] = {output_lid: c.clone()}
    topo_order = _reverse_topological_sort(net)
    registry = dual_tf._BACKWARD_REGISTRY
    target_set: Optional[set[int]] = (
        set(target_layer_ids) if target_layer_ids is not None else None
    )

    lA: Dict[int, torch.Tensor | Patches] = {}

    for lid in topo_order:
        layer = net.by_id[lid]
        k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind

        # Skip wrapper / terminal layers (no handler contribution to record).
        if k in (
            LayerKind.INPUT.value,
            LayerKind.INPUT_SPEC.value,
            LayerKind.ASSERT.value,
        ):
            continue

        if lid not in nu_accum:
            continue

        # Snapshot BEFORE popping — this is the coefficient on THIS layer's
        # output z used by BaBSR scoring. Detach to keep the scoring pass
        # gradient-free.
        if target_set is None or lid in target_set:
            current = nu_accum[lid]
            snapshot = current.detach() if isinstance(current, Patches) else current.detach().clone()
            if (
                isinstance(snapshot, torch.Tensor)
                and get_conv_mode() == "patches"
                and k == LayerKind.CONV2D.value
            ):
                output_shape = layer.params.get("output_shape")
                if isinstance(output_shape, (tuple, list)):
                    lA[lid] = linear_form_tensor_to_patches(
                        snapshot.reshape(snapshot.shape[0], -1),
                        tuple(int(dim) for dim in output_shape),
                    )
                else:
                    lA[lid] = snapshot
            else:
                lA[lid] = snapshot

        nu_here = nu_accum.pop(lid)
        if isinstance(nu_here, Patches) and k != LayerKind.CONV2D.value:
            output_shape = layer.params.get("output_shape")
            if isinstance(output_shape, (tuple, list)):
                nu_here = linear_form_patches_to_tensor(
                    nu_here,
                    tuple(int(dim) for dim in output_shape),
                )
            else:
                nu_here = linear_form_patches_to_tensor(
                    nu_here,
                    tuple(int(dim) for dim in bounds_dict[lid].lb.shape),
                )
        handler = registry.get(k)
        if handler is None:
            raise ValueError(
                f"compute_lA_per_layer: unknown layer kind '{k}' at layer {lid}; "
                f"supported: {sorted(registry.keys())}"
            )

        preds = list(net.preds.get(lid, []))
        if k == LayerKind.CONV2D.value:
            conv_handler = cast(
                Callable[[Layer, torch.Tensor | Patches, Dict[int, Bounds], List[int]], tuple[list[torch.Tensor | Patches], torch.Tensor]],
                handler,
            )
            pred_nus, _contrib = conv_handler(layer, nu_here, bounds_dict, preds)
        else:
            dense_handler = cast(
                Callable[[Layer, torch.Tensor, Dict[int, Bounds], List[int]], tuple[list[torch.Tensor], torch.Tensor]],
                handler,
            )
            pred_nus, _contrib = dense_handler(layer, cast(torch.Tensor, nu_here), bounds_dict, preds)
        if len(pred_nus) != len(preds):
            raise ValueError(
                f"compute_lA_per_layer: handler {k} at layer {lid} returned "
                f"{len(pred_nus)} pred_nus, expected {len(preds)}"
            )

        for pred_id, pred_nu in zip(preds, pred_nus):
            if pred_id in nu_accum:
                current_nu = nu_accum[pred_id]
                if isinstance(current_nu, torch.Tensor) and isinstance(pred_nu, torch.Tensor):
                    nu_accum[pred_id] = current_nu + pred_nu
                else:
                    pred_bounds = bounds_dict[pred_id]
                    feature_shape = tuple(int(dim) for dim in (
                        current_nu.input_shape if isinstance(current_nu, Patches) and current_nu.input_shape is not None
                        else pred_nu.input_shape if isinstance(pred_nu, Patches) and pred_nu.input_shape is not None
                        else pred_bounds.lb.shape
                    ))
                    lhs = (
                        linear_form_patches_to_tensor(current_nu, feature_shape)
                        if isinstance(current_nu, Patches)
                        else current_nu.reshape(current_nu.shape[0], -1)
                    )
                    rhs = (
                        linear_form_patches_to_tensor(pred_nu, feature_shape)
                        if isinstance(pred_nu, Patches)
                        else pred_nu.reshape(pred_nu.shape[0], -1)
                    )
                    nu_accum[pred_id] = lhs + rhs
            else:
                nu_accum[pred_id] = pred_nu.detach() if isinstance(pred_nu, Patches) else pred_nu.clone()

    return lA


def _lA_to_score_tensor(
    layer_id: int,
    value: torch.Tensor | Patches,
    bounds: Bounds,
    pre_layer: Layer,
    batch_size: int,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.reshape(batch_size, -1)

    output_shape = pre_layer.params.get("output_shape")
    if not isinstance(output_shape, (tuple, list)):
        if bounds.lb.dim() == 4:
            input_shape = tuple(int(dim) for dim in bounds.lb.shape)
        else:
            raise ValueError(
                f"BaBSR select_neurons: layer {layer_id} needs output_shape metadata to materialize Patches lA"
            )
    else:
        input_shape = tuple(int(dim) for dim in output_shape)
    dense = value.to_matrix(input_shape)
    _record_lA_materialization(layer_id)
    if dense.ndim != 3:
        raise ValueError(
            f"BaBSR select_neurons: expected dense lA [B, rows, D], got {tuple(dense.shape)}"
        )
    if dense.shape[1] == 1:
        return dense[:, 0, :]
    return dense.reshape(batch_size, -1)


# ---------------------------------------------------------------------------
# BaBSRBranching
# ---------------------------------------------------------------------------


class BaBSRBranching(BranchingStrategy):
    """Canonical BaBSR neuron-selection strategy for batched BaB.

    Given a batch of subproblems (all sharing the same ACT ``net`` but with
    per-row pre-activation bounds in ``bounds_dict``), returns one split
    decision per row: ``(pre_act_layer_id, neuron_idx, kind)`` where
    ``kind ∈ {'relu', 'smooth', 'none'}``.

    ReLU scoring (canonical beta-CROWN / BaBSR)::

        slope_ratio    = u / (u - l + eps)
        intercept      = -l * u / (u - l + eps)
        intercept_term = clamp(lA, max=0) * intercept
        bias_cand_1    = b_layer * (slope_ratio - 1)
        bias_cand_2    = b_layer * slope_ratio
        bias_term      = max(bias_cand_1, bias_cand_2)    # AND-clause (α-β-CROWN default)
        score          = |bias_term + intercept_term| * amb_mask

    where ``lA`` comes from a single reverse-topo backward sweep with
    ``c = spec_c`` (collected by :func:`compute_lA_per_layer`).

    Smooth activations (v1 best-effort): scored by pre-activation width
    ``u - l``, masked to neurons with ``u - l > SMOOTH_SPLIT_MIN_GAP``.
    Extending smooth scoring to the full canonical formula is future work
    (MUST DO #7 of Phase 3).
    """

    # Tunables (per Phase 3 plan).
    SCORE_THRESHOLD: float = 1e-4
    SMOOTH_SPLIT_MIN_GAP: float = 1e-2
    EPS: float = 1e-12


    # -- BranchingStrategy ABC shim ----------------------------------------

    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Not used by BaBSR — it ranks neurons across MULTIPLE layers and
        returns decisions directly via :meth:`select_neurons`, rather than
        a per-input-dimension score matrix. Raises to surface mis-wired
        callers loudly.
        """
        raise NotImplementedError(
            "BaBSRBranching uses select_neurons(), not compute_scores(). "
            "The input-dim scoring API is not applicable to neuron-split BaB."
        )

    # -- Primary API --------------------------------------------------------

    def select_neurons(
        self,
        net: Net,
        batch: SubproblemBatch,
        bounds_dict: Dict[int, Bounds],
        spec_c: torch.Tensor,
        num_classes: int,
        dual_solver=None,
    ) -> List[Tuple[int, int, str]]:
        """Return per-row ``(pre_act_layer_id, neuron_idx, kind)``.

        Args
        ----
        net
            ACT network (shared across the batch).
        batch
            ``SubproblemBatch`` with ``lb/ub/depths``; may also carry an
            optional ``.eta`` attribute (``EtaState``) whose ``sign`` tensor
            encodes previously-split ReLU neurons (sign != 0 ⇒ already split).
        bounds_dict
            Pre-activation bounds keyed by layer id (one entry per affine
            producer at least). Shape ``[B, *layer_shape]``.
        spec_c
            Objective coefficients ``[B, num_classes]``.
        num_classes
            Kept for API parity; inferred from ``spec_c.shape[-1]``.
        dual_solver
            Optional ``DualSolver``; if given its TF is reused so handler
            registries are shared. Otherwise a fresh ``DualTF()`` is built.

        Returns
        -------
        list[tuple[int, int, str]]
            One decision per row. ``(-1, -1, 'none')`` signals "nothing
            splittable" — caller should close the subproblem.
        """
        del num_classes

        B = batch.batch_size

        activations: List[Tuple[int, str]] = []
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k in _RELU_KINDS:
                kind = "relu"
            elif k in _SMOOTH_KINDS:
                kind = "smooth"
            else:
                continue
            try:
                pre_lid = get_pre_activation_layer_id(net, layer.id)
            except ValueError:
                continue
            activations.append((pre_lid, kind))

        if not activations:
            return [(-1, -1, "none")] * B

        pre_act_ids = [pre for pre, _ in activations]
        if dual_solver is not None and hasattr(dual_solver, "tf"):
            dual_tf = dual_solver.tf
        else:
            dual_tf = DualTF()
        lA_dict = compute_lA_per_layer(
            net, bounds_dict, spec_c, dual_tf, target_layer_ids=pre_act_ids,
        )

        # Concatenate in net-layer order so tiebreak (argmax = first index)
        # naturally prefers the lowest (layer_id, neuron_idx).
        scores_parts: List[torch.Tensor] = []
        backup_parts: List[torch.Tensor] = []
        col_layer_ids: List[int] = []
        col_neuron_idx: List[int] = []
        col_kinds: List[str] = []

        eta = getattr(batch, "eta", None)

        for pre_lid, kind in activations:
            if pre_lid not in lA_dict or pre_lid not in bounds_dict:
                continue

            pre_layer = net.by_id[pre_lid]
            lA = _lA_to_score_tensor(pre_lid, lA_dict[pre_lid], bounds_dict[pre_lid], pre_layer, B)
            bnd = bounds_dict[pre_lid]
            lb_t, ub_t = bnd.lb, bnd.ub
            if lb_t.dim() < 2:
                lb_flat = lb_t.flatten().unsqueeze(0).expand(B, -1)
                ub_flat = ub_t.flatten().unsqueeze(0).expand(B, -1)
            else:
                lb_flat = lb_t.reshape(B, -1)
                ub_flat = ub_t.reshape(B, -1)

            D = min(lA.shape[-1], lb_flat.shape[-1])
            if D <= 0:
                continue
            lA = lA[..., :D].contiguous()
            lb = lb_flat[..., :D].contiguous()
            ub = ub_flat[..., :D].contiguous()

            device, dtype = lA.device, lA.dtype
            lb = lb.to(device=device, dtype=dtype)
            ub = ub.to(device=device, dtype=dtype)

            eps = self.EPS
            denom = (ub - lb).clamp(min=eps)
            slope_ratio = ub / denom
            intercept = -lb * ub / denom
            intercept_term = lA.clamp(max=0) * intercept

            # Bias term via AND-clause reduction. Uses ``max`` to match
            # α-β-CROWN's canonical default (``branching_reduceop='max'``):
            # picking the larger of the two candidate bias terms selects
            # neurons whose split weakens the worse child more aggressively,
            # producing a tighter overall BaB tree. (``min`` is an alternate
            # reduction used in some BFS variants but empirically selects
            # lower-impact neurons on ResNet-scale networks.)
            b_layer = _get_pre_act_bias(pre_layer, D, device=device, dtype=dtype)
            b_layer_b = b_layer.unsqueeze(0).expand(B, -1)
            bias_cand_1 = b_layer_b * (slope_ratio - 1.0)
            bias_cand_2 = b_layer_b * slope_ratio
            bias_term = torch.maximum(bias_cand_1, bias_cand_2)

            if kind == "relu":
                amb_mask = (lb < 0) & (ub > 0)
            else:
                amb_mask = (ub - lb) > self.SMOOTH_SPLIT_MIN_GAP

            if eta is not None:
                sign_map = getattr(eta, "sign", None)
                if sign_map is not None and pre_lid in sign_map:
                    sign_t = sign_map[pre_lid]
                    if sign_t.dim() < 2:
                        sign_t = sign_t.unsqueeze(0).expand(B, -1)
                    else:
                        sign_t = sign_t.reshape(B, -1)
                    sign_t = sign_t[..., :D].to(device=device)
                    if kind == "relu":
                        amb_mask = amb_mask & (sign_t == 0)

            amb_mask_f = amb_mask.to(dtype)

            if kind == "relu":
                score = (bias_term + intercept_term).abs() * amb_mask_f
                backup = intercept_term.abs() * amb_mask_f
            else:
                width = (ub - lb)
                score = width * amb_mask_f
                backup = score

            scores_parts.append(score)
            backup_parts.append(backup)
            col_layer_ids.extend([pre_lid] * D)
            col_neuron_idx.extend(range(D))
            col_kinds.extend([kind] * D)

        if not scores_parts:
            return [(-1, -1, "none")] * B

        scores_big = torch.cat(scores_parts, dim=-1)
        backup_big = torch.cat(backup_parts, dim=-1)

        top_scores, top_idx = scores_big.max(dim=-1)

        results: List[Tuple[int, int, str]] = []
        for b in range(B):
            if top_scores[b].item() > self.SCORE_THRESHOLD:
                col = int(top_idx[b].item())
                results.append(
                    (col_layer_ids[col], col_neuron_idx[col], col_kinds[col])
                )
                continue
            b_scores, b_idx = backup_big[b].max(dim=-1)
            if b_scores.item() > 0.0:
                col = int(b_idx.item())
                results.append(
                    (col_layer_ids[col], col_neuron_idx[col], col_kinds[col])
                )
            else:
                results.append((-1, -1, "none"))
        return results


atexit.register(emit_materialization_summary, log)
