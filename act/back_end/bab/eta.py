"""Helpers for the eta (Lagrangian penalty) machinery used by batched BaB.

This file intentionally starts SMALL: only the pre-activation layer id mapping.
The EtaState dataclass and expand_eta_state helper are added in Phase 1.1 and
Phase 1.5 respectively.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from act.back_end.core import Net
from act.back_end.layer_schema import LayerKind
from act.util.device_manager import get_default_device, get_default_dtype

# Kinds whose OUTPUT can be the pre-activation `z` fed into a subsequent
# activation (ReLU / sigmoid / tanh / leaky_relu).
_PRE_ACTIVATION_KINDS = frozenset({
    LayerKind.DENSE.value,
    LayerKind.CONV2D.value,
    LayerKind.BIAS.value,
    LayerKind.SCALE.value,
    LayerKind.BN.value,
})

_ACTIVATION_KINDS = frozenset({
    LayerKind.RELU.value,
    LayerKind.LRELU.value,
    LayerKind.SIGMOID.value,
    LayerKind.TANH.value,
})


def get_pre_activation_layer_id(net: Net, activation_layer_id: int) -> int:
    """Return the layer id whose OUTPUT is the pre-activation value
    `z` fed into the activation at `activation_layer_id`.

    Convention: eta / beta penalty terms key on the AFFINE layer producing
    the pre-activation (DENSE / CONV2D / BIAS / SCALE / BN), not on the
    activation layer itself. The backward ν-modification hook in
    ``solver_dual.py`` is applied at that affine layer's output.

    Raises:
      ValueError: if the activation layer has != 1 predecessor, or if the
        single predecessor is not in ``_PRE_ACTIVATION_KINDS``. Multi-pred
        cases (ADD, CONCAT) are unsupported for splitting in v1.
    """
    try:
        layer = net.by_id[activation_layer_id]
    except Exception as e:
        raise ValueError(f"Activation layer id {activation_layer_id} not found in net") from e

    kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
    act_kinds = {k if isinstance(k, str) else k.value for k in _ACTIVATION_KINDS}
    if kind not in act_kinds:
        raise ValueError(
            f"Layer {activation_layer_id} has kind={kind!r}; expected an "
            f"activation kind in {_ACTIVATION_KINDS}.")

    preds = list(net.preds.get(activation_layer_id, []))
    if len(preds) != 1:
        raise ValueError(
            f"Layer {activation_layer_id} ({kind}) has {len(preds)} predecessors; "
            f"get_pre_activation_layer_id requires exactly 1. Multi-predecessor "
            f"cases (ADD/CONCAT) are not supported for splitting in v1.")

    pred_id = preds[0]
    pred_layer = net.by_id[pred_id]
    pred_kind = pred_layer.kind.upper() if isinstance(pred_layer.kind, str) else pred_layer.kind
    pre_kinds = {k if isinstance(k, str) else k.value for k in _PRE_ACTIVATION_KINDS}
    if pred_kind not in pre_kinds:
        raise ValueError(
            f"Layer {activation_layer_id} ({kind}) predecessor is layer "
            f"{pred_id} (kind={pred_kind!r}); expected a pre-activation kind "
            f"in {pre_kinds}. Chains like AFFINE → BN → RELU should be "
            f"pre-flattened or the BN should own the eta slot.")

    return pred_id


@dataclass
class EtaState:
    """Per-layer per-subproblem Lagrange multipliers for split constraints.

    Shapes (all tensors dense, NO sparsity):
        val[lid]   : (B, D_layer) >= 0, the eta values (optimizable)
        sign[lid]  : (B, D_layer) in {+1.0, -1.0, 0.0}
        point[lid] : (B, D_layer)  split point (0 for ReLU, midpoint for smooth)

    Sign convention (canonical beta-CROWN, Wang et al. 2021):
        sign = +1  ->  ReLU INACTIVE side, constraint z <= point
        sign = -1  ->  ReLU ACTIVE   side, constraint z >= point
        sign =  0  ->  no split (val ignored; stays 0)

    Keys ``lid`` are the AFFINE pre-activation layer ids, as returned by
    :func:`get_pre_activation_layer_id` - NOT the activation layer ids.

    The state is mutable; callers may update ``val`` in place during Adam
    optimisation, but MUST NOT change the keys or tensor shapes once
    constructed.
    """
    val:   Dict[int, torch.Tensor] = field(default_factory=dict)
    sign:  Dict[int, torch.Tensor] = field(default_factory=dict)
    point: Dict[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.val) != set(self.sign) or set(self.val) != set(self.point):
            raise ValueError(
                f"EtaState has mismatched keys: val={set(self.val)}, "
                f"sign={set(self.sign)}, point={set(self.point)}")
        for lid in self.val:
            if self.val[lid].shape != self.sign[lid].shape:
                raise ValueError(
                    f"EtaState[{lid}] shape mismatch val={self.val[lid].shape} "
                    f"vs sign={self.sign[lid].shape}")
            if self.val[lid].shape != self.point[lid].shape:
                raise ValueError(
                    f"EtaState[{lid}] shape mismatch val={self.val[lid].shape} "
                    f"vs point={self.point[lid].shape}")

    def is_empty(self) -> bool:
        """True iff every sign tensor is all zero (no active splits)."""
        if not self.sign:
            return True
        return all((s == 0).all().item() for s in self.sign.values())

    def fast_path_skip(self) -> bool:
        """Alias for :meth:`is_empty` - used by DualSolver.compute_bound."""
        return self.is_empty()

    def to(self, device=None, dtype=None) -> "EtaState":
        """Move all tensors to a new device / dtype. Returns a new EtaState
        (tensors may still alias originals if device/dtype already match)."""
        if device is None:
            device = get_default_device()
        if dtype is None:
            dtype = get_default_dtype()
        return EtaState(
            val={k: v.to(device=device, dtype=dtype) for k, v in self.val.items()},
            sign={k: s.to(device=device, dtype=dtype) for k, s in self.sign.items()},
            point={k: p.to(device=device, dtype=dtype) for k, p in self.point.items()},
        )

    def select(self, idx: torch.Tensor) -> "EtaState":
        """Return a new EtaState with all tensors indexed by ``idx`` along dim=0.
        ``idx`` must be a 1-D long tensor. Used by BaB when selecting an
        open-subproblem subset from a larger batch."""
        if idx.dim() != 1:
            raise ValueError(f"select: idx must be 1-D, got shape {tuple(idx.shape)}")
        return EtaState(
            val={k: v.index_select(0, idx) for k, v in self.val.items()},
            sign={k: s.index_select(0, idx) for k, s in self.sign.items()},
            point={k: p.index_select(0, idx) for k, p in self.point.items()},
        )

    @property
    def batch_size(self) -> int:
        """Leading batch dimension; 0 if no layers present."""
        for v in self.val.values():
            return v.shape[0]
        return 0

    @property
    def layer_ids(self) -> tuple[int, ...]:
        """Sorted tuple of layer ids in this eta state."""
        return tuple(sorted(self.val.keys()))


def expand_eta_state(eta: Optional[EtaState], M: int) -> Optional[EtaState]:
    """Replicate every tensor in ``eta`` from ``(B, D_layer)`` to ``(B*M, D_layer)``
    via ``torch.repeat_interleave(M, dim=0)``. Aligns with
    :func:`act.back_end.solver.spec_batching.expand_bounds_dict` so that
    row ``b*M + j`` of the expanded eta corresponds to subproblem ``b``
    seen under spec row ``j``.

    Pass-through if ``eta`` is None, empty, or ``M == 1``.
    """
    if eta is None:
        return None
    if M <= 0:
        raise ValueError(f"expand_eta_state: M must be positive, got {M}")
    if M == 1 or not eta.val:
        return eta
    return EtaState(
        val={k: v.repeat_interleave(M, dim=0) for k, v in eta.val.items()},
        sign={k: s.repeat_interleave(M, dim=0) for k, s in eta.sign.items()},
        point={k: p.repeat_interleave(M, dim=0) for k, p in eta.point.items()},
    )
