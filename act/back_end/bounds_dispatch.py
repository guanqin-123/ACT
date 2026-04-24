from __future__ import annotations

"""Shared bounds dispatcher and rank-3 expansion helpers.

All consumers of bounds MUST route through dispatch_* functions. Direct isinstance
checks or tensor-manipulation bypassing dispatch is a bug. Wave 2a fans out these dispatches
to every consumer. Wave 3-4 fills the Patches branches.
"""

from collections.abc import Mapping, Sequence
from typing import cast

import torch

from act.back_end.bab.eta import EtaState
from act.back_end.core import Bounds, Layer
from act.back_end.dual_tf.tf_cnn import (
    forward_avgpool2d,
    forward_conv2d,
    forward_maxpool2d,
)
from act.back_end.dual_tf.tf_forward import Frame, LinearBound, forward_add
from act.back_end.dual_tf.tf_mlp import backward_relu, forward_bn
from act.back_end.layer_schema import LayerKind
from act.back_end.patches import Patches


BoundsRep = LinearBound | Patches
ForwardDispatchResult = tuple[Bounds, Bounds, LinearBound, Frame]
ForwardParentFrames = list[Frame]
ForwardParentBoxes = list[Bounds]
ForwardParentLins = list[LinearBound]
DispatchParentLins = Sequence[BoundsRep]


def is_rank3_view(x: object) -> bool:
    return isinstance(x, torch.Tensor) and x.dim() == 3 and x.stride(0) == 0


def _contains_patches(value: object) -> bool:
    if isinstance(value, Patches):
        return True
    if isinstance(value, Mapping):
        return any(_contains_patches(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_patches(item) for item in value)
    return False


def _has_zero_stride_axis(tensor: torch.Tensor) -> bool:
    return any(stride == 0 for stride in tensor.stride())


def expand_rank3(
    bounds_dict: dict[int, LinearBound] | EtaState,
    M: int,
) -> dict[int, LinearBound]:
    """Expand LinearBound tensors with a stride-0 spec axis.

    Zero-copy expand via stride-0 view. Writes to the returned tensor will corrupt
    all M copies. Treat as read-only.
    """
    if isinstance(bounds_dict, EtaState):
        raise NotImplementedError(
            "expand_rank3 only applies to LinearBound entries; EtaState must remain dense."
        )
    if M <= 0:
        raise ValueError(f"expand_rank3: M must be positive, got {M}")
    if M == 1:
        return dict(bounds_dict)

    expanded: dict[int, LinearBound] = {}
    for lid, bounds in bounds_dict.items():
        A_lb = bounds.A_lb.unsqueeze(1).expand(-1, M, *([-1] * (bounds.A_lb.dim() - 1)))
        A_ub = bounds.A_ub.unsqueeze(1).expand(-1, M, *([-1] * (bounds.A_ub.dim() - 1)))
        b_lb = bounds.b_lb.unsqueeze(1).expand(-1, M, *([-1] * (bounds.b_lb.dim() - 1)))
        b_ub = bounds.b_ub.unsqueeze(1).expand(-1, M, *([-1] * (bounds.b_ub.dim() - 1)))

        assert A_lb.stride(-3) == 0, "expand_rank3: A_lb spec axis must be stride-0"
        assert A_ub.stride(-3) == 0, "expand_rank3: A_ub spec axis must be stride-0"
        assert b_lb.stride(1) == 0, "expand_rank3: b_lb spec axis must be stride-0"
        assert b_ub.stride(1) == 0, "expand_rank3: b_ub spec axis must be stride-0"

        expanded[lid] = LinearBound(A_lb=A_lb, b_lb=b_lb, A_ub=A_ub, b_ub=b_ub)
    return expanded


def materialize_if_needed(bounds: LinearBound) -> LinearBound:
    """Materialize expanded stride-0 views for writable consumers."""
    tensors = (bounds.A_lb, bounds.b_lb, bounds.A_ub, bounds.b_ub)
    needs_materialization = any(
        is_rank3_view(tensor) or _has_zero_stride_axis(tensor) for tensor in tensors
    )
    if not needs_materialization:
        return bounds
    return LinearBound(
        A_lb=bounds.A_lb.contiguous(),
        b_lb=bounds.b_lb.contiguous(),
        A_ub=bounds.A_ub.contiguous(),
        b_ub=bounds.b_ub.contiguous(),
    )


def dispatch_conv_forward(
    layer: Layer,
    parent_boxes: ForwardParentBoxes,
    parent_lins: DispatchParentLins,
    parent_frames: ForwardParentFrames,
    preds: list[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> ForwardDispatchResult:
    if _contains_patches(parent_lins):
        raise NotImplementedError("Filled in W3 — conv patches forward")
    return forward_conv2d(
        layer,
        parent_boxes,
        cast(ForwardParentLins, parent_lins),
        parent_frames,
        preds,
        post_activation,
        device,
        dtype,
    )


def dispatch_relu_backward(
    layer: object,
    nu: torch.Tensor | Patches,
    bounds_dict: object,
    preds: list[int],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    if isinstance(nu, Patches) or _contains_patches(bounds_dict):
        raise NotImplementedError("Filled in W3/W4")
    return backward_relu(layer, nu, cast(dict[int, Bounds], bounds_dict), preds)


def dispatch_bn_forward(
    layer: object,
    parent_boxes: ForwardParentBoxes,
    parent_lins: DispatchParentLins,
    parent_frames: ForwardParentFrames,
    preds: list[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> ForwardDispatchResult:
    if _contains_patches(parent_lins):
        raise NotImplementedError("Filled in W3/W4")
    return forward_bn(
        layer,
        parent_boxes,
        cast(ForwardParentLins, parent_lins),
        parent_frames,
        preds,
        post_activation,
        device,
        dtype,
    )


def dispatch_add_forward(
    layer: Layer,
    parent_boxes: ForwardParentBoxes,
    parent_lins: DispatchParentLins,
    parent_frames: ForwardParentFrames,
    preds: list[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> ForwardDispatchResult:
    if _contains_patches(parent_lins):
        raise NotImplementedError("Filled in W3/W4")
    return forward_add(
        layer,
        parent_boxes,
        cast(ForwardParentLins, parent_lins),
        parent_frames,
        preds,
        post_activation,
        device,
        dtype,
    )


def dispatch_pool_forward(
    layer: Layer,
    parent_boxes: ForwardParentBoxes,
    parent_lins: DispatchParentLins,
    parent_frames: ForwardParentFrames,
    preds: list[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> ForwardDispatchResult:
    if _contains_patches(parent_lins):
        raise NotImplementedError("Filled in W3/W4")
    if layer.kind == LayerKind.MAXPOOL2D.value:
        return forward_maxpool2d(
            layer,
            parent_boxes,
            cast(ForwardParentLins, list(parent_lins)),
            parent_frames,
            preds,
            post_activation,
            device,
            dtype,
        )
    if layer.kind == LayerKind.AVGPOOL2D.value:
        return forward_avgpool2d(
            layer,
            parent_boxes,
            cast(ForwardParentLins, list(parent_lins)),
            parent_frames,
            preds,
            post_activation,
            device,
            dtype,
        )
    raise ValueError(f"dispatch_pool_forward: unsupported pool kind {layer.kind!r}")


__all__ = [
    "BoundsRep",
    "dispatch_add_forward",
    "dispatch_bn_forward",
    "dispatch_conv_forward",
    "dispatch_pool_forward",
    "dispatch_relu_backward",
    "expand_rank3",
    "is_rank3_view",
    "materialize_if_needed",
]
