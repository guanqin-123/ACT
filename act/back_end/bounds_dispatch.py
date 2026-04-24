from __future__ import annotations

# pyright: reportImportCycles=false

"""Shared bounds dispatcher and rank-3 expansion helpers.

All consumers of bounds MUST route through dispatch_* functions. Direct isinstance
checks or tensor-manipulation bypassing dispatch is a bug. Wave 2a fans out these dispatches
to every consumer. Wave 3-4 fills the Patches branches.
"""

import importlib
import logging
from collections.abc import Mapping, Sequence
from typing import Protocol, TypeVar, cast, overload

import torch

from act.back_end.bab.eta import EtaState
from act.back_end.config import BackendConfig
from act.back_end.core import Bounds, Layer
from act.back_end.layer_schema import LayerKind
from act.back_end.patches import Patches


log = logging.getLogger(__name__)

_VALID_CONV_MODES = {"matrix", "patches"}
_conv_mode = BackendConfig.from_yaml().conv_mode
_strict_patches = False
_conv_materialization_count = 0


class LinearBoundLike(Protocol):
    A_lb: torch.Tensor
    b_lb: torch.Tensor
    A_ub: torch.Tensor
    b_ub: torch.Tensor


FrameLike = tuple[torch.Tensor, torch.Tensor]


LinearBoundT = TypeVar("LinearBoundT", bound=LinearBoundLike)


def get_conv_mode() -> str:
    return _conv_mode


def set_conv_mode(mode: str) -> None:
    if mode not in _VALID_CONV_MODES:
        raise ValueError(f"Invalid conv_mode {mode!r}; expected one of {_VALID_CONV_MODES}")
    global _conv_mode
    _conv_mode = mode


def get_strict_patches() -> bool:
    return _strict_patches


def set_strict_patches(enabled: bool) -> None:
    global _strict_patches
    _strict_patches = bool(enabled)


def get_conv_materialization_count() -> int:
    return _conv_materialization_count


def reset_conv_materialization_count() -> None:
    global _conv_materialization_count
    _conv_materialization_count = 0


def _record_conv_materialization(message: str) -> None:
    global _conv_materialization_count
    _conv_materialization_count += 1
    if get_strict_patches():
        raise RuntimeError(message)


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


def _expand_tensor_rank3(tensor: torch.Tensor, M: int) -> torch.Tensor:
    return tensor.unsqueeze(1).expand(-1, M, *([-1] * (tensor.dim() - 1)))


@overload
def expand_rank3(bounds_dict: Mapping[int, Bounds], M: int) -> dict[int, Bounds]: ...


@overload
def expand_rank3(bounds_dict: Mapping[int, LinearBoundT], M: int) -> Mapping[int, LinearBoundT]: ...


@overload
def expand_rank3(bounds_dict: EtaState, M: int) -> Mapping[int, LinearBoundLike]: ...


def expand_rank3(
    bounds_dict: Mapping[int, Bounds] | Mapping[int, LinearBoundLike] | EtaState,
    M: int,
) -> Mapping[int, Bounds] | Mapping[int, LinearBoundLike]:
    """Expand Bounds / LinearBound tensors with a stride-0 spec axis.

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
        if not bounds_dict:
            return {}
        first_value = next(iter(bounds_dict.values()))
        if isinstance(first_value, Bounds):
            return {lid: cast(Bounds, value) for lid, value in bounds_dict.items()}
        return {lid: cast(LinearBoundLike, value) for lid, value in bounds_dict.items()}

    from act.back_end.dual_tf.tf_forward import LinearBound

    expanded_bounds: dict[int, Bounds] = {}
    expanded_linear: dict[int, LinearBound] = {}
    is_linear = False
    for lid, bounds in bounds_dict.items():
        if isinstance(bounds, Bounds):
            lb = _expand_tensor_rank3(bounds.lb, M)
            ub = _expand_tensor_rank3(bounds.ub, M)
            assert lb.stride(1) == 0, "expand_rank3: lb spec axis must be stride-0"
            assert ub.stride(1) == 0, "expand_rank3: ub spec axis must be stride-0"
            expanded_bounds[lid] = Bounds(lb=lb, ub=ub)
            continue

        is_linear = True
        linear = cast(LinearBoundLike, bounds)
        A_lb = _expand_tensor_rank3(linear.A_lb, M)
        A_ub = _expand_tensor_rank3(linear.A_ub, M)
        b_lb = _expand_tensor_rank3(linear.b_lb, M)
        b_ub = _expand_tensor_rank3(linear.b_ub, M)

        assert A_lb.stride(-3) == 0, "expand_rank3: A_lb spec axis must be stride-0"
        assert A_ub.stride(-3) == 0, "expand_rank3: A_ub spec axis must be stride-0"
        assert b_lb.stride(1) == 0, "expand_rank3: b_lb spec axis must be stride-0"
        assert b_ub.stride(1) == 0, "expand_rank3: b_ub spec axis must be stride-0"

        expanded_linear[lid] = LinearBound(A_lb=A_lb, b_lb=b_lb, A_ub=A_ub, b_ub=b_ub)
    return expanded_linear if is_linear else expanded_bounds


def materialize_if_needed(bounds: LinearBoundLike) -> LinearBoundLike:
    """Materialize expanded stride-0 views for writable consumers."""
    tf_forward = importlib.import_module("act.back_end.dual_tf.tf_forward")

    tensors = (bounds.A_lb, bounds.b_lb, bounds.A_ub, bounds.b_ub)
    needs_materialization = any(
        is_rank3_view(tensor) or _has_zero_stride_axis(tensor) for tensor in tensors
    )
    if not needs_materialization:
        return bounds
    return tf_forward.LinearBound(
        A_lb=bounds.A_lb.contiguous(),
        b_lb=bounds.b_lb.contiguous(),
        A_ub=bounds.A_ub.contiguous(),
        b_ub=bounds.b_ub.contiguous(),
    )


def dispatch_conv_forward(
    layer: Layer,
    parent_boxes: list[Bounds],
    parent_lins: Sequence[LinearBoundLike | Patches],
    parent_frames: Sequence[FrameLike],
    preds: list[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Bounds, Bounds, LinearBoundLike | Patches, FrameLike]:
    tf_cnn = importlib.import_module("act.back_end.dual_tf.tf_cnn")
    tf_cnn_patches = importlib.import_module("act.back_end.dual_tf.tf_cnn_patches")

    conv_mode = get_conv_mode()
    if conv_mode == "patches":
        if _contains_patches(parent_lins):
            return tf_cnn_patches.forward_conv2d_patches(
                layer,
                parent_boxes,
                list(parent_lins),
                parent_frames,
                preds,
                post_activation,
                device,
                dtype,
            )
        message = (
            "dispatch_conv_forward: conv_mode=patches received LinearBound input; "
            "materializing and falling back to matrix path"
        )
        log.warning(message)
        _record_conv_materialization(message)
        materialized = [materialize_if_needed(cast(LinearBoundLike, parent_lins[0]))]
        return tf_cnn.forward_conv2d(
            layer,
            parent_boxes,
            cast(list[object], materialized),
            parent_frames,
            preds,
            post_activation,
            device,
            dtype,
        )

    if _contains_patches(parent_lins):
        log.warning(
            "dispatch_conv_forward: conv_mode=matrix received Patches input; converting to dense LinearBound"
        )
        input_shape = layer.params.get("input_shape")
        converted = [
            tf_cnn_patches._patches_to_linear_bound(
                cast(Patches, parent_lins[0]),
                cast(tuple[int, int, int, int], cast(Patches, parent_lins[0]).input_shape or input_shape),
            )
        ]
        return tf_cnn.forward_conv2d(
            layer,
            parent_boxes,
            cast(list[object], converted),
            parent_frames,
            preds,
            post_activation,
            device,
            dtype,
        )

    return tf_cnn.forward_conv2d(
        layer,
        parent_boxes,
        cast(list[object], list(parent_lins)),
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
) -> tuple[list[torch.Tensor | Patches], torch.Tensor]:
    tf_mlp = importlib.import_module("act.back_end.dual_tf.tf_mlp")
    return tf_mlp.backward_relu(layer, nu, cast(dict[int, Bounds], bounds_dict), preds)


def dispatch_bn_forward(
    layer: object,
    parent_boxes: list[Bounds],
    parent_lins: Sequence[LinearBoundLike | Patches],
    parent_frames: Sequence[FrameLike],
    preds: list[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
 ) -> tuple[Bounds, Bounds, LinearBoundLike | Patches, FrameLike]:
    tf_mlp = importlib.import_module("act.back_end.dual_tf.tf_mlp")

    return tf_mlp.forward_bn(
        layer,
        parent_boxes,
        cast(list[object], list(parent_lins)),
        parent_frames,
        preds,
        post_activation,
        device,
        dtype,
    )


def dispatch_add_forward(
    layer: Layer,
    parent_boxes: list[Bounds],
    parent_lins: Sequence[LinearBoundLike | Patches],
    parent_frames: Sequence[FrameLike],
    preds: list[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
 ) -> tuple[Bounds, Bounds, LinearBoundLike | Patches, FrameLike]:
    tf_forward = importlib.import_module("act.back_end.dual_tf.tf_forward")

    return tf_forward.forward_add(
        layer,
        parent_boxes,
        cast(list[object], list(parent_lins)),
        parent_frames,
        preds,
        post_activation,
        device,
        dtype,
    )


def dispatch_concat_forward(
    layer: Layer,
    parent_boxes: list[Bounds],
    parent_lins: Sequence[LinearBoundLike | Patches],
    parent_frames: Sequence[FrameLike],
    preds: list[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
 ) -> tuple[Bounds, Bounds, LinearBoundLike | Patches, FrameLike]:
    tf_forward = importlib.import_module("act.back_end.dual_tf.tf_forward")

    return tf_forward.forward_concat(
        layer,
        parent_boxes,
        cast(list[object], list(parent_lins)),
        parent_frames,
        preds,
        post_activation,
        device,
        dtype,
    )


def dispatch_pool_forward(
    layer: Layer,
    parent_boxes: list[Bounds],
    parent_lins: Sequence[LinearBoundLike | Patches],
    parent_frames: Sequence[FrameLike],
    preds: list[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
 ) -> tuple[Bounds, Bounds, LinearBoundLike | Patches, FrameLike]:
    tf_cnn = importlib.import_module("act.back_end.dual_tf.tf_cnn")

    if layer.kind == LayerKind.MAXPOOL2D.value:
        return tf_cnn.forward_maxpool2d(
            layer,
            parent_boxes,
            cast(list[object], list(parent_lins)),
            parent_frames,
            preds,
            post_activation,
            device,
            dtype,
        )
    if layer.kind == LayerKind.AVGPOOL2D.value:
        return tf_cnn.forward_avgpool2d(
            layer,
            parent_boxes,
            cast(list[object], list(parent_lins)),
            parent_frames,
            preds,
            post_activation,
            device,
            dtype,
        )
    raise ValueError(f"dispatch_pool_forward: unsupported pool kind {layer.kind!r}")


__all__ = [
    "dispatch_add_forward",
    "dispatch_bn_forward",
    "dispatch_concat_forward",
    "dispatch_conv_forward",
    "dispatch_pool_forward",
    "dispatch_relu_backward",
    "expand_rank3",
    "get_conv_materialization_count",
    "is_rank3_view",
    "materialize_if_needed",
    "get_strict_patches",
    "reset_conv_materialization_count",
    "set_strict_patches",
]
