from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import override

import torch
import torch.nn.functional as F


_DETERMINISTIC_PATCHES = threading.local()


def _tensor_shape_or_none(
    tensor: torch.Tensor | tuple[torch.Tensor, ...] | None,
) -> tuple[int, ...] | tuple[tuple[int, ...], ...] | None:
    if tensor is None:
        return None
    if isinstance(tensor, tuple):
        return tuple(tuple(int(dim) for dim in item.shape) for item in tensor)
    return tuple(int(dim) for dim in tensor.shape)


def _clone_tensor_like(
    value: torch.Tensor | tuple[torch.Tensor, ...] | None,
) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(item.clone() for item in value)
    return value.clone()


def _detach_tensor_like(
    value: torch.Tensor | tuple[torch.Tensor, ...] | None,
) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(item.detach() for item in value)
    return value.detach()


def _clone_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    return value.clone() if value is not None else None


def _detach_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    return value.detach() if value is not None else None


def _copy_stride(value: int | tuple[int, int]) -> int | tuple[int, int]:
    if isinstance(value, int):
        return int(value)
    return (int(value[0]), int(value[1]))


def _copy_padding(
    value: int | tuple[int, int] | tuple[int, int, int, int],
) -> int | tuple[int, int] | tuple[int, int, int, int]:
    if isinstance(value, int):
        return int(value)
    if len(value) == 2:
        return (int(value[0]), int(value[1]))
    return (int(value[0]), int(value[1]), int(value[2]), int(value[3]))


def _tensor_like_equal(
    left: torch.Tensor | tuple[torch.Tensor, ...] | None,
    right: torch.Tensor | tuple[torch.Tensor, ...] | None,
) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, tuple) or isinstance(right, tuple):
        if not isinstance(left, tuple) or not isinstance(right, tuple):
            return False
        return len(left) == len(right) and all(
            torch.equal(lhs, rhs) for lhs, rhs in zip(left, right, strict=True)
        )
    return torch.equal(left, right)


def _normalize_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    return (int(value[0]), int(value[1]))


def _normalize_padding4(
    padding: int | tuple[int, int] | tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if isinstance(padding, int):
        return (padding, padding, padding, padding)
    if len(padding) == 2:
        pad_h, pad_w = (int(item) for item in padding)
        return (pad_w, pad_w, pad_h, pad_h)
    return (
        int(padding[0]),
        int(padding[1]),
        int(padding[2]),
        int(padding[3]),
    )


def _normalize_input_shape(input_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(input_shape) == 4:
        batch_size, channels, height, width = input_shape
        return (int(batch_size), int(channels), int(height), int(width))
    if len(input_shape) == 3:
        channels, height, width = (int(dim) for dim in input_shape)
        return (1, channels, height, width)
    raise ValueError(
        f"Expected input_shape with 3 or 4 dims (B, C, H, W), got {input_shape!r}"
    )


def _canonicalize_patches_layout(
    pieces: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if pieces.ndim != 7:
        raise ValueError(
            (
                "Expected patches tensor with shape "
                "(out_c, B, out_h, out_w, in_c, k_h, k_w) or "
                "(B, out_c, out_h, out_w, in_c, k_h, k_w), got "
                f"{tuple(pieces.shape)!r}"
            )
        )
    if pieces.shape[1] == batch_size:
        return pieces
    if pieces.shape[0] == batch_size:
        return pieces.permute(1, 0, 2, 3, 4, 5, 6).contiguous()
    return pieces


@dataclass
class Patches:
    """Minimal sparse-conv scaffold for the Wave 1 dispatcher gate.

    TODO(W2b): Methods (.to_matrix, .clone, .detach, __eq__) and helpers
    (patches_to_matrix, insert_zeros, compute_patches_stride_padding,
    unify_shape, inplace_unfold) to be filled in Wave 2b.
    """

    patches: torch.Tensor | None = None
    stride: int | tuple[int, int] = 1
    padding: int | tuple[int, int] | tuple[int, int, int, int] = 0
    shape: tuple[int, ...] | None = None
    identity: int = 0
    unstable_idx: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    output_shape: tuple[int, ...] | None = None
    input_shape: tuple[int, ...] | None = None
    inserted_zeros: int = 0
    output_padding: int | tuple[int, int] | tuple[int, int, int, int] = 0

    def __post_init__(self) -> None:
        if self.inserted_zeros > 0:
            raise ValueError(
                "Patches v1 only supports inserted_zeros=0 (no dilation)."
            )
        if self.identity != 0 and self.patches is not None:
            raise ValueError(
                "Patches.identity != 0 is mutually exclusive with a patches tensor."
            )

    @property
    def is_identity(self) -> bool:
        return self.identity != 0

    @override
    def __repr__(self) -> str:
        return (
            "Patches("
            f"patches_shape={_tensor_shape_or_none(self.patches)}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"shape={self.shape}, "
            f"identity={self.identity}, "
            f"unstable_idx_shape={_tensor_shape_or_none(self.unstable_idx)}, "
            f"output_shape={self.output_shape}, "
            f"input_shape={self.input_shape}, "
            f"inserted_zeros={self.inserted_zeros}, "
            f"output_padding={self.output_padding}"
            ")"
        )

    def to_matrix(self, input_shape: tuple[int, ...]) -> torch.Tensor:
        return patches_to_matrix(self, input_shape)

    def clone(self) -> Patches:
        return Patches(
            patches=_clone_tensor(self.patches),
            stride=_copy_stride(self.stride),
            padding=_copy_padding(self.padding),
            shape=tuple(self.shape) if self.shape is not None else None,
            identity=int(self.identity),
            unstable_idx=_clone_tensor_like(self.unstable_idx),
            output_shape=(
                tuple(self.output_shape) if self.output_shape is not None else None
            ),
            input_shape=tuple(self.input_shape) if self.input_shape is not None else None,
            inserted_zeros=int(self.inserted_zeros),
            output_padding=_copy_padding(self.output_padding),
        )

    def detach(self) -> Patches:
        return Patches(
            patches=_detach_tensor(self.patches),
            stride=_copy_stride(self.stride),
            padding=_copy_padding(self.padding),
            shape=tuple(self.shape) if self.shape is not None else None,
            identity=int(self.identity),
            unstable_idx=_detach_tensor_like(self.unstable_idx),
            output_shape=(
                tuple(self.output_shape) if self.output_shape is not None else None
            ),
            input_shape=tuple(self.input_shape) if self.input_shape is not None else None,
            inserted_zeros=int(self.inserted_zeros),
            output_padding=_copy_padding(self.output_padding),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Patches):
            return False
        return (
            _tensor_like_equal(self.patches, other.patches)
            and self.stride == other.stride
            and self.padding == other.padding
            and self.shape == other.shape
            and self.identity == other.identity
            and _tensor_like_equal(self.unstable_idx, other.unstable_idx)
            and self.output_shape == other.output_shape
            and self.input_shape == other.input_shape
            and self.inserted_zeros == other.inserted_zeros
            and self.output_padding == other.output_padding
        )

    def __hash__(self) -> int:
        raise TypeError("unhashable type: 'Patches'")


def patches_to_matrix(patches: Patches, input_shape: tuple[int, ...]) -> torch.Tensor:
    if patches.inserted_zeros > 0:
        raise ValueError("dilation > 1 not supported in v1 (S1 regime)")

    batch_size, input_channels, input_height, input_width = _normalize_input_shape(
        input_shape
    )
    if patches.is_identity:
        features = input_channels * input_height * input_width
        eye = torch.eye(features, dtype=torch.get_default_dtype())
        if patches.input_shape is not None and patches.patches is not None:
            eye = eye.to(device=patches.patches.device, dtype=patches.patches.dtype)
        elif patches.patches is not None:
            eye = eye.to(device=patches.patches.device, dtype=patches.patches.dtype)
        return eye.unsqueeze(0).expand(batch_size, -1, -1).clone()

    if patches.patches is None:
        raise ValueError("Cannot materialize dense matrix when patches tensor is None")
    if patches.unstable_idx is not None:
        raise NotImplementedError("Sparse unstable_idx patches are not supported in v1")

    pieces = _canonicalize_patches_layout(patches.patches, batch_size)
    output_channels, _, output_height, output_width, patch_in_channels, k_h, k_w = (
        pieces.shape
    )
    if patch_in_channels != input_channels:
        raise ValueError(
            f"Input channel mismatch: patches expect {patch_in_channels}, got {input_channels}"
        )

    stride_h, stride_w = _normalize_pair(patches.stride)
    pad_left, pad_right, pad_top, pad_bottom = _normalize_padding4(patches.padding)
    del pad_right, pad_bottom

    dense = pieces.new_zeros(
        (batch_size, output_channels * output_height * output_width, input_channels * input_height * input_width)
    )

    for out_c in range(output_channels):
        for out_h in range(output_height):
            for out_w in range(output_width):
                row_idx = out_c * (output_height * output_width) + out_h * output_width + out_w
                patch_weights = pieces[out_c, :, out_h, out_w]
                top = out_h * stride_h - pad_top
                left = out_w * stride_w - pad_left
                for in_c in range(input_channels):
                    for kernel_h in range(k_h):
                        in_h = top + kernel_h
                        if in_h < 0 or in_h >= input_height:
                            continue
                        for kernel_w in range(k_w):
                            in_w = left + kernel_w
                            if in_w < 0 or in_w >= input_width:
                                continue
                            col_idx = (
                                in_c * (input_height * input_width)
                                + in_h * input_width
                                + in_w
                            )
                            dense[:, row_idx, col_idx] = patch_weights[
                                :, in_c, kernel_h, kernel_w
                            ]
    return dense


def insert_zeros(patches: Patches, zeros: int) -> Patches:
    if zeros == 0:
        return patches.clone()
    if zeros > 0:
        raise NotImplementedError(
            "dilation > 1 not supported in v1 (S1 regime)"
        )
    raise ValueError(f"zeros must be non-negative, got {zeros}")


def compute_patches_stride_padding(
    prev_stride: int | tuple[int, int],
    prev_padding: int | tuple[int, int],
    cur_stride: int | tuple[int, int],
    cur_padding: int | tuple[int, int],
    prev_inserted_zeros: int = 0,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if prev_inserted_zeros > 0:
        raise NotImplementedError(
            "fusion with dilation > 1 (inserted_zeros > 0) not supported in v1"
        )
    prev_stride_hw = _normalize_pair(prev_stride)
    prev_padding_hw = _normalize_pair(prev_padding)
    cur_stride_hw = _normalize_pair(cur_stride)
    cur_padding_hw = _normalize_pair(cur_padding)
    new_stride = (
        prev_stride_hw[0] * cur_stride_hw[0],
        prev_stride_hw[1] * cur_stride_hw[1],
    )
    new_padding = (
        prev_padding_hw[0] * cur_stride_hw[0] + cur_padding_hw[0],
        prev_padding_hw[1] * cur_stride_hw[1] + cur_padding_hw[1],
    )
    return new_stride, new_padding


def unify_shape(patches: Patches, target_shape: tuple[int, ...]) -> Patches:
    current_shape = patches.shape or (
        tuple(patches.patches.shape) if patches.patches is not None else None
    )
    if current_shape is None:
        if patches.is_identity:
            clone = patches.clone()
            clone.shape = tuple(target_shape)
            return clone
        raise ValueError("Cannot unify shape when current shape metadata is missing")

    if tuple(current_shape) == tuple(target_shape):
        return patches.clone()
    if len(current_shape) != len(target_shape):
        raise ValueError(
            f"Shape rank mismatch: current={current_shape!r}, target={target_shape!r}"
        )

    if patches.is_identity:
        clone = patches.clone()
        clone.shape = tuple(target_shape)
        clone.output_shape = tuple(target_shape)
        return clone
    if patches.patches is None:
        raise ValueError("Cannot unify non-identity patches without a tensor payload")

    tensor = patches.patches
    broadcast_dims: list[int] = []
    repeat_factors: list[int] = []
    for axis, (current_dim, target_dim) in enumerate(
        zip(current_shape, target_shape, strict=True)
    ):
        if current_dim == target_dim:
            repeat_factors.append(1)
            continue
        if axis in {0, 4}:
            raise ValueError(
                "Channel dimensions must match exactly when unifying Patches shapes"
            )
        if current_dim != 1:
            raise ValueError(
                f"Shapes are not broadcast-compatible: current={current_shape!r}, target={target_shape!r}"
            )
        broadcast_dims.append(axis)
        repeat_factors.append(target_dim)

    use_repeat = any(stride == 0 for stride in tensor.stride())
    new_tensor = (
        tensor.repeat(*repeat_factors)
        if use_repeat
        else tensor.expand(*target_shape)
    )

    clone = patches.clone()
    clone.patches = new_tensor
    clone.shape = tuple(target_shape)
    return clone


def is_deterministic_patches() -> bool:
    return getattr(_DETERMINISTIC_PATCHES, "value", True)


@contextmanager
def deterministic_patches_ctx(deterministic: bool = True):
    previous = is_deterministic_patches()
    _DETERMINISTIC_PATCHES.value = deterministic
    try:
        yield
    finally:
        _DETERMINISTIC_PATCHES.value = previous


def inplace_unfold(
    tensor: torch.Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> torch.Tensor:
    if is_deterministic_patches():
        return F.unfold(tensor, kernel_size=kernel_size, stride=stride, padding=padding)

    kernel_h, kernel_w = _normalize_pair(kernel_size)
    stride_h, stride_w = _normalize_pair(stride)
    padding_h, padding_w = _normalize_pair(padding)
    padded = F.pad(tensor, (padding_w, padding_w, padding_h, padding_h))
    batch_size, channels, padded_h, padded_w = padded.shape
    out_h = (padded_h - kernel_h) // stride_h + 1
    out_w = (padded_w - kernel_w) // stride_w + 1
    view = torch.as_strided(
        padded,
        size=(batch_size, channels, out_h, out_w, kernel_h, kernel_w),
        stride=(
            padded.stride(0),
            padded.stride(1),
            padded.stride(2) * stride_h,
            padded.stride(3) * stride_w,
            padded.stride(2),
            padded.stride(3),
        ),
    )
    return view.permute(0, 1, 4, 5, 2, 3).reshape(
        batch_size,
        channels * kernel_h * kernel_w,
        out_h * out_w,
    )


__all__ = [
    "Patches",
    "compute_patches_stride_padding",
    "deterministic_patches_ctx",
    "inplace_unfold",
    "insert_zeros",
    "is_deterministic_patches",
    "patches_to_matrix",
    "unify_shape",
]
