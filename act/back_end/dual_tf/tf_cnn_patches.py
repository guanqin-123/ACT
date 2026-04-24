from __future__ import annotations

import importlib
import logging
from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn.functional as F

from act.back_end.core import Bounds, Layer
from act.back_end.patches import Patches, compute_patches_stride_padding, inplace_unfold

from .tf_forward import Frame, LinearBound


log = logging.getLogger(__name__)

_S1_ERROR = (
    "S1 regime: only Conv2d stride∈{1,2} padding∈{0,1} groups=1 dilation=1 supported in v1"
)


def _normalize_pair(value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise NotImplementedError(_S1_ERROR)
    return (int(value[0]), int(value[1]))


def _normalize_padding2(
    value: int | tuple[int, int] | tuple[int, int, int, int],
) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if len(value) == 2:
        return (int(value[0]), int(value[1]))
    if len(value) == 4:
        left, right, top, bottom = (int(item) for item in value)
        if left != right or top != bottom:
            raise NotImplementedError(_S1_ERROR)
        return (top, left)
    raise NotImplementedError(_S1_ERROR)


def _read_int_param(layer: Layer, name: str, default: int) -> int:
    value = layer.params.get(name, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    raise NotImplementedError(_S1_ERROR)


def _normalize_shape(
    shape: tuple[int, ...] | list[int] | None,
    batch_size: int | None = None,
) -> tuple[int, int, int, int]:
    if shape is None:
        raise ValueError("Conv patches path requires explicit input/output shape metadata")
    if len(shape) == 4:
        b, c, h, w = (int(dim) for dim in shape)
        if batch_size is not None:
            b = batch_size
        return (b, c, h, w)
    if len(shape) == 3:
        c, h, w = (int(dim) for dim in shape)
        return (1 if batch_size is None else batch_size, c, h, w)
    raise ValueError(f"Expected shape with 3 or 4 dims, got {shape!r}")


def _flatten_frame_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 3 and tensor.stride(1) == 0:
        return tensor[:, 0, :]
    if tensor.dim() >= 2:
        return tensor.reshape(tensor.shape[0], -1)
    return tensor.flatten().unsqueeze(0)


def _resolve_batch_size(
    parent_boxes: Sequence[Bounds],
    parent_frames: Sequence[Frame],
    patches: Patches | None = None,
) -> int:
    if parent_boxes:
        return int(parent_boxes[0].lb.shape[0])
    if patches is not None and patches.patches is not None:
        if patches.patches.ndim == 7:
            return int(patches.patches.shape[1])
        if patches.patches.ndim >= 2:
            return int(patches.patches.shape[0])
    if patches is not None and patches.shape is not None:
        if len(patches.shape) >= 2:
            return int(patches.shape[1])
    if patches is not None and patches.output_shape is not None:
        return int(_normalize_shape(patches.output_shape)[0])
    if patches is not None and patches.input_shape is not None:
        return int(_normalize_shape(patches.input_shape)[0])
    if parent_frames:
        return int(parent_frames[0][0].shape[0])
    raise ValueError("Unable to infer batch size for Conv patches path")


def _canonicalize_patches_layout(
    pieces: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if pieces.ndim != 7:
        raise ValueError(
            "Expected patches tensor with shape (out_c, B, out_h, out_w, in_c, k_h, k_w)"
        )
    if pieces.shape[1] == batch_size:
        return pieces
    if pieces.shape[0] == batch_size:
        return pieces.permute(1, 0, 2, 3, 4, 5, 6).contiguous()
    raise ValueError(
        f"Cannot infer batch axis for patches shape {tuple(pieces.shape)} with batch={batch_size}"
    )


def _validate_conv2d_s1(layer: Layer) -> None:
    stride = _normalize_pair(cast(int | tuple[int, int], layer.params.get("stride", 1)))
    padding = _normalize_pair(_normalize_padding2(cast(int | tuple[int, int] | tuple[int, int, int, int], layer.params.get("padding", 0))))
    dilation = _normalize_pair(cast(int | tuple[int, int], layer.params.get("dilation", 1)))
    groups = _read_int_param(layer, "groups", 1)
    kernel_size = layer.params.get("kernel_size")
    if kernel_size is not None:
        kernel_hw = _normalize_pair(cast(int | tuple[int, int], kernel_size))
        if kernel_hw[0] != kernel_hw[1]:
            raise NotImplementedError(_S1_ERROR)
    if stride[0] not in {1, 2} or stride[1] not in {1, 2}:
        raise NotImplementedError(_S1_ERROR)
    if padding[0] not in {0, 1} or padding[1] not in {0, 1}:
        raise NotImplementedError(_S1_ERROR)
    if dilation != (1, 1) or groups != 1:
        raise NotImplementedError(_S1_ERROR)


def _resolve_input_shape(
    layer: Layer,
    parent_frames: Sequence[Frame],
    patches: Patches | None,
) -> tuple[int, int, int, int]:
    batch_size = _resolve_batch_size([], parent_frames, patches)
    if patches is not None and patches.input_shape is not None:
        return _normalize_shape(patches.input_shape, batch_size)
    frame_shape = tuple(int(dim) for dim in parent_frames[0][0].shape)
    if len(frame_shape) == 4:
        return _normalize_shape(frame_shape, batch_size)
    return _normalize_shape(cast(tuple[int, ...] | list[int] | None, layer.params.get("input_shape")), batch_size)


def _resolve_output_shape(layer: Layer, batch_size: int) -> tuple[int, int, int, int]:
    output_shape = cast(tuple[int, ...] | list[int] | None, layer.params.get("output_shape"))
    if output_shape is not None:
        return _normalize_shape(output_shape, batch_size)

    weight = cast(torch.Tensor, layer.params["weight"])
    _, _, in_h, in_w = _normalize_shape(cast(tuple[int, ...] | list[int], layer.params["input_shape"]), batch_size)
    stride_h, stride_w = _normalize_pair(cast(int | tuple[int, int], layer.params.get("stride", 1)))
    pad_h, pad_w = _normalize_pair(cast(int | tuple[int, int], layer.params.get("padding", 0)))
    k_h, k_w = int(weight.shape[-2]), int(weight.shape[-1])
    out_h = (in_h + 2 * pad_h - k_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - k_w) // stride_w + 1
    return (batch_size, int(weight.shape[0]), out_h, out_w)

def _patches_to_linear_bound(
    patches: Patches,
    input_shape: tuple[int, int, int, int],
) -> LinearBound:
    dense = patches.to_matrix(input_shape)
    if dense.ndim != 3:
        raise ValueError(f"Expected dense patches matrix [B, out_dim, in_dim], got {tuple(dense.shape)}")
    batch_size, out_dim, _ = dense.shape
    zeros = dense.new_zeros((batch_size, out_dim))
    return LinearBound(A_lb=dense, b_lb=zeros, A_ub=dense.clone(), b_ub=zeros.clone())


def _dense_matrix_to_patches(
    matrix: torch.Tensor,
    input_shape: tuple[int, int, int, int],
    out_channels: int,
    output_hw: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> torch.Tensor:
    batch_size, in_channels, in_h, in_w = input_shape
    out_h, out_w = output_hw
    k_h, k_w = kernel_size
    stride_h, stride_w = _normalize_pair(cast(int | tuple[int, int], stride))
    pad_h, pad_w = _normalize_pair(cast(int | tuple[int, int], padding))
    pieces = matrix.new_zeros((out_channels, batch_size, out_h, out_w, in_channels, k_h, k_w))
    for batch_idx in range(batch_size):
        flat = matrix[batch_idx]
        for out_c in range(out_channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    row_idx = out_c * (out_h * out_w) + oh * out_w + ow
                    top = oh * stride_h - pad_h
                    left = ow * stride_w - pad_w
                    for in_c in range(in_channels):
                        for kh in range(k_h):
                            in_y = top + kh
                            if 0 <= in_y < in_h:
                                for kw in range(k_w):
                                    in_x = left + kw
                                    if 0 <= in_x < in_w:
                                        col_idx = in_c * (in_h * in_w) + in_y * in_w + in_x
                                        pieces[out_c, batch_idx, oh, ow, in_c, kh, kw] = flat[row_idx, col_idx]
    return pieces


def _matrix_to_patches(
    matrix: torch.Tensor,
    input_shape: tuple[int, int, int, int],
    *,
    out_channels: int,
    output_hw: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> torch.Tensor:
    batch_size, in_channels, in_h, in_w = input_shape
    out_h, out_w = output_hw
    k_h, k_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    pieces = matrix.new_zeros((out_channels, batch_size, out_h, out_w, in_channels, k_h, k_w))
    for out_c in range(out_channels):
        for oh in range(out_h):
            for ow in range(out_w):
                row_idx = out_c * (out_h * out_w) + oh * out_w + ow
                top = oh * stride_h - pad_h
                left = ow * stride_w - pad_w
                for in_c in range(in_channels):
                    for kh in range(k_h):
                        in_y = top + kh
                        if 0 <= in_y < in_h:
                            for kw in range(k_w):
                                in_x = left + kw
                                if 0 <= in_x < in_w:
                                    col_idx = in_c * (in_h * in_w) + in_y * in_w + in_x
                                    pieces[out_c, :, oh, ow, in_c, kh, kw] = matrix[:, row_idx, col_idx]
    return pieces


def linear_form_tensor_to_patches(
    coeffs: torch.Tensor,
    feature_shape: tuple[int, ...],
) -> Patches:
    if coeffs.dim() == 1:
        coeffs = coeffs.unsqueeze(0)
    if coeffs.dim() != 2:
        raise ValueError(
            f"linear_form_tensor_to_patches expects [B, D] coefficients, got {tuple(coeffs.shape)}"
        )
    batch_size = int(coeffs.shape[0])
    _, channels, height, width = _normalize_shape(feature_shape, batch_size)
    expected = channels * height * width
    if int(coeffs.shape[1]) != expected:
        raise ValueError(
            f"Coefficient width {coeffs.shape[1]} does not match feature shape {feature_shape!r}"
        )
    pieces = coeffs.reshape(batch_size, channels, height, width).unsqueeze(0).unsqueeze(2).unsqueeze(2)
    return Patches(
        patches=pieces,
        stride=1,
        padding=0,
        shape=tuple(int(dim) for dim in pieces.shape),
        input_shape=_normalize_shape(feature_shape, batch_size),
        output_shape=(batch_size, 1, 1, 1),
    )


def linear_form_patches_to_tensor(
    patches: Patches,
    feature_shape: tuple[int, ...],
) -> torch.Tensor:
    dense = patches.to_matrix(_normalize_shape(feature_shape, _resolve_batch_size([], [], patches)))
    if dense.ndim != 3 or dense.shape[1] != 1:
        raise ValueError(
            f"Expected a single-row patches linear form, got dense shape {tuple(dense.shape)}"
        )
    return dense[:, 0, :]


def _bias_vector(
    bias: torch.Tensor | None,
    batch_size: int,
    output_shape: tuple[int, int, int, int],
) -> torch.Tensor | None:
    if bias is None:
        return None
    _, out_c, out_h, out_w = output_shape
    return bias.view(1, out_c, 1, 1).expand(batch_size, -1, out_h, out_w).reshape(batch_size, -1)


def _concretize_patches(
    patches: Patches,
    frame: Frame,
    bias: torch.Tensor | None,
    output_shape: tuple[int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    input_shape = _normalize_shape(cast(tuple[int, ...], patches.input_shape))
    dense = patches.to_matrix(input_shape)
    x_L = _flatten_frame_tensor(frame[0]).to(device=dense.device, dtype=dense.dtype)
    x_U = _flatten_frame_tensor(frame[1]).to(device=dense.device, dtype=dense.dtype)
    A_pos = dense.clamp(min=0)
    A_neg = dense.clamp(max=0)
    lb = torch.einsum("boi,bi->bo", A_pos, x_L) + torch.einsum("boi,bi->bo", A_neg, x_U)
    ub = torch.einsum("boi,bi->bo", A_pos, x_U) + torch.einsum("boi,bi->bo", A_neg, x_L)
    bias_vec = _bias_vector(bias, dense.shape[0], output_shape)
    if bias_vec is not None:
        lb = lb + bias_vec
        ub = ub + bias_vec
    return lb, ub


def _compose_forward_patches(layer: Layer, parent_patches: Patches, batch_size: int) -> torch.Tensor:
    if parent_patches.patches is None:
        raise ValueError("forward_conv2d_patches: non-identity patches input must carry a patches tensor")
    pieces = _canonicalize_patches_layout(parent_patches.patches, batch_size)
    weight = cast(torch.Tensor, layer.params["weight"])
    out_c, mid_c, k_h, k_w = (int(dim) for dim in weight.shape)
    prev_out_c, _, _prev_h, _prev_w, _in_c, prev_k_h, prev_k_w = (int(dim) for dim in pieces.shape)
    if prev_out_c != mid_c:
        raise ValueError(
            f"forward_conv2d_patches: channel mismatch (patches={prev_out_c}, weight expects={mid_c})"
        )

    input_shape = _normalize_shape(cast(tuple[int, ...], parent_patches.input_shape), batch_size)
    prev_output_shape = _normalize_shape(
        cast(tuple[int, ...] | list[int], parent_patches.output_shape or layer.params["input_shape"]),
        batch_size,
    )
    _, prev_channels, prev_h, prev_w = prev_output_shape
    dense = parent_patches.to_matrix(input_shape)
    _, prev_out_dim, input_dim = dense.shape
    if prev_out_dim != prev_channels * prev_h * prev_w:
        raise ValueError(
            "forward_conv2d_patches: parent patches output shape metadata does not match dense matrix rows"
        )

    stride = _normalize_pair(cast(int | tuple[int, int], layer.params.get("stride", 1)))
    padding = _normalize_pair(cast(int | tuple[int, int], layer.params.get("padding", 0)))
    dense_as_image = dense.transpose(1, 2).contiguous().view(batch_size * input_dim, prev_channels, prev_h, prev_w)
    dense_out = F.conv2d(dense_as_image, weight, None, stride=stride, padding=padding)
    out_h, out_w = int(dense_out.shape[-2]), int(dense_out.shape[-1])
    dense_out = dense_out.flatten(start_dim=1).reshape(batch_size, input_dim, -1).transpose(1, 2).contiguous()

    prev_stride_h, prev_stride_w = _normalize_pair(cast(int | tuple[int, int], parent_patches.stride))
    new_k_h = prev_k_h + (k_h - 1) * prev_stride_h
    new_k_w = prev_k_w + (k_w - 1) * prev_stride_w
    new_stride, new_padding = compute_patches_stride_padding(
        parent_patches.stride,
        _normalize_padding2(parent_patches.padding),
        cast(int | tuple[int, int], layer.params.get("stride", 1)),
        _normalize_padding2(cast(int | tuple[int, int] | tuple[int, int, int, int], layer.params.get("padding", 0))),
        parent_patches.inserted_zeros,
    )
    return _matrix_to_patches(
        dense_out,
        input_shape,
        out_channels=out_c,
        output_hw=(out_h, out_w),
        kernel_size=(new_k_h, new_k_w),
        stride=new_stride,
        padding=new_padding,
    )


def _identity_forward_patches(layer: Layer, batch_size: int, output_shape: tuple[int, int, int, int]) -> torch.Tensor:
    weight = cast(torch.Tensor, layer.params["weight"])
    _, _, out_h, out_w = output_shape
    return weight.view(weight.shape[0], 1, 1, 1, weight.shape[1], weight.shape[2], weight.shape[3]).expand(
        -1,
        batch_size,
        out_h,
        out_w,
        -1,
        -1,
        -1,
    ).clone()


def _compose_backward_patches(layer: Layer, nu: Patches, batch_size: int) -> torch.Tensor:
    if nu.patches is None:
        raise ValueError("backward_conv2d_patches: non-identity Patches input must carry a patches tensor")
    patches = _canonicalize_patches_layout(nu.patches, batch_size)
    flattened = patches.reshape(-1, patches.size(-3), patches.size(-2), patches.size(-1))
    weight = cast(torch.Tensor, layer.params["weight"])
    stride = _normalize_pair(cast(int | tuple[int, int], layer.params.get("stride", 1)))
    pieces = F.conv_transpose2d(flattened, weight, None, stride=stride)
    return pieces.view(*patches.shape[:-3], pieces.size(-3), pieces.size(-2), pieces.size(-1))


def _identity_backward_patches(layer: Layer, nu: Patches, batch_size: int) -> torch.Tensor:
    weight = cast(torch.Tensor, layer.params["weight"])
    if nu.shape is not None and len(nu.shape) >= 4:
        out_c, _, out_h, out_w = (int(nu.shape[0]), int(nu.shape[1]), int(nu.shape[2]), int(nu.shape[3]))
    elif nu.output_shape is not None:
        _, out_c, out_h, out_w = _normalize_shape(nu.output_shape, batch_size)
    else:
        _, out_c, out_h, out_w = _normalize_shape(cast(tuple[int, ...], layer.params["output_shape"]), batch_size)
    if out_c != int(weight.shape[0]):
        raise ValueError(
            f"backward_conv2d_patches: identity patches expect {weight.shape[0]} output channels, got {out_c}"
        )
    return weight.view(weight.shape[0], 1, 1, 1, weight.shape[1], weight.shape[2], weight.shape[3]).expand(
        -1,
        batch_size,
        out_h,
        out_w,
        -1,
        -1,
        -1,
    ).clone()


def _patch_bias_contrib(nu: Patches, bias: torch.Tensor | None, batch_size: int) -> torch.Tensor:
    if bias is None:
        if nu.patches is not None:
            device, dtype = nu.patches.device, nu.patches.dtype
        else:
            device, dtype = torch.device("cpu"), torch.get_default_dtype()
        return torch.zeros(batch_size, device=device, dtype=dtype)
    if nu.identity:
        if nu.shape is not None and len(nu.shape) >= 4:
            out_c, _, out_h, out_w = (int(nu.shape[0]), int(nu.shape[1]), int(nu.shape[2]), int(nu.shape[3]))
        elif nu.output_shape is not None:
            _, out_c, out_h, out_w = _normalize_shape(nu.output_shape, batch_size)
        else:
            raise ValueError(
                "backward_conv2d_patches: identity patches require shape or output_shape metadata for bias contribution"
            )
        bias_grid = bias.view(out_c, 1, 1, 1).expand(-1, batch_size, out_h, out_w)
        return -bias_grid.sum(dim=(0, 2, 3))
    if nu.patches is None:
        raise ValueError("backward_conv2d_patches: missing patches tensor for bias contribution")
    patches = _canonicalize_patches_layout(nu.patches, batch_size)
    summed = torch.einsum("obxyckl,c->obxy", patches, bias.to(device=patches.device, dtype=patches.dtype))
    return -summed.sum(dim=(0, 2, 3))


def _zero_contrib(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(batch_size, device=device, dtype=dtype)


def _is_identity_conv(layer: Layer) -> bool:
    weight = cast(torch.Tensor | None, layer.params.get("weight"))
    if weight is None or weight.ndim != 4:
        return False
    if _read_int_param(layer, "groups", 1) != 1:
        return False
    if _normalize_pair(cast(int | tuple[int, int], layer.params.get("stride", 1))) != (1, 1):
        return False
    if _normalize_padding2(cast(int | tuple[int, int] | tuple[int, int, int, int], layer.params.get("padding", 0))) != (0, 0):
        return False
    if _normalize_pair(cast(int | tuple[int, int], layer.params.get("dilation", 1))) != (1, 1):
        return False
    if weight.shape[0] != weight.shape[1] or tuple(weight.shape[-2:]) != (1, 1):
        return False
    eye = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype).view(
        weight.shape[0], weight.shape[1], 1, 1
    )
    bias = cast(torch.Tensor | None, layer.params.get("bias"))
    bias_is_zero = bias is None or torch.allclose(bias, torch.zeros_like(bias))
    return bool(torch.allclose(weight, eye) and bias_is_zero)


def forward_conv2d_patches(
    layer: Layer,
    parent_boxes: list[Bounds],
    parent_lins: Sequence[LinearBound | Patches],
    parent_frames: Sequence[Frame],
    preds: list[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Bounds, Bounds, LinearBound | Patches, Frame]:
    """Conv2D forward in sparse Patches representation.

    v1 keeps the general network path safe by falling back to the matrix kernel
    whenever the incoming affine state is still a LinearBound. Exact concrete
    bounds are computed for Patches inputs under the current v1 assumption that
    the incoming Patches encode only the affine A coefficients (no carried bias).
    """
    del post_activation
    assert len(parent_boxes) == 1, f"CONV2D expects 1 predecessor, got {len(parent_boxes)}"
    assert len(parent_lins) == 1, f"CONV2D expects 1 linear predecessor, got {len(parent_lins)}"
    _validate_conv2d_s1(layer)

    parent_lin = parent_lins[0]
    if not isinstance(parent_lin, Patches):
        from act.back_end.bounds_dispatch import materialize_if_needed

        log.warning(
            "forward_conv2d_patches: conv_mode=patches received LinearBound input; materializing and falling back to matrix path"
        )
        tf_cnn = importlib.import_module("act.back_end.dual_tf.tf_cnn")
        linear = materialize_if_needed(cast(LinearBound, parent_lin))
        return tf_cnn.forward_conv2d(
            layer,
            parent_boxes,
            [linear],
            parent_frames,
            preds,
            False,
            device,
            dtype,
        )

    if _is_identity_conv(layer):
        parent_box = parent_boxes[0]
        return parent_box, parent_box, parent_lin.clone(), parent_frames[0]

    batch_size = _resolve_batch_size(parent_boxes, parent_frames, parent_lin)
    input_shape = _resolve_input_shape(layer, parent_frames, parent_lin)
    output_shape = _resolve_output_shape(layer, batch_size)
    new_stride, new_padding = compute_patches_stride_padding(
        parent_lin.stride,
        _normalize_padding2(parent_lin.padding),
        cast(int | tuple[int, int], layer.params.get("stride", 1)),
        _normalize_padding2(cast(int | tuple[int, int] | tuple[int, int, int, int], layer.params.get("padding", 0))),
        parent_lin.inserted_zeros,
    )
    if parent_lin.is_identity:
        pieces = _identity_forward_patches(layer, batch_size, output_shape)
        lin = Patches(
            patches=pieces,
            stride=new_stride,
            padding=new_padding,
            shape=tuple(int(dim) for dim in pieces.shape),
            input_shape=input_shape,
            output_shape=output_shape,
        )
        lb, ub = _concretize_patches(
            lin,
            parent_frames[0],
            cast(torch.Tensor | None, layer.params.get("bias")),
            output_shape,
        )
        out = Bounds(lb=lb.to(device=device, dtype=dtype), ub=ub.to(device=device, dtype=dtype))
        return out, out, lin, parent_frames[0]

    tf_cnn = importlib.import_module("act.back_end.dual_tf.tf_cnn")
    dense_parent = _patches_to_linear_bound(parent_lin, input_shape)
    dense_result = tf_cnn.forward_conv2d(
        layer,
        parent_boxes,
        [dense_parent],
        parent_frames,
        preds,
        False,
        device,
        dtype,
    )
    prev_kernel_h = int(parent_lin.patches.shape[-2]) if parent_lin.patches is not None else 1
    prev_kernel_w = int(parent_lin.patches.shape[-1]) if parent_lin.patches is not None else 1
    cur_kernel_h = int(cast(torch.Tensor, layer.params["weight"]).shape[-2])
    cur_kernel_w = int(cast(torch.Tensor, layer.params["weight"]).shape[-1])
    prev_stride_h, prev_stride_w = _normalize_pair(cast(int | tuple[int, int], parent_lin.stride))
    fused_kernel = (
        prev_kernel_h + (cur_kernel_h - 1) * prev_stride_h,
        prev_kernel_w + (cur_kernel_w - 1) * prev_stride_w,
    )
    pieces = _dense_matrix_to_patches(
        dense_result[2].A_lb,
        input_shape,
        output_shape[1],
        output_shape[-2:],
        fused_kernel,
        new_stride,
        new_padding,
    )
    lin = Patches(
        patches=pieces,
        stride=new_stride,
        padding=new_padding,
        shape=tuple(int(dim) for dim in pieces.shape),
        input_shape=input_shape,
        output_shape=output_shape,
    )
    return dense_result[0], dense_result[1], lin, parent_frames[0]


def backward_conv2d_patches(
    layer: Any,
    nu: torch.Tensor | Patches,
    bounds: dict[int, Bounds],
    preds: list[int],
) -> tuple[list[torch.Tensor | Patches], torch.Tensor]:
    """Conv2D backward pass with a Patches-preserving escape hatch.

    Tensor ν keeps the existing matrix kernel for full backward compatibility.
    Patches ν uses the α-β-CROWN-style patch composition kernel.
    """
    assert len(preds) == 1, f"CONV2D expects 1 predecessor, got {len(preds)}"
    _validate_conv2d_s1(cast(Layer, layer))

    if isinstance(nu, torch.Tensor):
        tf_cnn = importlib.import_module("act.back_end.dual_tf.tf_cnn")
        return tf_cnn.backward_conv2d(layer, nu, bounds, preds)

    if _is_identity_conv(cast(Layer, layer)):
        if nu.patches is not None:
            device, dtype = nu.patches.device, nu.patches.dtype
        else:
            sample_bound = bounds.get(cast(Layer, layer).id)
            if sample_bound is None:
                raise ValueError("backward_conv2d_patches: missing bounds for identity fallback")
            device, dtype = sample_bound.lb.device, sample_bound.lb.dtype
        batch_size = _resolve_batch_size([], [], nu)
        return [nu.clone()], _zero_contrib(batch_size, device, dtype)

    batch_size = _resolve_batch_size([], [], nu)
    if nu.identity:
        pieces = _identity_backward_patches(cast(Layer, layer), nu, batch_size)
    else:
        pieces = _compose_backward_patches(cast(Layer, layer), nu, batch_size)

    new_stride, new_padding = compute_patches_stride_padding(
        nu.stride,
        _normalize_padding2(nu.padding),
        cast(int | tuple[int, int], cast(Layer, layer).params.get("stride", 1)),
        _normalize_padding2(cast(int | tuple[int, int] | tuple[int, int, int, int], cast(Layer, layer).params.get("padding", 0))),
        nu.inserted_zeros,
    )
    pred_nu = Patches(
        patches=pieces,
        stride=new_stride,
        padding=new_padding,
        shape=tuple(int(dim) for dim in pieces.shape),
        input_shape=_normalize_shape(cast(tuple[int, ...] | list[int], cast(Layer, layer).params["input_shape"]), batch_size),
        output_shape=nu.output_shape,
    )
    contrib = _patch_bias_contrib(
        nu,
        cast(torch.Tensor | None, cast(Layer, layer).params.get("bias")),
        batch_size,
    )
    return [pred_nu], contrib


__all__ = [
    "_is_identity_conv",
    "_patches_to_linear_bound",
    "backward_conv2d_patches",
    "forward_conv2d_patches",
    "linear_form_patches_to_tensor",
    "linear_form_tensor_to_patches",
]
