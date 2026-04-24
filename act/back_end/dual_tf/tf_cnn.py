#===- act/back_end/dual_tf/tf_cnn.py - CNN Dual Transfer Functions ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# Batch-aware Conv2D backward. nu: [B, *out_shape] -> v_out: [B, *in_flat], contrib: [B].
#===---------------------------------------------------------------------===#

# Note: Gradient enablement for dual backward helpers is governed by the
# caller's torch.set_grad_enabled() context (see DualSolver.evaluate_spec).
# @torch.no_grad() decorators on these helpers were removed to allow
# gradient flow during robust training; verify_once / verify_bab paths
# remain under no_grad via their own outer guards.

import logging

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
from act.back_end.core import Bounds, Layer
from act.back_end.patches import Patches, patches_to_matrix

from .tf_forward import (
    LinearBound, Frame,
    _fwd_conv2d, _fwd_conv2d_interval, _fwd_maxpool2d, _fwd_avgpool2d,
    _reset_forward_box, _concretize,
)


log = logging.getLogger(__name__)


# ==========================================================================
# Forward registry handlers (plan §4.2 uniform signature).
# Returns (stored, out, lin, frame); semantics copied from the pre-refactor
# monolithic branches of compute_forward_bounds in tf_forward.py.
# ==========================================================================


# ---- CONV2D ----
def forward_conv2d(
    L: Layer,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Conv2D forward bounds with dual-track fallback.

    Source: tf_forward.py lines 380-391 (pre-refactor monolithic CONV2D branch).

    Tries linear-relaxation conv via ``_fwd_conv2d``; when it returns ``None``
    (kernel/stride shape unsupported), falls back to the interval conv
    ``_fwd_conv2d_interval`` and resets the dual-track state at the new
    concrete box via ``_reset_forward_box``. ``stored`` equals ``out``
    (CONV2D has no activation split).
    """
    assert len(parent_boxes) == 1, f"CONV2D expects 1 predecessor, got {len(parent_boxes)}"
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    frame = parent_frames[0]
    x_L, x_U = frame

    new_lin = _fwd_conv2d(L, parent_lin)
    if new_lin is None:
        lb, ub = _fwd_conv2d_interval(L, parent_box.lb, parent_box.ub)
        out = Bounds(lb, ub)
        stored = out
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
    else:
        lin = new_lin
        lb, ub = _concretize(lin, x_L, x_U)
        out = Bounds(lb, ub)
        stored = out
    return stored, out, lin, frame


def backward_conv2d(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                    preds: List[int]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    stride = L.params.get("stride", 1)
    padding = L.params.get("padding", 0)
    if isinstance(stride, (list, tuple)): stride = stride[0]
    if isinstance(padding, (list, tuple)): padding = padding[0]
    nu_out, contrib = dual_conv2d_backward(
        nu, L.params["weight"], L.params.get("bias"),
        stride=stride, padding=padding,
        input_shape=L.params.get("input_shape"),
        output_shape=L.params.get("output_shape"),
    )
    assert len(preds) == 1, f"CONV2D expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_conv2d_backward(
    nu: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
    stride: int = 1, padding: int = 0,
    input_shape: Optional[tuple[int, ...]] = None, output_shape: Optional[tuple[int, ...]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Conv2D backward via F.conv_transpose2d (naturally batched).

    output_shape is REQUIRED (no silent sqrt-inference fallback). The
    .view(B, oC, oH, oW) into 4D is a legitimate shape restoration required
    by F.conv_transpose2d, not a silent reshape to paper over a mismatch.
    """
    assert weight.dim() == 4, f"weight must be 4D [oC,iC,kH,kW], got {weight.shape}"
    assert nu.dim() >= 2, f"nu must be batched (>=2D), got {nu.shape}"
    B = nu.shape[0]
    oC, iC, kH, kW = weight.shape

    v_flat = nu.flatten(start_dim=1)
    n = v_flat.shape[-1]

    if output_shape is None:
        raise ValueError(
            "dual_conv2d_backward: output_shape is required. The previous silent "
            "sqrt-based spatial inference was unsound for non-square feature maps."
        )
    if len(output_shape) == 4:
        _, oC2, oH, oW = output_shape
    elif len(output_shape) == 3:
        oC2, oH, oW = output_shape
    else:
        raise ValueError(
            f"dual_conv2d_backward: output_shape must have len 3 or 4 (got "
            f"{len(output_shape)}: {output_shape})"
        )
    if oC2 != oC:
        raise ValueError(
            f"dual_conv2d_backward: output_shape channels {oC2} do not match "
            f"weight oC {oC}"
        )

    expected = oC * oH * oW
    if n != expected:
        raise ValueError(
            f"dual_conv2d_backward: flat nu width {n} does not match expected "
            f"output size {expected} (oC={oC}, oH={oH}, oW={oW}). Callers must "
            f"pass nu shaped to match the conv output."
        )
    v_4d = v_flat.view(B, oC, oH, oW)

    if isinstance(stride, (list, tuple)): stride = stride[0]
    if isinstance(padding, (list, tuple)): padding = padding[0]

    # Derive output_padding so conv_transpose2d exactly recovers the original
    # input spatial size (stride > 1 loses an increment of up to stride-1).
    output_padding = 0
    if input_shape is not None:
        shape = list(input_shape)
        if len(shape) >= 4:
            iH, iW = shape[-2], shape[-1]
        elif len(shape) == 3:
            iH, iW = shape[-2], shape[-1]
        else:
            iH = iW = None
        if iH is not None:
            computed_h = (v_4d.shape[-2] - 1) * stride - 2 * padding + kH
            op_h = iH - computed_h
            if op_h > 0:
                output_padding = op_h

    v_out_4d = F.conv_transpose2d(v_4d, weight, None,
                                  stride=stride, padding=padding,
                                  output_padding=output_padding)
    v_out = v_out_4d.flatten(start_dim=1)

    if bias is not None:
        per_ch = v_4d.sum(dim=(-1, -2))
        contrib = -(per_ch @ bias.flatten())
    else:
        contrib = torch.zeros(B, dtype=nu.dtype, device=nu.device)
    return v_out, contrib


# ---- MAXPOOL2D ----
def forward_maxpool2d(
    L: Layer,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound | Patches],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound | Patches, Frame]:
    """MaxPool2D forward bounds (interval-only; dual-track resets).

    Source: tf_forward.py lines 448-452 (pre-refactor monolithic MAXPOOL2D
    branch).

    MaxPool has no linear relaxation here, so we compute interval bounds
    via ``_fwd_maxpool2d`` and reset the dual-track state at the resulting
    concrete box. ``stored`` equals ``out``.
    """
    assert len(parent_boxes) == 1, f"MAXPOOL2D expects 1 predecessor, got {len(parent_boxes)}"
    if isinstance(parent_lins[0], Patches):
        log.warning("forward_maxpool2d: pool materializes Patches in v1; consider refactor if hot")
        patches = parent_lins[0]
        input_shape = patches.input_shape or L.params.get("input_shape")
        if input_shape is None:
            raise ValueError("forward_maxpool2d: Patches input requires input_shape metadata")
        shape_tuple = tuple(int(dim) for dim in input_shape) if isinstance(input_shape, (list, tuple)) else None
        if shape_tuple is None:
            raise ValueError("forward_maxpool2d: unsupported input_shape metadata")
        dense = patches_to_matrix(patches, shape_tuple)
        zeros = dense.new_zeros((dense.shape[0], dense.shape[1]))
        parent_lins = [LinearBound(A_lb=dense, b_lb=zeros, A_ub=dense.clone(), b_ub=zeros.clone())]
    parent_box = parent_boxes[0]
    lb, ub = _fwd_maxpool2d(L, parent_box.lb, parent_box.ub)
    out = Bounds(lb, ub)
    stored = out
    lin, frame = _reset_forward_box(lb, ub, device, dtype)
    return stored, out, lin, frame


def backward_maxpool2d(L, nu, bounds_dict, preds):
    """MAXPOOL2D backward. (Pending)
    Will require: forward pooling windows (argmax indices) to route nu through
    the selected maxima.
    """
    raise NotImplementedError("backward for MAXPOOL2D not implemented in dual_tf")


def dual_maxpool2d_backward(*args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("dual_maxpool2d_backward: not yet implemented")


# ---- AVGPOOL2D ----
def forward_avgpool2d(
    L: Layer,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound | Patches],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound | Patches, Frame]:
    """AvgPool2D forward bounds (interval-only; dual-track resets).

    Source: tf_forward.py lines 454-458 (pre-refactor monolithic AVGPOOL2D
    branch).

    Analogous to :func:`forward_maxpool2d` using ``_fwd_avgpool2d``.
    ``stored`` equals ``out``.
    """
    assert len(parent_boxes) == 1, f"AVGPOOL2D expects 1 predecessor, got {len(parent_boxes)}"
    if isinstance(parent_lins[0], Patches):
        log.warning("forward_avgpool2d: pool materializes Patches in v1; consider refactor if hot")
        patches = parent_lins[0]
        input_shape = patches.input_shape or L.params.get("input_shape")
        if input_shape is None:
            raise ValueError("forward_avgpool2d: Patches input requires input_shape metadata")
        shape_tuple = tuple(int(dim) for dim in input_shape) if isinstance(input_shape, (list, tuple)) else None
        if shape_tuple is None:
            raise ValueError("forward_avgpool2d: unsupported input_shape metadata")
        dense = patches_to_matrix(patches, shape_tuple)
        zeros = dense.new_zeros((dense.shape[0], dense.shape[1]))
        parent_lins = [LinearBound(A_lb=dense, b_lb=zeros, A_ub=dense.clone(), b_ub=zeros.clone())]
    parent_box = parent_boxes[0]
    lb, ub = _fwd_avgpool2d(L, parent_box.lb, parent_box.ub)
    out = Bounds(lb, ub)
    stored = out
    lin, frame = _reset_forward_box(lb, ub, device, dtype)
    return stored, out, lin, frame


def backward_avgpool2d(L, nu, bounds_dict, preds):
    """AVGPOOL2D backward. (Pending)
    Will require: kernel/stride/padding for uniform redistribution of nu.
    """
    raise NotImplementedError("backward for AVGPOOL2D not implemented in dual_tf")


def dual_avgpool2d_backward(*args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("dual_avgpool2d_backward: not yet implemented")
