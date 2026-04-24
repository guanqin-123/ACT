#===- act/back_end/dual_tf/tf_mlp.py - MLP Dual Transfer Functions ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# Batch-aware MLP backward kernels for dual (Wong-Kolter) bound computation.
#
# Kernel convention (STRICT, batch-first):
#   nu      : Tensor[B, *layer_shape]   # dual variable, batch-first
#   v_out   : Tensor[B, *next_shape]
#   contrib : Tensor[B]                 # per-instance scalar
#
# ReLU uses FIXED upper-bound slope for crossing neurons.
#===---------------------------------------------------------------------===#

# Note: Gradient enablement for dual backward helpers is governed by the
# caller's torch.set_grad_enabled() context (see DualSolver.evaluate_spec).
# @torch.no_grad() decorators on these helpers were removed to allow
# gradient flow during robust training; verify_once / verify_bab paths
# remain under no_grad via their own outer guards.

import logging

import torch
from typing import Tuple, Optional, Dict, Any, List, cast
from act.back_end.core import Bounds
from act.back_end.patches import Patches, patches_to_matrix

from .tf_forward import (
    LinearBound, Frame,
    _fwd_dense, _fwd_relu, _fwd_bias, _fwd_scale, _fwd_bn, _fwd_lrelu,
    _concretize, _box_dense, _box_bias, _box_scale, _box_bn, _box_relu,
    _box_lrelu, _intersect_boxes, _reset_forward_box,
    _assert_broadcast_to_nu, _scale_mul, _bias_contrib, _flatten_batch_rows, _align,
)


log = logging.getLogger(__name__)


def _normalize_feature_shape(shape: tuple[int, ...] | list[int], batch_size: int) -> tuple[int, int, int, int]:
    if len(shape) == 4:
        _batch, channels, height, width = (int(dim) for dim in shape)
        return (batch_size, channels, height, width)
    if len(shape) == 3:
        channels, height, width = (int(dim) for dim in shape)
        return (batch_size, channels, height, width)
    raise ValueError(f"Expected feature shape with 3 or 4 dims, got {shape!r}")


def _resolve_patches_feature_shape(patches: Patches, reference: torch.Tensor) -> tuple[int, int, int, int]:
    batch_size = int(reference.shape[0])
    if patches.output_shape is not None:
        return _normalize_feature_shape(cast(tuple[int, ...] | list[int], patches.output_shape), batch_size)
    if patches.shape is not None and len(patches.shape) == 7:
        out_c, batch, out_h, out_w = (int(patches.shape[0]), int(patches.shape[1]), int(patches.shape[2]), int(patches.shape[3]))
        return (batch if batch > 0 else batch_size, out_c, out_h, out_w)
    flat_dim = int(reference[0].numel())
    return (batch_size, flat_dim, 1, 1)


def _resolve_patches_input_shape(patches: Patches, reference: torch.Tensor) -> tuple[int, int, int, int]:
    batch_size = int(reference.shape[0])
    if patches.input_shape is not None:
        return _normalize_feature_shape(cast(tuple[int, ...] | list[int], patches.input_shape), batch_size)
    if patches.shape is not None and len(patches.shape) == 7:
        in_c = int(patches.shape[4])
        k_h = int(patches.shape[5])
        k_w = int(patches.shape[6])
        return (batch_size, in_c, k_h, k_w)
    flat_dim = int(reference[0].numel())
    return (batch_size, flat_dim, 1, 1)


def _identity_patches_tensor(patches: Patches, reference: torch.Tensor) -> torch.Tensor:
    batch_size, out_c, out_h, out_w = _resolve_patches_feature_shape(patches, reference)
    input_shape = _resolve_patches_input_shape(patches, reference)
    in_c = input_shape[1]
    if out_c != in_c:
        raise ValueError(
            f"Identity patches require matching in/out channels, got out_c={out_c}, in_c={in_c}"
        )
    pieces = reference.new_zeros((out_c, batch_size, out_h, out_w, in_c, 1, 1))
    diag = torch.arange(out_c, device=reference.device)
    pieces[diag, :, :, :, diag, 0, 0] = 1
    return pieces


def _patches_payload(patches: Patches, reference: torch.Tensor) -> torch.Tensor:
    if patches.patches is not None:
        return patches.patches
    if patches.is_identity:
        return _identity_patches_tensor(patches, reference)
    raise ValueError("Patches tensor payload is missing")


def _reshape_like_patches(values: torch.Tensor, patches: Patches) -> torch.Tensor:
    batch_size, out_c, out_h, out_w = _resolve_patches_feature_shape(patches, values)
    flat = values.reshape(batch_size, -1)
    expected = out_c * out_h * out_w
    if flat.shape[1] != expected:
        raise ValueError(
            f"Cannot reshape tensor of width {flat.shape[1]} to patches output shape {(out_c, out_h, out_w)}"
        )
    return flat.view(batch_size, out_c, out_h, out_w)


def _flatten_param_to_shape(
    values: torch.Tensor,
    target: tuple[int, int, int, int],
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    value = values.to(device=reference.device, dtype=reference.dtype)
    batch_size, out_c, out_h, out_w = target
    candidates = [value]
    if value.numel() == out_c:
        candidates.append(value.reshape(1, out_c, 1, 1))
    if value.numel() == out_c * out_h * out_w:
        candidates.append(value.reshape(1, out_c, out_h, out_w))
    if value.numel() == batch_size * out_c:
        candidates.append(value.reshape(batch_size, out_c, 1, 1))
    if value.numel() == batch_size * out_c * out_h * out_w:
        candidates.append(value.reshape(target))
    for candidate in candidates:
        try:
            return torch.broadcast_to(candidate, target).contiguous().reshape(batch_size, -1)
        except RuntimeError:
            continue
    raise ValueError(
        f"{name}: shape {tuple(values.shape)} is not broadcastable to patches feature shape {target}"
    )


def _flatten_param_like_patches(
    values: torch.Tensor,
    patches: Patches,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    return _flatten_param_to_shape(
        values,
        _resolve_patches_feature_shape(patches, reference),
        reference,
        name,
    )


def _broadcast_forward_patch_scale(scale: torch.Tensor, patches: Patches) -> torch.Tensor:
    reshaped = _reshape_like_patches(scale, patches)
    return reshaped.permute(1, 0, 2, 3).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def _unfold_backward_patch_scale(scale: torch.Tensor, patches: Patches) -> torch.Tensor:
    if patches.shape is None and patches.patches is None:
        raise ValueError("Backward Patches path requires shape metadata")
    pieces = _patches_payload(patches, scale)
    out_c, batch_size, out_h, out_w, in_c, k_h, k_w = tuple(int(dim) for dim in pieces.shape)
    input_shape = _resolve_patches_input_shape(patches, scale)
    flat = scale.reshape(scale.shape[0], -1)
    expected = input_shape[1] * input_shape[2] * input_shape[3]
    if flat.shape[1] != expected:
        raise ValueError(
            f"Cannot reshape tensor of width {flat.shape[1]} to patches input shape {input_shape}"
        )
    scale_4d = flat.view(input_shape)
    stride = patches.stride if isinstance(patches.stride, tuple) else (patches.stride, patches.stride)
    padding = patches.padding
    if isinstance(padding, int):
        padding_hw = (padding, padding)
    elif len(padding) == 2:
        padding_hw = (int(padding[0]), int(padding[1]))
    else:
        padding_hw = (int(padding[2]), int(padding[0]))
    unfolded = torch.nn.functional.unfold(scale_4d, kernel_size=(k_h, k_w), stride=stride, padding=padding_hw)
    if unfolded.shape[-1] != out_h * out_w:
        raise ValueError(
            f"Unfolded scale positions {unfolded.shape[-1]} do not match patches spatial size {out_h * out_w}"
        )
    unfolded = unfolded.view(batch_size, in_c, k_h, k_w, out_h, out_w)
    return unfolded.permute(0, 4, 5, 1, 2, 3).unsqueeze(0).expand(out_c, -1, -1, -1, -1, -1, -1)


def _clone_with_patches_payload(source: Patches, pieces: torch.Tensor, *, output_shape: tuple[int, ...] | None = None) -> Patches:
    return Patches(
        patches=pieces,
        stride=source.stride,
        padding=source.padding,
        shape=tuple(int(dim) for dim in pieces.shape),
        identity=0,
        unstable_idx=source.unstable_idx,
        output_shape=tuple(output_shape) if output_shape is not None else source.output_shape,
        input_shape=source.input_shape,
        inserted_zeros=source.inserted_zeros,
        output_padding=source.output_padding,
    )


def _forward_relu_patches(
    patches: Patches,
    lb: torch.Tensor,
    ub: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    layer_id: Optional[int] = None,
) -> Patches:
    on, off, amb = get_relu_masks(lb, ub)
    denom = (ub - lb).clamp(min=1e-12)
    up_slope = torch.where(
        amb,
        ub / denom,
        torch.where(on, torch.ones_like(lb), torch.zeros_like(lb)),
    )
    alpha_heur = torch.where(
        amb,
        (up_slope > 0.5).to(lb.dtype),
        torch.where(on, torch.ones_like(lb), torch.zeros_like(lb)),
    )
    if alpha is None:
        alpha_used = alpha_heur
    else:
        alpha_b = _flatten_param_like_patches(alpha, patches, lb, "forward_relu alpha")
        try:
            broadcast_shape = torch.broadcast_shapes(tuple(alpha_b.shape), tuple(lb.shape))
        except RuntimeError as exc:
            layer_txt = "" if layer_id is None else f" for layer {layer_id}"
            raise ValueError(
                f"compute_forward_bounds: alpha{layer_txt} shape {tuple(alpha.shape)} "
                f"is not broadcastable to ReLU bounds shape {tuple(lb.shape)}"
            ) from exc
        if broadcast_shape != tuple(lb.shape):
            layer_txt = "" if layer_id is None else f" for layer {layer_id}"
            raise ValueError(
                f"compute_forward_bounds: alpha{layer_txt} shape {tuple(alpha.shape)} "
                f"must broadcast exactly to ReLU bounds shape {tuple(lb.shape)}"
            )
        alpha_b = torch.broadcast_to(alpha_b, lb.shape)
        if not torch.logical_and(alpha_b >= 0, alpha_b <= 1).all():
            raise ValueError("forward_relu: alpha values must lie in [0, 1]")
        alpha_used = torch.where(amb, alpha_b, alpha_heur)
    payload = _patches_payload(patches, lb)
    scale = _broadcast_forward_patch_scale(alpha_used, patches)
    return _clone_with_patches_payload(patches, payload * scale)


def _backward_relu_patches(nu: Patches, bounds: Bounds) -> tuple[Patches, torch.Tensor]:
    l, u = bounds.lb, bounds.ub
    on, off, amb = get_relu_masks(l, u)
    denom = (u - l).clamp(min=1e-12)
    slope = torch.where(
        amb,
        u / denom,
        torch.where(on, torch.ones_like(l), torch.zeros_like(l)),
    )
    payload = _patches_payload(nu, bounds.lb)
    scale = _unfold_backward_patch_scale(slope, nu)
    scaled = _clone_with_patches_payload(nu, payload * scale)
    input_shape = _resolve_patches_input_shape(nu, bounds.lb)
    dense = patches_to_matrix(scaled, input_shape)
    crossing = torch.where(
        amb.unsqueeze(1),
        -dense.clamp(max=0) * l.unsqueeze(1),
        torch.zeros_like(dense),
    )
    contrib = crossing.sum(dim=(1, 2))
    return scaled, contrib


def _forward_bn_patches(layer: Any, patches: Patches, bounds: Bounds) -> Patches:
    scale = _flatten_param_like_patches(cast(torch.Tensor, layer.params["A"]), patches, bounds.lb, "forward_bn scale")
    payload = _patches_payload(patches, bounds.lb)
    scale_bc = _broadcast_forward_patch_scale(scale, patches)
    return _clone_with_patches_payload(patches, payload * scale_bc)


def _backward_bn_patches(layer: Any, nu: Patches, bounds: Bounds) -> tuple[Patches, torch.Tensor]:
    input_shape = _resolve_patches_input_shape(nu, bounds.lb)
    scale_map = _flatten_param_to_shape(
        cast(torch.Tensor, layer.params["A"]),
        input_shape,
        bounds.lb,
        "backward_bn scale",
    )
    payload = _patches_payload(nu, bounds.lb)
    scale = _unfold_backward_patch_scale(scale_map, nu)
    scaled = _clone_with_patches_payload(nu, payload * scale)
    dense = patches_to_matrix(nu, input_shape)
    c_flat = _flatten_param_to_shape(
        cast(torch.Tensor, layer.params["c"]),
        input_shape,
        bounds.lb,
        "backward_bn bias",
    )
    c = c_flat.to(device=dense.device, dtype=dense.dtype).view(dense.shape[0], 1, -1)
    contrib = -(dense * c).sum(dim=(1, 2))
    return scaled, contrib


# ==========================================================================
# Forward dispatch handlers (uniform signature per plan §4.2):
#   (L, parent_boxes, parent_lins, parent_frames, preds,
#    post_activation, device, dtype) -> (stored, out, lin, frame)
#
# `parent_*` are parallel lists indexed by `preds`; unary handlers read [0].
# Function bodies are ported verbatim from the monolithic if/elif chain in
# tf_forward.compute_forward_bounds (pre-refactor; see source line ranges in
# each docstring). Driver still uses if/elif in Wave 2 — these are declared
# but not yet registered (registration lands in Wave 4 / Step D).
# ==========================================================================
# Dispatch functions : (L, nu, bounds_dict, preds) -> (pred_nus, contrib)
# Each pred_nus[i] is the ν routed to predecessor preds[i]. Unary layers
# (DENSE, RELU, BIAS, SCALE, BN) return [nu_out]. backward_identity handles
# both 0-pred (pure INPUT) and 1-pred (FLATTEN/RESHAPE/…) cases.
# ==========================================================================


# ---- IDENTITY ----
def forward_identity(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Pass-through handler for INPUT / INPUT_SPEC / ASSERT / TRANSPOSE / SQUEEZE / UNSQUEEZE.

    Source: tf_forward.py lines 357-358 (INPUT family) and 460-461
    (ASSERT / TRANSPOSE / SQUEEZE / UNSQUEEZE family). Both branches are
    `pass`, so stored/out/lin/frame are whatever the predecessor produced.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    return parent_box, parent_box, parent_lin, parent_frame


def backward_identity(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                      preds: List[int]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    nu_out, contrib = dual_identity_backward(nu)
    # 0 preds (pure INPUT) -> []; 1 pred (FLATTEN/RESHAPE/…) -> [nu_out].
    return [nu_out] * len(preds), contrib


def dual_identity_backward(nu: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten/Reshape/Transpose backward: v_out = nu, contrib = zeros[B]."""
    B = nu.shape[0]
    contrib = torch.zeros(B, dtype=nu.dtype, device=nu.device)
    return nu, contrib


# ---- RESHAPE ----
def forward_reshape(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for FLATTEN / RESHAPE.

    Source: tf_forward.py lines 420-422. Reshapes predecessor box lb/ub to
    ``[B, -1]`` and keeps lin/frame unchanged; downstream dense layers
    rematch the output-feature axis via ``_match_lin_input_dim``.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    B = parent_box.lb.shape[0]
    out = Bounds(parent_box.lb.reshape(B, -1), parent_box.ub.reshape(B, -1))
    stored = out
    return stored, out, parent_lin, parent_frame


# ---- DENSE ----
def forward_dense(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for DENSE (Wong-Kolter dual-track + interval intersection).

    Source: tf_forward.py lines 371-378. Composes lin via ``_fwd_dense``,
    concretizes against the predecessor frame, intersects with the interval
    box update, and returns ``stored == out`` (no pre/post distinction for
    an affine layer). Frame passes through unchanged.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    x_L, x_U = parent_frame
    prev_lb, prev_ub = parent_box.lb, parent_box.ub
    lin = _fwd_dense(L, parent_lin)
    crown_lb, crown_ub = _concretize(lin, x_L, x_U)
    int_lb, int_ub = _box_dense(L, prev_lb, prev_ub)
    lb, ub = _intersect_boxes(crown_lb, crown_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out
    return stored, out, lin, parent_frame


def backward_dense(L: Any, nu: torch.Tensor | Patches, bounds_dict: Dict[int, Bounds],
                   preds: List[int]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    if isinstance(nu, Patches):
        input_shape = cast(tuple[int, ...] | None, L.params.get("input_shape"))
        if input_shape is None:
            input_shape = nu.input_shape
        if input_shape is None:
            in_features = int(L.params["weight"].shape[1])
            input_shape = (bounds_dict[L.id].lb.shape[0], in_features, 1, 1)
        input_shape = _normalize_feature_shape(cast(tuple[int, ...] | list[int], input_shape), bounds_dict[L.id].lb.shape[0])
        log.debug("backward_dense: materializing Patches at Dense boundary for layer %s", L.id)
        nu = nu.to_matrix(input_shape)
    nu_out, contrib = dual_dense_backward(nu, L.params["weight"], L.params.get("bias"))
    assert len(preds) == 1, f"DENSE expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_dense_backward(nu: torch.Tensor, W: torch.Tensor,
                         b: Optional[torch.Tensor] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched dense backward: v_out = nu @ W, contrib = -(v · bias)."""
    assert W.dim() == 2, f"W must be 2D, got {W.shape}"
    assert nu.dim() >= 2, f"nu must be batched (>=2D), got {nu.shape}"
    if nu.dim() >= 3 and nu.shape[0] > 0 and nu.stride(1) == 0:
        nu = _flatten_batch_rows(nu, writable=True)
    nu_flat = nu.flatten(start_dim=1)
    assert nu_flat.shape[-1] == W.shape[0], \
        f"nu last dim {nu_flat.shape[-1]} != W.shape[0] {W.shape[0]}"
    v_out = nu_flat @ W
    contrib = _bias_contrib(nu, b, "dual_dense_backward")
    return v_out, contrib


# ---- RELU / LRELU ----
def get_relu_masks(l: torch.Tensor, u: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Element-wise masks (on, off, amb); shape-preserving."""
    on, off = l >= 0, u <= 0
    return on, off, ~(on | off)


def forward_relu(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound | Patches],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound | Patches, Frame]:
    """Forward handler for RELU (linear relaxation + interval intersection).

    Source: tf_forward.py lines 360-369. When ``post_activation`` is True,
    ``stored == out`` (post-ReLU box) and lin/frame are reset to identity
    over the new concrete box via ``_reset_forward_box``. Otherwise
    ``stored`` is the pre-activation box and lin/frame pass through.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    x_L, x_U = parent_frame
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    if isinstance(parent_lin, Patches):
        alpha = cast(Optional[torch.Tensor], L.params.get("alpha"))
        lin = _forward_relu_patches(parent_lin, pre_lb, pre_ub, alpha=alpha, layer_id=getattr(L, "id", None))
        int_lb, int_ub = _box_relu(pre_lb, pre_ub)
        out = Bounds(int_lb, int_ub)
        stored = out if post_activation else Bounds(pre_lb, pre_ub)
        return stored, out, lin, parent_frame
    lin = _fwd_relu(parent_lin, pre_lb, pre_ub)
    crown_lb, crown_ub = _concretize(lin, x_L, x_U)
    int_lb, int_ub = _box_relu(pre_lb, pre_ub)
    lb, ub = _intersect_boxes(crown_lb, crown_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    frame = parent_frame
    if post_activation:
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
    return stored, out, lin, frame


def forward_lrelu(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for LRELU / LEAKY_RELU (triangle linear relaxation).

    Source: tf_forward.py lines 436-446. Reads ``alpha`` from
    ``L.params.get("alpha", 0.01)``. ``post_activation`` handling mirrors
    :func:`forward_relu`: when True ``stored == out`` and lin/frame are
    reset to identity on the new box; otherwise ``stored`` is the
    pre-activation box and lin/frame pass through.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    x_L, x_U = parent_frame
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    alpha = L.params.get("alpha", 0.01)
    lin = _fwd_lrelu(parent_lin, pre_lb, pre_ub, alpha)
    crown_lb, crown_ub = _concretize(lin, x_L, x_U)
    int_lb, int_ub = _box_lrelu(pre_lb, pre_ub, alpha)
    lb, ub = _intersect_boxes(crown_lb, crown_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    frame = parent_frame
    if post_activation:
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
    return stored, out, lin, frame


def backward_relu(L: Any, nu: torch.Tensor | Patches, bounds_dict: Dict[int, Bounds],
                  preds: List[int]) -> Tuple[List[torch.Tensor | Patches], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_relu: layer {L.id} missing bounds in bounds_dict")
    if isinstance(nu, Patches):
        nu_out, contrib = _backward_relu_patches(nu, bounds)
        assert len(preds) == 1, f"RELU expects 1 predecessor, got {len(preds)}"
        return [nu_out], contrib
    nu_out, contrib = dual_relu_backward(nu, bounds, alpha=None)
    assert len(preds) == 1, f"RELU expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_relu_backward(nu: torch.Tensor, bounds: Bounds,
                       alpha: Optional[torch.Tensor] = None,
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched ReLU backward with fixed upper slope.

    Shape-preserving: operates at nu's native rank [B, *shape]. Caller MUST
    pass nu and bounds with identical shape; any mismatch raises (no
    implicit flatten, no silent broadcast). Returns v_out at nu's shape and
    contrib reduced over all non-batch dims.
    """
    _assert_broadcast_to_nu(bounds.lb.shape, nu.shape, "dual_relu_backward", "bounds")
    B = nu.shape[0]
    l, u = bounds.lb, bounds.ub
    v = nu
    assert (l <= u).all(), "Invalid bounds: l > u"

    on, off, amb = get_relu_masks(l, u)
    reduce_dims = tuple(range(1, v.dim()))
    if alpha is None:
        d = torch.zeros_like(l)
        d = torch.where(on, torch.ones_like(d), d)
        if amb.any():
            denom = (u - l).clamp(min=1e-12)
            d = torch.where(amb, u / denom, d)

        v_out = d * v
        if amb.any():
            # Canonical Wong-Kolter upper-relaxation bias for ambiguous ReLU:
            # for amb neurons (l < 0 < u), chord slope is u/(u-l). For
            # v_out < 0, the upper relaxation contributes bias -v_out * l;
            # this is equivalent to -v_out.clamp(max=0) * l, which correctly
            # handles negative nu without dropping the bias.
            crossing = torch.where(amb, -v_out.clamp(max=0) * l, torch.zeros_like(l))
            contrib = crossing.sum(dim=reduce_dims) if reduce_dims else crossing
        else:
            contrib = torch.zeros(B, dtype=nu.dtype, device=nu.device)
        return v_out, contrib

    _assert_broadcast_to_nu(alpha.shape, nu.shape, "dual_relu_backward", "alpha")
    if not torch.logical_and(alpha >= 0, alpha <= 1).all():
        raise ValueError("dual_relu_backward: alpha values must lie in [0, 1]")

    ones = torch.ones_like(l)
    zeros = torch.zeros_like(l)
    d_up = u / (u - l + 1e-12)
    d_low = torch.where(amb, alpha, torch.where(on, ones, zeros))
    v_out_amb = torch.where(v > 0, d_low * v, d_up * v)
    v_out = torch.where(amb, v_out_amb, torch.where(on, v, zeros))
    crossing = torch.where(amb, -v_out.clamp(max=0) * l, zeros)
    contrib = crossing.sum(dim=reduce_dims) if reduce_dims else crossing
    return v_out, contrib


# ---- BIAS ----
def forward_bias(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for BIAS (``y = x + c``).

    Source: tf_forward.py lines 393-400. Composes via ``_fwd_bias``,
    concretizes, intersects with interval box update; ``stored == out``.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    x_L, x_U = parent_frame
    prev_lb, prev_ub = parent_box.lb, parent_box.ub
    lin = _fwd_bias(L, parent_lin)
    crown_lb, crown_ub = _concretize(lin, x_L, x_U)
    int_lb, int_ub = _box_bias(L, prev_lb, prev_ub)
    lb, ub = _intersect_boxes(crown_lb, crown_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out
    return stored, out, lin, parent_frame


def backward_bias(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                  preds: List[int]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    nu_out, contrib = dual_bias_backward(nu, L.params["c"])
    assert len(preds) == 1, f"BIAS expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_bias_backward(nu: torch.Tensor, c: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """y = x + c ; v_out = nu, contrib = -(v · c)."""
    return nu, _bias_contrib(nu, c, "dual_bias_backward")


# ---- SCALE ----
def forward_scale(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for SCALE (``y = a * x``, element-wise).

    Source: tf_forward.py lines 402-409. Composes via ``_fwd_scale``,
    concretizes, intersects with interval box update; ``stored == out``.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    x_L, x_U = parent_frame
    prev_lb, prev_ub = parent_box.lb, parent_box.ub
    lin = _fwd_scale(L, parent_lin)
    crown_lb, crown_ub = _concretize(lin, x_L, x_U)
    int_lb, int_ub = _box_scale(L, prev_lb, prev_ub)
    lb, ub = _intersect_boxes(crown_lb, crown_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out
    return stored, out, lin, parent_frame


def backward_scale(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                   preds: List[int]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    nu_out, contrib = dual_scale_backward(nu, L.params["a"])
    assert len(preds) == 1, f"SCALE expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_scale_backward(nu: torch.Tensor, a: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """y = a * x ; v_out = a * nu, contrib = 0."""
    v_out = _scale_mul(nu, a, "dual_scale_backward")
    contrib = torch.zeros(nu.shape[0], dtype=nu.dtype, device=nu.device)
    return v_out, contrib


# ---- BN ----
def forward_bn(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound | Patches],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound | Patches, Frame]:
    """Forward handler for BN (``y = A * x + c``, element-wise).

    Source: tf_forward.py lines 411-418. Composes via ``_fwd_bn``,
    concretizes, intersects with interval box update; ``stored == out``.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    x_L, x_U = parent_frame
    prev_lb, prev_ub = parent_box.lb, parent_box.ub
    if isinstance(parent_lin, Patches):
        lin = _forward_bn_patches(L, parent_lin, parent_box)
        lb, ub = _box_bn(L, prev_lb, prev_ub)
        out = Bounds(lb, ub)
        stored = out
        return stored, out, lin, parent_frame
    lin = _fwd_bn(L, parent_lin)
    crown_lb, crown_ub = _concretize(lin, x_L, x_U)
    int_lb, int_ub = _box_bn(L, prev_lb, prev_ub)
    lb, ub = _intersect_boxes(crown_lb, crown_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out
    return stored, out, lin, parent_frame


def backward_bn(L: Any, nu: torch.Tensor | Patches, bounds_dict: Dict[int, Bounds],
                preds: List[int]) -> Tuple[List[torch.Tensor | Patches], torch.Tensor]:
    if isinstance(nu, Patches):
        bounds = bounds_dict.get(L.id)
        if bounds is None:
            raise ValueError(f"backward_bn: layer {L.id} missing bounds in bounds_dict")
        nu_out, contrib = _backward_bn_patches(L, nu, bounds)
        assert len(preds) == 1, f"BN expects 1 predecessor, got {len(preds)}"
        return [nu_out], contrib
    nu_out, contrib = dual_bn_backward(nu, L.params["A"], L.params["c"])
    assert len(preds) == 1, f"BN expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_bn_backward(nu: torch.Tensor, A: torch.Tensor, c: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """y = A*x + c ; v_out = A*nu, contrib = -(v · c)."""
    scale = _align(A, nu.reshape(nu.shape[0], -1).shape[1])
    bias = _align(c, nu.reshape(nu.shape[0], -1).shape[1])
    v_out = _scale_mul(nu, scale, "dual_bn_backward")
    contrib = _bias_contrib(nu, bias, "dual_bn_backward")
    return v_out, contrib
