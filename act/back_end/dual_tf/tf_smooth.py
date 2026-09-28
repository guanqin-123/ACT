#===- act/back_end/dual_tf/tf_smooth.py - Smooth Activation Dual TF -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# Batch-aware smooth (S-shaped) activation backward.
# nu: [B, *shape] -> v_out: [B, *shape], contrib: [B].
#===---------------------------------------------------------------------===#

# Note: Gradient enablement for dual backward helpers is governed by the
# caller's torch.set_grad_enabled() context (see DualSolver.evaluate_spec).
# @torch.no_grad() decorators on these helpers were removed to allow
# gradient flow during robust training; verify_once / verify_bab paths
# remain under no_grad via their own outer guards.

import math

import torch
from typing import Tuple, Callable, Dict, Any, List
from act.back_end.core import Bounds
from act.back_end.interval_tf.tf_mlp import _sin_interval, _cos_interval, _quantize_params_flat, _quantize_qdq_value
from .tf_forward import (
    LinearBound, Frame, _concretize, _fwd_elementwise_planes, _intersect_boxes,
    _reset_forward_box,
)


# ---- Shared primitives ----

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)

def dsigmoid(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x); return s * (1 - s)

def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)

def dtanh(x: torch.Tensor) -> torch.Tensor:
    return 1 - torch.tanh(x) ** 2

def erf(x: torch.Tensor) -> torch.Tensor:
    return torch.erf(x)

def derf(x: torch.Tensor) -> torch.Tensor:
    return (2.0 / math.sqrt(math.pi)) * torch.exp(-x * x)


def _dual_constant_box_backward(nu: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, M: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    BM = nu.shape[0]
    assert BM % M == 0, f"constant-box backward: nu batch {BM} not divisible by M={M}"
    B = BM // M
    v_flat = nu.flatten(start_dim=1)
    l_B = lower.flatten(start_dim=1) if lower.dim() >= 2 else lower.flatten().unsqueeze(0).expand(B, -1)
    u_B = upper.flatten(start_dim=1) if upper.dim() >= 2 else upper.flatten().unsqueeze(0).expand(B, -1)
    n = min(v_flat.shape[-1], l_B.shape[-1])
    if v_flat.shape[-1] != l_B.shape[-1]:
        v_flat = v_flat[..., :n]
        l_B = l_B[..., :n]
        u_B = u_B[..., :n]
    l = l_B.unsqueeze(1)
    u = u_B.unsqueeze(1)
    v = v_flat.view(B, M, n)
    b = torch.where(v >= 0, l, u)
    v_out = torch.zeros_like(v)
    contrib = (v * b).sum(dim=-1).view(BM)
    return v_out.view(BM, n), contrib


# Bisection steps for the tangency point of the crossing-interval planes. Each
# step halves the bracket, so 40 steps reach ~1e-12 of the interval width, far
# below float32 resolution; the slope is then read at the bracket end that
# over-approximates the exact tangent slope, so the plane stays sound.
_SMOOTH_TANGENT_BISECTIONS: int = 40


def _crossing_tangent_slopes(
    l: torch.Tensor, u: torch.Tensor,
    f_l: torch.Tensor, f_u: torch.Tensor,
    func: Callable[[torch.Tensor], torch.Tensor],
    dfunc: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sound slopes of the two endpoint-anchored planes on a crossing interval.

    For an S-shaped ``f`` (convex on ``x <= 0``, concave on ``x >= 0``) with
    ``l < 0 < u``:

    * the UPPER plane passes through ``(l, f(l))`` and is tangent to ``f`` at a
      point ``d_u`` in ``[0, u]``; the chord slope ``g(x) = (f(x) - f(l)) / (x - l)``
      increases up to ``d_u`` and decreases after it, so ``f'(d_u) = max g`` and
      any slope ``>= f'(d_u)`` keeps the plane above ``f`` on ``(l, u]``. When
      ``h(u) = f'(u) (u - l) - (f(u) - f(l)) >= 0`` the maximum sits at ``u`` and
      the chord itself is the plane;
    * the LOWER plane passes through ``(u, f(u))`` and is tangent at ``d_l`` in
      ``[l, 0]``, mirror image of the above (``h2(d) = f(u) - f(d) - f'(d) (u - d)``,
      chord when ``h2(l) <= 0``).

    ``h`` decreases on ``[0, u]`` and ``h2`` decreases on ``[l, 0]``, so a plain
    bisection brackets the root; the slope is read at the bracket end where
    ``f'`` is the LARGER of the two (``f'`` decreases on the concave side and
    increases on the convex side), which over-approximates the exact tangent
    slope and therefore stays sound. Returns ``(k_lower, k_upper)``.
    """
    # Upper plane anchored at (l, f(l)): root of h on [0, u].
    lo = torch.zeros_like(u)
    hi = u.clone()
    for _ in range(_SMOOTH_TANGENT_BISECTIONS):
        mid = (lo + hi) * 0.5
        h_mid = dfunc(mid) * (mid - l) - (func(mid) - f_l)
        lo = torch.where(h_mid >= 0, mid, lo)
        hi = torch.where(h_mid >= 0, hi, mid)
    k_upper = dfunc(lo)
    chord = (f_u - f_l) / (u - l).clamp(min=1e-12)
    h_u = dfunc(u) * (u - l) - (f_u - f_l)
    k_upper = torch.where(h_u >= 0, chord, k_upper)

    # Lower plane anchored at (u, f(u)): root of h2 on [l, 0].
    lo = l.clone()
    hi = torch.zeros_like(l)
    for _ in range(_SMOOTH_TANGENT_BISECTIONS):
        mid = (lo + hi) * 0.5
        h2_mid = f_u - func(mid) - dfunc(mid) * (u - mid)
        lo = torch.where(h2_mid >= 0, mid, lo)
        hi = torch.where(h2_mid >= 0, hi, mid)
    k_lower = dfunc(hi)
    h2_l = f_u - f_l - dfunc(l) * (u - l)
    k_lower = torch.where(h2_l <= 0, chord, k_lower)
    return k_lower, k_upper


def compute_smooth_relaxation(
    l: torch.Tensor, u: torch.Tensor,
    func: Callable[[torch.Tensor], torch.Tensor],
    dfunc: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sound linear relaxation ``(k_lo, b_lo, k_hi, b_hi)`` of an S-shaped f on [l, u].

    ``f`` must be increasing, convex on ``x <= 0`` and concave on ``x >= 0``
    (sigmoid, tanh, erf). Per element:

    * ``u <= 0`` (convex): lower plane = tangent at the midpoint, upper = chord;
    * ``l >= 0`` (concave): lower plane = chord, upper = tangent at the midpoint;
    * ``l < 0 < u`` (crossing): upper plane through ``(l, f(l))`` tangent to the
      concave side, lower plane through ``(u, f(u))`` tangent to the convex side
      (:func:`_crossing_tangent_slopes`); each degrades to the chord when the
      chord is already the tight envelope on that side.

    Works element-wise on any broadcastable shape (including batched [B, n]).
    """
    from .tf_mlp import _repair_degenerate_interval
    u = _repair_degenerate_interval(l, u, "smooth_relaxation")
    f_l, f_u = func(l), func(u)
    k_chord = (f_u - f_l) / (u - l).clamp(min=1e-12)
    b_chord = f_l - k_chord * l
    m = (l + u) * 0.5
    k_tan = dfunc(m)
    b_tan = func(m) - k_tan * m

    convex = u <= 0
    concave = l >= 0
    crossing = ~(convex | concave)
    k_lower = torch.where(convex, k_tan, k_chord)
    b_lower = torch.where(convex, b_tan, b_chord)
    k_upper = torch.where(concave, k_tan, k_chord)
    b_upper = torch.where(concave, b_tan, b_chord)

    if bool(crossing.any()):
        # Bisection on the crossing entries only; the others are masked out.
        l_c = torch.where(crossing, l, torch.full_like(l, -1.0))
        u_c = torch.where(crossing, u, torch.full_like(u, 1.0))
        k_lo_c, k_hi_c = _crossing_tangent_slopes(
            l_c, u_c, func(l_c), func(u_c), func, dfunc)
        k_lower = torch.where(crossing, k_lo_c, k_lower)
        b_lower = torch.where(crossing, f_u - k_lo_c * u, b_lower)
        k_upper = torch.where(crossing, k_hi_c, k_upper)
        b_upper = torch.where(crossing, f_l - k_hi_c * l, b_upper)

    return k_lower, b_lower, k_upper, b_upper


def forward_smooth_planes(
    parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
    int_lb: torch.Tensor, int_ub: torch.Tensor,
    planes: Callable[[torch.Tensor, torch.Tensor],
                     Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward state of an element-wise activation with interval box ``[int_lb, int_ub]``.

    When the predecessor carries an explicit linear frame, the sound planes
    ``planes(pre_lb, pre_ub) -> (k_lo, b_lo, k_hi, b_hi)`` are composed through
    it (:func:`_fwd_elementwise_planes`), concretized over the frame and
    intersected with the interval box, and the frame is kept -- the same
    pattern as :func:`tf_mlp.forward_relu`. A lazy-identity frame keeps the
    interval-reset behaviour unchanged.
    """
    parent_box, parent_lin, parent_frame = parent_boxes[0], parent_lins[0], parent_frames[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    new_lin = None
    if parent_lin.A_lb is not None and parent_lin.A_ub is not None:
        k_lo, b_lo, k_hi, b_hi = planes(pre_lb, pre_ub)
        new_lin = _fwd_elementwise_planes(parent_lin, k_lo, b_lo, k_hi, b_hi)
    if new_lin is None:
        out = Bounds(int_lb, int_ub)
        stored = out if post_activation else Bounds(pre_lb, pre_ub)
        lin, frame = _reset_forward_box(int_lb, int_ub, device, dtype)
        return stored, out, lin, frame
    lin_lb, lin_ub = _concretize(new_lin, *parent_frame)
    lb, ub = _intersect_boxes(lin_lb, lin_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    if post_activation:
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
        return stored, out, lin, frame
    return stored, out, new_lin, parent_frame


def dual_smooth_backward(
    nu: torch.Tensor, bounds: Bounds,
    func: Callable[[torch.Tensor], torch.Tensor],
    dfunc: Callable[[torch.Tensor], torch.Tensor],
    M: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched smooth activation backward (sigmoid/tanh) with lazy M-broadcast.

    Same broadcast pattern as :func:`dual_relu_backward`: relaxation
    coefficients ``(k_lower, b_lower, k_upper, b_upper)`` depend on the
    bounds only (spec-agnostic) and are computed once at ``[B, 1, n]``,
    then broadcast against ``nu`` viewed at ``[B, M, n]``.

    Args:
        nu: dual variable, shape ``[B*M, *shape]``.
        bounds: layer bounds, shape ``[B, *shape]``. NOT M-expanded.
        func/dfunc: activation function and its derivative.
        M: spec-row multiplicity (default 1).
    """
    BM = nu.shape[0]
    assert BM % M == 0, f"dual_smooth_backward: nu batch {BM} not divisible by M={M}"
    B = BM // M

    v_flat = nu.flatten(start_dim=1)                              # [BM, n]
    l_B = bounds.lb.flatten(start_dim=1) if bounds.lb.dim() >= 2 \
          else bounds.lb.flatten().unsqueeze(0).expand(B, -1)
    u_B = bounds.ub.flatten(start_dim=1) if bounds.ub.dim() >= 2 \
          else bounds.ub.flatten().unsqueeze(0).expand(B, -1)
    n = min(v_flat.shape[-1], l_B.shape[-1])
    if v_flat.shape[-1] != l_B.shape[-1]:
        v_flat = v_flat[..., :n]
        l_B = l_B[..., :n]
        u_B = u_B[..., :n]

    l = l_B.unsqueeze(1)                                          # [B, 1, n]
    u = u_B.unsqueeze(1)                                          # [B, 1, n]
    k_lower, b_lower, k_upper, b_upper = compute_smooth_relaxation(l, u, func, dfunc)

    v = v_flat.view(B, M, n)                                      # [B, M, n] view
    v_pos = v >= 0                                                # [B, M, n]
    k = torch.where(v_pos, k_lower, k_upper)                      # broadcast → [B, M, n]
    b = torch.where(v_pos, b_lower, b_upper)

    v_out = v * k                                                 # [B, M, n]
    contrib = (v * b).sum(dim=-1).view(BM)                        # [BM]
    return v_out.view(BM, n), contrib


# ---- SIGMOID ----

@torch.no_grad()
def forward_sigmoid(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward pass for SIGMOID: monotone interval box, frame kept when explicit."""
    parent_box = parent_boxes[0]
    return forward_smooth_planes(
        parent_boxes, parent_lins, parent_frames, post_activation, device, dtype,
        torch.sigmoid(parent_box.lb), torch.sigmoid(parent_box.ub),
        lambda l, u: compute_smooth_relaxation(l, u, sigmoid, dsigmoid),
    )


def backward_sigmoid(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                     preds: List[int], M: int = 1, alpha=None
                     ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_sigmoid: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_sigmoid_backward(nu, bounds, M)
    assert len(preds) == 1, f"SIGMOID expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_sigmoid_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    return dual_smooth_backward(nu, bounds, sigmoid, dsigmoid, M)


# ---- TANH ----

@torch.no_grad()
def forward_tanh(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward pass for TANH: monotone interval box, frame kept when explicit."""
    parent_box = parent_boxes[0]
    return forward_smooth_planes(
        parent_boxes, parent_lins, parent_frames, post_activation, device, dtype,
        torch.tanh(parent_box.lb), torch.tanh(parent_box.ub),
        lambda l, u: compute_smooth_relaxation(l, u, tanh, dtanh),
    )


def backward_tanh(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                  preds: List[int], M: int = 1, alpha=None
                  ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_tanh: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_tanh_backward(nu, bounds, M)
    assert len(preds) == 1, f"TANH expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_tanh_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    return dual_smooth_backward(nu, bounds, tanh, dtanh, M)


# ---- ERF ----

@torch.no_grad()
def forward_erf(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward pass for ERF activation, mirroring TANH."""
    parent_box = parent_boxes[0]
    return forward_smooth_planes(
        parent_boxes, parent_lins, parent_frames, post_activation, device, dtype,
        torch.erf(parent_box.lb), torch.erf(parent_box.ub),
        lambda l, u: compute_smooth_relaxation(l, u, erf, derf),
    )


def backward_erf(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                 preds: List[int], M: int = 1, alpha=None
                 ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_erf: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_erf_backward(nu, bounds, M)
    assert len(preds) == 1, f"ERF expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_erf_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    return dual_smooth_backward(nu, bounds, erf, derf, M)


# ---- SQRT ----

@torch.no_grad()
def forward_sqrt(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    lo_e = torch.clamp(pre_lb, min=0.0)
    hi_e = torch.clamp(pre_ub, min=0.0)
    out = Bounds(torch.sqrt(lo_e), torch.sqrt(hi_e))
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    lin, frame = _reset_forward_box(out.lb, out.ub, device, dtype)
    return stored, out, lin, frame


def backward_sqrt(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                  preds: List[int], M: int = 1, alpha=None
                  ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_sqrt: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_sqrt_backward(nu, bounds, M)
    assert len(preds) == 1, f"SQRT expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_sqrt_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    lo_e = torch.clamp(bounds.lb, min=0.0)
    hi_e = torch.clamp(bounds.ub, min=0.0)
    return _dual_constant_box_backward(nu, torch.sqrt(lo_e), torch.sqrt(hi_e), M)


# ---- SIN ----

@torch.no_grad()
def forward_sin(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    out = _sin_interval(pre_lb, pre_ub)
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    lin, frame = _reset_forward_box(out.lb, out.ub, device, dtype)
    return stored, out, lin, frame


def backward_sin(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                 preds: List[int], M: int = 1, alpha=None
                 ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_sin: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_sin_backward(nu, bounds, M)
    assert len(preds) == 1, f"SIN expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_sin_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    box = _sin_interval(bounds.lb, bounds.ub)
    return _dual_constant_box_backward(nu, box.lb, box.ub, M)


# ---- COS ----

@torch.no_grad()
def forward_cos(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    out = _cos_interval(pre_lb, pre_ub)
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    lin, frame = _reset_forward_box(out.lb, out.ub, device, dtype)
    return stored, out, lin, frame


def backward_cos(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                 preds: List[int], M: int = 1, alpha=None
                 ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_cos: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_cos_backward(nu, bounds, M)
    assert len(preds) == 1, f"COS expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_cos_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    box = _cos_interval(bounds.lb, bounds.ub)
    return _dual_constant_box_backward(nu, box.lb, box.ub, M)


# ---- QUANTIZE / QDQ real-valued map ----

@torch.no_grad()
def forward_quantize(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    n = pre_lb.numel()
    scale, zp, qmin, qmax = _quantize_params_flat(L, n, pre_lb.device, pre_lb.dtype)
    lo = pre_lb.reshape(-1)
    hi = torch.maximum(pre_ub.reshape(-1), lo)
    z_lo = _quantize_qdq_value(lo, scale, zp, qmin, qmax)
    z_hi = _quantize_qdq_value(hi, scale, zp, qmin, qmax)
    out = Bounds(torch.minimum(z_lo, z_hi).reshape_as(pre_lb), torch.maximum(z_lo, z_hi).reshape_as(pre_ub))
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    lin, frame = _reset_forward_box(out.lb, out.ub, device, dtype)
    return stored, out, lin, frame


def backward_quantize(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                      preds: List[int], M: int = 1, alpha=None
                      ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_quantize: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_quantize_backward(L, nu, bounds, M)
    assert len(preds) == 1, f"QUANTIZE expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_quantize_backward(L: Any, nu: torch.Tensor, bounds: Bounds, M: int = 1):
    n = bounds.lb.numel()
    scale, zp, qmin, qmax = _quantize_params_flat(L, n, bounds.lb.device, bounds.lb.dtype)
    lo = bounds.lb.reshape(-1)
    hi = torch.maximum(bounds.ub.reshape(-1), lo)
    z_lo = _quantize_qdq_value(lo, scale, zp, qmin, qmax).reshape_as(bounds.lb)
    z_hi = _quantize_qdq_value(hi, scale, zp, qmin, qmax).reshape_as(bounds.ub)
    return _dual_constant_box_backward(nu, torch.minimum(z_lo, z_hi), torch.maximum(z_lo, z_hi), M)
