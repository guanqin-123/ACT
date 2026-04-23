#===- act/back_end/dual_tf/tf_forward.py - Forward Bounds ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
"""Batched DAG-aware forward bound propagation for DualTF.

Per-layer linear state lives in ``lin_state`` and frame state in ``frame_dict``.
Traversal uses Kahn's algorithm over ``net.preds`` / ``net.succs``. ADD and
CONCAT read predecessor state explicitly for fan-out / fan-in DAGs such as
ResNet skips. Returned bounds stay batch-first flattened ``[B, n]`` tensors,
with activation bounds stored PRE-activation unless ``post_activation=True``.
"""
#===---------------------------------------------------------------------===#

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportMissingParameterType=false, reportUntypedFunctionDecorator=false, reportDeprecated=false

import contextlib
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, cast

from act.back_end.core import Bounds, Layer, Net
from act.back_end.layer_schema import LayerKind
from act.util.device_manager import get_default_device, get_default_dtype

if TYPE_CHECKING:
    from act.back_end.bab.eta import EtaState


@dataclass
class LinearBound:
    A_lb: torch.Tensor
    b_lb: torch.Tensor
    A_ub: torch.Tensor
    b_ub: torch.Tensor

Frame = Tuple[torch.Tensor, torch.Tensor]  # (x_L, x_U) over which lin is defined


def _concretize(lin: LinearBound, x_L: torch.Tensor, x_U: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concretize dual-track affine bounds over a batched input box."""
    A_lb_p = lin.A_lb.clamp(min=0)
    A_lb_n = lin.A_lb.clamp(max=0)
    A_ub_p = lin.A_ub.clamp(min=0)
    A_ub_n = lin.A_ub.clamp(max=0)
    lb = (
        torch.einsum("boi,bi->bo", A_lb_p, x_L)
        + torch.einsum("boi,bi->bo", A_lb_n, x_U)
        + lin.b_lb
    )
    ub = (
        torch.einsum("boi,bi->bo", A_ub_p, x_U)
        + torch.einsum("boi,bi->bo", A_ub_n, x_L)
        + lin.b_ub
    )
    return lb, ub

def _identity_lin(B: int, n: int, device, dtype) -> LinearBound:
    eye = torch.eye(n, device=device, dtype=dtype).unsqueeze(0).expand(B, n, n).contiguous()
    zeros = torch.zeros(B, n, device=device, dtype=dtype)
    return LinearBound(A_lb=eye, b_lb=zeros, A_ub=eye.clone(), b_ub=zeros.clone())

def _reset_lin(lb: torch.Tensor, ub: torch.Tensor, device, dtype
               ) -> Tuple[LinearBound, torch.Tensor, torch.Tensor]:
    B, n = lb.shape[0], lb.shape[1]
    return _identity_lin(B, n, device, dtype), lb.clone(), ub.clone()

def _match_lin_input_dim(lin: LinearBound, n_in: int) -> LinearBound:
    """Pad or truncate the current output-feature axis to size n_in."""
    curr_out = lin.A_lb.shape[1]
    if curr_out == n_in:
        return lin

    B, _, input_dim = lin.A_lb.shape
    if curr_out < n_in:
        pad = n_in - curr_out
        zeros_A = torch.zeros(B, pad, input_dim, device=lin.A_lb.device, dtype=lin.A_lb.dtype)
        zeros_b = torch.zeros(B, pad, device=lin.b_lb.device, dtype=lin.b_lb.dtype)
        return LinearBound(
            A_lb=torch.cat([lin.A_lb, zeros_A], dim=1),
            b_lb=torch.cat([lin.b_lb, zeros_b], dim=1),
            A_ub=torch.cat([lin.A_ub, zeros_A.clone()], dim=1),
            b_ub=torch.cat([lin.b_ub, zeros_b.clone()], dim=1),
        )

    return LinearBound(
        A_lb=lin.A_lb[:, :n_in, :],
        b_lb=lin.b_lb[:, :n_in],
        A_ub=lin.A_ub[:, :n_in, :],
        b_ub=lin.b_ub[:, :n_in],
    )

def _intersect_boxes(lb_a: torch.Tensor, ub_a: torch.Tensor,
                     lb_b: torch.Tensor, ub_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.maximum(lb_a, lb_b), torch.minimum(ub_a, ub_b)

def _align_batch(a: torch.Tensor, n: int) -> torch.Tensor:
    if a.shape[1] == n:
        return a
    if a.shape[1] > n:
        return a[:, :n]
    repeats = (n + a.shape[1] - 1) // a.shape[1]
    return a.repeat(1, repeats)[:, :n]

def _shape_list(shape_param: object) -> Optional[list[int]]:
    return [int(v) for v in shape_param] if isinstance(shape_param, (tuple, list)) else None

def _int_param(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    return value if isinstance(value, int) else default


def _box_dense(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    W = layer.params["weight"]
    b = layer.params.get("bias")
    lb = _align_batch(lb, W.shape[1])
    ub = _align_batch(ub, W.shape[1])
    W_pos = W.clamp(min=0)
    W_neg = W.clamp(max=0)
    out_lb = lb @ W_pos.transpose(0, 1) + ub @ W_neg.transpose(0, 1)
    out_ub = ub @ W_pos.transpose(0, 1) + lb @ W_neg.transpose(0, 1)
    if b is not None:
        bias_vec = _align(b.flatten(), W.shape[0])
        out_lb = out_lb + bias_vec
        out_ub = out_ub + bias_vec
    return out_lb, out_ub


def _box_bias(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    c = _align(layer.params["c"], lb.shape[1])
    return lb + c, ub + c


def _box_scale(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    a = _align(layer.params["a"], lb.shape[1])
    out_lb = torch.where(a >= 0, a * lb, a * ub)
    out_ub = torch.where(a >= 0, a * ub, a * lb)
    return out_lb, out_ub


def _box_bn(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    A_bn = _align(layer.params["A"], lb.shape[1])
    c = _align(layer.params["c"], lb.shape[1])
    out_lb = torch.where(A_bn >= 0, A_bn * lb + c, A_bn * ub + c)
    out_ub = torch.where(A_bn >= 0, A_bn * ub + c, A_bn * lb + c)
    return out_lb, out_ub


def _box_relu(lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact interval propagation for ReLU."""
    return lb.clamp(min=0), ub.clamp(min=0)


def _box_lrelu(lb: torch.Tensor, ub: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact interval propagation for LeakyReLU."""
    alpha_tensor = torch.full_like(lb, alpha)
    out_lb = torch.where(lb >= 0, lb, alpha_tensor * lb)
    out_ub = torch.where(ub <= 0, alpha_tensor * ub, ub)
    return out_lb, out_ub


def _store_forward_state(bounds_dict: Dict[int, Bounds],
                          box_state: Dict[int, Bounds],
                          lin_state: Dict[int, LinearBound],
                          frame_dict: Dict[int, Frame],
                          layer_id: int,
                          stored: Bounds,
                          out_box: Bounds,
                          lin: LinearBound,
                          frame: Frame) -> None:
    """Store public bounds plus internal forward state for a layer."""
    bounds_dict[layer_id] = stored.copy()
    box_state[layer_id] = out_box.copy()
    lin_state[layer_id] = lin
    frame_dict[layer_id] = frame


def _reset_forward_box(lb: torch.Tensor, ub: torch.Tensor, device, dtype
                       ) -> Tuple[LinearBound, Tuple[torch.Tensor, torch.Tensor]]:
    """Reset dual-track state on a concrete box."""
    lin, x_L, x_U = _reset_lin(lb, ub, device, dtype)
    return lin, (x_L, x_U)


def _topological_sort(net: Net) -> List[int]:
    """Return a Kahn topological order over the ACT DAG."""
    layer_ids = [layer.id for layer in net.layers]
    in_deg: Dict[int, int] = {lid: len(net.preds.get(lid, [])) for lid in layer_ids}
    queue = deque(lid for lid in layer_ids if in_deg[lid] == 0)
    order: List[int] = []
    while queue:
        lid = queue.popleft()
        order.append(lid)
        for succ in net.succs.get(lid, []):
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)
    if len(order) != len(layer_ids):
        raise ValueError(f"compute_forward_bounds: graph has cycle or disconnected layers ({len(order)}/{len(layer_ids)} sorted)")
    return order


def _sum_interval_bounds(boxes: List[Bounds]) -> Bounds:
    lbs = [box.lb.flatten(start_dim=1) for box in boxes]
    ubs = [box.ub.flatten(start_dim=1) for box in boxes]
    widths = [lb.shape[1] for lb in lbs]
    if len(set(widths)) > 1:
        raise ValueError(
            f"_sum_interval_bounds: predecessor widths must match, got {widths}. "
            f"Upstream graph produced shape-mismatched ADD inputs."
        )
    return Bounds(sum(lbs[1:], lbs[0]), sum(ubs[1:], ubs[0]))


def _sum_linear_bounds(lins: List[LinearBound]) -> LinearBound:
    """Sum dual-track affine bounds from multiple predecessors."""
    return LinearBound(
        A_lb=sum((lin.A_lb for lin in lins[1:]), lins[0].A_lb),
        b_lb=sum((lin.b_lb for lin in lins[1:]), lins[0].b_lb),
        A_ub=sum((lin.A_ub for lin in lins[1:]), lins[0].A_ub),
        b_ub=sum((lin.b_ub for lin in lins[1:]), lins[0].b_ub),
    )


def _apply_eta_clamp(layer_id: int, out: Bounds,
                     eta_state: "Optional[EtaState]") -> Bounds:
    """Clamp pre-activation concrete bounds per eta split constraints.

    Sign convention (canonical beta-CROWN):
        sign > 0  (INACTIVE, z <= point)  -> ub ← min(ub, point)
        sign < 0  (ACTIVE,   z >= point)  -> lb ← max(lb, point)
        sign == 0                         -> no change

    Fast path (returns ``out`` unchanged): ``eta_state`` is None,
    ``fast_path_skip()`` is True, or this layer has no eta entry.
    """
    if eta_state is None or eta_state.fast_path_skip():
        return out
    if layer_id not in eta_state.sign:
        return out

    sgn = eta_state.sign[layer_id]
    pt = eta_state.point[layer_id]

    orig_shape = out.lb.shape
    B = orig_shape[0]
    lb_flat = out.lb.reshape(B, -1)
    ub_flat = out.ub.reshape(B, -1)

    if lb_flat.shape[-1] != sgn.shape[-1]:
        raise ValueError(
            f"_apply_eta_clamp: layer {layer_id} bounds width {lb_flat.shape[-1]} "
            f"does not match eta sgn width {sgn.shape[-1]}. EtaState must be sized "
            f"to match the layer's flattened pre-activation width."
        )
    if lb_flat.shape[-1] == 0:
        return out

    sgn_n = sgn.to(device=lb_flat.device, dtype=lb_flat.dtype)
    pt_n = pt.to(device=lb_flat.device, dtype=lb_flat.dtype)
    inactive = sgn_n > 0
    active = sgn_n < 0

    new_ub = torch.where(inactive, torch.minimum(ub_flat, pt_n), ub_flat)
    new_lb = torch.where(active, torch.maximum(lb_flat, pt_n), lb_flat)

    return Bounds(lb=new_lb.reshape(orig_shape), ub=new_ub.reshape(orig_shape))


def compute_forward_bounds(net: Net, input_lb: torch.Tensor, input_ub: torch.Tensor,
                           post_activation: bool = False,
                           eta_state: "Optional[EtaState]" = None,
                           alphas: Optional[Dict[int, torch.Tensor]] = None) -> Dict[int, Bounds]:
    """Forward bounds, natively batched with singleton auto-promotion.

    When ``eta_state`` is supplied, pre-activation bounds of each layer id
    present in ``eta_state.sign`` are clamped in place at the per-layer
    dispatch site. ``eta_state=None`` (or empty via ``fast_path_skip()``)
    preserves bit-identical behavior to the un-clamped path.
    """
    with torch.no_grad() if alphas is None else contextlib.nullcontext():
        # Lazy import to break circular dep (dual_tf imports compute_forward_bounds)
        from .dual_tf import DualTF

        device, dtype = get_default_device(), get_default_dtype()
        if (
            input_lb.dtype != dtype or input_lb.device != device
            or input_ub.dtype != dtype or input_ub.device != device
        ):
            input_lb = input_lb.to(device=device, dtype=dtype)
            input_ub = input_ub.to(device=device, dtype=dtype)

        if input_lb.dim() < 2:
            input_lb = input_lb.unsqueeze(0)
            input_ub = input_ub.unsqueeze(0)

        B = input_lb.shape[0]
        lb_in = input_lb.reshape(B, -1)
        ub_in = input_ub.reshape(B, -1)
        input_dim = lb_in.shape[1]

        bounds_dict: Dict[int, Bounds] = {}
        box_state: Dict[int, Bounds] = {}
        lin_state: Dict[int, LinearBound] = {}
        frame_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        topo_order = _topological_sort(net)
        entry_box = Bounds(lb_in, ub_in)
        entry_lin = _identity_lin(B, input_dim, device, dtype)
        entry_frame = (lb_in, ub_in)

        for lid in topo_order:
            layer = net.by_id[lid]
            lid = layer.id
            kind = layer.kind.upper()
            preds = list(net.preds.get(lid, []) or [])

            if not preds:
                if kind not in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value):
                    raise ValueError(f"compute_forward_bounds: layer {lid} kind '{kind}' has no predecessors and is not INPUT / INPUT_SPEC")
                _store_forward_state(
                    bounds_dict,
                    box_state,
                    lin_state,
                    frame_dict,
                    lid,
                    entry_box,
                    entry_box,
                    entry_lin,
                    entry_frame,
                )
                continue

            missing = [pid for pid in preds if pid not in box_state or pid not in lin_state or pid not in frame_dict]
            if missing:
                raise ValueError(f"compute_forward_bounds: layer {lid} missing predecessor state for {missing}")

            if len(preds) >= 2 or kind in (LayerKind.ADD.value, LayerKind.CONCAT.value):
                handler = DualTF._FORWARD_REGISTRY.get(kind)
                if handler is None:
                    raise ValueError(
                        f"compute_forward_bounds: unknown multi-pred layer kind '{kind}' "
                        f"at layer {lid}. Registered kinds: "
                        f"{sorted(DualTF._FORWARD_REGISTRY.keys())}"
                    )
                pred_boxes  = [box_state[pid]   for pid in preds]
                pred_lins   = [lin_state[pid]   for pid in preds]
                pred_frames = [frame_dict[pid]  for pid in preds]
                stored, out, lin, frame = handler(
                    layer, pred_boxes, pred_lins, pred_frames, preds,
                    post_activation, device, dtype,
                )
                out = _apply_eta_clamp(lid, out, eta_state)
                stored = _apply_eta_clamp(lid, stored, eta_state)
                _store_forward_state(bounds_dict, box_state, lin_state, frame_dict,
                                     lid, stored, out, lin, frame)
                continue

            handler = DualTF._FORWARD_REGISTRY.get(kind)
            if handler is None:
                raise ValueError(
                    f"compute_forward_bounds: unknown layer kind '{kind}' at layer {lid}. "
                    f"Registered kinds: {sorted(DualTF._FORWARD_REGISTRY.keys())}"
                )
            pred_boxes  = [box_state[preds[0]]]
            pred_lins   = [lin_state[preds[0]]]
            pred_frames = [frame_dict[preds[0]]]
            # Keep registry handler signatures stable; Phase 2 per-layer learned
            # alphas are consumed only by RELU here. LRELU keeps its fixed param.
            if kind == LayerKind.RELU.value and alphas is not None and lid in alphas:
                pre_lb, pre_ub = pred_boxes[0].lb, pred_boxes[0].ub
                x_L, x_U = pred_frames[0]
                lin = _fwd_relu(pred_lins[0], pre_lb, pre_ub, alpha=alphas[lid], layer_id=lid)
                crown_lb, crown_ub = _concretize(lin, x_L, x_U)
                int_lb, int_ub = _box_relu(pre_lb, pre_ub)
                lb, ub = _intersect_boxes(crown_lb, crown_ub, int_lb, int_ub)
                out = Bounds(lb, ub)
                stored = out if post_activation else Bounds(pre_lb, pre_ub)
                frame = pred_frames[0]
                if post_activation:
                    lin, frame = _reset_forward_box(lb, ub, device, dtype)
            else:
                stored, out, lin, frame = handler(
                    layer, pred_boxes, pred_lins, pred_frames, preds,
                    post_activation, device, dtype,
                )
            out = _apply_eta_clamp(lid, out, eta_state)
            stored = _apply_eta_clamp(lid, stored, eta_state)
            _store_forward_state(bounds_dict, box_state, lin_state, frame_dict,
                                 lid, stored, out, lin, frame)

        return bounds_dict


def _fwd_dense(layer: Layer, lin: LinearBound) -> LinearBound:
    """Compose dual-track affine bounds through a dense layer."""
    W = layer.params["weight"]
    b = layer.params.get("bias")
    lin = _match_lin_input_dim(lin, W.shape[1])
    W_pos = W.clamp(min=0)
    W_neg = W.clamp(max=0)
    bias_vec = torch.zeros(W.shape[0], device=lin.b_lb.device, dtype=lin.b_lb.dtype)
    if b is not None:
        bias_vec = _align(b.flatten(), W.shape[0])
    return LinearBound(
        A_lb=torch.einsum("oc,bci->boi", W_pos, lin.A_lb) + torch.einsum("oc,bci->boi", W_neg, lin.A_ub),
        b_lb=torch.einsum("oc,bc->bo", W_pos, lin.b_lb) + torch.einsum("oc,bc->bo", W_neg, lin.b_ub) + bias_vec,
        A_ub=torch.einsum("oc,bci->boi", W_pos, lin.A_ub) + torch.einsum("oc,bci->boi", W_neg, lin.A_lb),
        b_ub=torch.einsum("oc,bc->bo", W_pos, lin.b_ub) + torch.einsum("oc,bc->bo", W_neg, lin.b_lb) + bias_vec,
    )


def _fwd_relu(lin: LinearBound, lb: torch.Tensor, ub: torch.Tensor,
              alpha: Optional[torch.Tensor] = None,
              layer_id: Optional[int] = None) -> LinearBound:
    """Apply forward ReLU linear relaxation with per-batch alpha choice."""
    on = lb >= 0
    off = ub <= 0
    amb = ~(on | off)
    denom = (ub - lb).clamp(min=1e-12)
    up_slope = torch.where(
        amb,
        ub / denom,
        torch.where(on, torch.ones_like(lb), torch.zeros_like(lb)),
    )
    up_inter = torch.where(amb, -up_slope * lb, torch.zeros_like(lb))
    alpha_heur = torch.where(
        amb,
        (up_slope > 0.5).to(lb.dtype),
        torch.where(on, torch.ones_like(lb), torch.zeros_like(lb)),
    )
    if alpha is None:
        alpha_used = alpha_heur
    else:
        alpha = alpha.to(device=lb.device, dtype=lb.dtype)
        try:
            broadcast_shape = torch.broadcast_shapes(tuple(alpha.shape), tuple(lb.shape))
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
        alpha_b = torch.broadcast_to(alpha, lb.shape)
        if __debug__:
            assert bool((alpha_b >= 0).all().item()), "ReLU alpha must be >= 0"
            assert bool((alpha_b <= 1).all().item()), "ReLU alpha must be <= 1"
        alpha_used = torch.where(amb, alpha_b, alpha_heur)
    return LinearBound(
        A_lb=alpha_used.unsqueeze(-1) * lin.A_lb,
        b_lb=alpha_used * lin.b_lb,
        A_ub=up_slope.unsqueeze(-1) * lin.A_ub,
        b_ub=up_slope * lin.b_ub + up_inter,
    )


def _fwd_bias(layer: Layer, lin: LinearBound) -> LinearBound:
    """Compose dual-track affine bounds through a bias layer."""
    c = _align(layer.params["c"], lin.b_lb.shape[1])
    return LinearBound(
        A_lb=lin.A_lb,
        b_lb=lin.b_lb + c,
        A_ub=lin.A_ub,
        b_ub=lin.b_ub + c,
    )


def _fwd_scale(layer: Layer, lin: LinearBound) -> LinearBound:
    """Compose dual-track affine bounds through an element-wise scale."""
    a = _align(layer.params["a"], lin.b_lb.shape[1])
    a_pos = a.clamp(min=0)
    a_neg = a.clamp(max=0)
    a_pos_A = a_pos.view(1, -1, 1)
    a_neg_A = a_neg.view(1, -1, 1)
    return LinearBound(
        A_lb=a_pos_A * lin.A_lb + a_neg_A * lin.A_ub,
        b_lb=a_pos * lin.b_lb + a_neg * lin.b_ub,
        A_ub=a_pos_A * lin.A_ub + a_neg_A * lin.A_lb,
        b_ub=a_pos * lin.b_ub + a_neg * lin.b_lb,
    )


def _fwd_bn(layer: Layer, lin: LinearBound) -> LinearBound:
    """Compose dual-track affine bounds through batch normalization."""
    A_bn = _align(layer.params["A"], lin.b_lb.shape[1])
    c = _align(layer.params["c"], lin.b_lb.shape[1])
    A_pos = A_bn.clamp(min=0)
    A_neg = A_bn.clamp(max=0)
    A_pos_A = A_pos.view(1, -1, 1)
    A_neg_A = A_neg.view(1, -1, 1)
    return LinearBound(
        A_lb=A_pos_A * lin.A_lb + A_neg_A * lin.A_ub,
        b_lb=A_pos * lin.b_lb + A_neg * lin.b_ub + c,
        A_ub=A_pos_A * lin.A_ub + A_neg_A * lin.A_lb,
        b_ub=A_pos * lin.b_ub + A_neg * lin.b_lb + c,
    )


def _fwd_lrelu(lin: LinearBound, lb: torch.Tensor, ub: torch.Tensor, alpha: float) -> LinearBound:
    """Apply forward triangle linear relaxation for LeakyReLU."""
    on = lb >= 0
    off = ub <= 0
    amb = ~(on | off)
    denom = (ub - lb).clamp(min=1e-12)
    alpha_tensor = torch.full_like(lb, alpha)
    up_slope = torch.where(
        amb,
        (ub - alpha * lb) / denom,
        torch.where(on, torch.ones_like(lb), alpha_tensor),
    )
    up_inter = torch.where(amb, alpha * lb - up_slope * lb, torch.zeros_like(lb))
    low_slope = torch.where(on, torch.ones_like(lb), alpha_tensor)
    return LinearBound(
        A_lb=low_slope.unsqueeze(-1) * lin.A_lb,
        b_lb=low_slope * lin.b_lb,
        A_ub=up_slope.unsqueeze(-1) * lin.A_ub,
        b_ub=up_slope * lin.b_ub + up_inter,
    )


def _fwd_conv2d(layer: Layer, lin: LinearBound) -> Optional[LinearBound]:
    """Propagate dual-track affine bounds through Conv2D via batched F.conv2d."""
    weight = layer.params["weight"]
    bias = layer.params.get("bias")
    stride = layer.params.get("stride", 1)
    padding = layer.params.get("padding", 0)
    dilation = layer.params.get("dilation", 1)
    groups = layer.params.get("groups", 1)
    if isinstance(stride, (list, tuple)):
        stride = stride[0]
    if isinstance(padding, (list, tuple)):
        padding = padding[0]
    if isinstance(dilation, (list, tuple)):
        dilation = dilation[0]

    out_c, in_c_per_g, _, _ = weight.shape
    in_c = in_c_per_g * groups
    B, curr_dim, input_dim = lin.A_lb.shape

    in_h = in_w = 0
    input_shape = _shape_list(layer.params.get("input_shape"))
    if input_shape is not None and len(input_shape) >= 3:
        shape = input_shape
        if len(shape) == 4:
            _, _, h, w = shape
        else:
            _, h, w = shape[-3], shape[-2], shape[-1]
        if h * w * in_c == curr_dim:
            in_h, in_w = h, w

    if in_h == 0 or in_w == 0:
        spatial = curr_dim // in_c if in_c > 0 else 0
        side = int(spatial ** 0.5) if spatial > 0 else 0
        if spatial > 0 and side * side * in_c == curr_dim:
            in_h = in_w = side

    if in_h == 0 or in_w == 0:
        return None

    W_pos = weight.clamp(min=0)
    W_neg = weight.clamp(max=0)

    def conv_A(A_mat: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        A_t = A_mat.transpose(1, 2).contiguous().view(B * input_dim, in_c, in_h, in_w)
        out = F.conv2d(A_t, kernel, None, stride, padding, dilation, groups)
        return out.flatten(start_dim=1).reshape(B, input_dim, -1).transpose(1, 2).contiguous()

    def conv_b(vec: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        b_4d = vec.view(B, in_c, in_h, in_w)
        return F.conv2d(b_4d, kernel, None, stride, padding, dilation, groups).flatten(start_dim=1)

    A_lb_new = conv_A(lin.A_lb, W_pos) + conv_A(lin.A_ub, W_neg)
    A_ub_new = conv_A(lin.A_ub, W_pos) + conv_A(lin.A_lb, W_neg)
    b_lb_new = conv_b(lin.b_lb, W_pos) + conv_b(lin.b_ub, W_neg)
    b_ub_new = conv_b(lin.b_ub, W_pos) + conv_b(lin.b_lb, W_neg)

    if bias is not None:
        out_spatial = b_lb_new.shape[1] // out_c
        bias_bc = bias.view(out_c, 1).expand(out_c, out_spatial).reshape(-1)
        b_lb_new = b_lb_new + bias_bc
        b_ub_new = b_ub_new + bias_bc

    return LinearBound(A_lb=A_lb_new, b_lb=b_lb_new, A_ub=A_ub_new, b_ub=b_ub_new)


def _fwd_conv2d_interval(layer: Layer, lb: torch.Tensor, ub: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback interval Conv2D for when the linear-relaxation path cannot infer shape."""
    weight = layer.params["weight"]
    bias = layer.params.get("bias")
    stride = layer.params.get("stride", 1)
    padding = layer.params.get("padding", 0)
    dilation = layer.params.get("dilation", 1)
    groups = layer.params.get("groups", 1)
    if isinstance(stride, (list, tuple)):
        stride = stride[0]
    if isinstance(padding, (list, tuple)):
        padding = padding[0]
    if isinstance(dilation, (list, tuple)):
        dilation = dilation[0]

    B = lb.shape[0]
    _, in_c_per_g, _, _ = weight.shape
    in_c = in_c_per_g * groups
    input_shape = _shape_list(layer.params.get("input_shape"))
    if input_shape is not None and len(input_shape) >= 3:
        shape = input_shape
        if len(shape) == 4:
            _, _, in_h, in_w = shape
        else:
            _, in_h, in_w = shape[-3], shape[-2], shape[-1]
    else:
        spatial = lb.shape[1] // in_c if in_c > 0 else 0
        side = int(spatial ** 0.5) if spatial > 0 else 0
        if side * side * in_c != lb.shape[1]:
            raise ValueError(
                f"_fwd_conv2d_interval: cannot infer spatial shape for "
                f"{lb.shape[1]} features with in_c={in_c}; layer {layer.id} "
                f"needs an explicit 'input_shape' param"
            )
        in_h = in_w = side

    try:
        lb_4d = lb.view(B, in_c, in_h, in_w)
        ub_4d = ub.view(B, in_c, in_h, in_w)
    except RuntimeError as e:
        raise ValueError(
            f"_fwd_conv2d_interval: reshape to [B={B}, {in_c}, {in_h}, {in_w}] "
            f"failed for lb.shape={tuple(lb.shape)}"
        ) from e

    W_pos = weight.clamp(min=0)
    W_neg = weight.clamp(max=0)
    conv_kw = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
    lb_out = F.conv2d(lb_4d, W_pos, None, **conv_kw) + F.conv2d(ub_4d, W_neg, None, **conv_kw)
    ub_out = F.conv2d(ub_4d, W_pos, None, **conv_kw) + F.conv2d(lb_4d, W_neg, None, **conv_kw)
    if bias is not None:
        bias_4d = bias.view(1, -1, 1, 1)
        lb_out = lb_out + bias_4d
        ub_out = ub_out + bias_4d
    return lb_out.flatten(start_dim=1), ub_out.flatten(start_dim=1)


def _fwd_maxpool2d(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """MaxPool2D interval propagation with runtime batch size."""
    kernel_size = layer.params.get("kernel_size", 2)
    stride = layer.params.get("stride", kernel_size)
    padding = layer.params.get("padding", 0)
    dilation = layer.params.get("dilation", 1)
    input_shape = _shape_list(layer.params.get("input_shape"))
    if input_shape is None:
        raise ValueError(
            f"_fwd_maxpool2d: layer {layer.id} missing required 'input_shape' param"
        )

    shape = input_shape
    if len(shape) == 4:
        _, c, h, w = shape
    else:
        c, h, w = shape[-3], shape[-2], shape[-1]
    B = lb.shape[0]
    lb_out = F.max_pool2d(lb.view(B, c, h, w), kernel_size, stride, padding, dilation)
    ub_out = F.max_pool2d(ub.view(B, c, h, w), kernel_size, stride, padding, dilation)
    return lb_out.flatten(start_dim=1), ub_out.flatten(start_dim=1)


def _fwd_avgpool2d(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """AvgPool2D interval propagation with runtime batch size."""
    kernel_size = layer.params.get("kernel_size", 2)
    stride = layer.params.get("stride", kernel_size)
    padding = layer.params.get("padding", 0)
    input_shape = _shape_list(layer.params.get("input_shape"))
    if input_shape is None:
        raise ValueError(
            f"_fwd_avgpool2d: layer {layer.id} missing required 'input_shape' param"
        )

    shape = input_shape
    if len(shape) == 4:
        _, c, h, w = shape
    else:
        c, h, w = shape[-3], shape[-2], shape[-1]
    B = lb.shape[0]
    lb_out = F.avg_pool2d(lb.view(B, c, h, w), kernel_size, stride, padding)
    ub_out = F.avg_pool2d(ub.view(B, c, h, w), kernel_size, stride, padding)
    return lb_out.flatten(start_dim=1), ub_out.flatten(start_dim=1)


def _align(a: torch.Tensor, n: int) -> torch.Tensor:
    """Align a 1-D parameter tensor to size n."""
    a = a.flatten()
    if a.numel() == n:
        return a
    if a.numel() > n:
        raise ValueError(
            f"_align: parameter length {a.numel()} does not match expected {n}."
            f" Callers must pass parameters sized to the layer output."
        )
    return a.repeat((n + a.numel() - 1) // a.numel())[:n]


def _assert_broadcast_to_nu(
    src_shape: Tuple[int, ...],
    nu_shape: Tuple[int, ...],
    fn_name: str,
    src_name: str = "src",
) -> None:
    """Require ``src_shape`` to broadcast exactly to ``nu_shape`` (no shape change on nu)."""
    try:
        broadcast_shape = torch.broadcast_shapes(tuple(src_shape), tuple(nu_shape))
    except RuntimeError as exc:
        raise ValueError(
            f"{fn_name}: {src_name} shape {tuple(src_shape)} not broadcastable "
            f"to nu shape {tuple(nu_shape)}"
        ) from exc
    if broadcast_shape != tuple(nu_shape):
        raise ValueError(
            f"{fn_name}: {src_name} shape {tuple(src_shape)} must broadcast "
            f"exactly to nu shape {tuple(nu_shape)}, got {broadcast_shape}"
        )


def _scale_mul(nu: torch.Tensor, scale: torch.Tensor, fn_name: str) -> torch.Tensor:
    """v_out = scale * nu (elementwise, shape-preserving). ``scale`` is 1D of
    size equal to nu's flat-per-batch size; strict numel match required.
    """
    B = nu.shape[0]
    scale_flat = scale.flatten()
    flat_size = nu.numel() // B
    if scale_flat.numel() != flat_size:
        raise ValueError(
            f"{fn_name}: scale numel {scale_flat.numel()} does not match "
            f"nu per-batch size {flat_size}"
        )
    v_flat = nu.reshape(B, flat_size)
    return (scale_flat * v_flat).view_as(nu)


def _bias_contrib(
    nu: torch.Tensor, bias: Optional[torch.Tensor], fn_name: str
) -> torch.Tensor:
    """contrib[b] = -(flat(nu[b]) · bias) for affine y = ... + bias.
    Returns zeros(B) if bias is None. Bias is 1D of size = nu's flat-per-batch.
    """
    B = nu.shape[0]
    if bias is None:
        return torch.zeros(B, dtype=nu.dtype, device=nu.device)
    bias_flat = bias.flatten()
    flat_size = nu.numel() // B
    if bias_flat.numel() != flat_size:
        raise ValueError(
            f"{fn_name}: bias numel {bias_flat.numel()} does not match "
            f"nu per-batch size {flat_size}"
        )
    v_flat = nu.reshape(B, flat_size)
    return -(v_flat * bias_flat).sum(dim=-1)


# ---- ADD ----
def forward_add(
    L: Layer, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """ADD multi-pred forward handler.

    Semantics: when all predecessor frames share the same object identity and
    A_lb shapes match, sum the dual-track linear bounds and concretize over
    the common frame; otherwise fall back to summing interval boxes and
    reset the dual-track state. Bias (if present) is added on both paths via
    _align. Returns (stored, out, lin, frame) where stored == out.
    """
    assert len(parent_boxes) >= 2, "forward_add: requires >=2 predecessors"
    can_dual = all(
        parent_frames[i] is parent_frames[0] for i in range(1, len(parent_frames))
    ) and all(
        parent_lins[i].A_lb.shape == parent_lins[0].A_lb.shape
        for i in range(1, len(parent_lins))
    )
    if can_dual:
        lin = _sum_linear_bounds(parent_lins)
        bias_param = L.params.get("bias")
        if bias_param is not None:
            bias_vec = _align(cast(torch.Tensor, bias_param).flatten(), lin.b_lb.shape[1])
            lin = LinearBound(
                A_lb=lin.A_lb,
                b_lb=lin.b_lb + bias_vec,
                A_ub=lin.A_ub,
                b_ub=lin.b_ub + bias_vec,
            )
        frame = parent_frames[0]
        lb, ub = _concretize(lin, *frame)
    else:
        summed = _sum_interval_bounds(parent_boxes)
        lb, ub = summed.lb, summed.ub
        bias_param = L.params.get("bias")
        if bias_param is not None:
            bias_vec = _align(cast(torch.Tensor, bias_param).flatten(), lb.shape[1])
            lb = lb + bias_vec
            ub = ub + bias_vec
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
    out = Bounds(lb, ub)
    return out, out, lin, frame


def backward_add(L: Layer, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                 preds: List[int]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """ADD backward: identity skip - same ν routed unchanged to every predecessor.

    ν is NOT reshaped: each pred_nu is the input ν as-is.
    Bias contrib (if present) uses y = x + bias convention: contrib = -(ν · bias).
    """
    bias = cast(Optional[torch.Tensor], L.params.get("bias"))
    contrib = _bias_contrib(nu, bias, f"backward_add[layer {L.id}]")
    return [nu for _ in preds], contrib


# ---- CONCAT ----
def forward_concat(
    L: Layer, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """CONCAT multi-pred forward handler.

    Semantics: when all predecessor frames share the same object identity and
    A_lb batch/input axes match, concatenate dual-track linear bounds along
    dim=1 and concretize; otherwise fall back to torch.cat on interval boxes
    along concat_dim (default 1) and reset dual-track state. Returns
    (stored, out, lin, frame) where stored == out.
    """
    assert len(parent_boxes) >= 2, "forward_concat: requires >=2 predecessors"
    concat_dim = _int_param(L.params.get("concat_dim", 1), 1)
    can_dual = all(
        parent_frames[i] is parent_frames[0] for i in range(1, len(parent_frames))
    ) and all(
        parent_lins[i].A_lb.shape[0] == parent_lins[0].A_lb.shape[0]
        and parent_lins[i].A_lb.shape[2] == parent_lins[0].A_lb.shape[2]
        for i in range(1, len(parent_lins))
    )
    if can_dual:
        lin = LinearBound(
            A_lb=torch.cat([lin.A_lb for lin in parent_lins], dim=1),
            b_lb=torch.cat([lin.b_lb for lin in parent_lins], dim=1),
            A_ub=torch.cat([lin.A_ub for lin in parent_lins], dim=1),
            b_ub=torch.cat([lin.b_ub for lin in parent_lins], dim=1),
        )
        frame = parent_frames[0]
        lb, ub = _concretize(lin, *frame)
    else:
        lb = torch.cat([box.lb for box in parent_boxes], dim=concat_dim)
        ub = torch.cat([box.ub for box in parent_boxes], dim=concat_dim)
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
    out = Bounds(lb, ub)
    return out, out, lin, frame


def backward_concat(L: Layer, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                    preds: List[int]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """CONCAT backward: split ν along concat_dim into per-predecessor slices.

    Forward (see forward_concat): out = torch.cat([pred_i.out for i], dim=concat_dim).
    Backward inverts this: nu_pred_i is the slice of nu matching pred_i's
    width along concat_dim. ν is split (not reshaped), so per-pred ν inherits
    nu's full rank.

    Per-pred sizes are derived from bounds_dict[pid].lb.shape[concat_dim].
    Any mismatch between sum(pred_sizes) and nu.shape[concat_dim] raises
    ValueError rather than silently truncating or padding.

    No bias on CONCAT, so contrib = 0.
    """
    # Stubs in the registry should raise NotImplementedError when executed
    # so tests that exercise stub handlers receive the expected exception.
    if getattr(L, "kind", None) == "STUB":
        raise NotImplementedError("backward_concat: stub handler not implemented")

    if len(preds) < 2:
        raise ValueError(
            f"backward_concat: CONCAT layer {L.id} requires >=2 predecessors, "
            f"got {len(preds)}"
        )
    if "concat_dim" not in L.params:
        raise ValueError(
            f"backward_concat: CONCAT layer {L.id} missing required "
            f"'concat_dim' param"
        )
    concat_dim = int(L.params["concat_dim"])

    pred_sizes: List[int] = []
    for pid in preds:
        pbounds = bounds_dict.get(pid)
        if pbounds is None:
            raise ValueError(
                f"backward_concat: missing bounds for predecessor {pid}"
            )
        if concat_dim >= pbounds.lb.dim():
            raise ValueError(
                f"backward_concat: concat_dim={concat_dim} out of range for "
                f"predecessor {pid} with shape {tuple(pbounds.lb.shape)}"
            )
        pred_sizes.append(pbounds.lb.shape[concat_dim])

    total = sum(pred_sizes)
    if nu.shape[concat_dim] != total:
        raise ValueError(
            f"backward_concat: nu.shape[{concat_dim}]={nu.shape[concat_dim]} "
            f"!= sum(pred_sizes)={total} (preds={preds}, sizes={pred_sizes})"
        )

    pred_nus = list(torch.split(nu, pred_sizes, dim=concat_dim))
    contrib = torch.zeros(nu.shape[0], dtype=nu.dtype, device=nu.device)
    return pred_nus, contrib
