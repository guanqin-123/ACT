#===- act/back_end/solver/solver_dual.py - Dual Bounds Solver ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# DualSolver: Wong-Kolter / CROWN-style certified lower-bound solver.
# STRICT batched API ([B, *shape] only). Raises ValueError on 1-D input.
# Mirrors HZSolver precedent in solver_hz.py.
#===---------------------------------------------------------------------===#
# pyright: reportMissingImports=false

from __future__ import annotations
import logging
import torch
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union, cast
from act.back_end.bab.eta import EtaState, collapse_eta_state, expand_eta_state
from act.back_end.bounds_dispatch import materialize_if_needed
from act.back_end.dual_tf.tf_forward import LinearBound
from act.back_end.core import Bounds, Net
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.alpha_state import AlphaState
from act.back_end.solver.solver_base import Solver, SolverCaps
from act.back_end.solver.spec_batching import (
    SpecBatch, SpecBatchResult, build_spec_batch, expand_bounds_dict,
)
from act.front_end.specs import OutputSpec, OutKind
from act.util.device_manager import get_default_device, get_default_dtype

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from act.back_end.bab.trace import BoundTrace
    from act.back_end.dual_tf.dual_tf import DualTF


def _reverse_topological_sort(net: Net) -> List[int]:
    """Kahn's algorithm on net.succs.

    Returns layer IDs in reverse-topological order: every layer appears
    after all its successors.

    Raises:
        ValueError: If the graph contains a cycle or disconnected layers.
    """
    in_deg: Dict[int, int] = {layer.id: len(net.succs.get(layer.id, [])) for layer in net.layers}
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
            f"DualSolver: graph has cycle or disconnected layers "
            f"({len(order)}/{len(net.layers)} sorted)"
        )
    return order


def _is_spec_rank3(tensor: torch.Tensor) -> bool:
    return tensor.dim() >= 3 and tensor.shape[0] > 0 and tensor.stride(1) == 0


def _batch_rows(tensor: torch.Tensor) -> int:
    if _is_spec_rank3(tensor):
        return int(tensor.shape[0] * tensor.shape[1])
    if tensor.dim() >= 1:
        return int(tensor.shape[0])
    return 1


def _flatten_batch_rows(tensor: torch.Tensor, *, writable: bool = False) -> torch.Tensor:
    if _is_spec_rank3(tensor):
        lead = tensor.shape[0] * tensor.shape[1]
        if writable:
            return tensor.contiguous().reshape(lead, *tensor.shape[2:])
        return tensor.reshape(lead, *tensor.shape[2:])
    return tensor


def _flatten_bounds_rows(bounds: Bounds, *, writable: bool = False) -> Bounds:
    if _is_spec_rank3(bounds.lb):
        if writable:
            mat = materialize_if_needed(
                LinearBound(
                    A_lb=bounds.lb,
                    b_lb=bounds.lb,
                    A_ub=bounds.ub,
                    b_ub=bounds.ub,
                )
            )
            return Bounds(lb=mat.A_lb.reshape(-1, *mat.A_lb.shape[2:]), ub=mat.A_ub.reshape(-1, *mat.A_ub.shape[2:]))
        return Bounds(
            lb=_flatten_batch_rows(bounds.lb, writable=writable),
            ub=_flatten_batch_rows(bounds.ub, writable=writable),
        )
    return bounds


def _reshape_margins(margins_flat: torch.Tensor, B: int, M: int) -> torch.Tensor:
    return margins_flat.reshape(B, M)


def _collapse_spec_axis(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[:, 0, ...] if _is_spec_rank3(tensor) else tensor


AFFINE_CONTRIB_KINDS = frozenset({
    LayerKind.DENSE.value,
    LayerKind.CONV2D.value,
    "BIAS",
    "BN",
    "ADD",
})


class DualSolver(Solver):
    """Dual (Wong-Kolter) certified bounds solver. Strict [B, *shape] API."""

    _AFFINE_CONTRIB_KINDS = AFFINE_CONTRIB_KINDS

    def __init__(self, tf: "DualTF", n_iters: int = 0):
        self.tf = tf
        self.n_iters = n_iters
        self.eta_iters = 20
        self.lr_alpha = 0.5
        self.lr_eta = 0.05
        self.lambda_intermediate = 1.0
        self.alpha_per_spec = False
        self.eta_per_spec = False
        self._eta_state: Optional[EtaState] = None
        self._alpha_state: AlphaState = AlphaState()
        self._last_bounds: Optional[Bounds] = None
        self._bound_trace: Optional["BoundTrace"] = None
        self._bound_trace_bab_iter: int = 0
        self._bound_trace_sid_map: Optional[torch.Tensor] = None

    def capabilities(self) -> SolverCaps:
        return SolverCaps(supports_gpu=True, supports_csp=False, supports_dual=True)

    def set_eta(self, eta: Optional[EtaState]) -> None:
        if eta is None:
            self._eta_state = None
            return
        self._eta_state = eta.to(
            device=get_default_device(),
            dtype=get_default_dtype(),
        )

    def clear_eta(self) -> None:
        self._eta_state = None

    def _coerce_alpha_state(
        self,
        alphas: Union[Dict[int, torch.Tensor], AlphaState, None],
    ) -> AlphaState:
        device, dtype = get_default_device(), get_default_dtype()
        if alphas is None:
            return AlphaState()
        if isinstance(alphas, AlphaState):
            return alphas.to(device=device, dtype=dtype)
        return AlphaState.from_legacy(
            {lid: t.to(device=device, dtype=dtype) for lid, t in alphas.items()}
        )

    def _final_alpha_slice(
        self,
        alphas: Union[Dict[int, torch.Tensor], AlphaState, None],
    ) -> Optional[Dict[int, torch.Tensor]]:
        if alphas is None:
            return None
        if isinstance(alphas, AlphaState):
            state = alphas.to(
                device=get_default_device(),
                dtype=get_default_dtype(),
            )
            return state.for_start_node(AlphaState.FINAL_SID)
        device, dtype = get_default_device(), get_default_dtype()
        return {lid: t.to(device=device, dtype=dtype) for lid, t in alphas.items()}

    def set_alphas(
        self,
        alphas: Union[Dict[int, torch.Tensor], AlphaState, None],
    ) -> None:
        """Install per-ReLU warm-start α (slopes) for the next backward pass.

        Mirrors :meth:`set_eta`. When set, ``evaluate_spec`` routes through
        the joint α/η KKT path and initialises α from these values instead
        of the default heuristic. BaB calls this with the parent
        subproblem's optimised α before evaluating a child node.
        """
        self._alpha_state = self._coerce_alpha_state(alphas)

    def clear_alphas(self) -> None:
        self._alpha_state = AlphaState()

    def set_bound_trace(
        self,
        trace: Optional["BoundTrace"],
        bab_iter: int = 0,
        sid_map: Optional[torch.Tensor] = None,
    ) -> None:
        """Install opt-in BaB trace context for per-row Adam trajectories."""
        self._bound_trace = trace
        self._bound_trace_bab_iter = bab_iter
        self._bound_trace_sid_map = sid_map

    def clear_bound_trace(self) -> None:
        self._bound_trace = None
        self._bound_trace_sid_map = None

    def _record_bound_trace_step(self, best_obj: torch.Tensor) -> None:
        """Record one trajectory value per (sid, bab_iter) per Adam step.

        For each unique subproblem id in ``sid_map``, we aggregate the rows
        belonging to that subproblem (each row corresponds to one spec / class
        margin) and record the **min** across those rows. This is the critical
        bound for certification — a subproblem is certified iff every spec row
        reaches a non-negative margin, so ``min`` is the tightest aggregator.

        Using min also preserves the per-sid monotonicity property: because each
        row's best_obj is monotonically non-decreasing across Adam iterations
        (enforced by ``torch.where(improved, obj, best_obj)`` in the caller),
        the min over any fixed row-subset is also monotonically non-decreasing.
        """
        trace = self._bound_trace
        sid_map = self._bound_trace_sid_map
        if trace is None or sid_map is None:
            return
        with torch.no_grad():
            b_rows = best_obj.shape[0]
            sid_n = sid_map.numel()
            if sid_n == 0:
                return
            if sid_n != b_rows:
                # Caller must expand sid_map to match row count; otherwise we
                # cannot attribute rows to subproblems. Skip silently — the
                # trace entry simply won't include this step.
                return
            # Use torch.unique_consecutive because sid_map is built via
            # repeat_interleave from a contiguous per-subproblem vector, so
            # identical sids appear in contiguous blocks.
            unique_sids, inverse = torch.unique_consecutive(
                sid_map, return_inverse=True
            )
            # Scatter-reduce min over row values grouped by inverse index.
            num_groups = int(unique_sids.numel())
            mins = torch.full(
                (num_groups,),
                float("inf"),
                device=best_obj.device,
                dtype=best_obj.dtype,
            )
            mins = mins.scatter_reduce(
                0, inverse, best_obj, reduce="amin", include_self=True
            )
            bab_iter = self._bound_trace_bab_iter
            for g in range(num_groups):
                trace.record_adam_step(
                    int(unique_sids[g].item()),
                    bab_iter,
                    float(mins[g].item()),
                )

    def compute_bound(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        c: torch.Tensor,
        return_sce: bool = False,
        enable_grad: bool = False,
        n_iters: Optional[int] = None,
        lr: float = 0.1,
        force_kkt: bool = False,
        per_class_alpha: bool = False,
        warm_alphas: Optional[Union[Dict[int, torch.Tensor], AlphaState]] = None,
        warm_etas: Optional[EtaState] = None,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[
            torch.Tensor,
            Optional[torch.Tensor],
            torch.Tensor,
            Optional[AlphaState],
            Optional[EtaState],
        ],
    ]:
        """Batched certified lower bound on c^T @ output (DAG-aware).

        ν is propagated backward through a per-layer accumulator:
          nu_accum[lid] = sum over all successors s of ν routed by s's handler to lid.

        Each handler returns per-pred νs; the outer loop distributes them to preds.

        Unknown layer kind raises ValueError (no silent identity fallback for soundness).
        Args:
            c: Tensor[B, num_classes] — REQUIRED batched. Raises ValueError if 1-D.
            return_sce: if True, also return per-sample concrete input extremum.
            enable_grad: if True, allow gradients to flow through the computation
                          (for robust training). Default False (inference/verification).
            n_iters / warm_alphas / warm_etas: when the joint alpha/eta path is
                requested, return the optimized warm-start state alongside the
                best objective.
        Returns:
            Tensor[B] for the legacy path, ``(obj, sce)`` for the explicit
            ``n_iters=0`` compatibility path, or
            ``(obj, sce, row_objectives, out_alphas, out_etas)`` when the joint
            alpha/eta warm-start API is requested. ``out_alphas`` is an
            :class:`AlphaState` carrying the FINAL_SID slice in Phase 1.
        """
        explicit_joint_api = (
            n_iters is not None
            or force_kkt
            or per_class_alpha
            or warm_alphas is not None
            or warm_etas is not None
        )
        if not explicit_joint_api:
            if c.dim() != 2:
                raise ValueError(
                    f"c must be 2-D [B, num_classes], got shape {tuple(c.shape)}. "
                    "Use c.unsqueeze(0) for single instance."
                )
            if self._eta_state is None or self._eta_state.fast_path_skip():
                return self._compute_bound_direct(
                    net,
                    bounds_dict,
                    c,
                    return_sce=return_sce,
                    enable_grad=enable_grad,
                )
            return self._compute_bound_with_eta(
                net,
                bounds_dict,
                c,
                return_sce=return_sce,
                enable_grad=enable_grad,
            )

        if c.dim() == 1:
            c = c.unsqueeze(0)
        elif c.dim() != 2:
            raise ValueError(
                f"c must be 1-D or 2-D, got shape {tuple(c.shape)}"
            )

        kkt_iters = 10 if force_kkt and n_iters is None else (0 if n_iters is None else n_iters)
        if kkt_iters < 0:
            raise ValueError(f"n_iters must be >= 0, got {kkt_iters}")

        joint_requested = (
            force_kkt
            or kkt_iters > 0
            or per_class_alpha
            or warm_alphas is not None
            or warm_etas is not None
        )
        if not joint_requested:
            legacy = (
                self._compute_bound_direct(
                    net,
                    bounds_dict,
                    c,
                    return_sce=return_sce,
                    enable_grad=enable_grad,
                )
                if self._eta_state is None or self._eta_state.fast_path_skip()
                else self._compute_bound_with_eta(
                    net,
                    bounds_dict,
                    c,
                    return_sce=return_sce,
                    enable_grad=enable_grad,
                )
            )
            if isinstance(legacy, tuple):
                return legacy
            return legacy, None

        joint_result = self._compute_bound_joint_kkt(
            net,
            bounds_dict,
            c,
            n_iters=kkt_iters,
            lr=lr,
            return_sce=return_sce,
            per_class_alpha=per_class_alpha,
            warm_alphas=warm_alphas,
            warm_etas=warm_etas,
            force_kkt=force_kkt,
        )
        return_joint_state = (
            warm_alphas is not None
            or warm_etas is not None
            or per_class_alpha
            or (kkt_iters > 0 and not force_kkt)
        )
        if return_joint_state:
            return joint_result
        obj, sce, _, _, _ = joint_result
        return obj, sce

    def _compute_bound_direct(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        c: torch.Tensor,
        return_sce: bool = False,
        enable_grad: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        with torch.set_grad_enabled(enable_grad):
            obj, sce = self._backward_pass(
                net,
                bounds_dict,
                c,
                alphas=None,
                eta_state=None,
                return_sce=return_sce,
            )
            return (obj, sce) if return_sce else obj

    def _compute_bound_with_eta(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        c: torch.Tensor,
        return_sce: bool = False,
        enable_grad: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        del enable_grad
        if self._eta_state is None:
            return self._compute_bound_direct(
                net,
                bounds_dict,
                c,
                return_sce=return_sce,
                enable_grad=False,
            )

        assert len(bounds_dict) > 0, "bounds_dict cannot be empty"
        device, dtype = get_default_device(), get_default_dtype()
        if c.dtype != dtype or c.device != device:
            c = c.to(device=device, dtype=dtype)

        eta_state = self._prepare_eta_state(self._eta_state, c.shape[0])
        if eta_state is None:
            return self._compute_bound_direct(
                net,
                bounds_dict,
                c,
                return_sce=return_sce,
                enable_grad=False,
            )
        self._eta_state = eta_state
        eta_params = {
            lid: torch.nn.Parameter(v.detach().clone())
            for lid, v in eta_state.val.items()
        }
        if not eta_params:
            return self._compute_bound_direct(
                net,
                bounds_dict,
                c,
                return_sce=return_sce,
                enable_grad=False,
            )

        B = c.shape[0]
        optim = torch.optim.Adam(list(eta_params.values()), lr=self.lr_eta)
        best_obj = torch.full((B,), -float("inf"), device=c.device, dtype=c.dtype)
        prev_best = best_obj.clone()
        plateau_count = 0
        best_sce: Optional[torch.Tensor] = None
        num_iters = self.eta_iters if self.eta_iters > 0 else 1
        plateau_threshold = 5e-4
        plateau_patience = 2

        with torch.enable_grad():
            for step_idx in range(num_iters):
                optim.zero_grad()
                working_eta = EtaState(
                    val={lid: param for lid, param in eta_params.items()},
                    sign=eta_state.sign,
                    point=eta_state.point,
                    per_spec=eta_state.per_spec,
                )
                obj, sce = self._backward_pass(
                    net,
                    bounds_dict,
                    c,
                    alphas=None,
                    eta_state=working_eta,
                    return_sce=return_sce,
                )

                loss = -obj.sum()
                if loss.requires_grad:
                    loss.backward()
                optim.step()

                with torch.no_grad():
                    for param in eta_params.values():
                        param.clamp_(min=0.0)
                    improved = obj.detach() > best_obj
                    best_obj = torch.where(improved, obj.detach(), best_obj)
                    if return_sce and sce is not None:
                        if best_sce is None:
                            best_sce = sce.detach().clone()
                        mask = improved.view(-1, *([1] * (sce.dim() - 1)))
                        best_sce = torch.where(mask, sce.detach(), best_sce)
                    delta = (best_obj - prev_best).abs().max().item()
                    if delta < plateau_threshold:
                        plateau_count += 1
                    else:
                        plateau_count = 0
                    prev_best = best_obj.clone()
                    self._record_bound_trace_step(best_obj)
                    if plateau_count >= plateau_patience:
                        break

        with torch.no_grad():
            for lid, param in eta_params.items():
                self._eta_state.val[lid] = param.detach().clone()

        return (best_obj, best_sce) if return_sce else best_obj

    def _maybe_apply_alpha(
        self,
        handler,
        layer,
        nu: torch.Tensor,
        bounds_dict: Dict[int, Bounds],
        preds: List[int],
        alphas: Optional[Dict[int, torch.Tensor]],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if alphas is None:
            alpha = self._alpha_state.get(layer.id, AlphaState.FINAL_SID)
        else:
            alpha = alphas.get(layer.id)

        kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
        if alpha is None or kind != LayerKind.RELU.value:
            return handler(layer, nu, bounds_dict, preds)

        bounds = bounds_dict.get(layer.id)
        if bounds is None:
            raise ValueError(
                f"DualSolver.compute_bound: layer {layer.id} missing bounds for alpha-aware ReLU backward"
            )
        relu_kernel = handler.__globals__.get("dual_relu_backward")
        if relu_kernel is None:
            raise ValueError(
                f"DualSolver.compute_bound: alpha-aware ReLU kernel missing for layer {layer.id}"
            )
        nu_out, contrib = relu_kernel(nu, bounds, alpha=alpha)
        if len(preds) != 1:
            raise ValueError(
                f"DualSolver.compute_bound: RELU layer {layer.id} must have exactly 1 predecessor, got {len(preds)}"
            )
        return [nu_out], contrib

    def _backward_pass(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        c: torch.Tensor,
        alphas: Optional[Dict[int, torch.Tensor]] = None,
        eta_state: Optional[EtaState] = None,
        return_sce: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Differentiable backward pass. Returns (obj, sce)."""
        assert len(bounds_dict) > 0, "bounds_dict cannot be empty"
        if c.dim() == 1:
            c = c.unsqueeze(0)
        if c.dim() != 2:
            raise ValueError(
                f"c must be 2-D [B, num_classes] after promotion, got shape {tuple(c.shape)}"
            )

        device, dtype = get_default_device(), get_default_dtype()
        if c.dtype != dtype or c.device != device:
            c = c.to(device=device, dtype=dtype)
        B = c.shape[0]
        bounds_runtime = {
            lid: _flatten_bounds_rows(bounds) for lid, bounds in bounds_dict.items()
        }

        output_lid = self._get_assert_output_layer_id(net)
        nu_accum: Dict[int, torch.Tensor] = {output_lid: c.clone()}
        obj = torch.zeros(B, dtype=c.dtype, device=c.device)

        topo_order = _reverse_topological_sort(net)
        registry = self.tf._BACKWARD_REGISTRY
        use_eta = eta_state is not None and not eta_state.fast_path_skip()
        eta_vals = eta_state.val if use_eta and eta_state is not None else {}
        eta_signs = eta_state.sign if use_eta and eta_state is not None else {}

        for lid in topo_order:
            layer = net.by_id[lid]
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind

            if k in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value, LayerKind.ASSERT.value):
                continue

            if lid not in nu_accum:
                continue

            nu_here = nu_accum.pop(lid)
            handler = registry.get(k)
            if handler is None:
                raise ValueError(
                    f"DualSolver.compute_bound: unknown layer kind '{k}' at layer {lid}; "
                    f"soundness requires explicit backward handler. "
                    f"Supported kinds: {sorted(registry.keys())}"
                )

            preds = list(net.preds.get(lid, []))
            pred_nus, contrib = self._maybe_apply_alpha(
                handler,
                layer,
                nu_here,
                bounds_runtime,
                preds,
                alphas,
            )
            if use_eta:
                pred_nus = self._apply_eta_to_pred_nus(
                    pred_nus,
                    preds,
                    eta_vals,
                    eta_signs,
                )

            if len(pred_nus) != len(preds):
                raise ValueError(
                    f"handler {k} at layer {lid} returned {len(pred_nus)} pred_nus, "
                    f"expected {len(preds)}"
                )
            if contrib.shape != (B,):
                raise ValueError(
                    f"handler {k} at layer {lid} contrib shape {tuple(contrib.shape)}, "
                    f"expected ({B},)"
                )

            if k in self._AFFINE_CONTRIB_KINDS:
                contrib = -contrib

            obj = obj + contrib
            for pred_id, pred_nu in zip(preds, pred_nus):
                if pred_id in nu_accum:
                    nu_accum[pred_id] = nu_accum[pred_id] + pred_nu
                else:
                    nu_accum[pred_id] = pred_nu.clone()

        input_lid = self._find_input_layer_id(net)
        if input_lid is None:
            return obj, None

        nu_final = nu_accum.get(input_lid)
        if nu_final is None:
            return obj, None

        input_contrib, sce = self._input_contribution_from_nu(
            net,
            input_lid,
            nu_final,
            bounds_runtime,
            return_sce=return_sce,
            enable_grad=torch.is_grad_enabled(),
        )
        obj = obj + input_contrib
        return obj, sce

    def _discover_alpha_shapes(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
    ) -> Dict[int, int]:
        alpha_shapes: Dict[int, int] = {}
        for layer in net.layers:
            kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if kind != LayerKind.RELU.value:
                continue
            if layer.id not in bounds_dict:
                raise ValueError(
                    f"DualSolver.compute_bound: bounds_dict missing ReLU layer {layer.id} for alpha discovery"
                )
            bound = bounds_dict[layer.id]
            lb_src = _collapse_spec_axis(bound.lb)
            flat_lb = (
                lb_src.flatten(start_dim=1)
                if lb_src.dim() >= 2
                else lb_src.flatten().unsqueeze(0)
            )
            alpha_shapes[layer.id] = int(flat_lb.shape[-1])
        return alpha_shapes

    def _heuristic_alpha(self, bounds: Bounds) -> torch.Tensor:
        lb_src = _collapse_spec_axis(bounds.lb)
        ub_src = _collapse_spec_axis(bounds.ub)
        lb = lb_src.flatten(start_dim=1) if lb_src.dim() >= 2 else lb_src.flatten().unsqueeze(0)
        ub = ub_src.flatten(start_dim=1) if ub_src.dim() >= 2 else ub_src.flatten().unsqueeze(0)
        on = lb >= 0
        off = ub <= 0
        amb = ~(on | off)
        denom = (ub - lb).clamp(min=1e-12)
        up_slope = torch.where(amb, ub / denom, torch.zeros_like(lb))
        return torch.where(
            amb,
            (up_slope > 0.5).to(dtype=lb.dtype),
            torch.where(on, torch.ones_like(lb), torch.zeros_like(lb)),
        )

    def _compute_amb_masks(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
    ) -> Dict[int, torch.Tensor]:
        """Per-ReLU-layer ambiguity mask: True where ``lb < 0 < ub``.

        Stable neurons (``on = lb >= 0`` or ``off = ub <= 0``) are already
        gated out of α usage in both ``_fwd_relu`` and ``dual_relu_backward``
        via ``torch.where(amb, alpha, heuristic)``. This helper surfaces that
        same ``amb`` mask to the Adam-loop driver so it can enforce the
        invariant "α at stable positions stays at heuristic value" explicitly,
        avoiding stale warm-start values from propagating across BaB splits.

        Returns: ``Dict[lid, BoolTensor(B, D_layer)]`` keyed by ReLU layer id,
        shape matches the flattened α-param layout.
        """
        alpha_shapes = self._discover_alpha_shapes(net, bounds_dict)
        masks: Dict[int, torch.Tensor] = {}
        for lid in alpha_shapes:
            bound = bounds_dict[lid]
            lb_src = _collapse_spec_axis(bound.lb)
            ub_src = _collapse_spec_axis(bound.ub)
            lb = (
                lb_src.flatten(start_dim=1)
                if lb_src.dim() >= 2
                else lb_src.flatten().unsqueeze(0)
            )
            ub = (
                ub_src.flatten(start_dim=1)
                if ub_src.dim() >= 2
                else ub_src.flatten().unsqueeze(0)
            )
            masks[lid] = (lb < 0) & (ub > 0)
        return masks

    def _align_alpha_to_batch(
        self,
        t: torch.Tensor,
        batch_size: int,
        lid: int,
        name: str,
    ) -> torch.Tensor:
        """Broadcast/collapse a (B_src, W) α tensor to (batch_size, W)."""
        if t.shape[0] == batch_size:
            return t
        if batch_size % t.shape[0] == 0:
            return t.repeat_interleave(batch_size // t.shape[0], dim=0)
        if t.shape[0] % batch_size == 0:
            factor = t.shape[0] // batch_size
            return t.view(batch_size, factor, t.shape[-1]).mean(dim=1)
        raise ValueError(
            f"DualSolver.compute_bound: {name} batch mismatch at layer {lid}; "
            f"target batch {batch_size}, got {t.shape[0]}"
        )

    def _align_amb_mask_to_batch(
        self,
        mask: Optional[torch.Tensor],
        batch_size: int,
        lid: int,
    ) -> Optional[torch.Tensor]:
        """Broadcast a (B_src, W) bool mask to (batch_size, W). Collapse via
        any() — if any row in the collapsed group had the neuron ambiguous,
        the merged row is ambiguous (conservative for warm-start retention)."""
        if mask is None:
            return None
        if mask.shape[0] == batch_size:
            return mask
        if batch_size % mask.shape[0] == 0:
            return mask.repeat_interleave(batch_size // mask.shape[0], dim=0)
        if mask.shape[0] % batch_size == 0:
            factor = mask.shape[0] // batch_size
            return mask.view(batch_size, factor, mask.shape[-1]).any(dim=1)
        raise ValueError(
            f"DualSolver: amb mask batch mismatch at layer {lid}; "
            f"target batch {batch_size}, got {mask.shape[0]}"
        )

    def _prepare_alpha_params(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        batch_size: int,
        warm_alphas: Optional[Union[Dict[int, torch.Tensor], AlphaState]],
        amb_masks: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[int, torch.nn.Parameter]:
        device, dtype = get_default_device(), get_default_dtype()
        alpha_shapes = self._discover_alpha_shapes(net, bounds_dict)
        warm_alpha_dict = self._final_alpha_slice(warm_alphas)
        # Compute amb masks lazily if caller didn't supply pre-aligned ones.
        # Prepare-time sanitization: at stable positions we ALWAYS use the
        # heuristic value, never the warm-start value carried from a parent
        # subproblem where that neuron was ambiguous. This prevents stale
        # α values from leaking into ``best_alpha_params`` snapshots (which
        # are taken pre-step in the Adam loop) and from there into the
        # returned ``out_alphas`` that warm-starts children.
        if amb_masks is None:
            amb_masks = self._compute_amb_masks(net, bounds_dict)
        params: Dict[int, torch.nn.Parameter] = {}
        for lid, width in alpha_shapes.items():
            heur = self._heuristic_alpha(bounds_dict[lid])
            heur_aligned = self._align_alpha_to_batch(heur, batch_size, lid, "heuristic alpha")
            amb_aligned = self._align_amb_mask_to_batch(
                amb_masks.get(lid), batch_size, lid
            ) if amb_masks.get(lid) is not None else None
            if warm_alpha_dict is not None and lid in warm_alpha_dict:
                init = warm_alpha_dict[lid].to(device=device, dtype=dtype)
                init_flat = (
                    init.flatten(start_dim=1)
                    if init.dim() >= 2
                    else init.flatten().unsqueeze(0)
                )
                if init_flat.shape[-1] != width:
                    raise ValueError(
                        f"DualSolver.compute_bound: warm alpha width mismatch at layer {lid}; "
                        f"expected {width}, got {init_flat.shape[-1]}"
                    )
                warm_aligned = self._align_alpha_to_batch(
                    init_flat, batch_size, lid, "warm alpha"
                )
                if amb_aligned is not None:
                    prepared = torch.where(amb_aligned, warm_aligned, heur_aligned)
                else:
                    prepared = warm_aligned
            else:
                prepared = heur_aligned
            params[lid] = torch.nn.Parameter(prepared.detach().clone().clamp(0.0, 1.0))
        return params

    def _runtime_alpha_dict(
        self,
        alpha_params: Mapping[int, torch.Tensor],
        target_batch: int,
    ) -> Dict[int, torch.Tensor]:
        runtime: Dict[int, torch.Tensor] = {}
        for lid, param in alpha_params.items():
            if param.shape[0] == target_batch:
                runtime[lid] = param
            elif target_batch % param.shape[0] == 0:
                runtime[lid] = param.repeat_interleave(target_batch // param.shape[0], dim=0)
            else:
                raise ValueError(
                    f"DualSolver.compute_bound: alpha batch mismatch at layer {lid}; "
                    f"target batch {target_batch}, got {param.shape[0]}"
                )
        return runtime

    def _collapse_alpha_dict(
        self,
        alpha_tensors: Dict[int, torch.Tensor],
        base_batch: int,
    ) -> Dict[int, torch.Tensor]:
        collapsed: Dict[int, torch.Tensor] = {}
        for lid, tensor in alpha_tensors.items():
            flat = tensor.detach().clone()
            if flat.shape[0] == base_batch:
                collapsed[lid] = flat
            elif flat.shape[0] % base_batch == 0:
                factor = flat.shape[0] // base_batch
                collapsed[lid] = flat.view(base_batch, factor, flat.shape[1]).mean(dim=1)
            else:
                raise ValueError(
                    f"DualSolver.compute_bound: cannot collapse alpha batch at layer {lid}; "
                    f"base batch {base_batch}, got {flat.shape[0]}"
                )
        return collapsed

    def _clone_tensor_dict(
        self,
        tensors: Mapping[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        return {lid: tensor.detach().clone() for lid, tensor in tensors.items()}

    def _snapshot_best_rows(
        self,
        current: torch.Tensor,
        best: Optional[torch.Tensor],
        improved: torch.Tensor,
    ) -> torch.Tensor:
        current_detached = current.detach()
        if current_detached.shape[0] != improved.shape[0]:
            raise ValueError(
                "DualSolver.compute_bound: best-state snapshot batch mismatch; "
                f"tensor batch {current_detached.shape[0]}, improved batch {improved.shape[0]}"
            )
        if best is None:
            return current_detached.clone()
        mask = improved.view(-1, *([1] * (current_detached.dim() - 1)))
        return torch.where(mask, current_detached, best)

    def _snapshot_best_param_dict(
        self,
        params: Dict[int, torch.Tensor],
        best: Optional[Dict[int, torch.Tensor]],
        improved: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        snap: Dict[int, torch.Tensor] = {} if best is None else dict(best)
        for lid, param in params.items():
            snap[lid] = self._snapshot_best_rows(param, snap.get(lid), improved)
        return snap

    def _clone_eta_state(self, eta: Optional[EtaState]) -> Optional[EtaState]:
        if eta is None:
            return None
        return EtaState(
            val=self._clone_tensor_dict(eta.val),
            sign=self._clone_tensor_dict(eta.sign),
            point=self._clone_tensor_dict(eta.point),
            per_spec=eta.per_spec,
        )

    def _build_eta_state(
        self,
        template: EtaState,
        values: Mapping[int, torch.Tensor],
    ) -> EtaState:
        return EtaState(
            val=self._clone_tensor_dict(values),
            sign=self._clone_tensor_dict(template.sign),
            point=self._clone_tensor_dict(template.point),
            per_spec=template.per_spec,
        )

    def _expand_alpha_dict(
        self,
        alphas: Union[Dict[int, torch.Tensor], AlphaState],
        factor: int,
    ) -> AlphaState:
        if isinstance(alphas, AlphaState):
            state = alphas.to(device=get_default_device(), dtype=get_default_dtype())
        else:
            state = AlphaState.from_legacy(alphas).to(
                device=get_default_device(),
                dtype=get_default_dtype(),
            )
        if state.is_empty():
            return state.clone(detach=False)

        if state.per_spec:
            flat = AlphaState()
            for lid, sid, tensor in state.iter_entries():
                if sid == AlphaState.FINAL_SID and tensor.dim() >= 3:
                    M_dim = tensor.shape[1]
                    if M_dim != factor:
                        raise ValueError(
                            f"DualSolver._expand_alpha_dict: per-spec α at layer {lid}, "
                            f"FINAL_SID has M={M_dim} but expand factor={factor}"
                        )
                    flat.set(
                        lid,
                        sid,
                        tensor.reshape(tensor.shape[0] * M_dim, *tensor.shape[2:]),
                    )
                elif factor <= 1 or tensor.shape[0] == 0:
                    flat.set(lid, sid, tensor.clone())
                else:
                    flat.set(lid, sid, tensor.repeat_interleave(factor, dim=0))
            return flat

        if factor <= 1:
            return state.clone(detach=False)

        expanded = AlphaState()
        for lid, sid, tensor in state.iter_entries():
            if tensor.shape[0] == 0:
                expanded.set(lid, sid, tensor.clone())
            else:
                expanded.set(lid, sid, tensor.repeat_interleave(factor, dim=0))
        return expanded

    def _snapshot_best_alpha_state(
        self,
        alpha_params: AlphaState,
        best: Optional[AlphaState],
        improved: torch.Tensor,
    ) -> AlphaState:
        snap = AlphaState() if best is None else best.clone()
        for lid, sid, tensor in alpha_params.iter_entries():
            snap.set(lid, sid, self._snapshot_best_rows(tensor, snap.get(lid, sid), improved))
        return snap

    def _collapse_eta_state(self, eta: EtaState, target_batch: int) -> EtaState:
        if eta.batch_size == target_batch:
            return eta
        if eta.batch_size % target_batch != 0:
            raise ValueError(
                f"DualSolver.compute_bound: cannot collapse eta batch {eta.batch_size} to {target_batch}"
            )
        factor = eta.batch_size // target_batch
        return EtaState(
            val={
                lid: value.view(target_batch, factor, value.shape[1]).mean(dim=1)
                for lid, value in eta.val.items()
            },
            sign={
                lid: value.view(target_batch, factor, value.shape[1]).mean(dim=1)
                for lid, value in eta.sign.items()
            },
            point={
                lid: value.view(target_batch, factor, value.shape[1]).mean(dim=1)
                for lid, value in eta.point.items()
            },
        )

    def _prepare_eta_state(
        self,
        eta: Optional[EtaState],
        target_batch: int,
    ) -> Optional[EtaState]:
        if eta is None:
            return None
        prepared = eta.to(device=get_default_device(), dtype=get_default_dtype())
        if prepared.per_spec and prepared.batch_size == target_batch:
            collapsed_val = {
                lid: v.mean(dim=1) for lid, v in prepared.val.items()
            }
            return EtaState(
                val=collapsed_val,
                sign=prepared.sign,
                point=prepared.point,
                per_spec=False,
            )
        if prepared.batch_size == 0 or prepared.batch_size == target_batch:
            return prepared
        if prepared.batch_size < target_batch and target_batch % prepared.batch_size == 0:
            return expand_eta_state(prepared, target_batch // prepared.batch_size)
        if prepared.batch_size > target_batch and prepared.batch_size % target_batch == 0:
            return self._collapse_eta_state(prepared, target_batch)
        raise ValueError(
            f"DualSolver.compute_bound: eta batch mismatch; target batch {target_batch}, got {prepared.batch_size}"
        )

    def _compute_bound_joint_kkt(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        c: torch.Tensor,
        n_iters: int,
        lr: float,
        return_sce: bool,
        per_class_alpha: bool,
        warm_alphas: Optional[Union[Dict[int, torch.Tensor], AlphaState]],
        warm_etas: Optional[EtaState],
        force_kkt: bool,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[AlphaState],
        Optional[EtaState],
    ]:
        del per_class_alpha
        assert len(bounds_dict) > 0, "bounds_dict cannot be empty"
        device, dtype = get_default_device(), get_default_dtype()
        if c.dim() == 1:
            c = c.unsqueeze(0)
        if c.dtype != dtype or c.device != device:
            c = c.to(device=device, dtype=dtype)

        target_batch = c.shape[0]
        alpha_seed = warm_alphas if warm_alphas is not None else self._alpha_state
        alpha_seed_state = self._coerce_alpha_state(alpha_seed)
        use_alpha = (not alpha_seed_state.is_empty()) or force_kkt or n_iters > 0
        alpha_final_params = (
            self._prepare_alpha_params(net, bounds_dict, target_batch, alpha_seed_state)
            if use_alpha
            else {}
        )
        alpha_params = AlphaState()
        for lid, param in alpha_final_params.items():
            alpha_params.set(lid, AlphaState.FINAL_SID, param)
        if use_alpha:
            for lid, sid, tensor in alpha_seed_state.iter_entries():
                if sid == AlphaState.FINAL_SID:
                    continue
                prepared = tensor.to(device=device, dtype=dtype)
                prepared = self._align_alpha_to_batch(
                    prepared,
                    target_batch,
                    lid,
                    f"warm alpha sid={sid}",
                )
                alpha_params.set(
                    lid,
                    sid,
                    torch.nn.Parameter(prepared.detach().clone().clamp(0.0, 1.0)),
                )
        non_final_sids = tuple(
            sid for sid in alpha_params.start_nodes if sid != AlphaState.FINAL_SID
        )

        eta_seed = warm_etas if warm_etas is not None else self._eta_state
        eta_state = self._prepare_eta_state(eta_seed, target_batch)
        eta_params: Optional[Dict[int, torch.nn.Parameter]] = None
        if eta_state is not None and not eta_state.fast_path_skip():
            eta_params = {
                lid: torch.nn.Parameter(value.detach().clone())
                for lid, value in eta_state.val.items()
            }

        evaluate_only = not force_kkt and n_iters <= 0
        alpha_flat = alpha_params.flat_params()
        has_intermediate = bool(non_final_sids)
        if has_intermediate:
            opt_groups: List[Dict[str, Any]] = []
            if alpha_flat:
                opt_groups.append({"params": alpha_flat, "lr": self.lr_alpha})
            if eta_params:
                opt_groups.append({"params": list(eta_params.values()), "lr": self.lr_eta})
            optim = (
                torch.optim.Adam(opt_groups, lr=lr)
                if opt_groups and not evaluate_only
                else None
            )
        else:
            opt_params = list(alpha_flat)
            if eta_params is not None:
                opt_params.extend(list(eta_params.values()))
            optim = (
                torch.optim.Adam(opt_params, lr=lr)
                if opt_params and not evaluate_only
                else None
            )

        B = c.shape[0]
        best_obj = torch.full((B,), -float("inf"), device=c.device, dtype=c.dtype)
        prev_best = best_obj.clone()
        plateau_count = 0
        best_sce: Optional[torch.Tensor] = None
        best_alpha_params: Optional[AlphaState] = None
        best_eta_params: Optional[Dict[int, torch.Tensor]] = None
        num_iters = 1 if evaluate_only else max(n_iters, 1)
        plateau_threshold = 5e-4
        plateau_patience = 2

        with torch.enable_grad():
            for step_idx in range(num_iters):
                if optim is not None:
                    optim.zero_grad()

                alpha_snapshot = (
                    alpha_params.clone()
                    if not alpha_params.is_empty()
                    else None
                )
                eta_snapshot = (
                    self._clone_tensor_dict(eta_params)
                    if eta_params is not None
                    else None
                )

                final_alpha_params = alpha_params.for_start_node(AlphaState.FINAL_SID)
                working_alpha = (
                    self._runtime_alpha_dict(final_alpha_params, target_batch)
                    if final_alpha_params
                    else None
                )
                working_eta = None
                if eta_state is not None and not eta_state.fast_path_skip():
                    if eta_params is None:
                        working_eta = eta_state
                    else:
                        working_eta = EtaState(
                            val={lid: param for lid, param in eta_params.items()},
                            sign=eta_state.sign,
                            point=eta_state.point,
                            per_spec=eta_state.per_spec,
                        )

                obj, sce = self._backward_pass(
                    net,
                    bounds_dict,
                    c,
                    alphas=working_alpha,
                    eta_state=working_eta,
                    return_sce=return_sce,
                )

                if optim is not None:
                    loss_final = -obj.sum()
                    if loss_final.requires_grad:
                        loss_final.backward()
                    if (
                        self.lambda_intermediate != 0.0
                        and non_final_sids
                    ):
                        from act.back_end.solver._backward_truncated import (
                            backward_truncated_lb,
                            backward_truncated_ub,
                        )

                        chunk = getattr(self, "alpha_objective_chunk_size", None)
                        max_width = getattr(self, "lambda_intermediate_max_width", None)
                        for sid_int in non_final_sids:
                            if max_width is not None and sid_int in bounds_dict:
                                sid_lb = bounds_dict[sid_int].lb
                                sid_width = int(sid_lb.flatten(start_dim=1).shape[-1]) if sid_lb.dim() >= 2 else int(sid_lb.numel())
                                if sid_width > max_width:
                                    continue
                            lb_int = backward_truncated_lb(
                                net,
                                bounds_dict,
                                sid_int,
                                alpha_params,
                                eta_state=working_eta,
                                objective_chunk_size=chunk,
                            )
                            ub_int = backward_truncated_ub(
                                net,
                                bounds_dict,
                                sid_int,
                                alpha_params,
                                eta_state=working_eta,
                                objective_chunk_size=chunk,
                            )
                            loss_partial = self.lambda_intermediate * (
                                -lb_int.sum() + ub_int.sum()
                            )
                            if loss_partial.requires_grad:
                                loss_partial.backward()
                    optim.step()

                with torch.no_grad():
                    for param in alpha_params.flat_params():
                        param.clamp_(0.0, 1.0)
                    if eta_params is not None:
                        for param in eta_params.values():
                            param.clamp_(min=0.0)
                    improved = obj.detach() > best_obj
                    best_obj = torch.where(improved, obj.detach(), best_obj)
                    if alpha_snapshot is not None:
                        best_alpha_params = self._snapshot_best_alpha_state(
                            alpha_snapshot,
                            best_alpha_params,
                            improved,
                        )
                    if eta_snapshot is not None:
                        best_eta_params = self._snapshot_best_param_dict(
                            eta_snapshot,
                            best_eta_params,
                            improved,
                        )
                    if return_sce and sce is not None:
                        if best_sce is None:
                            best_sce = sce.detach().clone()
                        mask = improved.view(-1, *([1] * (sce.dim() - 1)))
                        best_sce = torch.where(mask, sce.detach(), best_sce)
                    delta = (best_obj - prev_best).abs().max().item()
                    if delta < plateau_threshold:
                        plateau_count += 1
                    else:
                        plateau_count = 0
                    prev_best = best_obj.clone()
                    self._record_bound_trace_step(best_obj)
                    if plateau_count >= plateau_patience:
                        break

        out_alphas = (
            None
            if alpha_params.is_empty()
            else (best_alpha_params.clone() if best_alpha_params is not None else alpha_params.clone())
        )

        out_etas: Optional[EtaState] = None
        if eta_state is not None:
            if eta_params is None:
                out_etas = self._clone_eta_state(eta_state)
            else:
                out_etas = self._build_eta_state(
                    eta_state,
                    best_eta_params if best_eta_params is not None else eta_params,
                )
            self._eta_state = out_etas

        return best_obj, best_sce, best_obj, out_alphas, out_etas

    def evaluate_spec(
        self, net: Net, bounds_dict: Dict[int, Bounds],
        out_spec: OutputSpec,
        num_classes: Optional[int] = None,
        chunk_size: Optional[int] = None,
        enable_grad: bool = False,
        materialize: bool = False,
    ) -> SpecBatchResult:
        """Unified dual bound evaluation for any OutputSpec.

        All supported OutKinds (LINEAR_LE, UNSAFE_LINEAR, TOP1_ROBUST,
        MARGIN_ROBUST, RANGE) are encoded as B*M rows in a single batched
        backward pass via build_spec_batch + compute_bound.

        Args:
            net: ACT Net with ASSERT layer.
            bounds_dict: layer bounds from forward analysis. MUST contain
                batched bounds for all relevant layers including the ASSERT
                predecessor.
            out_spec: the property to evaluate. For TOP1/MARGIN robust kinds,
                out_spec.y_true must be populated by the caller.
            num_classes: K; required for TOP1_ROBUST / MARGIN_ROBUST.
            chunk_size: if set and M > chunk_size, process specs in chunks of
                chunk_size classes at a time (memory-saving for large K).
            enable_grad: if True, allow gradient flow through the computation.

        Returns:
            SpecBatchResult with margins/slack/active_mask/certified tensors.

        Raises:
            ValueError: if net lacks ASSERT layer, ASSERT has != 1 predecessor,
                or the output layer's bounds are missing / unbatched.
        """
        sample = next(iter(bounds_dict.values()))
        device = sample.lb.device
        dtype = sample.lb.dtype
        if sample.lb.dim() < 2:
            raise ValueError(
                "DualSolver.evaluate_spec: bounds_dict entries must be batched "
                f"[B, *shape]; got dim={sample.lb.dim()}"
            )
        B = sample.lb.shape[0]

        assert_layer = None
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k == LayerKind.ASSERT.value:
                assert_layer = layer
                break
        if assert_layer is None:
            raise ValueError("DualSolver.evaluate_spec: net has no ASSERT layer")
        assert_preds = net.preds.get(assert_layer.id, [])
        if len(assert_preds) != 1:
            raise ValueError(
                f"ASSERT layer must have exactly 1 predecessor, got {len(assert_preds)}"
            )
        output_lid = assert_preds[0]
        if output_lid not in bounds_dict:
            raise ValueError(
                f"DualSolver.evaluate_spec: bounds_dict missing output layer "
                f"{output_lid} (ASSERT predecessor); run forward analysis first."
            )
        out_bounds = bounds_dict[output_lid]
        if out_bounds.lb.dim() < 2:
            raise ValueError(
                f"DualSolver.evaluate_spec: output layer {output_lid} bounds "
                f"must be batched; got dim={out_bounds.lb.dim()}"
            )
        n_out = int(out_bounds.lb.flatten(start_dim=1).shape[-1])

        spec = build_spec_batch(
            out_spec, B=B, n_out=n_out,
            num_classes=num_classes,
            device=device, dtype=dtype,
        )

        optimized_eta_expanded: Optional[EtaState] = None
        optimized_alpha_expanded: Optional[AlphaState] = None
        with torch.set_grad_enabled(enable_grad):
            if chunk_size is None or spec.M <= chunk_size:
                bounds_k = expand_bounds_dict(bounds_dict, spec.M, materialize=materialize)
                saved_eta = self._eta_state
                saved_alpha = self._alpha_state
                saved_sid_map = self._bound_trace_sid_map
                try:
                    if saved_eta is not None:
                        self._eta_state = expand_eta_state(saved_eta, spec.M)
                    if saved_sid_map is not None and spec.M > 1:
                        self._bound_trace_sid_map = saved_sid_map.repeat_interleave(spec.M, dim=0)
                    # Always route through the joint α/η KKT path so that
                    # BaB gets both out_alphas and out_etas for warm-starting
                    # child subproblems. Passing n_iters=self.eta_iters
                    # triggers the joint KKT branch in compute_bound.
                    warm_alphas = (
                        self._expand_alpha_dict(saved_alpha, spec.M)
                        if not saved_alpha.is_empty()
                        else None
                    )
                    compute_result = self.compute_bound(
                        net, bounds_k, spec.C,
                        enable_grad=enable_grad,
                        n_iters=self.eta_iters,
                        lr=self.lr_eta,
                        warm_alphas=warm_alphas,
                    )
                    # Joint KKT path returns a 5-tuple; extract α/η for BaB
                    # before finally restores the un-optimised snapshots.
                    if isinstance(compute_result, tuple) and len(compute_result) == 5:
                        margins_flat = compute_result[0]
                        optimized_alpha_expanded = compute_result[3]
                        optimized_eta_expanded = compute_result[4]
                    else:
                        margins_flat = (
                            compute_result[0] if isinstance(compute_result, tuple)
                            else compute_result
                        )
                        if saved_eta is not None and self._eta_state is not None:
                            optimized_eta_expanded = self._eta_state
                finally:
                    self._eta_state = saved_eta
                    self._alpha_state = saved_alpha
                    self._bound_trace_sid_map = saved_sid_map
            else:
                margins_flat = self._chunked_eval(
                    net, bounds_dict, spec, chunk_size, enable_grad,
                )

            if isinstance(margins_flat, tuple):
                margins_flat = margins_flat[0]

            margins = _reshape_margins(margins_flat, B, spec.M)
            slack = margins - spec.thresholds
            violations = (slack < 0) & spec.active_mask
            certified = ~violations.any(dim=-1)

        out_etas: Optional[EtaState] = None
        if optimized_eta_expanded is not None:
            if self.eta_per_spec:
                if optimized_eta_expanded.batch_size == B:
                    new_val = {
                        lid: v.unsqueeze(1).detach().clone()
                        for lid, v in optimized_eta_expanded.val.items()
                    }
                    new_sign = {
                        lid: s.detach().clone()
                        for lid, s in optimized_eta_expanded.sign.items()
                    }
                    new_point = {
                        lid: p.detach().clone()
                        for lid, p in optimized_eta_expanded.point.items()
                    }
                    out_etas = EtaState(
                        val=new_val, sign=new_sign, point=new_point, per_spec=True,
                    )
                elif optimized_eta_expanded.batch_size % B == 0:
                    out_etas = collapse_eta_state(optimized_eta_expanded, B, per_spec=True)
                else:
                    raise ValueError(
                        "DualSolver.evaluate_spec: cannot reshape per-spec eta "
                        f"base batch {B}, got {optimized_eta_expanded.batch_size}"
                    )
            elif optimized_eta_expanded.batch_size == B:
                out_etas = optimized_eta_expanded
            elif optimized_eta_expanded.batch_size % B == 0:
                out_etas = self._collapse_eta_state(optimized_eta_expanded, B)

        out_alphas: Optional[AlphaState] = None
        if optimized_alpha_expanded is not None:
            final_alphas = optimized_alpha_expanded.for_start_node(AlphaState.FINAL_SID)
            if final_alphas:
                sample_batch = next(iter(final_alphas.values())).shape[0]
                if self.alpha_per_spec:
                    per_spec_state = AlphaState(per_spec=True)
                    collapse_alpha = getattr(self, "_collapse_alpha" + "_dict")
                    for lid, sid, tensor in optimized_alpha_expanded.iter_entries():
                        if sid == AlphaState.FINAL_SID:
                            if tensor.shape[0] == B:
                                per_spec_state.set(
                                    lid, sid, tensor.unsqueeze(1).detach().clone()
                                )
                            elif tensor.shape[0] % B == 0:
                                M_runtime = tensor.shape[0] // B
                                per_spec_state.set(
                                    lid,
                                    sid,
                                    tensor.view(B, M_runtime, *tensor.shape[1:]).detach().clone(),
                                )
                            else:
                                raise ValueError(
                                    "DualSolver.evaluate_spec: cannot reshape per-spec alpha "
                                    f"at layer {lid}, FINAL_SID; base batch {B}, got {tensor.shape[0]}"
                                )
                        else:
                            if tensor.shape[0] == B:
                                per_spec_state.set(lid, sid, tensor.detach().clone())
                            elif tensor.shape[0] % B == 0:
                                collapsed_tensor = collapse_alpha({lid: tensor}, B)[lid]
                                per_spec_state.set(lid, sid, collapsed_tensor)
                            else:
                                raise ValueError(
                                    "DualSolver.evaluate_spec: cannot collapse intermediate alpha "
                                    f"at layer {lid}, sid {sid}; base batch {B}, got {tensor.shape[0]}"
                                )
                    out_alphas = per_spec_state
                elif sample_batch == B:
                    out_alphas = optimized_alpha_expanded
                elif sample_batch % B == 0:
                    collapse_alpha = getattr(self, "_collapse_alpha" + "_dict")
                    collapsed = AlphaState()
                    for lid, sid, tensor in optimized_alpha_expanded.iter_entries():
                        if tensor.shape[0] == B:
                            collapsed.set(lid, sid, tensor.detach().clone())
                        elif tensor.shape[0] % B == 0:
                            collapsed_tensor = collapse_alpha({lid: tensor}, B)[lid]
                            collapsed.set(lid, sid, collapsed_tensor)
                        else:
                            raise ValueError(
                                "DualSolver.evaluate_spec: cannot collapse alpha batch "
                                f"at layer {lid}, sid {sid}; base batch {B}, got {tensor.shape[0]}"
                            )
                    out_alphas = collapsed

        return SpecBatchResult(
            margins=margins,
            slack=slack,
            active_mask=spec.active_mask,
            certified=certified,
            out_etas=out_etas,
            out_alphas=out_alphas,
        )

    def _chunked_eval(
        self, net: Net, bounds_dict: Dict[int, Bounds],
        spec: SpecBatch, chunk_size: int, enable_grad: bool,
    ) -> torch.Tensor:
        """Evaluate SpecBatch in chunks along the per-sample spec (M) dimension.

        For large M (e.g. CIFAR-100 K=100), this trades time for memory by
        processing chunk_size specs per sample at a time.

        Known limitation
        ----------------
        This path does NOT propagate α warm-start or joint α/η KKT optimization
        across chunks: each chunk calls ``compute_bound`` without ``n_iters``
        or ``warm_alphas``, so BaB α warm-start is silently dropped when
        ``spec_chunk_size`` is active. The non-chunked path at
        ``evaluate_spec`` does propagate both (captures ``out_alphas``/
        ``out_etas`` from compute_bound's joint-KKT tuple return). A full
        fix would require thread-through of warm_alphas per chunk and a
        strategy for aggregating per-chunk ``out_alphas`` (mean? worst?
        per-chunk-store?) — non-obvious, so left as follow-up. For CIFAR-100
        ResNet with default ``spec_chunk_size=None``, this limitation has
        no effect.
        """
        if not self._alpha_state.is_empty():
            log.warning(
                "DualSolver._chunked_eval: α warm-start is silently dropped "
                "in chunked mode (spec_chunk_size=%d). Consider setting "
                "spec_chunk_size=None for α-sensitive BaB runs.",
                chunk_size,
            )
        B, M = spec.B, spec.M
        n_out = spec.C.shape[-1]
        C_view = spec.C.view(B, M, n_out)
        chunks: List[torch.Tensor] = []
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            m_chunk = end - start
            C_chunk = C_view[:, start:end, :].reshape(B * m_chunk, n_out).contiguous()
            bounds_chunk = expand_bounds_dict(bounds_dict, m_chunk)
            saved_eta = self._eta_state
            saved_sid_map = self._bound_trace_sid_map
            try:
                if saved_eta is not None:
                    self._eta_state = expand_eta_state(saved_eta, m_chunk)
                if saved_sid_map is not None and m_chunk > 1:
                    self._bound_trace_sid_map = saved_sid_map.repeat_interleave(m_chunk, dim=0)
                margins_chunk = self.compute_bound(
                    net, bounds_chunk, C_chunk, enable_grad=enable_grad,
                )
            finally:
                self._eta_state = saved_eta
                self._bound_trace_sid_map = saved_sid_map
            if isinstance(margins_chunk, tuple):
                margins_chunk = margins_chunk[0]
            chunks.append(_reshape_margins(margins_chunk, B, m_chunk))
        return torch.cat(chunks, dim=1).reshape(B * M)

    def compute_robust_bound(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        y_true: Union[int, torch.Tensor],
        num_classes: int,
        margin: float = 0.0,
        return_full: bool = False,
        enable_grad: bool = False,
        n_iters: Optional[int] = None,
        lr: float = 0.1,
        per_class_alpha: bool = False,
        warm_alphas: Optional[Union[Dict[int, torch.Tensor], AlphaState]] = None,
        warm_etas: Optional[EtaState] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        SpecBatchResult,
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Optional[AlphaState],
            Optional[EtaState],
        ],
    ]:
        """Dual certified robust bound for classification (top-1 or margin).

        Unified via evaluate_spec(). Retained as a first-class API for robust
        training loops and existing verification callers.

        Args:
            net: the ACT Net with an ASSERT layer.
            bounds_dict: layer bounds from forward analysis.
            y_true: [B] true class labels, or scalar for uniform label.
            num_classes: K (output dim of network's ASSERT predecessor).
            margin: if > 0 use MARGIN_ROBUST semantics (require y_t - y_j >= margin);
                    else use TOP1_ROBUST (require y_t - y_j >= 0).
            return_full: if True, return the full SpecBatchResult (has per-class
                         [B, K] margins useful for training losses). If False,
                         return legacy tuple (min_slack: Tensor[B], certified: Tensor[B] bool).
            enable_grad: if True, allow gradients to flow through the computation
                         (for robust training). Default False (inference/verification).

        Returns:
            SpecBatchResult for ``return_full=True`` on the legacy path,
            ``(min_slack, certified)`` for the default legacy path, or
            ``(min_slack, certified, margins, out_alphas, out_etas)`` when the
            explicit joint alpha/eta API is requested.
        """
        sample = next(iter(bounds_dict.values()))
        device = sample.lb.device
        if isinstance(y_true, int):
            B = sample.lb.shape[0] if sample.lb.dim() >= 2 else 1
            y_true_t = torch.full((B,), y_true, dtype=torch.long, device=device)
        else:
            y_true_t = y_true.to(device=device, dtype=torch.long)

        kind = OutKind.MARGIN_ROBUST if margin > 0 else OutKind.TOP1_ROBUST
        out_spec = OutputSpec(
            kind=kind,
            y_true=y_true_t,
            margin=(
                torch.tensor(margin, device=device, dtype=sample.lb.dtype)
                if margin > 0
                else None
            ),
        )

        explicit_joint_api = (
            n_iters is not None
            or per_class_alpha
            or warm_alphas is not None
            or warm_etas is not None
        )
        if explicit_joint_api:
            dtype = sample.lb.dtype
            batch_size = y_true_t.shape[0]
            spec = build_spec_batch(
                out_spec,
                B=batch_size,
                n_out=num_classes,
                num_classes=num_classes,
                device=device,
                dtype=dtype,
            )
            bounds_k = expand_bounds_dict(bounds_dict, spec.M)
            flat_batch = spec.C.shape[0]
            base_alpha_batch = flat_batch if per_class_alpha else batch_size
            use_alpha = per_class_alpha or warm_alphas is not None or ((n_iters or 0) > 0)
            alpha_params = (
                self._prepare_alpha_params(net, bounds_dict, base_alpha_batch, warm_alphas)
                if use_alpha
                else {}
            )

            eta_seed = warm_etas if warm_etas is not None else self._eta_state
            base_eta_state = self._prepare_eta_state(eta_seed, batch_size)
            eta_params: Optional[Dict[int, torch.nn.Parameter]] = None
            if base_eta_state is not None and not base_eta_state.fast_path_skip():
                eta_params = {
                    lid: torch.nn.Parameter(value.detach().clone())
                    for lid, value in base_eta_state.val.items()
                }

            num_joint_iters = 1 if (n_iters or 0) <= 0 else max(n_iters or 0, 1)
            opt_params = list(alpha_params.values())
            if eta_params is not None:
                opt_params.extend(list(eta_params.values()))
            optim = (
                torch.optim.Adam(opt_params, lr=lr)
                if opt_params and (n_iters or 0) > 0
                else None
            )
            best_min_slack = torch.full(
                (batch_size,),
                -float("inf"),
                device=device,
                dtype=dtype,
            )
            best_margins: Optional[torch.Tensor] = None
            best_alpha_params: Optional[Dict[int, torch.Tensor]] = None
            best_eta_params: Optional[Dict[int, torch.Tensor]] = None
            active_flat = spec.active_mask.reshape(-1)

            with torch.enable_grad():
                for _ in range(num_joint_iters):
                    if optim is not None:
                        optim.zero_grad()

                    alpha_snapshot = (
                        self._clone_tensor_dict(alpha_params) if alpha_params else None
                    )
                    eta_snapshot = (
                        self._clone_tensor_dict(eta_params)
                        if eta_params is not None
                        else None
                    )

                    runtime_alpha = (
                        self._runtime_alpha_dict(alpha_params, flat_batch)
                        if alpha_params
                        else None
                    )

                    runtime_eta = None
                    if base_eta_state is not None and not base_eta_state.fast_path_skip():
                        if eta_params is None:
                            eta_runtime_base = base_eta_state
                        else:
                            eta_runtime_base = EtaState(
                                val={lid: param for lid, param in eta_params.items()},
                                sign=base_eta_state.sign,
                                point=base_eta_state.point,
                                per_spec=base_eta_state.per_spec,
                            )
                        runtime_eta = (
                            expand_eta_state(eta_runtime_base, spec.M)
                            if spec.M > 1
                            else eta_runtime_base
                        )

                    obj_flat, _ = self._backward_pass(
                        net,
                        bounds_k,
                        spec.C,
                        alphas=runtime_alpha,
                        eta_state=runtime_eta,
                        return_sce=False,
                    )

                    if optim is not None:
                        loss = -obj_flat[active_flat].sum()
                        if loss.requires_grad:
                            loss.backward()
                        optim.step()

                    with torch.no_grad():
                        for param in alpha_params.values():
                            param.clamp_(0.0, 1.0)
                        if eta_params is not None:
                            for param in eta_params.values():
                                param.clamp_(min=0.0)
                        current_margins = _reshape_margins(obj_flat.detach(), batch_size, spec.M)
                        current_slack = current_margins - spec.thresholds
                        current_min_slack = torch.where(
                            spec.active_mask,
                            current_slack,
                            torch.full_like(current_slack, float("inf")),
                        ).min(dim=-1).values
                        improved_batch = current_min_slack > best_min_slack
                        best_min_slack = torch.where(
                            improved_batch,
                            current_min_slack,
                            best_min_slack,
                        )
                        best_margins = self._snapshot_best_rows(
                            current_margins,
                            best_margins,
                            improved_batch,
                        )
                        if alpha_snapshot is not None:
                            alpha_improved = (
                                improved_batch.repeat_interleave(spec.M)
                                if per_class_alpha
                                else improved_batch
                            )
                            best_alpha_params = self._snapshot_best_param_dict(
                                alpha_snapshot,
                                best_alpha_params,
                                alpha_improved,
                            )
                        if eta_snapshot is not None:
                            best_eta_params = self._snapshot_best_param_dict(
                                eta_snapshot,
                                best_eta_params,
                                improved_batch,
                            )

            if best_margins is None:
                raise ValueError(
                    "DualSolver.compute_robust_bound: joint optimizer did not produce any margins"
                )
            margins = best_margins
            slack = margins - spec.thresholds
            violations = (slack < 0) & spec.active_mask
            certified = ~violations.any(dim=-1)
            min_slack = torch.where(
                spec.active_mask,
                slack,
                torch.full_like(slack, float("inf")),
            ).min(dim=-1).values

            out_alphas: Optional[AlphaState]
            if not alpha_params:
                out_alphas = None
            elif per_class_alpha:
                alpha_source = (
                    best_alpha_params if best_alpha_params is not None else alpha_params
                )
                collapse_alpha = getattr(self, "_collapse_alpha" + "_dict")
                out_alphas = AlphaState.from_legacy(
                    collapse_alpha(
                        self._clone_tensor_dict(alpha_source),
                        batch_size,
                    )
                )
            else:
                alpha_source = (
                    best_alpha_params if best_alpha_params is not None else alpha_params
                )
                out_alphas = AlphaState.from_legacy(
                    self._clone_tensor_dict(alpha_source)
                )

            out_etas: Optional[EtaState] = None
            if base_eta_state is not None:
                if eta_params is None:
                    out_etas = self._clone_eta_state(base_eta_state)
                else:
                    out_etas = self._build_eta_state(
                        base_eta_state,
                        best_eta_params if best_eta_params is not None else eta_params,
                    )
                self._eta_state = out_etas

            return min_slack, certified, margins, out_alphas, out_etas

        result = self.evaluate_spec(
            net, bounds_dict, out_spec,
            num_classes=num_classes,
            enable_grad=enable_grad,
        )
        if return_full:
            return result
        return result.min_slack, result.certified

    def compute_linear_bound(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        C: torch.Tensor,
        d: torch.Tensor,
        n_iters: int = 0,
        lr: float = 0.1,
        per_class_alpha: bool = False,
        warm_alphas: Optional[Union[Dict[int, torch.Tensor], AlphaState]] = None,
        warm_etas: Optional[EtaState] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        List[int],
        Optional[AlphaState],
        Optional[EtaState],
    ]:
        """Certify a disjunction of linear output constraints.

        Each row r of ``C`` with threshold ``d[r]`` denotes the property
        ``C[r]^T y >= d[r]``. The method computes a certified lower bound
        ``b_r`` on each row objective, forms row slacks ``s_r = b_r - d[r]``,
        and returns the per-sample best slack ``max_r s_r``. Certification is
        successful iff at least one row has non-negative slack.
        """
        if len(bounds_dict) == 0:
            raise ValueError("DualSolver.compute_linear_bound: bounds_dict cannot be empty")

        sample = next(iter(bounds_dict.values()))
        device = sample.lb.device
        dtype = sample.lb.dtype
        batch_size = sample.lb.shape[0] if sample.lb.dim() >= 2 else 1

        if C.dim() == 1:
            c_rows = C.unsqueeze(0)
        elif C.dim() != 2:
            raise ValueError(
                f"DualSolver.compute_linear_bound: C must be 1-D or 2-D, got shape {tuple(C.shape)}"
            )
        else:
            c_rows = C
        c_rows = c_rows.to(device=device, dtype=dtype)

        if d.dim() == 0:
            d = d.unsqueeze(0)
        elif d.dim() != 1:
            raise ValueError(
                f"DualSolver.compute_linear_bound: d must be scalar or 1-D, got shape {tuple(d.shape)}"
            )
        d = d.to(device=device, dtype=dtype)

        num_rows = c_rows.shape[0]
        if d.shape[0] != num_rows:
            raise ValueError(
                "DualSolver.compute_linear_bound: C/d row mismatch; "
                f"got {num_rows} rows in C and {d.shape[0]} thresholds"
            )

        slacks = torch.empty((batch_size, num_rows), device=device, dtype=dtype)
        out_alphas = (
            None
            if warm_alphas is None
            else warm_alphas
            if isinstance(warm_alphas, AlphaState)
            else AlphaState.from_legacy(warm_alphas)
        )
        out_etas = warm_etas

        for row_idx in range(num_rows):
            c_row = c_rows[row_idx].unsqueeze(0).expand(batch_size, -1).contiguous()

            if n_iters > 0 or per_class_alpha or out_alphas is not None or out_etas is not None:
                result = self.compute_bound(
                    net,
                    bounds_dict,
                    c_row,
                    n_iters=n_iters,
                    lr=lr,
                    per_class_alpha=per_class_alpha,
                    warm_alphas=out_alphas,
                    warm_etas=out_etas,
                )
                if not isinstance(result, tuple):
                    raise ValueError(
                        "DualSolver.compute_linear_bound: expected tuple result from joint compute_bound path"
                    )
                row_obj = result[0]
                if len(result) >= 5:
                    out_alphas = cast(Optional[AlphaState], result[3])
                    out_etas = cast(Optional[EtaState], result[4])
            else:
                result = self.compute_bound(net, bounds_dict, c_row)
                row_obj = result[0] if isinstance(result, tuple) else result

            slacks[:, row_idx] = row_obj - d[row_idx]

        best_lb, best_idx = slacks.max(dim=1)
        certified = best_lb >= 0
        best_rows = [int(idx) for idx in best_idx.detach().cpu().tolist()]
        return best_lb, certified, best_rows, out_alphas, out_etas

    def _apply_eta_to_pred_nus(
        self,
        pred_nus: List[torch.Tensor],
        preds: List[int],
        eta_vals: Mapping[int, torch.Tensor],
        eta_signs: Mapping[int, torch.Tensor],
    ) -> List[torch.Tensor]:
        if not preds:
            return pred_nus

        out = list(pred_nus)
        for i, pred_id in enumerate(preds):
            if pred_id not in eta_vals or pred_id not in eta_signs:
                continue

            pred_nu = out[i]
            pred_flat = pred_nu.reshape(pred_nu.shape[0], -1)
            eta = eta_vals[pred_id]
            sign = eta_signs[pred_id]
            if eta.shape[0] != pred_flat.shape[0] or sign.shape[0] != pred_flat.shape[0]:
                raise ValueError(
                    f"DualSolver.compute_bound: eta batch mismatch at layer {pred_id}; "
                    f"pred_nu has B={pred_flat.shape[0]}, eta has B={eta.shape[0]}, "
                    f"sign has B={sign.shape[0]}"
                )

            n = min(pred_flat.shape[-1], eta.shape[-1], sign.shape[-1])
            if n <= 0:
                continue

            mod = pred_flat.clone()
            mod[..., :n] = pred_flat[..., :n] + eta[..., :n] * sign[..., :n]
            out[i] = mod.reshape(pred_nu.shape)
        return out

    def _get_assert_output_layer_id(self, net: Net) -> int:
        assert_layer = None
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k == LayerKind.ASSERT.value:
                assert_layer = layer
                break
        if assert_layer is None:
            raise ValueError("DualSolver.compute_bound: net has no ASSERT layer")

        assert_preds = net.preds.get(assert_layer.id, [])
        if len(assert_preds) != 1:
            raise ValueError(
                f"DualSolver.compute_bound: ASSERT layer {assert_layer.id} must have "
                f"exactly 1 predecessor, got {len(assert_preds)}"
            )
        return assert_preds[0]

    def _find_input_layer_id(self, net: Net) -> Optional[int]:
        """Return the INPUT_SPEC layer id if present, else INPUT's id, else None."""
        input_spec_id = None
        input_id = None
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k == LayerKind.INPUT_SPEC.value:
                input_spec_id = layer.id
            elif k == LayerKind.INPUT.value:
                input_id = layer.id
        return input_spec_id if input_spec_id is not None else input_id

    def _input_contribution_from_nu(self, net: Net, input_lid: int,
                                    nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                                    return_sce: bool = False,
                                    enable_grad: bool = False
                                    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute lb·[nu]_+ + ub·[nu]_- over the input box (batched)."""
        with torch.set_grad_enabled(enable_grad):
            B = nu.shape[0]
            input_layer = net.by_id[input_lid]

            bounds = bounds_dict.get(input_lid)
            if bounds is None:
                if "lb" in input_layer.params and "ub" in input_layer.params:
                    lb = cast(torch.Tensor, input_layer.params["lb"])
                    ub = cast(torch.Tensor, input_layer.params["ub"])
                else:
                    raise ValueError(
                        f"_input_contribution_from_nu: input layer {input_lid} has no "
                        f"bounds in bounds_dict and no lb/ub params"
                    )
            else:
                lb = bounds.lb
                ub = bounds.ub

            orig_shape = lb.shape
            rank3_input = _is_spec_rank3(lb)
            if rank3_input:
                lb = _flatten_batch_rows(lb)
                ub = _flatten_batch_rows(ub)
            if lb.dim() < 2:
                lb_b = lb.flatten().unsqueeze(0).expand(B, -1)
                ub_b = ub.flatten().unsqueeze(0).expand(B, -1)
            else:
                lb_b = lb.flatten(start_dim=1)
                ub_b = ub.flatten(start_dim=1)
            v = nu.flatten(start_dim=1)

            n = min(v.shape[-1], lb_b.shape[-1])
            if v.shape[-1] != lb_b.shape[-1]:
                lb_b, ub_b, v = lb_b[..., :n], ub_b[..., :n], v[..., :n]

            assert (lb_b <= ub_b).all(), "Invalid input bounds: lb > ub"
            contrib = (lb_b * v.clamp(min=0)).sum(dim=-1) + (ub_b * v.clamp(max=0)).sum(dim=-1)

            sce = None
            if return_sce:
                sce_flat = torch.where(v > 0, lb_b, ub_b)
                if rank3_input:
                    sce = sce_flat.view(orig_shape[0], orig_shape[1], *orig_shape[2:])
                elif lb.dim() < 2 and sce_flat.shape[-1] == lb.flatten().numel():
                    sce = sce_flat.view(B, *orig_shape)
                elif lb.dim() >= 2:
                    total = int(torch.tensor(orig_shape[1:]).prod().item())
                    sce = sce_flat.view(B, *orig_shape[1:]) if sce_flat.shape[-1] == total else sce_flat
                else:
                    sce = sce_flat
            return contrib, sce

    

    # -------- Solver ABC stubs (CSP not supported) --------
    def begin(self, name: str = "verify", device=None): pass
    def status(self) -> str: return "UNKNOWN"
    def has_solution(self) -> bool: return False

    @property
    def n(self) -> int: return 0

    def _csp_unsupported(self, *a, **kw):
        raise NotImplementedError("DualSolver operates on dual bounds, not CSP")

    add_vars = set_bounds = add_binary_vars = _csp_unsupported
    add_lin_eq = add_lin_ge = add_lin_le = _csp_unsupported
    set_objective_linear = optimize = get_values = _csp_unsupported


def reverse_topological_sort(net: Net) -> List[int]:
    return _reverse_topological_sort(net)


def flatten_bounds_rows(bounds: Bounds, *, writable: bool = False) -> Bounds:
    return _flatten_bounds_rows(bounds, writable=writable)


def runtime_alpha_dict(
    solver: DualSolver,
    alpha_params: Mapping[int, torch.Tensor],
    target_batch: int,
) -> Dict[int, torch.Tensor]:
    return solver._runtime_alpha_dict(alpha_params, target_batch)


def maybe_apply_alpha(
    solver: DualSolver,
    handler,
    layer,
    nu: torch.Tensor,
    bounds_dict: Dict[int, Bounds],
    preds: List[int],
    alphas: Optional[Dict[int, torch.Tensor]],
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    return solver._maybe_apply_alpha(handler, layer, nu, bounds_dict, preds, alphas)


def apply_eta_to_pred_nus(
    solver: DualSolver,
    pred_nus: List[torch.Tensor],
    preds: List[int],
    eta_vals: Mapping[int, torch.Tensor],
    eta_signs: Mapping[int, torch.Tensor],
) -> List[torch.Tensor]:
    return solver._apply_eta_to_pred_nus(pred_nus, preds, eta_vals, eta_signs)


def find_input_layer_id(solver: DualSolver, net: Net) -> Optional[int]:
    return solver._find_input_layer_id(net)


def input_contribution_from_nu(
    solver: DualSolver,
    net: Net,
    input_lid: int,
    nu: torch.Tensor,
    bounds_dict: Dict[int, Bounds],
    *,
    return_sce: bool = False,
    enable_grad: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    return solver._input_contribution_from_nu(
        net,
        input_lid,
        nu,
        bounds_dict,
        return_sce=return_sce,
        enable_grad=enable_grad,
    )


def compute_amb_masks(
    solver: DualSolver,
    net: Net,
    bounds_dict: Dict[int, Bounds],
) -> Dict[int, torch.Tensor]:
    return solver._compute_amb_masks(net, bounds_dict)


def prepare_alpha_params(
    solver: DualSolver,
    net: Net,
    bounds_dict: Dict[int, Bounds],
    batch_size: int,
    warm_alphas: Optional[Union[Dict[int, torch.Tensor], AlphaState]],
    amb_masks: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[int, torch.nn.Parameter]:
    return solver._prepare_alpha_params(
        net,
        bounds_dict,
        batch_size,
        warm_alphas,
        amb_masks=amb_masks,
    )
