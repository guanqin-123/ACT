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
import torch
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, cast
from act.back_end.core import Bounds, Net
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_base import Solver, SolverCaps
from act.util.device_manager import get_default_device, get_default_dtype

if TYPE_CHECKING:
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


class DualSolver(Solver):
    """Dual (Wong-Kolter) certified bounds solver. Strict [B, *shape] API."""

    _AFFINE_CONTRIB_KINDS = {
        LayerKind.DENSE.value,
        LayerKind.CONV2D.value,
        "BIAS",
        "BN",
        "ADD",
    }

    def __init__(self, tf: "DualTF", n_iters: int = 0):
        self.tf = tf
        self.n_iters = n_iters
        self._last_bounds: Optional[Bounds] = None

    def capabilities(self) -> SolverCaps:
        return SolverCaps(supports_gpu=True, supports_csp=False, supports_dual=True)

    @torch.no_grad()
    def compute_bound(self, net: Net, bounds_dict: Dict[int, Bounds],
                      c: torch.Tensor, return_sce: bool = False
                      ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Batched certified lower bound on c^T @ output (DAG-aware).

        ν is propagated backward through a per-layer accumulator:
          nu_accum[lid] = sum over all successors s of ν routed by s's handler to lid.

        Each handler returns per-pred νs; the outer loop distributes them to preds.

        Unknown layer kind raises ValueError (no silent identity fallback for soundness).
        Args:
            c: Tensor[B, num_classes] — REQUIRED batched. Raises ValueError if 1-D.
        Returns:
            Tensor[B] or (Tensor[B], Tensor[B, *in_shape]) when return_sce=True.
        """
        if c.dim() != 2:
            raise ValueError(
                f"c must be 2-D [B, num_classes], got shape {tuple(c.shape)}. "
                "Use c.unsqueeze(0) for single instance.")
        assert len(bounds_dict) > 0, "bounds_dict cannot be empty"
        device, dtype = get_default_device(), get_default_dtype()
        if c.dtype != dtype or c.device != device:
            c = c.to(device=device, dtype=dtype)
        B = c.shape[0]

        for _ in range(self.n_iters):
            pass

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

        output_lid = assert_preds[0]
        nu_accum: Dict[int, torch.Tensor] = {output_lid: c.clone()}
        obj = torch.zeros(B, dtype=c.dtype, device=c.device)

        topo_order = _reverse_topological_sort(net)
        registry = self.tf._BACKWARD_REGISTRY

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
            pred_nus, contrib = handler(layer, nu_here, bounds_dict, preds)

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
            return (obj, None) if return_sce else obj

        nu_final = nu_accum.get(input_lid)
        if nu_final is None:
            return (obj, None) if return_sce else obj

        input_contrib, sce = self._input_contribution_from_nu(
            net,
            input_lid,
            nu_final,
            bounds_dict,
            return_sce=return_sce,
        )
        obj = obj + input_contrib
        return (obj, sce) if return_sce else obj

    @torch.no_grad()
    def compute_robust_bound(self, net: Net, bounds_dict: Dict[int, Bounds],
                             y_true: Union[int, torch.Tensor], num_classes: int
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Oracle-approved batched margin: mask true-class rows to +inf.

        Returns (min_margin: Tensor[B], certified: Tensor[B] bool).
        """
        sample = next(iter(bounds_dict.values()))
        device, dtype = sample.lb.device, sample.lb.dtype
        B_bounds = sample.lb.shape[0] if sample.lb.dim() >= 2 else 1

        if isinstance(y_true, int):
            y_true_t = torch.full((B_bounds,), y_true, dtype=torch.long, device=device)
        else:
            y_true_t = y_true.to(device=device, dtype=torch.long)
        B = y_true_t.shape[0]

        margins_list = []
        for j in range(num_classes):
            c = torch.zeros(B, num_classes, dtype=dtype, device=device)
            any_active = False
            for b in range(B):
                if j == int(y_true_t[b].item()):
                    continue
                c[b, int(y_true_t[b].item())] = 1.0
                c[b, j] = -1.0
                any_active = True
            if not any_active:
                continue
            margin_j = self.compute_bound(net, bounds_dict, c)
            mask = (y_true_t != j)
            margin_j = torch.where(
                mask, margin_j, torch.full_like(margin_j, float("inf")))
            margins_list.append(margin_j)

        if not margins_list:
            return (torch.full((B,), float("inf"), dtype=dtype, device=device),
                    torch.ones(B, dtype=torch.bool, device=device))
        margins = torch.stack(margins_list, dim=-1)
        min_margins, _ = margins.min(dim=-1)
        return min_margins, (min_margins > 0)

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
                                    return_sce: bool = False
                                    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute lb·[nu]_+ + ub·[nu]_- over the input box (batched)."""
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
            if lb.dim() < 2 and sce_flat.shape[-1] == lb.flatten().numel():
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
