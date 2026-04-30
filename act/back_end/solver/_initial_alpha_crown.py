from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

import torch

from act.back_end.bab.eta import get_pre_activation_layer_id
from act.back_end.core import Bounds, Net
from act.back_end.dual_tf import dual_tf as dual_tf_module
from act.back_end.layer_schema import LayerKind
from act.back_end.solver._backward_truncated import (
    backward_truncated_lb,
    backward_truncated_ub,
)
from act.back_end.solver.alpha_state import AlphaState
from act.back_end.solver import solver_dual as solver_dual_module

log = logging.getLogger(__name__)

_RELU_KINDS = frozenset({LayerKind.RELU.value, LayerKind.LRELU.value})


def enumerate_intermediate_start_nodes(net: Net) -> list[int]:
    seen: set[int] = set()
    start_nodes: list[int] = []
    for layer in net.layers:
        kind = layer.kind.upper()
        if kind not in _RELU_KINDS:
            continue
        sid_int = get_pre_activation_layer_id(net, layer.id)
        if sid_int not in seen:
            seen.add(sid_int)
            start_nodes.append(sid_int)
    return start_nodes


def _enumerate_upstream_relus(net: Net, sid_int: int) -> list[int]:
    seen: set[int] = {sid_int}
    stack = [sid_int]
    ancestors: set[int] = set()
    while stack:
        lid = stack.pop()
        for pred in net.preds.get(lid, []):
            if pred in seen:
                continue
            seen.add(pred)
            ancestors.add(pred)
            stack.append(pred)
    return [
        layer.id
        for layer in net.layers
        if layer.id in ancestors
        and layer.kind.upper() in _RELU_KINDS
    ]


def _snapshot_best_rows(
    current: torch.Tensor,
    best: torch.Tensor | None,
    improved: torch.Tensor,
) -> torch.Tensor:
    current_detached = current.detach()
    if best is None:
        return current_detached.clone()
    mask = improved.view(-1, *([1] * (current_detached.dim() - 1)))
    return torch.where(mask, current_detached, best)


def _snapshot_best_sid(
    snapshot_source: dict[int, torch.Tensor],
    relu_ids: list[int],
    best: dict[int, torch.Tensor] | None,
    improved: torch.Tensor,
) -> dict[int, torch.Tensor]:
    snapshot: dict[int, torch.Tensor] = {} if best is None else dict(best)
    for lid in relu_ids:
        tensor = snapshot_source.get(lid)
        if tensor is None:
            continue
        snapshot[lid] = _snapshot_best_rows(tensor, snapshot.get(lid), improved)
    return snapshot


def _clone_sid_snapshot(
    alpha_state: AlphaState,
    sid_int: int,
    relu_ids: list[int],
) -> dict[int, torch.Tensor]:
    snapshot: dict[int, torch.Tensor] = {}
    for lid in relu_ids:
        tensor = alpha_state.get(lid, sid_int)
        if tensor is None:
            continue
        snapshot[lid] = tensor.detach().clone()
    return snapshot


def optimize_initial_intermediate_bounds(
    net: Net,
    bounds_dict: dict[int, Bounds],
    *,
    alpha_iters: int,
    lr_alpha: float,
    objective_chunk_size: int | None = None,
    per_spec: bool = False,
    log: logging.Logger | None = None,
) -> tuple[dict[int, Bounds], AlphaState]:
    logger = log or globals()["log"]
    if alpha_iters <= 0:
        return dict(bounds_dict), AlphaState(per_spec=per_spec)

    intermediate_sids = enumerate_intermediate_start_nodes(net)
    if not intermediate_sids:
        return dict(bounds_dict), AlphaState(per_spec=per_spec)

    solver = solver_dual_module.DualSolver(dual_tf_module.DualTF())
    sample = next(iter(bounds_dict.values()))
    batch_size = int(sample.lb.shape[0]) if sample.lb.dim() >= 2 else 1
    device = sample.lb.device
    dtype = sample.lb.dtype

    amb_masks = solver_dual_module.compute_amb_masks(solver, net, bounds_dict)
    alpha_templates = solver_dual_module.prepare_alpha_params(
        solver,
        net,
        bounds_dict,
        batch_size,
        warm_alphas=None,
        amb_masks=amb_masks,
    )

    upstream_relus = {
        sid_int: _enumerate_upstream_relus(net, sid_int)
        for sid_int in intermediate_sids
    }
    alpha_state = AlphaState(per_spec=per_spec)
    for sid_int, relu_ids in upstream_relus.items():
        for lid in relu_ids:
            template = alpha_templates.get(lid)
            if template is None:
                continue
            alpha_state.set(lid, sid_int, torch.nn.Parameter(template.detach().clone()))

    if alpha_state.is_empty():
        _ = logger.debug("optimize_initial_intermediate_bounds: no upstream ReLUs")
        return dict(bounds_dict), alpha_state

    optim: torch.optim.Optimizer = torch.optim.Adam(alpha_state.flat_params(), lr=lr_alpha)
    step_fn = cast(Callable[[], object | None], optim.step)
    best_lb = {
        sid_int: bounds_dict[sid_int].lb.reshape(batch_size, -1).detach().clone()
        for sid_int in intermediate_sids
    }
    best_ub = {
        sid_int: bounds_dict[sid_int].ub.reshape(batch_size, -1).detach().clone()
        for sid_int in intermediate_sids
    }
    best_alpha_by_sid = {
        sid_int: _clone_sid_snapshot(alpha_state, sid_int, upstream_relus[sid_int])
        for sid_int in intermediate_sids
    }
    prev_best_total = torch.stack(
        [best_lb[sid_int].sum(dim=-1) - best_ub[sid_int].sum(dim=-1) for sid_int in intermediate_sids],
        dim=0,
    ).sum(dim=0)
    plateau_count = 0
    plateau_threshold = 5e-4
    plateau_patience = 2

    with torch.enable_grad():
        for _ in range(alpha_iters):
            optim.zero_grad()
            alpha_snapshot_by_sid = {
                sid_int: _clone_sid_snapshot(alpha_state, sid_int, upstream_relus[sid_int])
                for sid_int in intermediate_sids
            }
            current_lb: dict[int, torch.Tensor] = {}
            current_ub: dict[int, torch.Tensor] = {}
            for sid_int in intermediate_sids:
                lb_sid = backward_truncated_lb(net, bounds_dict, sid_int, alpha_state, objective_chunk_size=objective_chunk_size)
                ub_sid = backward_truncated_ub(net, bounds_dict, sid_int, alpha_state, objective_chunk_size=objective_chunk_size)
                current_lb[sid_int] = lb_sid.detach()
                current_ub[sid_int] = ub_sid.detach()
                loss_partial = -lb_sid.sum() + ub_sid.sum()
                if loss_partial.requires_grad:
                    torch.autograd.backward(loss_partial)
            _ = step_fn()

            with torch.no_grad():
                for param in alpha_state.flat_params():
                    _ = param.clamp_(0.0, 1.0)

                for sid_int in intermediate_sids:
                    lb_now = current_lb[sid_int]
                    ub_now = current_ub[sid_int]
                    non_looser = (lb_now >= best_lb[sid_int] - 1e-7).all(dim=-1) & (
                        ub_now <= best_ub[sid_int] + 1e-7
                    ).all(dim=-1)
                    strictly_better = (lb_now > best_lb[sid_int] + 1e-7).any(dim=-1) | (
                        ub_now < best_ub[sid_int] - 1e-7
                    ).any(dim=-1)
                    improved = non_looser & strictly_better
                    best_lb[sid_int] = _snapshot_best_rows(lb_now, best_lb[sid_int], improved)
                    best_ub[sid_int] = _snapshot_best_rows(ub_now, best_ub[sid_int], improved)
                    best_alpha_by_sid[sid_int] = _snapshot_best_sid(
                        alpha_snapshot_by_sid[sid_int],
                        upstream_relus[sid_int],
                        best_alpha_by_sid.get(sid_int),
                        improved,
                    )

                best_total = torch.stack(
                    [best_lb[sid_int].sum(dim=-1) - best_ub[sid_int].sum(dim=-1) for sid_int in intermediate_sids],
                    dim=0,
                ).sum(dim=0)
                delta = (best_total - prev_best_total).abs().max().item()
                plateau_count = plateau_count + 1 if delta < plateau_threshold else 0
                prev_best_total = best_total.clone()
                if plateau_count >= plateau_patience:
                    break

    with torch.no_grad():
        for sid_int, relu_tensors in best_alpha_by_sid.items():
            for lid, best_tensor in relu_tensors.items():
                param = alpha_state.get(lid, sid_int)
                if param is None:
                    raise RuntimeError(
                        f"optimize_initial_intermediate_bounds: missing alpha ({lid}, {sid_int})"
                    )
                _ = param.copy_(best_tensor)

    new_bounds = dict(bounds_dict)
    with torch.no_grad():
        for sid_int in intermediate_sids:
            old_bounds = bounds_dict[sid_int]
            candidate_lb = backward_truncated_lb(net, bounds_dict, sid_int, alpha_state, objective_chunk_size=objective_chunk_size).reshape_as(old_bounds.lb)
            candidate_ub = backward_truncated_ub(net, bounds_dict, sid_int, alpha_state, objective_chunk_size=objective_chunk_size).reshape_as(old_bounds.ub)
            new_lb = torch.maximum(candidate_lb, old_bounds.lb)
            new_ub = torch.minimum(candidate_ub, old_bounds.ub)
            invalid = (~torch.isfinite(new_lb)) | (~torch.isfinite(new_ub)) | (new_lb > new_ub + 1e-5)
            if invalid.any():
                _ = logger.warning(
                    "initial α-CROWN produced %d invalid entries at layer %s; reverting those entries to forward bounds",
                    int(invalid.sum().item()),
                    sid_int,
                )
                new_lb = torch.where(invalid, old_bounds.lb, new_lb)
                new_ub = torch.where(invalid, old_bounds.ub, new_ub)
            new_bounds[sid_int] = Bounds(lb=new_lb, ub=new_ub)

    return new_bounds, alpha_state


__all__ = [
    "enumerate_intermediate_start_nodes",
    "optimize_initial_intermediate_bounds",
]
