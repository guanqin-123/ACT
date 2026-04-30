from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

import torch

from act.back_end.dual_tf import dual_tf as dual_tf_module
from act.back_end.bab.eta import EtaState, expand_eta_state
from act.back_end.core import Bounds, Net
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.alpha_state import AlphaState
from act.back_end.solver import solver_dual as solver_dual_module
from act.back_end.solver.spec_batching import expand_bounds_dict

log = logging.getLogger(__name__)

BackwardHandler = Callable[
    [object, torch.Tensor, dict[int, Bounds], list[int]],
    tuple[list[torch.Tensor], torch.Tensor],
]
MaybeApplyAlphaFn = Callable[
    [
        solver_dual_module.DualSolver,
        BackwardHandler,
        object,
        torch.Tensor,
        dict[int, Bounds],
        list[int],
        dict[int, torch.Tensor] | None,
    ],
    tuple[list[torch.Tensor], torch.Tensor],
]
maybe_apply_alpha_fn = cast(MaybeApplyAlphaFn, solver_dual_module.maybe_apply_alpha)


def _flatten_layer_width(bounds: Bounds) -> tuple[int, int]:
    flat = solver_dual_module.flatten_bounds_rows(bounds).lb
    if flat.dim() < 2:
        flat = flat.flatten().unsqueeze(0)
    else:
        flat = flat.flatten(start_dim=1)
    return int(flat.shape[0]), int(flat.shape[1])


def _coordinate_objective_rows(
    bounds_dict: dict[int, Bounds],
    sid_int: int,
    *,
    sign: float,
    coord_start: int = 0,
    coord_end: int | None = None,
) -> tuple[torch.Tensor, int, int]:
    if sid_int not in bounds_dict:
        raise ValueError(f"backward_truncated_lb: missing bounds for layer {sid_int}")
    base_batch, width = _flatten_layer_width(bounds_dict[sid_int])
    end = width if coord_end is None else coord_end
    if not (0 <= coord_start < end <= width):
        raise ValueError(
            f"_coordinate_objective_rows: invalid chunk range [{coord_start}, {end}) for width {width}"
        )
    sample = solver_dual_module.flatten_bounds_rows(bounds_dict[sid_int]).lb
    chunk_size = end - coord_start
    indices = torch.arange(coord_start, end, device=sample.device)
    eye_chunk = torch.nn.functional.one_hot(indices, num_classes=width).to(dtype=sample.dtype)
    coeffs = sign * eye_chunk.unsqueeze(0).expand(base_batch, -1, -1)
    return coeffs.reshape(base_batch * chunk_size, width).contiguous(), base_batch, width


def _backward_truncated_objective(
    net: Net,
    bounds_dict: dict[int, Bounds],
    sid_int: int,
    coeffs: torch.Tensor,
    alpha_state: AlphaState,
    *,
    eta_state: EtaState | None = None,
) -> torch.Tensor:
    solver = solver_dual_module.DualSolver(dual_tf_module.DualTF())
    if sid_int not in net.by_id:
        raise ValueError(f"backward_truncated_lb: unknown layer id {sid_int}")
    if coeffs.dim() == 1:
        coeffs = coeffs.unsqueeze(0)
    if coeffs.dim() != 2:
        raise ValueError(
            f"backward_truncated_lb: coeffs must be 2-D, got shape {tuple(coeffs.shape)}"
        )

    sample = bounds_dict[sid_int].lb
    device, dtype = sample.device, sample.dtype
    if coeffs.device != device or coeffs.dtype != dtype:
        coeffs = coeffs.to(device=device, dtype=dtype)

    base_batch, width = _flatten_layer_width(bounds_dict[sid_int])
    if coeffs.shape[-1] != width:
        raise ValueError(f"backward_truncated_lb: coeff width {coeffs.shape[-1]} does not match layer {sid_int} width {width}")
    if coeffs.shape[0] % base_batch != 0:
        raise ValueError(f"backward_truncated_lb: coeff batch {coeffs.shape[0]} is not divisible by base batch {base_batch}")

    # SOUNDNESS INVARIANT: bounds_dict may already carry a spec axis
    # (e.g. _compute_bound_joint_kkt passes already-expanded final-margin
    # bounds [B, M, *layer]). Flatten BEFORE replicating: otherwise
    # expand_bounds_dict adds a second spec axis, breaking broadcast vs nu.
    # row_repeat = objective rows per flat batch row (chunk_size, NOT M).
    flat_bounds_dict = {
        lid: solver_dual_module.flatten_bounds_rows(bounds)
        for lid, bounds in bounds_dict.items()
    }
    row_repeat = coeffs.shape[0] // base_batch
    bounds_runtime = {
        lid: solver_dual_module.flatten_bounds_rows(bounds)
        for lid, bounds in expand_bounds_dict(flat_bounds_dict, row_repeat).items()
    }
    alpha_slice = alpha_state.for_start_node(sid_int)
    working_alpha = (
        solver_dual_module.runtime_alpha_dict(solver, alpha_slice, coeffs.shape[0])
        if alpha_slice
        else None
    )
    working_eta = (
        expand_eta_state(eta_state, row_repeat)
        if eta_state is not None and row_repeat > 1
        else eta_state
    )
    registry = cast(dict[str, BackwardHandler], dual_tf_module.BACKWARD_REGISTRY)

    nu_accum: dict[int, torch.Tensor] = {sid_int: coeffs.clone()}
    obj = torch.zeros(coeffs.shape[0], device=coeffs.device, dtype=coeffs.dtype)
    use_eta = working_eta is not None and not working_eta.fast_path_skip()
    eta_vals = working_eta.val if use_eta and working_eta is not None else {}
    eta_signs = working_eta.sign if use_eta and working_eta is not None else {}

    for lid in solver_dual_module.reverse_topological_sort(net):
        layer = net.by_id[lid]
        kind = layer.kind.upper()
        if kind in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value, LayerKind.ASSERT.value):
            continue
        if lid not in nu_accum:
            continue

        nu_here = nu_accum.pop(lid)
        handler = registry.get(kind)
        if handler is None:
            raise ValueError(
                f"backward_truncated_lb: unknown layer kind '{kind}' at layer {lid}"
            )

        preds = list(net.preds.get(lid, []))
        pred_nus, contrib = maybe_apply_alpha_fn(
            solver,
            handler,
            layer,
            nu_here,
            bounds_runtime,
            preds,
            working_alpha,
        )
        if use_eta:
            pred_nus = solver_dual_module.apply_eta_to_pred_nus(
                solver,
                pred_nus,
                preds,
                eta_vals,
                eta_signs,
            )

        if kind in solver_dual_module.AFFINE_CONTRIB_KINDS:
            contrib = -contrib
        obj = obj + contrib

        for pred_id, pred_nu in zip(preds, pred_nus):
            if pred_id in nu_accum:
                nu_accum[pred_id] = nu_accum[pred_id] + pred_nu
            else:
                nu_accum[pred_id] = pred_nu.clone()

    input_lid = solver_dual_module.find_input_layer_id(solver, net)
    if input_lid is None:
        return obj
    nu_final = nu_accum.get(input_lid)
    if nu_final is None:
        return obj

    input_contrib, _ = solver_dual_module.input_contribution_from_nu(
        solver,
        net,
        input_lid,
        nu_final,
        bounds_runtime,
        return_sce=False,
        enable_grad=torch.is_grad_enabled(),
    )
    return obj + input_contrib


def backward_truncated_objective(
    net: Net,
    bounds_dict: dict[int, Bounds],
    sid_int: int,
    coeffs: torch.Tensor,
    alpha_state: AlphaState,
    *,
    eta_state: EtaState | None = None,
) -> torch.Tensor:
    return _backward_truncated_objective(
        net,
        bounds_dict,
        sid_int,
        coeffs,
        alpha_state,
        eta_state=eta_state,
    )


def _truncated_per_sign(
    net: Net,
    bounds_dict: dict[int, Bounds],
    sid_int: int,
    alpha_state: AlphaState,
    *,
    sign: float,
    eta_state: EtaState | None,
    objective_chunk_size: int | None,
) -> torch.Tensor:
    base_batch, width = _flatten_layer_width(bounds_dict[sid_int])
    chunk = width if objective_chunk_size is None else min(int(objective_chunk_size), width)
    if chunk <= 0:
        raise ValueError(f"objective_chunk_size must be positive, got {objective_chunk_size}")
    if chunk >= width:
        coeffs, _, _ = _coordinate_objective_rows(bounds_dict, sid_int, sign=sign)
        obj = _backward_truncated_objective(
            net, bounds_dict, sid_int, coeffs, alpha_state, eta_state=eta_state,
        )
        return obj.reshape(base_batch, width)
    obj_chunks: list[torch.Tensor] = []
    for coord_start in range(0, width, chunk):
        coord_end = min(coord_start + chunk, width)
        coeffs, _, _ = _coordinate_objective_rows(
            bounds_dict, sid_int, sign=sign,
            coord_start=coord_start, coord_end=coord_end,
        )
        obj_chunk = _backward_truncated_objective(
            net, bounds_dict, sid_int, coeffs, alpha_state, eta_state=eta_state,
        )
        obj_chunks.append(obj_chunk.reshape(base_batch, coord_end - coord_start))
    return torch.cat(obj_chunks, dim=1)


def backward_truncated_lb(
    net: Net,
    bounds_dict: dict[int, Bounds],
    sid_int: int,
    alpha_state: AlphaState,
    *,
    eta_state: EtaState | None = None,
    objective_chunk_size: int | None = None,
    log: logging.Logger | None = None,
) -> torch.Tensor:
    del log
    return _truncated_per_sign(
        net, bounds_dict, sid_int, alpha_state,
        sign=1.0, eta_state=eta_state, objective_chunk_size=objective_chunk_size,
    )


def backward_truncated_ub(
    net: Net,
    bounds_dict: dict[int, Bounds],
    sid_int: int,
    alpha_state: AlphaState,
    *,
    eta_state: EtaState | None = None,
    objective_chunk_size: int | None = None,
) -> torch.Tensor:
    return -_truncated_per_sign(
        net, bounds_dict, sid_int, alpha_state,
        sign=-1.0, eta_state=eta_state, objective_chunk_size=objective_chunk_size,
    )


__all__ = [
    "_backward_truncated_objective",
    "backward_truncated_lb",
    "backward_truncated_objective",
    "backward_truncated_ub",
]
