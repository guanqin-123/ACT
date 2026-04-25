# ===- act/back_end/bab/bab.py - BaB Verification Engine -----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from act.back_end.bab.branching.babsr import BaBSRBranching
from act.back_end.bab.branching.bounding import BFSBounding, BoundingStrategy, DFSBounding, RandomBounding
from act.back_end.bab.branching.branching import BranchingStrategy, RandomBranching
from act.back_end.bab.eta import get_pre_activation_layer_id
from act.back_end.bab.node import BabNode, SubproblemBatch, split_subproblems
from act.back_end.bab.trace import BoundTrace
from act.back_end.config import BaBConfig
from act.back_end.core import Bounds, Net
from act.back_end.dual_tf.dual_tf import DualTF
from act.back_end.dual_tf.tf_forward import compute_forward_bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.solver._initial_alpha_crown import (
    enumerate_intermediate_start_nodes,
    optimize_initial_intermediate_bounds,
)
from act.back_end.solver.alpha_state import AlphaState
from act.back_end.solver.solver_base import SolveStatus, Solver
from act.back_end.solver.solver_dual import DualSolver
from act.back_end.solver.spec_batching import build_spec_batch
from act.back_end.verifier import (
    gather_input_spec_layers,
    get_assert_layer,
    seed_from_input_specs,
    setup_and_solve,
)
from act.front_end.specs import OutKind, OutputSpec
from act.pipeline.verification.act2torch import ACTToTorch
from act.util.device_manager import get_default_device, get_default_dtype
from act.util.stats import VerifyResult, VerifyStatus
from act.back_end.counterexample_io import save_counterexample

log = logging.getLogger(__name__)


def _infer_output_dim(net: Net, assert_layer) -> int:
    preds = list(net.preds.get(assert_layer.id, []))
    if len(preds) != 1:
        raise ValueError(
            f"ASSERT layer {assert_layer.id} must have exactly 1 predecessor, got {len(preds)}"
        )
    out_layer = net.by_id[preds[0]]
    if isinstance(out_layer.params.get("out_features"), int):
        return int(out_layer.params["out_features"])
    weight = out_layer.params.get("weight")
    if isinstance(weight, torch.Tensor):
        return int(weight.shape[0])
    output_shape = out_layer.get_output_shape()
    if output_shape is not None:
        n = int(np.prod(output_shape))
        if n > 0:
            return n
    return len(assert_layer.in_vars)


def _infer_num_classes(net: Net, assert_layer) -> int:
    return _infer_output_dim(net, assert_layer)


def _extract_out_spec(net: Net, assert_layer) -> tuple[OutputSpec, int]:
    params = assert_layer.params
    kind = params.get("kind")
    out_spec = OutputSpec(
        kind=kind,
        c=params.get("c"),
        d=params.get("d"),
        y_true=params.get("y_true"),
        margin=params.get("margin"),
        lb=params.get("lb"),
        ub=params.get("ub"),
    )
    return out_spec, _infer_num_classes(net, assert_layer)


def _ensure_model(net: Net) -> torch.nn.Module:
    model = getattr(net, "_torch_model_cache", None)
    if model is None:
        model = getattr(net, "_bab_torch_model", None)
    current_device = get_default_device()
    current_dtype = get_default_dtype()
    if model is None:
        model = ACTToTorch(net).run().to(device=current_device, dtype=current_dtype)
        model.eval()
        setattr(net, "_torch_model_cache", model)
        setattr(net, "_bab_torch_model", model)
        return model
    try:
        first_param = next(model.parameters())
        if first_param.device != current_device or first_param.dtype != current_dtype:
            model = model.to(device=current_device, dtype=current_dtype)
            setattr(net, "_torch_model_cache", model)
            setattr(net, "_bab_torch_model", model)
    except StopIteration:
        model = model.to(device=current_device, dtype=current_dtype)
        setattr(net, "_torch_model_cache", model)
        setattr(net, "_bab_torch_model", model)
    return model


def _infer_model_input_shape(net: Net) -> Optional[tuple[int, ...]]:
    for target_kind in (LayerKind.INPUT_SPEC.value, LayerKind.INPUT.value):
        for layer in net.layers:
            kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if kind != target_kind:
                continue
            output_shape = layer.get_output_shape()
            if output_shape is not None:
                return tuple(int(dim) for dim in output_shape)
    return None


def _run_model(net: Net, x_batch: torch.Tensor) -> Optional[torch.Tensor]:
    xb = x_batch.to(device=get_default_device(), dtype=get_default_dtype())
    if xb.dim() == 1:
        xb = xb.unsqueeze(0)
    input_shape = _infer_model_input_shape(net)
    if xb.dim() == 2 and input_shape is not None and len(input_shape) > 2:
        xb = xb.view(xb.shape[0], *input_shape[1:])
    model = _ensure_model(net)
    with torch.no_grad():
        output = model(xb)
        if isinstance(output, dict):
            output = output.get("output", None)
    return output if isinstance(output, torch.Tensor) else None


# ---------------------------------------------------------------------------
# CE validation
# ---------------------------------------------------------------------------


def check_violation_at_point_batched(
    net: Net, x_batch: torch.Tensor, assert_layer
) -> torch.Tensor:
    if x_batch.dim() == 1:
        x_batch = x_batch.unsqueeze(0)
    x_batch = x_batch.to(device=get_default_device(), dtype=get_default_dtype())
    output = _run_model(net, x_batch)
    if output is None:
        return torch.zeros(x_batch.shape[0], dtype=torch.bool, device=x_batch.device)
    y = output.reshape(output.shape[0], -1)
    k = assert_layer.params.get("kind")

    if k == OutKind.TOP1_ROBUST:
        t = int(torch.as_tensor(assert_layer.params["y_true"]).reshape(-1)[0].item())
        true_scores = y[:, t]
        mask = torch.ones(y.shape[1], dtype=torch.bool, device=y.device)
        mask[t] = False
        return (y[:, mask] - true_scores.unsqueeze(1)).max(dim=1).values >= 0.0

    if k == OutKind.MARGIN_ROBUST:
        t = int(torch.as_tensor(assert_layer.params["y_true"]).reshape(-1)[0].item())
        margin = float(torch.as_tensor(assert_layer.params["margin"]).reshape(-1)[0].item())
        true_scores = y[:, t]
        mask = torch.ones(y.shape[1], dtype=torch.bool, device=y.device)
        mask[t] = False
        return (y[:, mask] - true_scores.unsqueeze(1)).max(dim=1).values >= margin

    if k == OutKind.LINEAR_LE:
        c = torch.as_tensor(assert_layer.params["c"], dtype=y.dtype, device=y.device).reshape(-1)
        d = float(torch.as_tensor(assert_layer.params["d"]).reshape(-1)[0].item())
        return (y @ c) >= d + 1e-8

    if k == OutKind.RANGE:
        violation = torch.zeros(y.shape[0], dtype=torch.bool, device=y.device)
        lb = assert_layer.params.get("lb")
        ub = assert_layer.params.get("ub")
        if lb is not None:
            lb_t = torch.as_tensor(lb, dtype=y.dtype, device=y.device).reshape(1, -1)
            violation |= (y < lb_t - 1e-8).any(dim=1)
        if ub is not None:
            ub_t = torch.as_tensor(ub, dtype=y.dtype, device=y.device).reshape(1, -1)
            violation |= (y > ub_t + 1e-8).any(dim=1)
        return violation

    if k == OutKind.UNSAFE_LINEAR:
        C = torch.as_tensor(assert_layer.params["c"], dtype=y.dtype, device=y.device)
        if C.dim() == 1:
            C = C.unsqueeze(0)
        d_vec = torch.as_tensor(assert_layer.params["d"], dtype=y.dtype, device=y.device).reshape(1, -1)
        return ((y @ C.T) <= d_vec + 1e-8).all(dim=1)

    raise NotImplementedError(f"ASSERT kind not supported: {k}")


def check_violation_at_point(net: Net, x: torch.Tensor, assert_layer) -> bool:
    return bool(check_violation_at_point_batched(net, x, assert_layer).reshape(-1)[0].item())


# ---------------------------------------------------------------------------
# Strategy factories
# ---------------------------------------------------------------------------


def _build_branching_strategy(method: str) -> BranchingStrategy:
    if method == "random":
        return RandomBranching()
    if method == "babsr":
        return BaBSRBranching()
    raise ValueError(f"Unknown branching method: {method!r}")


def _build_bounding(method: str) -> BoundingStrategy:
    if method == "random":
        return RandomBounding()
    if method == "bfs":
        return BFSBounding()
    if method == "dfs":
        return DFSBounding()
    raise ValueError(f"Unknown bounding method: {method!r}")


def _compute_pre_act_widths(net: Net) -> dict[int, int]:
    widths: dict[int, int] = {}
    splittable = {
        LayerKind.RELU.value,
        LayerKind.LRELU.value,
        LayerKind.SIGMOID.value,
        LayerKind.TANH.value,
    }
    for layer in net.layers:
        kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
        if kind not in splittable:
            continue
        pre_lid = get_pre_activation_layer_id(net, layer.id)
        pre_layer = net.by_id[pre_lid]
        output_shape = pre_layer.get_output_shape()
        widths[pre_lid] = (
            int(np.prod(output_shape))
            if output_shape is not None
            else len(pre_layer.out_vars)
        )
    return widths


def _select_bounds_rows(bounds_dict: dict[int, Bounds], idx: torch.Tensor) -> dict[int, Bounds]:
    return {
        lid: Bounds(v.lb.index_select(0, idx), v.ub.index_select(0, idx))
        for lid, v in bounds_dict.items()
    }


def _align_fixed_bound_batch(current: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
    fixed_t = fixed.to(device=current.device, dtype=current.dtype)
    if fixed_t.shape[0] == current.shape[0]:
        return fixed_t
    if fixed_t.shape[0] == 1:
        return fixed_t.expand(current.shape[0], *fixed_t.shape[1:])
    raise ValueError(
        f"fixed bounds batch mismatch: current={current.shape[0]} fixed={fixed_t.shape[0]}"
    )


def _apply_fixed_bounds(
    bounds_dict: dict[int, Bounds],
    fixed_bounds: dict[int, Bounds],
) -> dict[int, Bounds]:
    merged = dict(bounds_dict)
    for lid, fixed in fixed_bounds.items():
        current = merged.get(lid)
        if current is None:
            continue
        fixed_lb = _align_fixed_bound_batch(current.lb, fixed.lb)
        fixed_ub = _align_fixed_bound_batch(current.ub, fixed.ub)
        merged[lid] = Bounds(
            lb=torch.maximum(current.lb, fixed_lb),
            ub=torch.minimum(current.ub, fixed_ub),
        )
    return merged


def _merge_alpha_states(
    existing: Optional[AlphaState],
    updated: AlphaState,
) -> AlphaState:
    if existing is None or existing.is_empty():
        return updated.clone()
    merged = existing.clone()
    for lid, by_sid in updated._store.items():
        for sid, tensor in by_sid.items():
            merged.set(lid, sid, tensor.detach().clone())
    return merged


def _compute_split_points(
    open_batch: SubproblemBatch,
    bounds_dict_subset: dict[int, Bounds],
    decisions: list[tuple[int, int, str]],
) -> torch.Tensor:
    device = open_batch.lb.device
    dtype = open_batch.lb.dtype
    points = torch.zeros(open_batch.batch_size, device=device, dtype=dtype)
    for row, (layer_id, neuron_idx, kind) in enumerate(decisions):
        if kind == "relu":
            continue
        if layer_id not in bounds_dict_subset:
            raise ValueError(f"Missing bounds for split layer {layer_id}")
        lb = bounds_dict_subset[layer_id].lb[row].reshape(-1)
        ub = bounds_dict_subset[layer_id].ub[row].reshape(-1)
        points[row] = (lb[neuron_idx] + ub[neuron_idx]) / 2.0
    return points


def _extract_ces(
    dual_solver: DualSolver,
    net: Net,
    bounds_dict: dict[int, Bounds],
    batch: SubproblemBatch,
    out_spec: OutputSpec,
    result,
    open_mask: torch.Tensor,
    num_classes: int,
) -> Optional[torch.Tensor]:
    open_idx = open_mask.nonzero(as_tuple=True)[0]
    if open_idx.numel() == 0:
        return None

    open_bounds = _select_bounds_rows(bounds_dict, open_idx)
    spec = build_spec_batch(
        out_spec,
        B=int(open_idx.numel()),
        n_out=_infer_output_dim(net, get_assert_layer(net)),
        num_classes=num_classes,
        device=get_default_device(),
        dtype=get_default_dtype(),
    )
    worst_row = result.slack[open_idx].argmin(dim=-1)
    spec_rows = spec.C.view(open_idx.numel(), spec.M, -1)
    row_idx = torch.arange(open_idx.numel(), device=worst_row.device)
    c_worst = spec_rows[row_idx, worst_row]

    open_eta = batch.eta.select(open_idx) if batch.eta is not None else None
    dual_solver.set_eta(open_eta)
    try:
        bound_out = dual_solver.compute_bound(
            net,
            open_bounds,
            c_worst,
            return_sce=True,
            enable_grad=False,
        )
    finally:
        dual_solver.clear_eta()

    if not isinstance(bound_out, tuple):
        return None
    _margins, sce = bound_out
    return sce


def _select_random_neurons(
    net: Net,
    bounds_dict_subset: dict[int, Bounds],
    batch: SubproblemBatch,
    config: BaBConfig,
) -> list[tuple[int, int, str]]:
    decisions: list[tuple[int, int, str]] = []
    smooth_kinds = {LayerKind.SIGMOID.value, LayerKind.TANH.value}
    relu_kinds = {LayerKind.RELU.value, LayerKind.LRELU.value}
    for row in range(next(iter(bounds_dict_subset.values())).lb.shape[0]):
        chosen: tuple[int, int, str] = (-1, -1, "none")
        for layer in net.layers:
            kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if kind not in relu_kinds and kind not in smooth_kinds:
                continue
            try:
                pre_lid = get_pre_activation_layer_id(net, layer.id)
            except ValueError:
                continue
            if pre_lid not in bounds_dict_subset:
                continue
            lb = bounds_dict_subset[pre_lid].lb[row].reshape(-1)
            ub = bounds_dict_subset[pre_lid].ub[row].reshape(-1)
            if kind in relu_kinds:
                cand = ((lb < 0) & (ub > 0)).nonzero(as_tuple=True)[0]
                if batch.eta is not None and pre_lid in batch.eta.sign:
                    sign_row = batch.eta.sign[pre_lid][row].reshape(-1)
                    cand = cand[sign_row[cand] == 0]
                if cand.numel() > 0:
                    chosen = (pre_lid, int(cand[0].item()), "relu")
                    break
            else:
                cand = ((ub - lb) > config.smooth_split_min_gap).nonzero(as_tuple=True)[0]
                if cand.numel() > 0:
                    chosen = (pre_lid, int(cand[0].item()), "smooth")
                    break
        decisions.append(chosen)
    return decisions


def _log_batched_step(
    *,
    iter_num: int,
    processed: int,
    batch: SubproblemBatch,
    safe_min_slack: torch.Tensor,
    safe_certified: torch.Tensor,
    raw_min_slack: torch.Tensor,
    clamp_mask: torch.Tensor,
    elapsed: float,
) -> None:
    """Emit one line per subproblem in the just-evaluated batch.

    Called from `_verify_bab_batched` when ``BaBConfig.verbose=True``.
    Each line reports:
      - ``depth`` = number of neuron splits on the path to this subproblem,
      - ``lb`` = certified lower bound on min-slack AFTER the parent-margin
                clamp (what BaB actually uses downstream),
      - ``raw`` = the dual solver's unclamped output for this subproblem;
                when ``lb != raw`` the parent-margin clamp kicked in, marked
                with ``CLAMP``. A high CLAMP rate at deep BaB levels signals
                that Adam is producing worse bounds than the parent and the
                clamp is masking the regression,
      - ``CERT``/``open`` status marker for quick scanning.
    """
    n = batch.batch_size
    depths = batch.depths
    n_clamped = int(clamp_mask.sum().item())
    for row in range(n):
        d = int(depths[row].item())
        lb = float(safe_min_slack[row].item())
        raw = float(raw_min_slack[row].item())
        clamped = bool(clamp_mask[row].item())
        mark = "CERT" if bool(safe_certified[row].item()) else "open"
        clamp_tag = " CLAMP" if clamped else ""
        node_id = processed - n + row + 1
        print(
            f"[BaB] iter={iter_num:3d} node={node_id:4d} "
            f"depth={d:2d} lb={lb:+.4f} raw={raw:+.4f}{clamp_tag} "
            f"[{mark}] t={elapsed:.2f}s"
        )
    if n > 0:
        print(
            f"[BaB] iter={iter_num:3d} summary: clamp_hits={n_clamped}/{n} "
            f"({100.0 * n_clamped / n:.0f}%)"
        )


def _verify_bab_legacy(
    net: Net,
    solver: Solver,
    config: BaBConfig,
    *,
    budget: float,
    network_path: Optional[Path] = None,
) -> VerifyResult:
    brancher = _build_branching_strategy(config.branching_method)
    pool = _build_bounding(config.bounding_method)
    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)
    root_bounds = seed_from_input_specs(spec_layers)
    pool.push(SubproblemBatch.from_bounds(root_bounds))

    start = time.time()
    processed = 0
    while not pool.empty and (time.time() - start) < budget and (config.max_nodes is None or processed < config.max_nodes):
        batch = pool.pop(batch_size=1)
        for bounds in batch.to_bounds_list():
            processed += 1
            iter_solver = type(solver)()
            status, ce_input, _ = setup_and_solve(net, bounds, iter_solver, timelimit=None)
            if config.verbose:
                depth = int(batch.depths[0].item())
                print(
                    f"[BaB] node={processed:4d} depth={depth:2d} "
                    f"status={getattr(status, 'name', status)} "
                    f"t={time.time() - start:.2f}s"
                )
            if status == SolveStatus.UNSAT:
                continue
            if status == SolveStatus.SAT and ce_input is not None:
                ce_tensor = torch.from_numpy(ce_input).to(device=root_bounds.lb.device)
                if check_violation_at_point(net, ce_tensor, assert_layer):
                    # CE already validated by check_violation_at_point — just save
                    ce_batch = ce_tensor.unsqueeze(0)
                    model_output = _run_model(net, ce_batch)
                    saved_path: Optional[Path] = None
                    if model_output is not None:
                        saved_path = save_counterexample(
                            net, ce_tensor, model_output[0], network_path,
                            {
                                "verifier_mode": "verify_bab_legacy",
                                "property_kind": str(assert_layer.params.get("kind")),
                                "solver": type(solver).__name__,
                                "nodes": processed,
                            },
                        )
                    return VerifyResult(
                        VerifyStatus.FALSIFIED,
                        counterexample=ce_tensor,
                        metadata={"nodes": processed, "saved_to": str(saved_path) if saved_path is not None else None},
                    )
            node_batch = SubproblemBatch(
                lb=bounds.lb.unsqueeze(0),
                ub=bounds.ub.unsqueeze(0),
                depths=batch.depths[:1],
            )
            if int(node_batch.depths[0].item()) >= config.max_depth:
                continue
            scores = brancher.compute_scores(node_batch, net)
            split_dims = brancher.select(scores)
            left, right = split_subproblems(node_batch, split_dims)
            pool.push(left)
            pool.push(right)
    if (time.time() - start) >= budget or processed >= config.max_nodes:
        return VerifyResult(VerifyStatus.UNKNOWN, metadata={"nodes": processed})
    return VerifyResult(VerifyStatus.CERTIFIED, metadata={"nodes": processed})


def _verify_bab_batched(
    net: Net,
    config: BaBConfig,
    *,
    budget: float,
    dual_solver: Optional[DualSolver] = None,
    trace: Optional[BoundTrace] = None,
    out_bounds_dict: Optional[dict[int, dict[str, torch.Tensor]]] = None,
    network_path: Optional[Path] = None,
) -> VerifyResult:
    dual_solver = dual_solver or DualSolver(DualTF())
    dual_solver.eta_iters = config.eta_iters
    dual_solver.lr_eta = config.lr_eta
    if config.record_bound_trace and trace is None:
        trace = BoundTrace()

    pool = _build_bounding(config.bounding_method)
    brancher = _build_branching_strategy(config.branching_method)
    if isinstance(brancher, BaBSRBranching):
        brancher.SMOOTH_SPLIT_MIN_GAP = config.smooth_split_min_gap

    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)
    root_bounds = seed_from_input_specs(spec_layers)
    out_spec, num_classes = _extract_out_spec(net, assert_layer)
    layer_widths = _compute_pre_act_widths(net)
    root_batch = SubproblemBatch.from_bounds_with_eta(root_bounds, layer_widths, trace=trace)
    precomputed_root_bounds: Optional[dict[int, Bounds]] = None
    fixed_bounds: Optional[dict[int, Bounds]] = None
    if config.alpha_split_objective:
        precomputed_root_bounds = compute_forward_bounds(
            net,
            root_batch.lb,
            root_batch.ub,
            post_activation=False,
            eta_state=root_batch.eta,
        )
        precomputed_root_bounds, root_alpha_state = optimize_initial_intermediate_bounds(
            net,
            precomputed_root_bounds,
            alpha_iters=config.alpha_iters,
            lr_alpha=config.lr_alpha,
        )
        fixed_bounds = {
            lid: precomputed_root_bounds[lid]
            for lid in enumerate_intermediate_start_nodes(net)
            if lid in precomputed_root_bounds
        }
        root_batch.alphas = root_alpha_state
    pool.push(root_batch)

    start = time.time()
    processed = 0
    iteration_counter = 0

    def _finalize(result: VerifyResult) -> VerifyResult:
        if trace is not None:
            result.metadata["bound_trace"] = trace
        return result

    while not pool.empty and (time.time() - start) < budget and (config.max_nodes is None or processed < config.max_nodes):
        batch = pool.pop(batch_size=config.subproblem_batch_size)
        processed += batch.batch_size

        if iteration_counter == 0 and precomputed_root_bounds is not None:
            bounds_dict = dict(precomputed_root_bounds)
        else:
            bounds_dict = compute_forward_bounds(
                net,
                batch.lb,
                batch.ub,
                post_activation=False,
                eta_state=batch.eta,
            )
            if fixed_bounds is not None:
                bounds_dict = _apply_fixed_bounds(bounds_dict, fixed_bounds)

        dual_solver.set_eta(batch.eta)
        dual_solver.set_alphas(batch.alphas)
        try:
            if trace is not None and batch.subproblem_ids is not None:
                dual_solver.set_bound_trace(
                    trace,
                    bab_iter=iteration_counter,
                    sid_map=batch.subproblem_ids,
                )
            result = dual_solver.evaluate_spec(
                net,
                bounds_dict,
                out_spec,
                num_classes=num_classes,
                chunk_size=config.spec_chunk_size,
                enable_grad=True,
            )
        finally:
            dual_solver.clear_bound_trace()
            dual_solver.clear_eta()
            dual_solver.clear_alphas()

        safe_min_slack = result.min_slack
        safe_certified = result.certified
        raw_min_slack = result.min_slack.detach().clone()
        regressed_mask = torch.zeros(
            result.min_slack.shape[0],
            dtype=torch.bool,
            device=result.min_slack.device,
        )
        if batch.parent_margins is not None:
            with torch.no_grad():
                current = result.min_slack
                parent_pm = batch.parent_margins.to(current.device)
                valid_parent = ~torch.isnan(parent_pm)
                regressed_mask = valid_parent & (current < parent_pm)
                if regressed_mask.any():
                    safe_min_slack = torch.where(regressed_mask, parent_pm, current)
                    safe_certified = safe_min_slack >= 0

        if trace is not None and batch.subproblem_ids is not None:
            for row in range(batch.batch_size):
                sid = int(batch.subproblem_ids[row].item())
                min_slack = float(safe_min_slack[row].item())
                trace.record(sid, iteration_counter, min_slack)

        if config.verbose:
            _log_batched_step(
                iter_num=iteration_counter,
                processed=processed,
                batch=batch,
                safe_min_slack=safe_min_slack,
                safe_certified=safe_certified,
                raw_min_slack=raw_min_slack,
                clamp_mask=regressed_mask,
                elapsed=time.time() - start,
            )

        # Warm-start propagation: overwrite the batch's α and η with the
        # Adam-optimised values returned by evaluate_spec. Children spawned
        # below via split_neuron_batched inherit these (α copied wholesale;
        # η val carried forward for pre-existing splits, newly-split neuron
        # reset to 0).
        if result.out_etas is not None:
            batch.eta = result.out_etas
        if result.out_alphas is not None:
            batch.alphas = _merge_alpha_states(batch.alphas, result.out_alphas)

        iteration_counter += 1

        proven = safe_certified
        open_mask = ~proven

        if open_mask.any():
            ce_candidates = _extract_ces(
                dual_solver,
                net,
                bounds_dict,
                batch,
                out_spec,
                result,
                open_mask,
                num_classes,
            )
            if ce_candidates is not None:
                true_violations = check_violation_at_point_batched(net, ce_candidates, assert_layer)
                if true_violations.any():
                    first = int(true_violations.nonzero(as_tuple=True)[0][0].item())
                    ce_tensor = ce_candidates[first].detach().clone()
                    # CE already validated by check_violation_at_point_batched — just save
                    ce_batch = ce_tensor.unsqueeze(0)
                    model_output = _run_model(net, ce_batch)
                    saved_path: Optional[Path] = None
                    if model_output is not None:
                        saved_path = save_counterexample(
                            net, ce_tensor, model_output[0], network_path,
                            {
                                "verifier_mode": "verify_bab_batched",
                                "property_kind": str(assert_layer.params.get("kind")),
                                "solver": "DualSolver",
                                "nodes": processed,
                            },
                        )
                    if out_bounds_dict is not None:
                        out_bounds_dict.clear()
                        out_bounds_dict.update(
                            {
                                lid: {"lb": bounds.lb.detach().cpu(), "ub": bounds.ub.detach().cpu()}
                                for lid, bounds in bounds_dict.items()
                            }
                        )
                    return _finalize(VerifyResult(
                        VerifyStatus.FALSIFIED,
                        counterexample=ce_tensor,
                        metadata={"nodes": processed, "saved_to": str(saved_path) if saved_path is not None else None},
                    ))

        open_idx = open_mask.nonzero(as_tuple=True)[0]
        if open_idx.numel() == 0:
            continue
        open_batch = batch.select(open_idx)
        depth_ok_mask = open_batch.depths < config.max_depth
        if not depth_ok_mask.any():
            continue
        depth_idx = depth_ok_mask.nonzero(as_tuple=True)[0]
        open_batch = open_batch.select(depth_idx)
        bounds_open = _select_bounds_rows(bounds_dict, open_idx)
        bounds_dict_subset = _select_bounds_rows(bounds_open, depth_idx)

        worst_row = result.slack[open_idx][depth_idx].argmin(dim=-1)
        spec = build_spec_batch(
            out_spec,
            B=open_batch.batch_size,
            n_out=_infer_output_dim(net, assert_layer),
            num_classes=num_classes,
            device=get_default_device(),
            dtype=get_default_dtype(),
        )
        spec_c = spec.C.view(open_batch.batch_size, spec.M, -1)[
            torch.arange(open_batch.batch_size, device=worst_row.device), worst_row
        ]

        if isinstance(brancher, BaBSRBranching):
            decisions = brancher.select_neurons(
                net,
                open_batch,
                bounds_dict_subset,
                spec_c,
                num_classes,
                dual_solver=dual_solver,
            )
        else:
            decisions = _select_random_neurons(net, bounds_dict_subset, open_batch, config)
        valid_mask = torch.tensor(
            [decision[0] != -1 for decision in decisions],
            dtype=torch.bool,
            device=open_batch.lb.device,
        )
        if not valid_mask.any():
            continue
        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        open_batch = open_batch.select(valid_idx)
        bounds_dict_subset = _select_bounds_rows(bounds_dict_subset, valid_idx)
        decisions = [decisions[int(i)] for i in valid_idx.tolist()]
        split_points = _compute_split_points(open_batch, bounds_dict_subset, decisions)
        left, right = open_batch.split_neuron_batched(
            decisions,
            per_row_split_points=split_points,
        )
        child_parent_margins = safe_min_slack[open_idx][depth_idx][valid_idx]
        left.parent_margins = child_parent_margins.clone()
        right.parent_margins = child_parent_margins.clone()
        if trace is not None and open_batch.subproblem_ids is not None:
            parent_ids = open_batch.subproblem_ids
            parent_depths = open_batch.depths
            n_open = open_batch.batch_size
            left.subproblem_ids = torch.tensor(
                [
                    trace.new_id(
                        parent=int(parent_ids[i].item()),
                        depth=int(parent_depths[i].item()) + 1,
                    )
                    for i in range(n_open)
                ],
                dtype=torch.long,
                device=parent_ids.device,
            )
            right.subproblem_ids = torch.tensor(
                [
                    trace.new_id(
                        parent=int(parent_ids[i].item()),
                        depth=int(parent_depths[i].item()) + 1,
                    )
                    for i in range(n_open)
                ],
                dtype=torch.long,
                device=parent_ids.device,
            )
        pool.push(left)
        pool.push(right)

    if not pool.empty and ((time.time() - start) >= budget or (config.max_nodes is not None and processed >= config.max_nodes)):
        if out_bounds_dict is not None:
            out_bounds_dict.clear()
            out_bounds_dict.update(
                {
                    lid: {"lb": bounds.lb.detach().cpu(), "ub": bounds.ub.detach().cpu()}
                    for lid, bounds in bounds_dict.items()
                }
            )
        return _finalize(VerifyResult(VerifyStatus.UNKNOWN, metadata={"nodes": processed}))
    if out_bounds_dict is not None:
        out_bounds_dict.clear()
        out_bounds_dict.update(
            {
                lid: {"lb": bounds.lb.detach().cpu(), "ub": bounds.ub.detach().cpu()}
                for lid, bounds in bounds_dict.items()
            }
        )
    return _finalize(VerifyResult(VerifyStatus.CERTIFIED, metadata={"nodes": processed}))


# ---------------------------------------------------------------------------
# BaB engine
# ---------------------------------------------------------------------------


@torch.no_grad()
def verify_bab(
    net: Net,
    solver: Solver,
    config: Optional[BaBConfig] = None,
    *,
    max_depth: Optional[int] = None,
    max_nodes: Optional[int] = None,
    max_subproblems: Optional[int] = None,
    time_budget_s: Optional[float] = None,
    timelimit: Optional[float] = None,
    verbose: bool = False,
    dual_solver: Optional[DualSolver] = None,
    trace: Optional[BoundTrace] = None,
    out_bounds_dict: Optional[dict[int, dict[str, torch.Tensor]]] = None,
    network_path: Optional[Path] = None,
) -> VerifyResult:
    """Branch-and-bound verification with optional CE persistence.

    Returns CERTIFIED / FALSIFIED / UNKNOWN. On FALSIFIED, the validated
    counterexample is also persisted to disk via
    ``act.back_end.counterexample_io.save_counterexample`` and the saved
    directory path is stored in ``result.metadata["saved_to"]``.

    Args:
        network_path: Optional path of the source network JSON. Used as the
            ``<stem>`` in the saved CE directory name. When ``None``, the
            stem defaults to ``"unnamed_net"``.
    """
    if config is None:
        config = BaBConfig(
            max_depth=max_depth if max_depth is not None else 20,
            max_nodes=(max_nodes or max_subproblems or 2000),
            verbose=verbose,
        )
    elif max_depth is not None or max_nodes is not None or max_subproblems is not None or verbose:
        config = BaBConfig(
            **{
                **config.__dict__,
                **({"max_depth": max_depth} if max_depth is not None else {}),
                **({"max_nodes": (max_nodes or max_subproblems)} if (max_nodes is not None or max_subproblems is not None) else {}),
                **({"verbose": verbose} if verbose else {}),
            }
        )

    budget = time_budget_s or timelimit or 300.0
    use_batched = (
        config.branching_method == "babsr"
        or config.bounding_method in ("bfs", "dfs")
    )
    if use_batched:
        return _verify_bab_batched(
            net,
            config,
            budget=budget,
            dual_solver=dual_solver,
            trace=trace,
            out_bounds_dict=out_bounds_dict,
            network_path=network_path,
        )
    return _verify_bab_legacy(net, solver, config, budget=budget, network_path=network_path)


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------


class _StubNet:
    layers = []


def test_imports():
    for sym in (
        verify_bab,
        BaBConfig,
        BabNode,
        SubproblemBatch,
        split_subproblems,
        check_violation_at_point,
        check_violation_at_point_batched,
        BranchingStrategy,
        BoundingStrategy,
        RandomBranching,
        RandomBounding,
    ):
        assert sym is not None


def test_config_yaml_roundtrip():
    c1 = BaBConfig()
    assert c1.max_depth == 20
    c2 = BaBConfig.from_yaml()
    assert c2.branching_method == "random"
    c3 = BaBConfig.from_yaml(max_depth=50, branching_method="kfsb")
    assert c3.max_depth == 50 and c3.branching_method == "kfsb"
    tmp = tempfile.mktemp(suffix=".yaml")
    try:
        c3.to_yaml(tmp)
        c4 = BaBConfig.from_yaml(tmp)
        assert c4.max_depth == 50
        assert c4.branching_method == "kfsb"
    finally:
        os.unlink(tmp)


def test_subproblem_batch():
    lb = torch.tensor([[-1.0, -2.0, -3.0]])
    ub = torch.tensor([[1.0, 2.0, 3.0]])
    batch = SubproblemBatch(lb=lb, ub=ub, depths=torch.tensor([0]))
    assert batch.batch_size == 1
    assert batch.input_dim == 3
    assert batch.total_width().item() == 12.0
    bounds = Bounds(lb.squeeze(0), ub.squeeze(0))
    batch2 = SubproblemBatch.from_bounds(bounds)
    assert torch.equal(batch2.lb, lb)
    back = batch2.to_bounds_list()
    assert len(back) == 1
    assert torch.equal(back[0].lb, bounds.lb)


def test_split_subproblems():
    lb = torch.tensor([[-1.0, -2.0, -3.0]])
    ub = torch.tensor([[1.0, 2.0, 3.0]])
    batch = SubproblemBatch(lb=lb, ub=ub, depths=torch.tensor([0]))
    split_dim = torch.tensor([1])
    left, right = split_subproblems(batch, split_dim)
    mid = (lb[0, 1] + ub[0, 1]) / 2
    assert torch.isclose(left.ub[0, 1], mid)
    assert torch.isclose(right.lb[0, 1], mid)
    assert left.depths[0] == 1
    assert right.depths[0] == 1
    assert torch.equal(left.lb[0, 0], lb[0, 0])
    assert torch.equal(right.ub[0, 2], ub[0, 2])


def test_random_branching():
    lb = torch.tensor([[-1.0, -2.0, -3.0]])
    ub = torch.tensor([[1.0, 2.0, 3.0]])
    batch = SubproblemBatch(lb=lb, ub=ub, depths=torch.tensor([0]))
    brancher = RandomBranching()
    scores = brancher.compute_scores(batch, _StubNet())
    assert scores.shape == (1, 3)
    assert (scores >= 0).all()
    dims = brancher.select(scores)
    assert dims.shape == (1,)
    assert 0 <= dims.item() <= 2


def test_random_branching_with_mask():
    lb = torch.tensor([[-1.0, -2.0, -3.0]])
    ub = torch.tensor([[1.0, 2.0, 3.0]])
    batch = SubproblemBatch(lb=lb, ub=ub, depths=torch.tensor([0]))
    mask = torch.tensor([False, True, False])
    brancher = RandomBranching()
    scores = brancher.compute_scores(batch, _StubNet(), unstable_mask=mask)
    assert scores[0, 0].item() == 0.0
    assert scores[0, 2].item() == 0.0
    assert brancher.select(scores).item() == 1


def test_random_bounding():
    lb = torch.tensor([[-1.0, -2.0], [0.0, 0.0]])
    ub = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    batch = SubproblemBatch(lb=lb, ub=ub, depths=torch.tensor([0, 1]))
    pool = RandomBounding()
    assert pool.empty
    pool.push(batch)
    assert len(pool) == 2
    popped = pool.pop(1)
    assert popped.batch_size == 1
    assert len(pool) == 1
    pool.pop(1)
    assert pool.empty


def test_babnode_compat():
    bounds = Bounds(torch.tensor([-1.0, -2.0]), torch.tensor([1.0, 2.0]))
    node = BabNode(box=bounds, depth=3, score=0.5)
    batch = node.to_batch()
    assert batch.batch_size == 1
    assert batch.depths[0].item() == 3


_TESTS = [
    test_imports,
    test_config_yaml_roundtrip,
    test_subproblem_batch,
    test_split_subproblems,
    test_random_branching,
    test_random_branching_with_mask,
    test_random_bounding,
    test_babnode_compat,
]


def run_all_tests() -> int:
    passed = failed = 0
    for fn in _TESTS:
        try:
            fn()
            passed += 1
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    print("Running BaB module tests\n")
    sys.exit(run_all_tests())
