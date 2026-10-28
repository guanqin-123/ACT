# ===- act/back_end/bab/bab.py - BaB Verification Engine -----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   BaB loop on a single-spec instance. Subproblems are explored in K-batched
#   waves via solve_batch with CE validation per SAT lane; optional certificate
#   reuse is delegated to ``act.back_end.bab.climb.ClimbSession``. Dual-tier
#   bound policy lives here, concrete CE checks in ``violation``, branching
#   decisions in ``branching/``, and child construction in ``node.py``.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

import logging
import math
import time
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Union, cast

import torch

from act.config.config import (
    BaBConfig,
    DualConfig,
    NEURON_BRANCHING_METHODS,
    TOP_K_BOUNDINGS,
    TOP_K_INCOMPATIBLE_BOUNDINGS,
    VALID_BOUNDINGS,
    VALID_SOLVER_TIERS,
)
from act.back_end.bab.node import (
    SubproblemBatch,
    slice_branching_state,
    split_neurons,
)
from act.back_end.bab.climb import ClimbSession
from act.back_end.bab.branching.branching import (
    BranchingStrategy,
    SplitDecision,
    _assert_witness_residual_decision_unstable,
    _build_branching_strategy as _build_branching_strategy_impl,
    witness_relu_preactivations,
)
from act.back_end.bab.branching.multi_split import (
    _multi_split_from_decision,
    enumerate_unstable_candidates,
    gain_tested_decision,
    groups_to_tensors,
    lanes_with_relu_candidates,
    presplit_root,
)
from act.back_end.bab.branching.bounding import (
    BoundingStrategy,
    RandomBounding,
    TopKBounding,
    DiverseTopKBounding,
    MCTSBounding,
    OrderFunction,
    DepthLowerBoundOrder,
    GreedyOrder,
    SAOrder,
)

from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf.tf_forward import compute_forward_bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.bab.violation import _check_input_specs_batched, check_violations_batched
from act.back_end.solver.solver_base import Solver, SolveStatus
from act.back_end.solver.solver_dual import (
    DualBatchResult,
    DualSolver,
    _alpha_tree_select_spec_rows,
    expand_bounds_dict,
    unproven_spec_rows,
)
from act.back_end.verifier import (
    gather_input_spec_layers,
    get_assert_layer,
    get_input_ids,
    seed_from_input_specs,
    setup_and_solve_batch,
)
from act.front_end.specs import OutKind, OutputSpec
from act.util.stats import VerifyStatus, VerifyResult

log = logging.getLogger(__name__)

_monotonic = time.monotonic


# ---------------------------------------------------------------------------
# Strategy factories
# ---------------------------------------------------------------------------


def _build_branching_strategy(
    method: str, *, dual_solver: Any = None, neuron_only: bool = False
) -> BranchingStrategy:
    return _build_branching_strategy_impl(
        method, dual_solver=dual_solver, neuron_only=neuron_only
    )


def _build_bounding(
    bounding: str,
    *,
    depth_weight: float = 1.0,
    bound_weight: float = 1.0,
    cooling_rate: float = 0.99,
    mcts_exploration: float = 1.0,
    mcts_lambda: float = 0.5,
    mcts_virtual_loss: float = 1.0,
    top_k: int = 0,
) -> BoundingStrategy:
    if bounding not in VALID_BOUNDINGS:
        raise ValueError(
            f"Unknown bounding={bounding!r}. Valid: {VALID_BOUNDINGS}."
        )
    if top_k > 0 and bounding in TOP_K_INCOMPATIBLE_BOUNDINGS:
        raise ValueError(
            f"top_k={top_k} is not supported by bounding={bounding!r}: it does not "
            f"rank the pool by an order function. Use one of "
            f"{TOP_K_BOUNDINGS}, or leave top_k=0 (unbounded)."
        )
    if bounding == "random":
        return RandomBounding()
    if bounding == "diverse_split_signs":
        return DiverseTopKBounding(
            DepthLowerBoundOrder(depth_weight=depth_weight, bound_weight=bound_weight),
            k=top_k,
        )
    # MCTS pins depth_bound_blend: W2 replaces its scoring with UCB1, so the order is moot.
    order_name = "depth_bound_blend" if bounding == "mcts" else bounding
    order: OrderFunction
    if order_name == "depth_bound_blend":
        order = DepthLowerBoundOrder(depth_weight=depth_weight, bound_weight=bound_weight)
    elif order_name == "greedy":
        order = GreedyOrder()
    elif order_name == "annealed":
        order = SAOrder(cooling_rate=cooling_rate)
    else:
        raise ValueError(f"No order registered for bounding {bounding!r}")
    if bounding == "mcts":
        return MCTSBounding(
            order,
            exploration=mcts_exploration,
            lambda_=mcts_lambda,
            virtual_loss=mcts_virtual_loss,
        )
    return TopKBounding(order, k=top_k)


# ---------------------------------------------------------------------------
# BaB engine
# ---------------------------------------------------------------------------


def _net_bound_elements(net: Net) -> int:
    """Total bound-carrying variables; a proxy for per-lane memory cost."""
    return sum(len(l.out_vars) for l in net.layers if l.out_vars)


def _select_spec_rows(
    state: Optional[Dict[int, torch.Tensor]],
    keep_rows: torch.Tensor,
) -> Optional[Dict[int, torch.Tensor]]:
    """Slice the spec axis of per-layer incremental dual state."""
    if state is None:
        return None
    return {
        lid: tensor.index_select(1, keep_rows.to(tensor.device))
        if tensor.dim() >= 3
        else tensor
        for lid, tensor in state.items()
    }


def _neuron_branching_supported(config: BaBConfig) -> bool:
    return (
        config.branching_method in NEURON_BRANCHING_METHODS
        and config.solver_tier in ("dual_alpha", "dual_alpha_eta")
    )


def _witness_residual_branching_active(config: BaBConfig) -> bool:
    return config.branching_method == "witness_residual"


def _keep_branch_lanes(
    branch_batch: SubproblemBatch,
    bounds_dict: Optional[Dict[int, Bounds]],
    nu_per_layer: Optional[Dict[int, torch.Tensor]],
    witness_preact: Optional[Dict[int, torch.Tensor]],
    keep: torch.Tensor,
) -> tuple[
    SubproblemBatch,
    Optional[Dict[int, Bounds]],
    Optional[Dict[int, torch.Tensor]],
    Optional[Dict[int, torch.Tensor]],
]:
    """Apply one lane selector to branch state, including spec-packed sensitivities."""
    kept_idx = torch.where(keep.to(branch_batch.lb.device))[0]
    bounds_out, nu_out = slice_branching_state(
        bounds_dict, nu_per_layer, kept_idx, branch_batch.batch_size
    )
    witness_out = (
        None
        if witness_preact is None
        else {
            lid: tensor.index_select(0, kept_idx.to(tensor.device))
            for lid, tensor in witness_preact.items()
        }
    )
    return branch_batch.select(kept_idx), bounds_out, nu_out, witness_out


def _assert_multi_split_invariants(
    stats: Dict[str, int], active_k: Optional[int] = None
) -> None:
    requested = stats["multi_split_k_requested"]
    used = stats["multi_split_k_used"]
    waves = stats["multi_split_wave_count"]
    clamped = stats["multi_split_clamped_wave_count"]
    starved = stats["multi_split_lane_starved_count"]
    counters_ok = waves >= 0 and clamped >= 0 and 0 <= starved <= waves
    metadata_ok = (waves == 0 and used == 1) or (
        waves > 0 and 2 <= used <= requested
    )
    active_ok = active_k is None or 1 <= active_k <= requested
    assert requested >= 1 and counters_ok and metadata_ok and active_ok, (
        "MULTI-SPLIT invariant violated: split depth or wave counters are inconsistent "
        f"(k_requested={requested}, k_used={used}, active_k={active_k}, "
        f"waves={waves}, clamped_waves={clamped}, starved_waves={starved})"
    )


def _solve_dual_batch(
    *,
    net: Net,
    assert_layer: Layer,
    batched_bounds: Bounds,
    k_actual: int,
    batch: SubproblemBatch,
    config: BaBConfig,
    dual_config: DualConfig,
    optimize: bool,
    keep_rows: Optional[torch.Tensor] = None,
    root_bounds_dict: Optional[Dict[int, Bounds]] = None,
    round_policy: Optional[Any] = None,
    time_cap: Optional[float] = None,
) -> DualBatchResult:
    """Prepare BaB-specific bounds/state policy, then call ``DualSolver``."""
    solver = DualSolver()
    call_started = _monotonic() if time_cap is not None else 0.0

    def _remaining_cap() -> Optional[float]:
        if time_cap is None:
            return None
        return max(0.0, time_cap - (_monotonic() - call_started))

    # Reused root boxes have no lane-matched symbolic frame.
    forward_frame = None
    input_split_child = (
        batch.depths.numel() > 0
        and bool((batch.depths.max() > 0).item())
        and not batch.split_signs
    )
    use_root_dict = root_bounds_dict is not None and not input_split_child
    if use_root_dict:
        assert root_bounds_dict is not None
        bounds_dict = expand_bounds_dict(root_bounds_dict, k_actual)
        lane_box = Bounds(batched_bounds.lb, batched_bounds.ub)
        for layer in net.layers:
            kind = (
                layer.kind.upper()
                if isinstance(layer.kind, str)
                else layer.kind
            )
            if (
                kind in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value)
                and layer.id in bounds_dict
            ):
                bounds_dict[layer.id] = lane_box
        if config.root_bounds_reuse == "split_refresh" and batch.split_signs:
            refreshed = solver._interval_refresh_bounds(
                net, bounds_dict, batch.split_signs
            )
            if refreshed is not None:
                bounds_dict = refreshed
            refine_mode = config.per_subproblem_refine
            refine_rows_cap = config.per_subproblem_refine_rows_cap
            refine_iters = config.per_subproblem_refine_iters
            if round_policy is not None:
                if round_policy.refine_mode is not None:
                    refine_mode = round_policy.refine_mode
                if round_policy.refine_rows_cap is not None:
                    refine_rows_cap = round_policy.refine_rows_cap
                if round_policy.refine_iters is not None:
                    refine_iters = round_policy.refine_iters
            if refine_mode != "none":
                bounds_dict = solver.refine_intermediate_bounds_batched(
                    net,
                    bounds_dict,
                    split_signs=batch.split_signs,
                    mode=refine_mode,
                    rows_cap=refine_rows_cap,
                    optimize_iters=refine_iters,
                )
    else:
        bounds_dict, forward_frame, _ = compute_forward_bounds(
            net,
            batched_bounds.lb,
            batched_bounds.ub,
            forward_lin_max_perturbed=dual_config.forward_lin_max_perturbed,
        )

    out_kind = assert_layer.params["kind"]
    if not isinstance(out_kind, str):
        raise TypeError(
            f"ASSERT kind must be str, got {type(out_kind).__name__}"
        )
    out_kind = OutKind(out_kind)
    fields = OutputSpec(kind=out_kind)._gather_rows(
        rows=None,
        batch_size=1,
        device=batched_bounds.lb.device,
        dtype=batched_bounds.lb.dtype,
        shared_ndim={},
        source=assert_layer.params,
        source_batch_size=1,
        drop_singleton_batch=True,
    )
    out_spec = OutputSpec(
        kind=out_kind,
        y_true=fields.get("y_true"),
        margin=fields.get("margin"),
        c=fields.get("c"),
        d=fields.get("d"),
        lb=fields.get("lb"),
        ub=fields.get("ub"),
    )
    eta_enabled = config.solver_tier == "dual_alpha_eta"
    is_child_batch = (
        bool((batch.depths > 0).all().item()) if batch.depths.numel() else False
    )
    child_n_iters = int(config.child_n_iters)
    remaining_cap = _remaining_cap()
    if remaining_cap is not None and remaining_cap <= 0.0:
        raise TimeoutError("BaB dual batch time cap exhausted before solving")
    max_time = dual_config.max_time
    if remaining_cap is not None:
        max_time = (
            remaining_cap
            if dual_config.max_time <= 0.0
            else min(dual_config.max_time, remaining_cap)
        )
    capped_dual_config = replace(
        dual_config,
        max_time=max_time,
        n_iters=(
            child_n_iters
            if optimize and is_child_batch and child_n_iters > 0
            else dual_config.n_iters
        ),
    )
    result = solver.solve_spec_batch(
        net,
        bounds_dict,
        out_spec,
        optimize=optimize,
        dual_config=capped_dual_config,
        input_shape=tuple(batched_bounds.lb.shape[1:]),
        keep_rows=keep_rows,
        split_signs=batch.split_signs if eta_enabled else None,
        eta=batch.incremental_eta if eta_enabled else None,
        incremental_alphas=(
            batch.incremental_alpha
            if dual_config.incremental_start_enabled
            else None
        ),
        incremental_etas=(
            batch.incremental_eta
            if eta_enabled and dual_config.incremental_start_enabled
            else None
        ),
        optimize_alpha=not (
            config.eta_only_children and is_child_batch
        ),
        refresh_forward=not use_root_dict,
        return_nu=_neuron_branching_supported(config),
        reuse_bounds_for_branching=use_root_dict,
        forward_frame=forward_frame,
    )
    if optimize:
        batch.incremental_alpha = result.alpha_state
        if eta_enabled:
            batch.incremental_eta = result.eta_state
    return result


def _auto_batch_budget_bytes(safety: float) -> float:
    """Memory the auto sizer may use: min(safety*total, 90% of what this
    process can reclaim), so it shares the GPU with other processes."""
    free, total = torch.cuda.mem_get_info()
    reclaimable = free + torch.cuda.memory_reserved()
    return min(float(total) * safety, float(reclaimable) * 0.9)


def _auto_initial_batch(net: Net, config: BaBConfig) -> int:
    """Conservative first batch from net size; the loop recalibrates it from
    the measured per-lane peak after the first real round."""
    safety = float(config.auto_batch_safety)
    cap = int(config.auto_batch_cap)
    floor = int(config.auto_batch_floor)
    per_lane = 4.0 * max(1, _net_bound_elements(net)) * 256.0
    k = int(_auto_batch_budget_bytes(safety) / per_lane)
    return max(floor, min(cap, k))


def _auto_recalibrate_batch(peak_bytes: float, max_k_seen: int, config: BaBConfig) -> int:
    """Batch for the next round = budget / measured bytes-per-lane.

    ``peak_bytes / max_k_seen`` over-estimates the marginal per-lane cost (it
    folds in the one-time root/presolve peak), so the sizer errs toward fewer
    lanes - safe against OOM while still ramping up on small nets with spare
    memory."""
    safety = float(config.auto_batch_safety)
    cap = int(config.auto_batch_cap)
    floor = int(config.auto_batch_floor)
    bpl = max(peak_bytes / max(1, max_k_seen), 1.0)
    k = int(_auto_batch_budget_bytes(safety) / bpl)
    return max(floor, min(cap, k))


@torch.no_grad()
def verify_bab_batched(
    net: Net,
    solver_factory: Callable[[], Solver],
    config: Optional[BaBConfig] = None,
    *,
    max_batch_size: Union[int, str],
    time_budget_s: Optional[float] = None,
    dual_config: Optional[DualConfig] = None,
    verbose: bool = False,
    trace: Optional[List[Dict[str, Any]]] = None,
) -> VerifyResult:
    """[BATCHED-API] K-batched Branch-and-Bound verification (single instance).

    Per iteration::

        K       = min(len(pool), max_batch_size, max_nodes - processed)
        batch   = pool.pop(K)                       # [K, D_flat]
        sol     = setup_and_solve_batch(net, [K,*input_shape] bounds, solver_factory())
        # decode per-lane:
        #   UNSAT       -> prune (region certified)
        #   SAT + violation (check_violations_batched) -> FALSIFIED (terminate)
        #   SAT spurious / UNKNOWN -> branch (or drop at max_depth)

    Soundness: returns CERTIFIED only when the pool drains via UNSAT pruning
    with every processed sub-box resolved (``all_resolved_unsat`` and
    ``pool.empty``). If the time/node budget exhausts with unproven sub-boxes
    remaining (branched-then-never-revisited, or dropped at ``max_depth``),
    returns UNKNOWN with
    ``metadata['reason'] == 'budget_exhausted_with_unproven_subboxes'``.

    Args:
        net: ACT network with a single-instance INPUT_SPEC (B=1 seed).
        solver_factory: callable returning a fresh ``Solver`` per iteration
            (no state leakage across iterations).
        config: ``BaBConfig``; ``max_depth`` and ``max_nodes`` cap the search
            tree.
        max_batch_size: required cap on K, an int >= 1 or ``"auto"`` (sizes K
            from GPU memory; ``config.auto_batch_cap`` on CPU). There is no
            implicit default: callers pass the central
            ``BackendConfig.bab_max_batch_size`` or an explicit value.
        time_budget_s: wall-clock budget (default 300 s).
        verbose: reserved.
        trace: if a list is supplied, one dict per iteration is appended with
            keys iter, K, pool_before, pool_after, pruned_unsat, branched,
            falsified_lane, max_depth, spec_rows_kept, elapsed_s.
    """
    run_started = _monotonic()
    if config is None:
        config = BaBConfig()
    budget_s = time_budget_s if time_budget_s is not None else 300.0
    if not math.isfinite(budget_s) or budget_s < 0.0:
        raise ValueError(
            f"time_budget_s must be finite and non-negative, got {budget_s}"
        )
    presolve_budget_s = (
        config.presolve_time_fraction * budget_s
        if config.presolve_time_fraction > 0.0
        else budget_s
    )

    def _elapsed() -> float:
        return _monotonic() - run_started

    def _remaining() -> float:
        return max(0.0, budget_s - _elapsed())

    def _root_time_cap() -> float:
        return max(0.0, min(presolve_budget_s - _elapsed(), _remaining()))

    assert_layer = get_assert_layer(net)
    climb = ClimbSession.from_config(
        config, net, cast(str, assert_layer.params["kind"])
    )
    if dual_config is None:
        dual_config = DualConfig()
    auto_batch = isinstance(max_batch_size, str) and max_batch_size == "auto"
    if auto_batch:
        effective_batch = (
            _auto_initial_batch(net, config)
            if torch.cuda.is_available()
            else int(config.auto_batch_cap)
        )
    else:
        effective_batch = int(max_batch_size)
    if effective_batch < 1:
        raise ValueError(f"max_batch_size must be >= 1, got {effective_batch}")
    max_k_seen = 0

    fsb_dual_solver = None
    if config.branching_method == "fsb":
        fsb_dual_solver = DualSolver()
    # "gain" measures child bounds directly; its fallback (when no measured
    # decision is available) is BaBSR — it reuses the dual ν scores and
    # degrades to width-based only when ν/bounds are absent, which is strictly
    # better than a random fallback.
    brancher_method = (
        "babsr" if config.branching_method == "gain" else config.branching_method
    )
    brancher = _build_branching_strategy(
        brancher_method, dual_solver=fsb_dual_solver, neuron_only=climb is not None
    )

    multi_split_stats: Dict[str, int] = {
        "multi_split_k_requested": int(config.multi_split_levels),
        "multi_split_k_used": 1,
        "multi_split_wave_count": 0,
        "multi_split_clamped_wave_count": 0,
        "multi_split_lane_starved_count": 0,
    }

    def _branching_metadata() -> Dict[str, Any]:
        _assert_multi_split_invariants(multi_split_stats)
        meta: Dict[str, Any] = dict(multi_split_stats)
        meta["bounding_top_k_effective"] = int(getattr(pool, "k", 0))
        if _witness_residual_branching_active(config):
            meta["witness_residual_fallback_count"] = int(getattr(brancher, "fallback_count", 0))
            meta["witness_residual_diff_from_babsr_count"] = int(
                getattr(brancher, "different_from_babsr_count", 0)
            )
        if climb is not None:
            meta.update(climb.metadata())
        return meta

    pool = _build_bounding(
        config.bounding,
        depth_weight=config.bounding_depth_weight,
        bound_weight=config.bounding_bound_weight,
        cooling_rate=config.sa_cooling_rate,
        mcts_exploration=config.mcts_exploration,
        mcts_lambda=config.mcts_lambda,
        mcts_virtual_loss=config.mcts_virtual_loss,
        top_k=config.top_k,
    )
    llm_probe: Any = None
    _llm: Any = None
    _wave_index = 0
    if config.llm_probe_enabled:
        from act.pipeline.verification import llm_probe as _llm
        llm_probe = _llm.build_llm_probe(config)

    provenance = bool(config.provenance_enabled) or isinstance(
        pool, MCTSBounding
    )
    if provenance and not isinstance(pool, (TopKBounding, MCTSBounding)):
        raise ValueError(
            "provenance_enabled requires a bounding that preserves node_id/parent_id, "
            "not 'random'"
        )
    if isinstance(pool, MCTSBounding) and config.presplit_levels > 0:
        raise ValueError(
            "bounding='mcts' is incompatible with presplit_levels>0: the "
            "pre-split root batch carries no node_id/parent_id provenance"
        )
    node_counter = 0
    fanout = max(2, int(config.input_split_fanout))
    frontier_cap = int(config.frontier_cap)

    spec_layers = gather_input_spec_layers(net)
    root_bounds = seed_from_input_specs(spec_layers)
    if root_bounds.lb.shape[0] != 1:
        raise ValueError("verify_bab_batched requires one root sample; slice the net before BaB")
    input_shape: tuple[int, ...] = tuple(root_bounds.lb.shape[1:])

    per_lane_dim = int(root_bounds.lb[0].numel())
    n_input_vars = len(get_input_ids(net))
    if n_input_vars != per_lane_dim:
        raise ValueError(
            f"verify_bab_batched: INPUT layer declares {n_input_vars} variables "
            f"but the per-lane input dim is {per_lane_dim}. The net was likely "
            f"converted with a batched input shape (B baked into INPUT vars); "
            f"synthesize per-instance (B=1) models before BaB."
        )

    root_batch = SubproblemBatch.from_bounds(root_bounds)
    if provenance:
        n = root_batch.batch_size
        root_batch.node_id = torch.arange(
            node_counter,
            node_counter + n,
            device=root_batch.lb.device,
            dtype=torch.long,
        )
        root_batch.parent_id = torch.full(
            (n,), -1, device=root_batch.lb.device, dtype=torch.long
        )
        node_counter += n

    # Root spec-pruning presolve (ALL-rows kinds, dual tiers): rows certified
    # on the root box stay certified on every sub-box, so descendants only
    # carry the unproven rows.
    spec_keep_rows: Optional[torch.Tensor] = None
    presolve_tier = config.solver_tier
    root_fwd: Optional[Dict[int, Bounds]] = None
    refine_mode = config.intermediate_refine
    reuse_mode = config.root_bounds_reuse
    if presolve_tier in ("dual", "dual_alpha", "dual_alpha_eta") and (
        reuse_mode != "none" or refine_mode != "none"
    ):
        root_fwd, _, _ = compute_forward_bounds(
            net,
            root_bounds.lb,
            root_bounds.ub,
            forward_lin_max_perturbed=dual_config.forward_lin_max_perturbed,
        )
        if refine_mode != "none":
            root_fwd = DualSolver().refine_intermediate_bounds(
                net,
                root_fwd,
                mode=refine_mode,
                blowup_ratio=config.intermediate_refine_ratio,
            )
    # Per-node bound reuse is governed solely by root_bounds_reuse: root_fwd may
    # exist just for the root presolve/refine above, and passing it to descendant
    # solves would freeze every child's intermediate bounds at root tightness
    # (fatal for input-split BaB, where the whole gain comes from recomputing
    # intermediates on the smaller box).
    node_root_fwd: Optional[Dict[int, Bounds]] = (
        root_fwd if reuse_mode != "none" else None
    )
    presolve: Optional[DualBatchResult] = None
    if (
        presolve_tier in ("dual", "dual_alpha", "dual_alpha_eta")
        and assert_layer.params.get("kind") != OutKind.UNSAFE_LINEAR
        and _root_time_cap() > 0.0
    ):
        try:
            presolve = _solve_dual_batch(
                net=net,
                assert_layer=assert_layer,
                batched_bounds=Bounds(root_bounds.lb, root_bounds.ub),
                k_actual=root_batch.batch_size,
                batch=root_batch,
                config=config,
                dual_config=dual_config,
                optimize=presolve_tier in ("dual_alpha", "dual_alpha_eta"),
                root_bounds_dict=root_fwd,
                time_cap=_root_time_cap(),
            )
        except TimeoutError:
            presolve = None
    if presolve is not None:
        presolve_slack = presolve.spec_row_slack()
        if presolve_slack is not None:
            unproven = unproven_spec_rows(presolve_slack).any(dim=0)
            total_rows = int(unproven.numel())
            if not bool(unproven.any().item()):
                return VerifyResult(
                    VerifyStatus.CERTIFIED,
                    metadata={
                        "nodes": root_batch.batch_size,
                        "iterations": 0,
                        "pool_remaining": 0,
                        "spec_rows_total": total_rows,
                        "spec_rows_kept": 0,
                        "resolved_by": "root_presolve",
                        **(climb.metadata() if climb is not None else {}),
                    },
                )
            keep = torch.where(unproven)[0]
            if int(keep.numel()) < total_rows:
                spec_keep_rows = keep
                root_batch.incremental_alpha = _alpha_tree_select_spec_rows(
                    root_batch.incremental_alpha, keep,
                )
                root_batch.incremental_eta = _select_spec_rows(
                    root_batch.incremental_eta, keep,
                )
                root_batch.split_signs = _select_spec_rows(
                    root_batch.split_signs, keep,
                )
        presplit_k = int(config.presplit_levels)
        presolve_bounds = presolve.intermediate_bounds()
        presolve_sensitivities = presolve.neuron_sensitivities()
        if (
            presplit_k > 0
            and root_batch.batch_size == 1
            and presolve_bounds is not None
            and presolve_sensitivities is not None
        ):
            presplit = presplit_root(
                root_batch,
                net,
                presolve_bounds,
                presolve_sensitivities,
                presplit_k,
            )
            if presplit is not None:
                root_batch = presplit
                node_counter += root_batch.batch_size

    # A completed root solve can replace a second solve after its optimization
    # window expires, but only while its spec axis and lane domain still match.
    if spec_keep_rows is not None or bool((root_batch.depths > 0).any().item()):
        presolve = None

    pool.push(root_batch)
    any_dropped_frontier_cap = False
    if frontier_cap > 0 and len(pool) > frontier_cap:
        if pool.evict_to(frontier_cap) > 0:
            any_dropped_frontier_cap = True

    start = run_started
    processed = 0
    iteration = 0

    def _trace_row(
        k: int,
        pool_before: int,
        pruned_unsat: int,
        branched: int,
        falsified_lane: Optional[int],
        max_depth: int,
    ) -> Dict[str, Any]:
        return {
            "iter": iteration,
            "K": k,
            "pool_before": pool_before,
            "pool_after": len(pool),
            "pruned_unsat": pruned_unsat,
            "branched": branched,
            "falsified_lane": falsified_lane,
            "max_depth": max_depth,
            "spec_rows_kept": (
                int(spec_keep_rows.numel()) if spec_keep_rows is not None else None
            ),
            "elapsed_s": _monotonic() - start,
        }

    any_dropped_max_depth = False
    any_retired_no_relu_candidate = False
    _last_input_widths: Optional[list[float]] = None

    def _retire_lanes(
        keep: torch.Tensor,
        branch_batch: SubproblemBatch,
        bounds_dict: Optional[Dict[int, Bounds]],
        nu_per_layer: Optional[Dict[int, torch.Tensor]],
        witness_preact: Optional[Dict[int, torch.Tensor]],
    ) -> tuple[
        Optional[SubproblemBatch],
        Optional[Dict[int, Bounds]],
        Optional[Dict[int, torch.Tensor]],
        Optional[Dict[int, torch.Tensor]],
    ]:
        """Retire unresolved CLIMB lanes without interpreting them as proofs."""
        nonlocal any_retired_no_relu_candidate
        assert climb is not None
        retired = int((~keep).sum().item())
        if retired == 0:
            return branch_batch, bounds_dict, nu_per_layer, witness_preact
        climb.retire_lanes_without_relu_candidate(retired)
        any_retired_no_relu_candidate = True
        if not bool(keep.any().item()):
            return None, None, None, None
        return _keep_branch_lanes(
            branch_batch, bounds_dict, nu_per_layer, witness_preact, keep
        )

    while not pool.empty:
        elapsed = _monotonic() - start
        if elapsed >= budget_s or processed >= config.max_nodes:
            break

        if climb is not None:
            climb.before_pop(cast(TopKBounding, pool))
        if pool.empty:
            break

        remaining_nodes = config.max_nodes - processed
        k_requested = min(len(pool), effective_batch, remaining_nodes)
        if k_requested <= 0:
            break

        _wave_t0 = time.time()
        _pool_before = len(pool)
        _wave_policy = None
        _wave_split_used = None
        if llm_probe is not None and _llm is not None:
            _wave_policy = llm_probe.begin_wave(_llm.build_frontier_stats(
                wave_index=_wave_index,
                pool_size=len(pool),
                effective_batch=effective_batch,
                remaining_nodes=remaining_nodes,
                elapsed_s=elapsed,
                remaining_s=max(0.0, budget_s - elapsed),
                input_widths=_last_input_widths,
            ))
            if _wave_policy.k_requested is not None:
                k_requested = max(1, min(_wave_policy.k_requested, len(pool), effective_batch, remaining_nodes))

        batch = pool.pop(batch_size=k_requested)
        k_actual = batch.batch_size
        iteration += 1
        batch_max_depth = int(batch.depths.max().item()) if trace is not None else 0
        if climb is not None:
            climb.after_pop(len(pool) + k_actual)

        if input_shape:
            k_lb = batch.lb.reshape(k_actual, *input_shape)
            k_ub = batch.ub.reshape(k_actual, *input_shape)
        else:
            k_lb = batch.lb
            k_ub = batch.ub
        batched_bounds = Bounds(k_lb, k_ub)

        solver_tier = config.solver_tier
        neuron_branching_supported = _neuron_branching_supported(config)
        bounds_dict_for_branching: Optional[Dict[int, Bounds]] = None
        nu_per_layer_for_branching: Optional[Dict[int, torch.Tensor]] = None
        witness_input_for_branching: Optional[torch.Tensor] = None
        dual_solve_result: Optional[DualBatchResult] = None
        if solver_tier == "lp":
            solver = solver_factory()
            solution = setup_and_solve_batch(
                net, batched_bounds, solver, timelimit=None,
            )
        elif solver_tier in ("dual", "dual_alpha", "dual_alpha_eta"):
            has_root = bool((batch.depths == 0).any().item())
            optimize = solver_tier != "dual"
            if has_root and presolve is not None and _root_time_cap() <= 0.0:
                dual_solve_result = presolve
            else:
                # The root window caps optimization, not search. If setup uses
                # up that window, a fixed bound still permits root branching.
                modes = (True, False) if has_root and optimize else (optimize,)
                for optimize in modes:
                    dual_time_cap = (
                        _root_time_cap() if has_root and optimize else _remaining()
                    )
                    if dual_time_cap <= 0.0:
                        continue
                    try:
                        dual_solve_result = _solve_dual_batch(
                            net=net,
                            assert_layer=assert_layer,
                            batched_bounds=batched_bounds,
                            k_actual=k_actual,
                            batch=batch,
                            config=config,
                            dual_config=dual_config,
                            optimize=optimize,
                            keep_rows=spec_keep_rows,
                            root_bounds_dict=node_root_fwd,
                            round_policy=_wave_policy,
                            time_cap=dual_time_cap,
                        )
                    except TimeoutError:
                        continue
                    break
            presolve = None
            if dual_solve_result is None:
                pool.push(batch)
                break
            solution = dual_solve_result.lane_solution()
            bounds_dict_for_branching = dual_solve_result.intermediate_bounds()
            nu_per_layer_for_branching = dual_solve_result.neuron_sensitivities()
            witness_input_for_branching = dual_solve_result.lane_witness_inputs()
        else:
            raise ValueError(
                f"Unknown solver_tier={solver_tier!r}. Valid: {VALID_SOLVER_TIERS}."
            )

        if climb is not None:
            climb.after_main_bound(k_actual)

        n_unsat = sum(1 for s in solution.statuses if s == SolveStatus.UNSAT)
        branched_lanes = 0
        node_lower_bound = (-solution.max_viol).detach()
        if batch.lower_bound is not None:
            # Bound inheritance: a child region is a subset of its parent, so
            # the parent's certified lower bound stays valid; clamping removes
            # per-subproblem optimization regressions (observed: re-optimized
            # children reporting bounds below their parent's).
            node_lower_bound = torch.maximum(
                node_lower_bound, batch.lower_bound.to(node_lower_bound.device)
            )

        sat_lane_idx = [
            i for i, s in enumerate(solution.statuses) if s == SolveStatus.SAT
        ]
        if sat_lane_idx:
            input_ids = get_input_ids(net)
            input_index = torch.tensor(
                input_ids, device=solution.x.device, dtype=torch.long,
            )
            sat_idx_t = torch.tensor(
                sat_lane_idx, device=solution.x.device, dtype=torch.long,
            )
            x_full = solution.x.index_select(0, sat_idx_t)
            x_input_flat = x_full.index_select(1, input_index)
            x_input_shaped = (
                x_input_flat.reshape(len(sat_lane_idx), *input_shape)
                if input_shape
                else x_input_flat
            )
            in_region = _check_input_specs_batched(x_input_shaped, spec_layers)
            violations = check_violations_batched(net, x_input_shaped, assert_layer) & in_region
            for j, lane in enumerate(sat_lane_idx):
                if bool(violations[j].item()):
                    if trace is not None:
                        trace.append(_trace_row(
                            k_actual, _pool_before, n_unsat, 0, lane, batch_max_depth,
                        ))
                    return VerifyResult(
                        VerifyStatus.FALSIFIED,
                        counterexample=x_input_shaped[j].detach().cpu().clone(),
                        metadata={
                            "nodes": processed + k_actual,
                            "iterations": iteration,
                            "lane": lane,
                            "K": k_actual,
                            "nodes_minted": node_counter,
                            "any_dropped_frontier_cap": any_dropped_frontier_cap,
                            **_branching_metadata(),
                        },
                    )

        # Must run post-validation: a SAT lane whose counterexample fails the
        # concrete forward check is spurious, stays unresolved, and so must earn
        # the ordinary lb reward instead of a terminal one.
        if isinstance(pool, MCTSBounding):
            assert batch.node_id is not None
            n_unstable = 1
            if bounds_dict_for_branching is not None:
                # Entries are batched over the K lanes; divide to recover the
                # per-lane unstable count the depth reward normalises by.
                n_unstable = max(1, sum(
                    int(((b.lb < 0) & (b.ub > 0)).sum().item())
                    for b in bounds_dict_for_branching.values()
                ) // k_actual)
            pool.observe(
                batch.node_id,
                node_lower_bound,
                solution.statuses,
                batch.depths,
                n_unstable,
            )

        unresolved_idx = torch.tensor(
            [i for i, status in enumerate(solution.statuses) if status != SolveStatus.UNSAT],
            device=batch.lb.device,
            dtype=torch.long,
        )
        if climb is not None:
            assert dual_solve_result is not None
            climb.learn(batch, solution.statuses, dual_solve_result, k_actual)

        if int(unresolved_idx.numel()) > 0:
            unresolved = batch.select(unresolved_idx)
            unresolved.lower_bound = node_lower_bound.index_select(
                0, unresolved_idx.to(node_lower_bound.device)
            )
            if climb is not None:
                unresolved, kept_rows = climb.before_split(cast(TopKBounding, pool), unresolved)
                unresolved_idx = unresolved_idx.index_select(0, kept_rows)
            branch_mask = unresolved.depths < int(config.max_depth)
            if bool((~branch_mask).any().item()):
                any_dropped_max_depth = True
            branch_idx = torch.where(branch_mask)[0]
            if int(branch_idx.numel()) > 0:
                branch_batch = unresolved.select(branch_idx)
                children: Optional[SubproblemBatch] = None
                parent_index = torch.empty(0, dtype=torch.long, device=branch_batch.lb.device)
                if neuron_branching_supported:
                    full_branch_idx = unresolved_idx.index_select(
                        0, branch_idx.to(unresolved_idx.device)
                    )
                    bd_branch, nu_branch = slice_branching_state(
                        bounds_dict_for_branching,
                        nu_per_layer_for_branching,
                        full_branch_idx,
                        k_actual,
                    )
                    witness_preact_branch: Optional[Dict[int, torch.Tensor]] = None
                    if _witness_residual_branching_active(config) and witness_input_for_branching is not None:
                        witness_branch = witness_input_for_branching.index_select(
                            0,
                            full_branch_idx.to(witness_input_for_branching.device),
                        )
                        witness_preact_branch = witness_relu_preactivations(
                            net,
                            witness_branch,
                            input_shape,
                            dual_config,
                        )
                    multi = None
                    multi_k = int(config.multi_split_levels)
                    if llm_probe is not None and _llm is not None and llm_probe.wants_neuron:
                        # neuron_topk>0 => never bail on candidate count: enumerate the
                        # full set (limit=None) and let advise_neuron_groups truncate to
                        # the top-K by score, so the LLM always decides (with a bounded
                        # view) instead of falling back to FSB. neuron_topk==0 keeps the
                        # legacy "bail to FSB when > max_candidates_total" behavior.
                        _neuron_topk = int(config.llm_probe_neuron_topk)
                        _cand_dicts = enumerate_unstable_candidates(
                            branch_batch, bd_branch, nu_branch,
                            limit=None if _neuron_topk > 0
                            else config.llm_probe_max_candidates_total,
                        )
                        if _cand_dicts:
                            _ngroups = llm_probe.advise_neuron_groups(_llm.build_frontier_stats(
                                wave_index=_wave_index,
                                pool_size=len(pool),
                                effective_batch=effective_batch,
                                remaining_nodes=remaining_nodes,
                                elapsed_s=elapsed,
                                branch_batch_size=branch_batch.batch_size,
                                candidates=[_llm.CandidateSummary(**_d) for _d in _cand_dicts],
                            ))
                            if _ngroups is not None:
                                _tl, _tn, _keff = groups_to_tensors(_ngroups, branch_batch)
                                if _tl is not None and _tn is not None:
                                    multi = split_neurons(branch_batch, net, _tl, _tn, _keff)
                                    _wave_split_used = _keff
                    # Joint splitting is orthogonal to how a split is SCORED, so
                    # it is no longer keyed on branching_method == "gain": the k
                    # neurons come from the BaBSR heuristic inside
                    # _multi_split_from_decision either way. The enclosing
                    # neuron_branching_supported guard already restricts this to a
                    # neuron-branching method on a dual_alpha* tier.
                    if multi is None and multi_k > 1:
                        # Adaptive split depth: fan out so children roughly
                        # fill one bounding batch; n_branch lanes x 2^k <=
                        # max_batch_size keeps the frontier from flooding
                        # the pool. Note this needs
                        # effective_batch >= 4 * branch_batch.batch_size before
                        # k_adaptive can exceed 1 at all, so joint splitting
                        # stays dormant on waves where most lanes are
                        # unresolved. The clamp is a memory guard and must stay
                        # (forcing exact k risks OOM); the user's lever is
                        # --bab-max-batch-size.
                        k_adaptive = max(
                            1,
                            min(
                                multi_k,
                                int(math.log2(max(2, effective_batch // max(1, branch_batch.batch_size)))),
                            ),
                        )
                        if _wave_policy is not None and _wave_policy.split_k is not None and _llm is not None:
                            k_adaptive = _llm.clip_split_k(
                                _wave_policy.split_k,
                                branch_batch_size=branch_batch.batch_size,
                                effective_batch=effective_batch,
                                multi_split_levels=multi_k,
                            )
                        if k_adaptive < multi_k:
                            if multi_split_stats["multi_split_clamped_wave_count"] == 0:
                                log.warning(
                                    "joint multi-split clamped: requested k=%d but "
                                    "effective_batch=%d / branch lanes=%d allows only "
                                    "k=%d (needs effective_batch >= %d for k=%d); "
                                    "raise --bab-max-batch-size to lift this",
                                    multi_k, effective_batch, branch_batch.batch_size,
                                    k_adaptive, branch_batch.batch_size * (2 ** multi_k),
                                    multi_k,
                                )
                            multi_split_stats["multi_split_clamped_wave_count"] += 1
                        _wave_split_used = k_adaptive
                        if k_adaptive > 1:
                            multi = _multi_split_from_decision(
                                branch_batch, net, bd_branch, nu_branch, k_adaptive,
                            )
                            if multi is not None:
                                multi_split_stats["multi_split_wave_count"] += 1
                                multi_split_stats["multi_split_k_used"] = k_adaptive
                                if multi[0].batch_size != branch_batch.batch_size * (2 ** k_adaptive):
                                    multi_split_stats["multi_split_lane_starved_count"] += 1
                    if multi is not None:
                        _assert_multi_split_invariants(
                            multi_split_stats, active_k=_wave_split_used
                        )
                        children, parent_index = multi
                    else:
                        decision = None
                        if climb is not None and config.branching_method == "gain":
                            branch_batch, bd_branch, nu_branch, witness_preact_branch = _retire_lanes(
                                lanes_with_relu_candidates(branch_batch, bd_branch, nu_branch),
                                branch_batch, bd_branch, nu_branch, witness_preact_branch,
                            )
                        if branch_batch is not None and config.branching_method == "gain":
                            decision = gain_tested_decision(
                                branch_batch,
                                net,
                                assert_layer,
                                config,
                                dual_config,
                                spec_keep_rows,
                                node_root_fwd,
                                bd_branch,
                                nu_branch,
                                input_shape,
                                solve_dual=_solve_dual_batch,
                            )
                        if branch_batch is not None and decision is None:
                            extra_branch_kwargs = (
                                {"witness_preact_per_layer": witness_preact_branch}
                                if _witness_residual_branching_active(config)
                                else {}
                            )
                            scores = cast(Any, brancher).compute_scores(
                                branch_batch,
                                net,
                                bounds_dict=bd_branch,
                                nu_per_layer=nu_branch,
                                **extra_branch_kwargs,
                            )
                            decision = cast(SplitDecision, cast(Any, brancher).select(scores))
                            if climb is not None and decision.lane_mask is not None:
                                branch_batch, bd_branch, nu_branch, witness_preact_branch = _retire_lanes(
                                    decision.lane_mask,
                                    branch_batch, bd_branch, nu_branch, witness_preact_branch,
                                )
                            if branch_batch is not None and _witness_residual_branching_active(config):
                                _assert_witness_residual_decision_unstable(
                                    decision, bd_branch
                                )
                        if branch_batch is not None and decision is not None:
                            if decision.kind == "input_axis":
                                if climb is not None:
                                    climb.reject_input_axis_split()
                                decision.fanout = fanout
                            children, parent_index = decision.apply(branch_batch, net)
                else:
                    scores = brancher.compute_scores(branch_batch, net)
                    legacy_decision = cast(Any, brancher).select(scores)
                    if isinstance(legacy_decision, SplitDecision):
                        decision = legacy_decision
                        split_dims = decision.input_axes(branch_batch)
                    else:
                        split_dims = torch.as_tensor(
                            legacy_decision,
                            device=branch_batch.lb.device,
                            dtype=torch.long,
                        ).reshape(-1)
                        decision = SplitDecision(
                            kind="input_axis",
                            input_axis=split_dims,
                            fanout=fanout,
                        )
                    widths = branch_batch.widths()
                    _last_input_widths = (
                        widths.mean(dim=0).tolist() if widths.shape[1] <= 32 else None
                    )
                    if _wave_policy is not None and _wave_policy.input_split_dim is not None:
                        # LLM-advised input dimension (already range-clipped). Lanes where
                        # the advised dim has zero width keep the brancher's choice:
                        # splitting a zero-width dim yields identical children (livelock).
                        advised = torch.full_like(split_dims, int(_wave_policy.input_split_dim))
                        has_width = widths.gather(1, advised.unsqueeze(1)).squeeze(1) > 0
                        split_dims = torch.where(has_width, advised, split_dims)
                    decision.input_axis = split_dims
                    if _wave_policy is not None and _wave_policy.input_split_fanout is not None:
                        decision.fanout = int(_wave_policy.input_split_fanout)
                    children, parent_index = decision.apply(branch_batch, net)

                if children is not None:
                    assert branch_batch is not None
                    branched_lanes = branch_batch.batch_size
                    if provenance:
                        pid = branch_batch.node_id
                        assert pid is not None
                        children.parent_id = pid.index_select(0, parent_index.to(pid.device))
                        nc = children.batch_size
                        children.node_id = torch.arange(
                            node_counter,
                            node_counter + nc,
                            device=children.lb.device,
                            dtype=torch.long,
                        )
                        node_counter += nc
                    pool.push(children)
                    if climb is not None:
                        climb.after_children(children.batch_size, len(pool))
                    if frontier_cap > 0 and len(pool) > frontier_cap:
                        if pool.evict_to(frontier_cap) > 0:
                            any_dropped_frontier_cap = True

        processed += k_actual

        if trace is not None:
            trace.append(_trace_row(
                k_actual, _pool_before, n_unsat, branched_lanes, None, batch_max_depth,
            ))

        if isinstance(pool, MCTSBounding):
            log.info(
                "mcts: nodes=%d n_tot=%d frontier N[parent] histogram=%s",
                processed,
                pool.n_tot,
                pool.frontier_parent_visit_histogram(),
            )

        if auto_batch and torch.cuda.is_available():
            max_k_seen = max(max_k_seen, k_actual)
            effective_batch = _auto_recalibrate_batch(
                torch.cuda.max_memory_allocated(), max_k_seen, config,
            )

        if llm_probe is not None and _llm is not None:
            llm_probe.end_wave(_llm.WaveOutcome(
                wave_index=_wave_index,
                pool_before=_pool_before,
                pool_after=len(pool),
                k_requested_used=k_actual,
                split_k_used=_wave_split_used if _wave_split_used is not None else 1,
                refine_iters_used=(_wave_policy.refine_iters if (_wave_policy is not None and _wave_policy.refine_iters is not None) else 0),
                certified_count=0,
                falsified_found=False,
                branched_count=0,
                best_lb_before=None,
                best_lb_after=None,
                wave_time_s=time.time() - _wave_t0,
                fallback_used=False,
            ))
            _wave_index += 1

    pool_remaining = len(pool)
    elapsed_total = _monotonic() - start
    exhausted_time = elapsed_total >= budget_s
    exhausted_nodes = processed >= config.max_nodes

    spec_rows_kept = (
        int(spec_keep_rows.numel()) if spec_keep_rows is not None else None
    )

    if (
        not any_dropped_max_depth
        and not any_dropped_frontier_cap
        and not any_retired_no_relu_candidate
        and pool_remaining == 0
    ):
        return VerifyResult(
            VerifyStatus.CERTIFIED,
            metadata={
                "nodes": processed,
                "spec_rows_kept": spec_rows_kept,
                "pool_remaining": 0,
                "iterations": iteration,
                "exhausted_budget_time": exhausted_time,
                "exhausted_budget_nodes": exhausted_nodes,
                "nodes_minted": node_counter,
                "any_dropped_frontier_cap": any_dropped_frontier_cap,
                **_branching_metadata(),
            },
        )

    return VerifyResult(
        VerifyStatus.UNKNOWN,
        metadata={
            "nodes": processed,
            "spec_rows_kept": spec_rows_kept,
            "pool_remaining": pool_remaining,
            "iterations": iteration,
            "exhausted_budget_time": exhausted_time,
            "exhausted_budget_nodes": exhausted_nodes,
            "nodes_minted": node_counter,
            "any_dropped_frontier_cap": any_dropped_frontier_cap,
            "reason": (
                "climb_retired_without_relu_candidate"
                if any_retired_no_relu_candidate
                else "budget_exhausted_with_unproven_subboxes"
            ),
            **_branching_metadata(),
        },
    )


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
    dual_config: Optional[DualConfig] = None,
    trace: Optional[List[Dict[str, Any]]] = None,
) -> VerifyResult:
    """Single-solver Branch-and-Bound entry: one subproblem per iteration.

    Thin wrapper over ``verify_bab_batched`` with K=1. Constructs a solver factory
    from the supplied solver instance's type so each BaB iteration gets a fresh
    instance. Prefer ``verify_bab_batched`` directly for batched (K>1) solving.
    """
    if config is None:
        config = BaBConfig(
            max_depth=max_depth if max_depth is not None else 20,
            max_nodes=(max_nodes or max_subproblems or 2000),
            verbose=verbose,
        )
    budget = (
        time_budget_s if time_budget_s is not None
        else (timelimit if timelimit is not None else 300.0)
    )
    solver_tier = config.solver_tier
    if solver_tier not in VALID_SOLVER_TIERS:
        raise ValueError(
            f"Unknown solver_tier={solver_tier!r}. Valid: {VALID_SOLVER_TIERS}."
        )
    solver_type = type(solver)
    return verify_bab_batched(
        net=net,
        solver_factory=lambda: solver_type(),
        config=config,
        max_batch_size=1,
        time_budget_s=budget,
        verbose=verbose,
        dual_config=dual_config,
        trace=trace,
    )


if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    from act.back_end.net_factory import LAYER_TESTING_SPECS, NetFactory
    from act.config.config import BackendConfig
    from act.util.device_manager import initialize_device

    initialize_device("cpu", "float64")
    torch.set_num_threads(1)
    _ = torch.manual_seed(42)
    with TemporaryDirectory(prefix="act-bab-shared-") as directory:
        factory = NetFactory(BackendConfig.from_yaml().generation.net_factory,
                             output_dir=directory, write_manifest=False)
        # Shared [B,n] alpha must broadcast child eta/signs over both spec rows.
        spec = LAYER_TESTING_SPECS["layer_testing_climb_reuse"]()
        spec["layers"][-1]["params"].update(c=[[1.], [-1.]], d=[.1, .1])
        net = factory.create_network("shared_multirow", spec)
        trace: list[dict[str, int | float | None]] = []
        verdict = verify_bab_batched(
            net, DualSolver,
            BaBConfig(
                solver_tier="dual_alpha_eta", branching_method="babsr",
                root_bounds_reuse="plain", bounding="depth_bound_blend",
                max_depth=4, max_nodes=16,
            ),
            max_batch_size=4, time_budget_s=60,
            dual_config=DualConfig(n_iters=1, per_class_alpha=False), trace=trace,
        )
        assert verdict.status == VerifyStatus.UNKNOWN
        assert len(trace) > 1
        assert any(wave["K"] > 1 for wave in trace if isinstance(wave["K"], int))

        # Control the clock at solver boundaries, not by sleeping. A root-only
        # deadline must leave the remaining search budget available to children.
        original_solve = _solve_dual_batch
        original_forward = compute_forward_bounds
        for case in (
            "completed", "timeout", "skipped", "pruned", "presplit",
            "main_timeout", "global", "default",
        ):
            clock = [0.0]
            calls: list[tuple[tuple[int, ...], bool, Optional[list[int]]]] = []
            controlled_trace: list[Dict[str, Any]] = []
            controlled_net = net
            if case == "pruned":
                pruned_spec = LAYER_TESTING_SPECS["layer_testing_climb_reuse"]()
                pruned_spec["layers"][-1]["params"].update(
                    c=[[1.], [-1.]], d=[10., .1]
                )
                controlled_net = factory.create_network("pruned_rows", pruned_spec)

            def controlled_solve(**kwargs: Any) -> DualBatchResult:
                """Expire only the requested window at a deterministic boundary."""
                rows = kwargs.get("keep_rows")
                calls.append((
                    tuple(kwargs["batch"].depths.tolist()), kwargs["optimize"],
                    None if rows is None else rows.tolist(),
                ))
                if (case == "timeout" and len(calls) == 1) or (
                    case == "main_timeout" and len(calls) == 2
                ):
                    clock[0] = 2.0
                    raise TimeoutError("controlled root work exhausted")
                result = original_solve(**kwargs)
                if case != "main_timeout":
                    clock[0] = 10.0 if case == "global" else 2.0
                return result

            def controlled_forward(*args: Any, **kwargs: Any) -> Any:
                """Cover root setup consuming its window before presolve starts."""
                result = original_forward(*args, **kwargs)
                if case == "skipped":
                    clock[0] = 2.0
                return result

            with (
                patch(f"{__name__}._monotonic", lambda: clock[0]),
                patch("act.back_end.solver.solver_dual._monotonic", lambda: clock[0]),
                patch(f"{__name__}._solve_dual_batch", controlled_solve),
                patch(f"{__name__}.compute_forward_bounds", controlled_forward),
            ):
                controlled = verify_bab_batched(
                    controlled_net, DualSolver,
                    BaBConfig(
                        solver_tier="dual_alpha_eta", branching_method="babsr",
                        root_bounds_reuse="plain", bounding="depth_bound_blend",
                        max_depth=1, max_nodes=3,
                        presolve_time_fraction=0.0 if case == "default" else .1,
                        presplit_levels=int(case == "presplit"),
                    ),
                    max_batch_size=2, time_budget_s=10,
                    dual_config=DualConfig(n_iters=1), trace=controlled_trace,
                )
            assert controlled.status == VerifyStatus.UNKNOWN, (case, controlled)
            if case == "global":
                assert not controlled_trace and len(calls) == 1
                assert controlled.metadata["exhausted_budget_time"]
                continue
            assert controlled_trace, (case, controlled.metadata, calls)
            assert any(all(depth > 0 for depth in depths) for depths, _, _ in calls), (
                case, calls
            )
            assert not controlled.metadata["exhausted_budget_time"]
            root_calls = [(optimize, rows) for depths, optimize, rows in calls if depths == (0,)]
            if case in ("completed", "presplit"):
                assert root_calls == [(True, None)], (case, calls)
            elif case == "timeout":
                assert root_calls == [(True, None), (False, None)], calls
            elif case == "skipped":
                assert root_calls == [(False, None)], calls
            elif case == "pruned":
                assert root_calls == [(True, None), (False, [1])], calls
                assert controlled.metadata["spec_rows_kept"] == 1
            elif case == "main_timeout":
                assert root_calls == [(True, None), (True, None), (False, None)], calls
            else:
                assert root_calls == [(True, None), (True, None)], calls

        # Root/mixed batches retain the base optimizer policy. Only optimized
        # all-child batches use the override, carried solely by DualConfig.
        bounds = seed_from_input_specs(gather_input_spec_layers(net))
        root = SubproblemBatch.from_bounds(bounds)
        child = SubproblemBatch.from_bounds(bounds, depth=1)
        children = child.concat(child)
        mixed = root.concat(child)
        base_dual = DualConfig(n_iters=3, max_time=7.0)
        for batch, optimize, child_iters, time_cap, expected_iters in (
            (root, True, 2, 5.0, 3),
            (children, True, 2, 5.0, 2),
            (mixed, True, 2, 5.0, 3),
            (children, False, 2, 5.0, 3),
            (children, True, 0, None, 3),
            (children, True, 2, None, 2),
        ):
            with (
                patch(f"{__name__}._monotonic", return_value=0.0),
                patch("act.back_end.solver.solver_dual._monotonic", return_value=0.0),
                patch.object(
                    DualSolver, "solve_spec_batch", autospec=True,
                    side_effect=DualSolver.solve_spec_batch,
                ) as solve_spy,
            ):
                _solve_dual_batch(
                    net=net, assert_layer=get_assert_layer(net),
                    batched_bounds=Bounds(batch.lb, batch.ub),
                    k_actual=batch.batch_size, batch=batch,
                    config=BaBConfig(
                        solver_tier="dual_alpha_eta", child_n_iters=child_iters,
                        eta_only_children=True,
                    ),
                    dual_config=base_dual, optimize=optimize, time_cap=time_cap,
                )
            assert solve_spy.call_count == 1
            kwargs = solve_spy.call_args.kwargs
            assert "n_iters" not in kwargs
            assert kwargs["dual_config"].n_iters == expected_iters
            assert kwargs["dual_config"].max_time == (7.0 if time_cap is None else 5.0)
            assert kwargs["optimize_alpha"] == bool((batch.depths == 0).any().item())
            assert base_dual.n_iters == 3 and base_dual.max_time == 7.0
