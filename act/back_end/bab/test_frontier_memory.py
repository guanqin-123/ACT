"""Frontier memory planning: pool byte accounting, warm-start shedding, eviction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
import torch

import act.back_end.bab.bab as bab_module
from act.back_end.bab.bab import verify_bab_batched
from act.back_end.bab.branching.bounding import (
    DiverseTopKBounding, MCTSBounding, RandomBounding, TopKBounding,
)
from act.back_end.bab.node import SubproblemBatch
from act.back_end.core import Layer, Net
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_torchlp import TorchLPSolver
from act.config.config import BaBConfig, DualConfig
from act.front_end.specs import OutKind, OutputSpec
from act.util.device_manager import initialize_device
from act.util.stats import VerifyStatus

initialize_device("cpu", "float64")


def _dense_params(weight: torch.Tensor, bias: torch.Tensor) -> Dict[str, Any]:
    return {
        "weight": weight,
        "bias": bias,
        "in_features": weight.shape[1],
        "out_features": weight.shape[0],
        "weight_pos": weight.clamp(min=0),
        "weight_neg": weight.clamp(max=0),
        "input_shape": (weight.shape[1],),
    }


def _make_relu_net(*, eps: float = 1.0, threshold: float = 0.1) -> Net:
    """Two duplicate ReLU pairs compute y=0 exactly but relax independently:
    ``threshold > 0`` is certified only after splitting, ``< 0`` is falsified."""
    w1 = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    w2 = torch.tensor([[1.0, -1.0, 1.0, -1.0]], dtype=torch.float64)
    lb = torch.full((1, 2), -eps, dtype=torch.float64)
    ub = torch.full((1, 2), eps, dtype=torch.float64)
    encoded = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.tensor([1.0], dtype=torch.float64),
        d=torch.tensor([threshold], dtype=torch.float64),
    ).encode_linear(B=1, n_out=1, device=lb.device, dtype=lb.dtype)
    layers = [
        Layer(0, LayerKind.INPUT.value, {"shape": (1, 2), "dtype": str(lb.dtype)}, [], [0, 1]),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX", "lb": lb, "ub": ub}, [0, 1], [0, 1]),
        Layer(2, LayerKind.DENSE.value, _dense_params(w1, torch.zeros(4, dtype=torch.float64)), [0, 1], [2, 3, 4, 5]),
        Layer(3, LayerKind.RELU.value, {}, [2, 3, 4, 5], [6, 7, 8, 9]),
        Layer(4, LayerKind.DENSE.value, _dense_params(w2, torch.zeros(1, dtype=torch.float64)), [6, 7, 8, 9], [10]),
        Layer(5, LayerKind.ASSERT.value, encoded, [10], [10]),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]}
    succs = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []}
    return Net(layers, preds, succs)


def _config(**overrides: Any) -> BaBConfig:
    values: Dict[str, Any] = dict(
        solver_tier="dual_alpha_eta",
        branching_method="babsr",
        bounding="depth_bound_blend",
        root_bounds_reuse="none",
        intermediate_refine="none",
        per_subproblem_refine="none",
        max_nodes=64,
        max_depth=8,
    )
    values.update(overrides)
    return BaBConfig(**values)


def _batch(n: int, *, seed: int = 0) -> SubproblemBatch:
    g = torch.Generator().manual_seed(seed)
    lb = -torch.rand(n, 3, generator=g, dtype=torch.float64)
    return SubproblemBatch(
        lb=lb,
        ub=lb + 1.0,
        depths=torch.arange(n, dtype=torch.long),
        incremental_alpha={3: torch.rand(n, 1, 4, generator=g, dtype=torch.float64)},
        incremental_eta={3: torch.rand(n, 1, 4, generator=g, dtype=torch.float32)},
        split_signs={3: torch.zeros(n, 1, 4, dtype=torch.float64)},
        lower_bound=torch.linspace(-1.0, 0.0, n, dtype=torch.float64),
    )


# -- (a) byte accounting ------------------------------------------------------


def test_nbytes_matches_hand_count() -> None:
    pool = TopKBounding()
    pool.push(_batch(5))
    n = 5
    small = (
        n * 3 * 8 * 2  # lb, ub
        + n * 8  # depths (long)
        + n * 8 * 2  # lower_bound, parent_margins
        + n * 8  # propagation revisions (long)
        + n * 4 * 8  # split_signs
    )
    warm = n * 4 * 8 + n * 4 * 4  # alpha float64 + eta float32
    assert pool.warm_state_nbytes() == warm
    assert pool.nbytes() == small + warm
    assert pool.bytes_per_node == pytest.approx((small + warm) / n)
    assert pool.warm_nodes() == n

    assert pool.offload_warm_state()
    assert not pool.offload_warm_state()
    assert pool.nbytes() == small
    assert pool.offloaded_nbytes() == warm

    dropped = pool.drop_warm_state(2)
    assert dropped == 2 and pool.warm_nodes() == 3
    assert pool.offloaded_nbytes() == 3 * (4 * 8 + 4 * 4)
    assert pool.nbytes() == small + n  # + the [N] bool warm mask


# -- pool mechanics -----------------------------------------------------------


@pytest.mark.parametrize("pool_cls", [TopKBounding, DiverseTopKBounding])
def test_drop_keeps_hot_rows_and_pops_classes_separately(pool_cls: type) -> None:
    pool = pool_cls()
    source = _batch(6)
    pool.push(source)
    scores = pool._priority_scores()
    coldest = set(torch.argsort(scores)[:2].tolist())
    assert pool.drop_warm_state(2) == 2

    seen_cold: List[int] = []
    while not pool.empty:
        popped = pool.pop(6)
        rows = [int(d) for d in popped.depths.tolist()]
        assert popped.warm_start is None
        if popped.incremental_alpha is None:
            assert popped.incremental_eta is None
            seen_cold.extend(rows)
        else:
            # Warm rows keep their exact state.
            assert source.incremental_alpha is not None
            source_alpha = source.incremental_alpha[3]
            assert isinstance(source_alpha, torch.Tensor)
            for row, alpha in zip(rows, popped.incremental_alpha[3]):
                assert torch.equal(alpha, source_alpha[row])
    assert set(seen_cold) == coldest
    assert pool.cold_pops == 2


def test_mixed_view_roundtrip_preserves_cold_rows() -> None:
    pool = TopKBounding()
    source = _batch(6)
    pool.push(source)
    pool.drop_warm_state(3)
    view = pool.view_all()
    assert view.warm_start is not None and int(view.warm_start.sum()) == 3
    assert view.incremental_alpha is not None
    view_alpha = view.incremental_alpha[3]
    assert isinstance(view_alpha, torch.Tensor)
    cold_alpha = view_alpha[~view.warm_start]
    assert torch.count_nonzero(cold_alpha) == 0
    before = pool.nbytes()
    pool.replace_all(view)
    assert pool.warm_nodes() == 3
    assert pool.nbytes() == before


def test_cold_batch_pushed_back_stays_cold() -> None:
    pool = TopKBounding()
    pool.push(_batch(4))
    pool.drop_warm_state(4)
    popped = pool.pop(4)
    assert popped.incremental_alpha is None
    pool.push(popped)  # e.g. an OOM wave returned unchanged
    assert pool.warm_nodes() == 0 and len(pool) == 4


def test_offloaded_state_is_uploaded_on_pop() -> None:
    pool = TopKBounding()
    source = _batch(4)
    pool.push(source)
    pool.offload_warm_state()
    pool.push(_batch(2, seed=1))
    popped = pool.pop(6)
    assert popped.incremental_alpha is not None
    popped_alpha = popped.incremental_alpha[3]
    assert isinstance(popped_alpha, torch.Tensor)
    assert popped_alpha.device == popped.lb.device
    assert popped_alpha.shape[0] == 6


def test_projected_push_matches_actual_with_heterogeneous_dicts() -> None:
    pool = TopKBounding()
    pool.push(_batch(3))
    other = _batch(2, seed=1)
    other.split_signs = {7: torch.zeros(2, 1, 5, dtype=torch.float64)}
    other.incremental_eta = {7: torch.rand(2, 1, 5, dtype=torch.float32)}
    assert other.incremental_alpha is not None
    other.incremental_alpha = {
        3: other.incremental_alpha[3], 9: torch.rand(2, 1, 6, dtype=torch.float64)
    }
    projected = pool.projected_push_nbytes(other)
    pool.push(other)
    assert pool.nbytes() == projected
    pool.drop_warm_state(2)
    third = _batch(4, seed=2)
    third.warm_start = torch.tensor([True, False, True, False])
    projected = pool.projected_push_nbytes(third)
    pool.push(third)
    assert pool.nbytes() == projected
    pool.offload_warm_state()
    projected = pool.projected_push_nbytes(_batch(2, seed=3))
    pool.push(_batch(2, seed=3))
    assert pool.nbytes() == projected


def test_evict_all_empties_the_pool() -> None:
    pool = TopKBounding()
    pool.push(_batch(3))
    assert pool.evict_all() == 3 and pool.empty and pool.nbytes() == 0


def test_empty_warm_leaves_are_counted_on_next_push() -> None:
    pool = TopKBounding()
    pool.push(_batch(3))
    pool.drop_warm_state(3)
    incoming = _batch(2)
    projected = pool.projected_push_nbytes(incoming)
    pool.push(incoming)
    assert pool.nbytes() == projected
    assert pool.warm_state_nbytes() == 2 * (4 * 8 + 4 * 4)


def test_nested_alpha_and_provenance_accounting() -> None:
    batch = _batch(3)
    batch.incremental_alpha = {3: {0: torch.zeros(3, 2, 4, dtype=torch.float64)}}
    batch.node_id = torch.arange(3)
    batch.parent_id = torch.full((3,), -1, dtype=torch.long)
    pool = TopKBounding()
    pool.push(batch)
    # Base fields 80, split signs 32, nested alpha 64, eta 16, ids 16 per row.
    assert pool.nbytes() == 3 * (80 + 32 + 64 + 16 + 16)
    assert pool.bytes_per_node == 208


def test_random_pool_accounting() -> None:
    pool = RandomBounding()
    batch = _batch(3)
    assert pool.projected_push_nbytes(batch) == 3 * (3 * 8 * 2 + 8)
    pool.push(batch)
    assert pool.nbytes() == 168
    assert pool.bytes_per_node == 56
    assert pool.evict_all() == 3 and pool.nbytes() == 0


def test_mcts_offload_preserves_observer_provenance() -> None:
    pool = MCTSBounding()
    batch = _batch(3)
    batch.node_id = torch.arange(3)
    batch.parent_id = torch.full((3,), -1, dtype=torch.long)
    pool.push(batch)
    resident = pool.nbytes()
    assert pool.offload_warm_state()
    assert pool.nbytes() < resident
    popped = pool.pop(3)
    assert popped.node_id is not None
    assert torch.equal(popped.node_id, batch.node_id)
    assert popped.incremental_alpha is not None
    assert pool.parent == {0: -1, 1: -1, 2: -1}


def test_frontier_cli_flags_are_generated() -> None:
    import argparse
    from act.config.backend_cli import _add_dataclass_config_args, _collect_backend_overrides

    parser = argparse.ArgumentParser()
    _add_dataclass_config_args(parser)
    args = parser.parse_args(["--bab-frontier-mem-safety", "0.25", "--no-bab-frontier-offload"])
    overrides = _collect_backend_overrides(args, lambda _flag: False)
    assert overrides["bab_frontier_mem_safety"] == 0.25
    assert overrides["bab_frontier_offload"] is False


# -- (b)-(d) BaB integration --------------------------------------------------

_LANE_BYTES = 4.0 * 256.0  # initial per-lane estimate with _net_bound_elements -> 1


def _run(
    net: Net,
    monkeypatch: pytest.MonkeyPatch,
    *,
    budget: Optional[str] = None,
    k_log: Optional[List[int]] = None,
    max_batch_size: int = 2,
    observed: Optional[Dict[str, float]] = None,
    **overrides: Any,
) -> Any:
    """``budget``: None = no plan (CPU default); 'huge' = never under
    pressure; 'warm' = the frontier fits with a quarter of its warm state next
    to a full wave; 'lanes' = the frontier fits but only one lane does;
    'tiny' = nothing fits."""
    captured: Dict[str, Any] = {}
    original_build = bab_module._build_bounding

    def build(*args: Any, **kwargs: Any) -> Any:
        captured["pool"] = original_build(*args, **kwargs)
        return captured["pool"]

    def budget_fn(config: BaBConfig, device: torch.device) -> Optional[float]:
        value = planned_budget(config, device)
        if observed is not None and value is not None:
            observed["budget"] = value
        return value

    def planned_budget(_config: BaBConfig, _device: torch.device) -> Optional[float]:
        pool = captured.get("pool")
        if budget == "huge":
            return float(2**40)
        if budget == "tiny":
            return 1.0
        if pool is None or pool.empty:
            return float(2**40)
        resident_warm = 0 if pool.warm_offloaded else pool.warm_state_nbytes()
        mask_bytes = 0 if pool._warm is not None else len(pool)
        kept = float(pool.nbytes() - resident_warm + mask_bytes) + 0.25 * resident_warm
        if budget == "warm":
            return kept + max_batch_size * _LANE_BYTES
        return 2.0 * float(pool.nbytes() + mask_bytes) + 1.5 * _LANE_BYTES

    monkeypatch.setattr(bab_module, "_build_bounding", build)
    monkeypatch.setattr(bab_module, "_net_bound_elements", lambda _net: 1)
    if budget is not None:
        monkeypatch.setattr(bab_module, "_frontier_budget_bytes", budget_fn)
        monkeypatch.setattr(bab_module, "_frontier_host_budget_bytes", lambda *_a: float(2**40))
    torch.manual_seed(123)
    result = verify_bab_batched(
        net,
        solver_factory=lambda: TorchLPSolver(),
        config=_config(frontier_mem_safety=1.0, **overrides),
        max_batch_size=max_batch_size,
        time_budget_s=60,
        dual_config=DualConfig(),
        _k_log=k_log,
    )
    monkeypatch.undo()
    return result


def test_no_pressure_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    net = _make_relu_net()
    clean_k: List[int] = []
    planned_k: List[int] = []
    clean = _run(net, monkeypatch, k_log=clean_k)
    planned = _run(net, monkeypatch, budget="huge", k_log=planned_k)
    assert planned.status is clean.status is VerifyStatus.CERTIFIED
    assert planned_k == clean_k
    assert planned.metadata["nodes"] == clean.metadata["nodes"]
    for key in (
        "frontier_offload_events", "frontier_warmstart_dropped_nodes",
        "frontier_wave_k_caps", "frontier_evicted_nodes", "frontier_cold_pops",
    ):
        assert planned.metadata[key] == 0
    assert "frontier_bytes_peak" not in clean.metadata


@pytest.mark.parametrize("threshold", [0.1, -0.1])
def test_shedding_warm_state_keeps_verdict(
    monkeypatch: pytest.MonkeyPatch, threshold: float
) -> None:
    net = _make_relu_net(threshold=threshold)
    clean = _run(net, monkeypatch)
    shed = _run(net, monkeypatch, budget="warm", frontier_offload=False)
    assert clean.status is shed.status
    assert clean.status is (VerifyStatus.CERTIFIED if threshold > 0 else VerifyStatus.FALSIFIED)
    assert shed.metadata["any_dropped_frontier_cap"] is False


def test_cold_popped_nodes_are_solved_and_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run(_make_relu_net(), monkeypatch, budget="warm", frontier_offload=False)
    meta = result.metadata
    assert result.status is VerifyStatus.CERTIFIED
    assert meta["frontier_warmstart_dropped_nodes"] > 0
    assert meta["frontier_cold_pops"] > 0
    assert meta["frontier_offload_events"] == 0
    assert meta["frontier_evicted_nodes"] == 0
    assert meta["frontier_bytes_peak"] > 0 and meta["frontier_nodes_peak"] > 0


def test_sat_after_warm_state_shedding_has_a_real_witness(monkeypatch: pytest.MonkeyPatch) -> None:
    net = _make_relu_net(threshold=-0.1)
    original = bab_module.check_violations_batched
    calls = 0

    def defer_root(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal calls
        calls += 1
        violations = original(*args, **kwargs)
        # Exercise a child wave rather than accepting the root witness.
        return torch.zeros_like(violations) if calls == 1 else violations

    monkeypatch.setattr(bab_module, "check_violations_batched", defer_root)
    result = _run(net, monkeypatch, budget="warm", frontier_offload=False)
    assert result.status is VerifyStatus.FALSIFIED
    assert result.metadata["frontier_warmstart_dropped_nodes"] > 0
    assert result.counterexample is not None
    assert bool(original(net, result.counterexample.unsqueeze(0), net.layers[-1]).all())


def test_offload_first_is_lossless(monkeypatch: pytest.MonkeyPatch) -> None:
    net = _make_relu_net()
    clean = _run(net, monkeypatch)
    offloaded = _run(net, monkeypatch, budget="warm", frontier_offload=True)
    assert offloaded.status is clean.status is VerifyStatus.CERTIFIED
    assert offloaded.metadata["frontier_offload_events"] == 1
    assert offloaded.metadata["frontier_warmstart_dropped_nodes"] == 0
    assert offloaded.metadata["any_dropped_frontier_cap"] is False


def test_fixed_k_is_capped_per_wave_before_any_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    net = _make_relu_net()
    k_log: List[int] = []
    result = _run(net, monkeypatch, budget="lanes", k_log=k_log, max_batch_size=4)
    assert result.status is VerifyStatus.CERTIFIED
    assert result.metadata["frontier_wave_k_caps"] > 0
    assert result.metadata["any_dropped_frontier_cap"] is False
    assert k_log and max(k_log) == 1


@pytest.mark.parametrize("bounding", ["depth_bound_blend", "mcts", "random"])
def test_zero_retain_drops_everything_at_the_root_push(
    monkeypatch: pytest.MonkeyPatch, bounding: str,
) -> None:
    result = _run(
        _make_relu_net(), monkeypatch, budget="tiny", frontier_offload=True,
        bounding=bounding,
    )
    assert result.metadata["any_dropped_frontier_cap"] is True
    assert result.metadata["frontier_evicted_nodes"] >= 1
    assert result.metadata["nodes"] == 0
    assert result.status is VerifyStatus.UNKNOWN



@pytest.mark.parametrize("budget", ["warm", "lanes"])
def test_every_push_is_preflighted_against_the_budget(
    monkeypatch: pytest.MonkeyPatch, budget: str
) -> None:
    observed: Dict[str, float] = {}
    checks: List[bool] = []
    original_push = TopKBounding.push

    def checked_push(self: TopKBounding, batch: SubproblemBatch) -> None:
        if "budget" in observed:
            checks.append(self.nbytes() + self.projected_push_nbytes(batch) <= observed["budget"])
        original_push(self, batch)

    monkeypatch.setattr(TopKBounding, "push", checked_push)
    result = _run(
        _make_relu_net(), monkeypatch, budget=budget, observed=observed, frontier_offload=False
    )
    assert result.status is VerifyStatus.CERTIFIED
    assert checks and all(checks)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.5])
def test_frontier_mem_safety_range_is_validated(value: float) -> None:
    from act.config.config import ConfigError

    with pytest.raises(ConfigError, match="frontier_mem_safety"):
        BaBConfig(frontier_mem_safety=value)


def test_projection_counts_rows_of_emptied_warm_leaves() -> None:
    pool = TopKBounding()
    pool.push(_batch(3))
    pool.drop_warm_state(3)
    assert pool.warm_nodes() == 0
    incoming = _batch(2, seed=4)
    projected = pool.projected_push_nbytes(incoming)
    pool.push(incoming)
    assert pool.nbytes() == projected
    assert pool.warm_nodes() == 2


def test_offload_on_an_empty_pool_routes_incoming_warm_state_to_host() -> None:
    pool = TopKBounding()
    assert pool.offload_warm_state()
    pool.push(_batch(3))
    assert pool.warm_offloaded and pool.offloaded_nbytes() > 0
    assert pool.nbytes() + pool.offloaded_nbytes() == TopKBounding().projected_push_nbytes(_batch(3))
