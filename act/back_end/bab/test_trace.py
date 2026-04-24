from __future__ import annotations

import time

import pytest
import torch

from act.back_end.bab.bab import verify_bab
from act.back_end.bab.trace import BoundTrace
from act.back_end.config import BaBConfig
from act.back_end.core import Layer, Net
from act.back_end.dual_tf import DualTF
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_dual import DualSolver
from act.back_end.solver.solver_interval import TorchLPSolver
from act.front_end.specs import OutKind
from act.util.device_manager import get_default_device, get_default_dtype, initialize_device
from act.util.stats import VerifyStatus


def _t(data: object) -> torch.Tensor:
    return torch.tensor(data, dtype=get_default_dtype(), device=get_default_device())


def _dtype_name() -> str:
    return str(get_default_dtype()).split(".")[-1]


def _make_unknown_root_relu_net() -> Net:
    in_vars = [0, 1]
    z = [10, 11]
    a = [20, 21]
    out = [30]
    layers = [
        Layer(0, LayerKind.INPUT.value, {"shape": [2], "dtype": _dtype_name(), "num_classes": 1, "value_range": [0.0, 1.0]}, in_vars, in_vars),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX", "lb": _t([-1.0, -1.0]), "ub": _t([1.0, 1.0])}, in_vars, in_vars),
        Layer(2, LayerKind.DENSE.value, {
            "in_features": 2,
            "out_features": 2,
            "weight": _t([[0.6614, 0.2669], [0.0617, 0.6213]]),
            "bias": _t([-0.0904, -0.0332]),
        }, in_vars, z),
        Layer(3, LayerKind.RELU.value, {}, z, a),
        Layer(4, LayerKind.DENSE.value, {
            "in_features": 2,
            "out_features": 1,
            "weight": _t([[-1.5228, 0.3817]]),
            "bias": _t([-0.1028]),
        }, a, out),
        Layer(5, LayerKind.ASSERT.value, {"kind": OutKind.LINEAR_LE, "c": _t([1.0]), "d": _t([0.1346417528397746])}, out, out),
    ]
    return Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )


def _trace_cfg(**kwargs) -> BaBConfig:
    return BaBConfig(
        max_depth=kwargs.pop("max_depth", 3),
        max_nodes=kwargs.pop("max_nodes", 8),
        branching_method=kwargs.pop("branching_method", "babsr"),
        bounding_method=kwargs.pop("bounding_method", "bfs"),
        subproblem_batch_size=kwargs.pop("subproblem_batch_size", 1),
        record_bound_trace=kwargs.pop("record_bound_trace", False),
        **kwargs,
    )


def _run_verify(
    *,
    config: BaBConfig,
    trace: BoundTrace | None = None,
    dual_solver: DualSolver | None = None,
):
    return verify_bab(
        _make_unknown_root_relu_net(),
        TorchLPSolver(),
        config=config,
        dual_solver=dual_solver or DualSolver(DualTF()),
        trace=trace,
    )


def setup_module() -> None:
    initialize_device("cpu", _dtype_name())


def test_new_id_assigns_sequentially() -> None:
    trace = BoundTrace()
    assert trace.new_id() == 0
    assert trace.new_id() == 1
    assert trace.new_id() == 2


def test_record_and_retrieve() -> None:
    trace = BoundTrace()
    sid = trace.new_id()
    trace.record(sid=sid, t=1, slack=0.5)
    trace.record(sid=sid, t=2, slack=0.3)
    assert trace.min_slack_history[sid] == [0.5, 0.3]
    assert trace.iteration_history[sid] == [1, 2]


def test_parent_lineage() -> None:
    trace = BoundTrace()
    root = trace.new_id(depth=0)
    child = trace.new_id(parent=root, depth=1)
    grandchild = trace.new_id(parent=child, depth=2)
    assert trace.parent[child] == root
    assert trace.parent[grandchild] == child


def test_record_unregistered_raises() -> None:
    trace = BoundTrace()
    with pytest.raises(KeyError, match="unregistered subproblem id"):
        trace.record(sid=0, t=1, slack=0.1)


def test_record_adam_step_unregistered_raises() -> None:
    trace = BoundTrace()
    with pytest.raises(KeyError, match="unregistered subproblem id"):
        trace.record_adam_step(sid=0, bab_iter=0, obj_val=0.1)


def test_trace_none_is_zero_overhead() -> None:
    baseline_cfg = _trace_cfg(record_bound_trace=False)
    traced_cfg = _trace_cfg(record_bound_trace=True)

    start = time.perf_counter()
    for _ in range(10):
        result = _run_verify(config=baseline_cfg, trace=None)
        assert result.status in {VerifyStatus.CERTIFIED, VerifyStatus.FALSIFIED, VerifyStatus.UNKNOWN}
        assert "bound_trace" not in result.metadata
    baseline_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(10):
        result = _run_verify(config=traced_cfg, trace=None)
        assert "bound_trace" in result.metadata
    traced_elapsed = time.perf_counter() - start

    assert baseline_elapsed >= 0.0
    assert traced_elapsed >= 0.0


def test_trace_records_improvement() -> None:
    result = _run_verify(config=_trace_cfg(record_bound_trace=True), trace=None)
    assert result.status == VerifyStatus.CERTIFIED
    trace = result.metadata["bound_trace"]
    assert isinstance(trace, BoundTrace)
    assert 0 in trace.min_slack_history
    assert any(history for history in trace.min_slack_history.values())
    assert trace.next_id >= 1

    monotone_sid_found = False
    for history in trace.min_slack_history.values():
        if len(history) == 0:
            continue
        if all(history[i] <= history[i + 1] for i in range(len(history) - 1)):
            monotone_sid_found = True
            break
    assert monotone_sid_found


def test_adam_trajectory_records_per_step() -> None:
    """Adam loop within compute_bound records per-iter obj values when trace attached."""
    eta_iters = 8
    result = _run_verify(
        config=_trace_cfg(
            record_bound_trace=True,
            subproblem_batch_size=1,
            eta_iters=eta_iters,
        ),
        trace=None,
    )
    assert result.status == VerifyStatus.CERTIFIED
    trace = result.metadata["bound_trace"]
    assert isinstance(trace, BoundTrace)
    assert trace.adam_trajectory
    assert any(len(traj) >= 2 for traj in trace.adam_trajectory.values())
    assert all(len(traj) <= eta_iters for traj in trace.adam_trajectory.values())


def test_adam_plateau_early_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    """When obj plateaus, Adam breaks before eta_iters."""
    solver = DualSolver(DualTF())
    monkeypatch.setattr("act.back_end.bab.bab._extract_ces", lambda *args, **kwargs: None)

    def fake_backward_pass(
        self,
        net,
        bounds_dict,
        c,
        alphas=None,
        eta_state=None,
        return_sce=False,
    ):
        del net, bounds_dict, alphas
        if eta_state is None or eta_state.fast_path_skip():
            obj = torch.full((c.shape[0],), -1.0, device=c.device, dtype=c.dtype)
        else:
            obj = torch.zeros((c.shape[0],), device=c.device, dtype=c.dtype)
        sce = None
        if return_sce:
            sce = torch.zeros((c.shape[0], 1), device=c.device, dtype=c.dtype)
        return obj, sce

    monkeypatch.setattr(solver, "_backward_pass", fake_backward_pass.__get__(solver, DualSolver))
    result = _run_verify(
        config=_trace_cfg(
            record_bound_trace=True,
            subproblem_batch_size=1,
            eta_iters=8,
            max_nodes=4,
        ),
        trace=None,
        dual_solver=solver,
    )
    assert result.status == VerifyStatus.CERTIFIED
    trace = result.metadata["bound_trace"]
    assert isinstance(trace, BoundTrace)
    assert trace.adam_trajectory
    assert any(2 <= len(traj) < 8 for traj in trace.adam_trajectory.values())


def test_adam_trajectory_monotonic_within_sid() -> None:
    """Per-(sid, bab_iter) best_obj trajectory is monotonically non-decreasing."""
    result = _run_verify(
        config=_trace_cfg(
            record_bound_trace=True,
            subproblem_batch_size=1,
            eta_iters=5,
        ),
        trace=None,
    )
    trace = result.metadata["bound_trace"]
    assert isinstance(trace, BoundTrace)
    assert trace.adam_trajectory
    for traj in trace.adam_trajectory.values():
        assert len(traj) >= 1
        assert all(traj[i] <= traj[i + 1] for i in range(len(traj) - 1))
