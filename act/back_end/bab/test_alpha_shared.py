"""Shared-alpha certificates retain every spec row in replay and branching."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from act.back_end.bab.bab import verify_bab_batched
from act.back_end.bab.climb import LiteralCodec, certificate_replay
from act.back_end.bab.node import SubproblemBatch
from act.back_end.bab.test_frontier_memory import _config, _dense_params, _make_relu_net
from act.back_end.dual_tf.tf_forward import compute_forward_bounds
from act.back_end.solver import solver_dual
from act.back_end.solver.solver_dual import DualSolver, _phase_slope_replay_alpha
from act.back_end.solver.solver_torchlp import TorchLPSolver
from act.config.config import DualConfig
from act.front_end.specs import OutKind, OutputSpec
from act.util.stats import VerifyStatus


@pytest.mark.parametrize("out_kind", [OutKind.UNSAFE_LINEAR, OutKind.TOP1_ROBUST])
def test_shared_alpha_replay_matches_expanded_certificate(out_kind: str) -> None:
    net = _make_relu_net()
    lb = torch.full((2, 2), -1.0, dtype=torch.float64)
    ub = -lb
    bounds = compute_forward_bounds(net, lb, ub, post_activation=False)
    alpha = torch.tensor([[0.2, 0.3, 0.7, 0.8], [0.6, 0.4, 0.1, 0.9]], dtype=lb.dtype)
    source = torch.tensor([[[1., -1., 0., 0.]]], dtype=lb.dtype).expand(2, 3, 4).clone()
    current = source.clone()
    current[:, 0, 0] = 0
    current[:, 1, 1] = 0
    current[:, 2, :2] = 0
    expanded = alpha[:, None, :].expand(2, 3, 4)
    torch.testing.assert_close(
        _phase_slope_replay_alpha(alpha, current, source),
        _phase_slope_replay_alpha(expanded, current, source), rtol=0, atol=0,
    )
    batch = SubproblemBatch(
        lb, ub, torch.ones(2, dtype=torch.long),
        incremental_alpha={3: alpha},
        incremental_eta={3: torch.full_like(source, 0.25)},
        split_signs={3: current},
    )
    codec = LiteralCodec.from_net(net)
    literals = torch.tensor([[codec.encode(3, 0, 1), codec.encode(3, 1, -1)]]).expand(2, 2)
    def replay(candidate: SubproblemBatch):
        return certificate_replay(
            batch=candidate, net=net, reference_bounds=bounds,
            c_rows=torch.tensor([[1.], [-1.], [0.5]], dtype=lb.dtype).repeat(2, 1),
            thresholds=torch.zeros(2, 3, dtype=lb.dtype), m_specs=3,
            out_kind=out_kind, literals=literals, codec=codec,
            phase_slope_signs={3: source},
        )

    shared = replay(batch)
    explicit = replay(replace(batch, incremental_alpha={3: expanded}))
    assert shared is not None and explicit is not None
    for actual, expected in zip(shared, explicit):
        assert actual is not None and expected is not None
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_shared_alpha_replay_rejects_wrong_lane_or_width() -> None:
    for shape in [(3, 4), (2, 5)]:
        with pytest.raises(ValueError, match="phase-slope replay shape mismatch"):
            _phase_slope_replay_alpha(torch.zeros(shape), torch.zeros(2, 3, 4), torch.zeros(2, 3, 4))


@pytest.mark.parametrize("cap, mode", [(0.0, "shared"), (1024.0, "per_row")])
def test_alpha_mode_survives_warm_state_shedding(
    monkeypatch: pytest.MonkeyPatch, cap: float, mode: str,
) -> None:
    monkeypatch.setattr(solver_dual, "_PER_CLASS_ALPHA_MAX_MIB", cap)
    monkeypatch.setattr(solver_dual, "_INCREMENTAL_ALPHA_PER_LANE_MAX_MIB", 0.0)
    net = _make_relu_net()
    lb = torch.full((2, 2), -1.0, dtype=torch.float64)
    bounds = compute_forward_bounds(net, lb, -lb, post_activation=False)
    spec = OutputSpec(
        kind=OutKind.UNSAFE_LINEAR,
        c=torch.tensor([[1.], [-1.]], dtype=lb.dtype),
        d=torch.tensor([-0.1, -0.1], dtype=lb.dtype),
    )
    result = DualSolver().solve_spec_batch(
        net, bounds, spec, optimize=True, dual_config=DualConfig(n_iters=1),
        input_shape=(2,),
    )
    assert result.alpha_state is None
    assert result.alpha_mode == mode


def test_singleton_split_rows_replay_all_6237_objectives() -> None:
    """231 shared-slope lanes x 27 rows must not subtract 231 eta rows."""
    net = _make_relu_net()
    lanes, rows = 231, 27
    lb = torch.full((lanes, 2), -1.0, dtype=torch.float64)
    bounds = compute_forward_bounds(net, lb, -lb, post_activation=False)
    signs = torch.zeros(lanes, 1, 4, dtype=lb.dtype)
    signs[:, :, 0] = 1
    eta = torch.full_like(signs, 0.2)
    objective = torch.ones(lanes * rows, 1, dtype=lb.dtype)
    def solve(split: torch.Tensor, multiplier: torch.Tensor):
        return DualSolver().compute_certified_bound(
            net, bounds, objective, split_signs={3: split}, eta={3: multiplier},
            M=rows, alpha={3: torch.full((lanes, 4), 0.4, dtype=lb.dtype)},
            return_nu_per_layer=True,
        )

    shared = solve(signs, eta)
    explicit = solve(signs.expand(-1, rows, -1), eta.expand(-1, rows, -1))
    torch.testing.assert_close(shared.margins, explicit.margins, rtol=0, atol=0)


@pytest.mark.parametrize("out_kind", [OutKind.UNSAFE_LINEAR, OutKind.TOP1_ROBUST])
def test_climb_forced_shared_alpha_matches_default(monkeypatch: pytest.MonkeyPatch, out_kind: str) -> None:
    def run(cap: float):
        monkeypatch.setattr(solver_dual, "_PER_CLASS_ALPHA_MAX_MIB", cap)
        net = _make_relu_net()
        if out_kind == OutKind.UNSAFE_LINEAR:
            spec = OutputSpec(kind=out_kind, c=torch.tensor([[1.], [-1.]], dtype=torch.float64),
                              d=torch.tensor([-0.1, -0.1], dtype=torch.float64))
            n_out = 1
        else:
            weight = torch.tensor([[1., -1., 1., -1.], [0., 0., 0., 0.], [0., 0., 0., 0.]], dtype=torch.float64)
            net.by_id[4].params = _dense_params(weight, torch.tensor([0.1, 0., 0.], dtype=torch.float64))
            net.by_id[4].out_vars = [10, 11, 12]
            net.by_id[5].in_vars = [10, 11, 12]
            net.by_id[5].out_vars = [10, 11, 12]
            spec = OutputSpec(kind=out_kind, y_true=torch.tensor([0]))
            n_out = 3
        net.by_id[5].params = spec.encode_linear(B=1, n_out=n_out, device=torch.device("cpu"), dtype=torch.float64)
        k_log: list[int] = []
        result = verify_bab_batched(
            net, solver_factory=lambda: TorchLPSolver(),
            config=_config(climb_enabled=True, max_nodes=256, max_depth=16),
            max_batch_size=4, time_budget_s=60, dual_config=DualConfig(n_iters=4), _k_log=k_log,
        )
        return result, k_log

    default, default_k = run(1024.0)
    shared, _ = run(0.0)
    assert default.status == shared.status == VerifyStatus.CERTIFIED
    assert shared.metadata["alpha_shared_waves"] > 0
    assert shared.metadata["alpha_per_row_waves"] == 0
    assert shared.metadata["climb_acquisition_replay_calls"] > 0
    assert shared.metadata["climb_cores_inserted"] > 0
    assert default.metadata["alpha_shared_waves"] == 0
    assert default.metadata["alpha_per_row_waves"] > 0
    uncapped, uncapped_k = run(float("inf"))
    assert uncapped.status == default.status
    assert uncapped_k == default_k
    assert uncapped.metadata["nodes"] == default.metadata["nodes"]
