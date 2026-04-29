from __future__ import annotations

import pytest
import torch

from act.back_end.bab.eta import EtaState
from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf import DualTF, compute_forward_bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.solver import DualSolver, expand_bounds_dict
from act.front_end.specs import OutKind, OutputSpec
from act.util.device_manager import get_default_device, get_default_dtype, initialize_device


@pytest.fixture(scope="module", autouse=True)
def _cpu_float32() -> None:
    initialize_device("cpu", "float32")


def _t(data: object) -> torch.Tensor:
    return torch.tensor(data, dtype=get_default_dtype(), device=get_default_device())


def _make_test_net(output_dim: int = 2) -> tuple[Net, int]:
    in_vars = [0, 1]
    hidden_vars = [10, 11, 12]
    relu_vars = [20, 21, 22]
    out_vars = list(range(30, 30 + output_dim))
    layers = [
        Layer(0, LayerKind.INPUT.value, {"shape": (2,), "dtype": "float32", "num_classes": 1, "value_range": (0.0, 1.0)}, in_vars, in_vars),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, in_vars, in_vars),
        Layer(2, LayerKind.DENSE.value, {"in_features": 2, "out_features": 3, "weight": _t([[1.0, 0.0], [0.5, -0.5], [0.2, 0.3]]), "bias": _t([0.0, 0.0, 0.1])}, in_vars, hidden_vars),
        Layer(3, LayerKind.RELU.value, {}, hidden_vars, relu_vars),
        Layer(4, LayerKind.DENSE.value, {"in_features": 3, "out_features": output_dim, "weight": _t([[-1.0, 0.0, 0.2], [0.0, 1.0, -0.1]])[:output_dim], "bias": torch.zeros(output_dim, dtype=get_default_dtype(), device=get_default_device())}, relu_vars, out_vars),
        Layer(5, LayerKind.ASSERT.value, {"kind": "RANGE"}, out_vars, out_vars),
    ]
    net = Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )
    return net, 2


def _compute_bounds(net: Net, batch_size: int = 2) -> dict[int, Bounds]:
    lb = _t([[-1.0, -1.0], [-0.5, -0.25]])[:batch_size]
    ub = _t([[1.0, 1.0], [0.75, 0.5]])[:batch_size]
    if lb.shape[0] != batch_size:
        lb = lb[:1].expand(batch_size, -1).clone()
        ub = ub[:1].expand(batch_size, -1).clone()
    return compute_forward_bounds(net, lb, ub)


def _make_eta(batch_size: int, pre_lid: int) -> EtaState:
    val = {pre_lid: torch.full((batch_size, 3), 0.1, dtype=get_default_dtype(), device=get_default_device())}
    sign = {pre_lid: torch.zeros((batch_size, 3), dtype=get_default_dtype(), device=get_default_device())}
    sign[pre_lid][:, 0] = 1.0
    point = {pre_lid: torch.zeros((batch_size, 3), dtype=get_default_dtype(), device=get_default_device())}
    return EtaState(val=val, sign=sign, point=point)


def test_expand_bounds_dict_is_zero_copy() -> None:
    net, _ = _make_test_net()
    bounds = _compute_bounds(net, batch_size=2)
    expanded = expand_bounds_dict(bounds, 4)
    sample = expanded[2]
    assert sample.lb.shape[:2] == (2, 4)
    assert sample.lb.stride(1) == 0
    assert sample.ub.stride(1) == 0


def test_expand_bounds_dict_materialize_fallback() -> None:
    net, _ = _make_test_net()
    bounds = _compute_bounds(net, batch_size=2)
    expanded = expand_bounds_dict(bounds, 3, materialize=True)
    sample = expanded[2]
    assert sample.lb.shape == (6, 3)
    assert sample.lb.is_contiguous()
    assert sample.ub.is_contiguous()


def test_backward_pass_rank3_shape() -> None:
    net, _ = _make_test_net()
    bounds = _compute_bounds(net, batch_size=2)
    expanded = expand_bounds_dict(bounds, 3)
    solver = DualSolver(DualTF())
    c = _t([[1.0, -1.0]]).expand(6, -1).contiguous()
    obj, _ = solver._backward_pass(net, expanded, c)
    assert obj.shape == (6,)


def test_input_contribution_from_nu_rank3() -> None:
    net, _ = _make_test_net()
    bounds = _compute_bounds(net, batch_size=2)
    expanded = expand_bounds_dict(bounds, 3)
    solver = DualSolver(DualTF())
    nu = torch.ones(6, 2, dtype=get_default_dtype(), device=get_default_device())
    contrib, sce = solver._input_contribution_from_nu(net, 1, nu, expanded, return_sce=True)
    assert contrib.shape == (6,)
    assert sce is not None and sce.shape == (2, 3, 2)


def test_evaluate_spec_margins_shape() -> None:
    net, _ = _make_test_net()
    bounds = _compute_bounds(net, batch_size=2)
    solver = DualSolver(DualTF())
    spec = OutputSpec(kind=OutKind.RANGE, lb=_t([-1.0, -1.0]), ub=_t([1.0, 1.0]))
    result = solver.evaluate_spec(net, bounds, spec)
    assert result.margins.shape == (2, 4)


def test_evaluate_spec_equal_to_repeat_interleave_baseline() -> None:
    net, _ = _make_test_net()
    bounds = _compute_bounds(net, batch_size=2)
    solver = DualSolver(DualTF())
    spec = OutputSpec(kind=OutKind.RANGE, lb=_t([-1.0, -1.0]), ub=_t([1.0, 1.0]))
    rank3 = solver.evaluate_spec(net, bounds, spec)
    dense = solver.evaluate_spec(net, bounds, spec, materialize=True)
    torch.testing.assert_close(rank3.margins, dense.margins)


def test_evaluate_spec_out_alphas_out_etas_unchanged() -> None:
    net, pre_lid = _make_test_net()
    bounds = _compute_bounds(net, batch_size=2)
    solver = DualSolver(DualTF())
    solver.set_eta(_make_eta(2, pre_lid))
    solver.set_alphas({3: torch.full((2, 3), 0.25, dtype=get_default_dtype(), device=get_default_device())})
    spec = OutputSpec(kind=OutKind.RANGE, lb=_t([-1.0, -1.0]), ub=_t([1.0, 1.0]))
    result = solver.evaluate_spec(net, bounds, spec)
    assert result.out_etas is not None and result.out_etas.batch_size == 2
    assert result.out_alphas is not None and result.out_alphas[3].shape == (2, 3)


def test_evaluate_spec_with_M1() -> None:
    net, _ = _make_test_net(output_dim=1)
    bounds = _compute_bounds(net, batch_size=2)
    solver = DualSolver(DualTF())
    spec = OutputSpec(kind=OutKind.LINEAR_LE, c=_t([1.0]), d=_t([0.0]))
    result = solver.evaluate_spec(net, bounds, spec)
    assert result.margins.shape == (2, 1)


def test_align_alpha_to_batch_rank3_passthrough() -> None:
    solver = DualSolver(DualTF())
    alpha = torch.full((2, 3), 0.2, dtype=get_default_dtype(), device=get_default_device())
    aligned = solver._align_alpha_to_batch(alpha, 2, 3, "alpha")
    assert aligned.shape == alpha.shape


def test_align_amb_mask_to_batch_rank3_passthrough() -> None:
    solver = DualSolver(DualTF())
    mask = torch.tensor([[True, False, True], [False, True, False]], device=get_default_device())
    aligned = solver._align_amb_mask_to_batch(mask, 2, 3)
    assert aligned is not None and torch.equal(aligned, mask)


def test_prepare_eta_state_materialized() -> None:
    net, pre_lid = _make_test_net()
    bounds = _compute_bounds(net, batch_size=2)
    solver = DualSolver(DualTF())
    solver.compute_bound(net, bounds, _t([[1.0, -1.0], [1.0, -1.0]]))
    eta = _make_eta(2, pre_lid)
    prepared = solver._prepare_eta_state(eta, 6)
    assert prepared is not None and prepared.batch_size == 6
    assert prepared.val[pre_lid].is_contiguous()


def test_expand_alpha_dict_rank3_no_expand() -> None:
    solver = DualSolver(DualTF())
    alpha = {3: torch.full((2, 3), 0.2, dtype=get_default_dtype(), device=get_default_device())}
    expanded = solver._expand_alpha_dict(alpha, 4)
    assert expanded.for_start_node(-1)[3].shape == (8, 3)


def test_backward_pass_rank3_parity_materialized() -> None:
    net, _ = _make_test_net()
    bounds = _compute_bounds(net, batch_size=2)
    solver = DualSolver(DualTF())
    c = _t([[1.0, -1.0]]).expand(6, -1).contiguous()
    obj_rank3, _ = solver._backward_pass(net, expand_bounds_dict(bounds, 3), c)
    obj_dense, _ = solver._backward_pass(net, expand_bounds_dict(bounds, 3, materialize=True), c)
    torch.testing.assert_close(obj_rank3, obj_dense)


def test_input_contribution_from_nu_rank3_parity() -> None:
    net, _ = _make_test_net()
    bounds = _compute_bounds(net, batch_size=2)
    solver = DualSolver(DualTF())
    nu = torch.randn(6, 2, dtype=get_default_dtype(), device=get_default_device())
    rank3, _ = solver._input_contribution_from_nu(net, 1, nu, expand_bounds_dict(bounds, 3))
    dense, _ = solver._input_contribution_from_nu(net, 1, nu, expand_bounds_dict(bounds, 3, materialize=True))
    torch.testing.assert_close(rank3, dense)


def test_backward_pass_rank3_parity_with_eta() -> None:
    net, pre_lid = _make_test_net()
    bounds = _compute_bounds(net, batch_size=2)
    solver = DualSolver(DualTF())
    eta = _make_eta(6, pre_lid)
    c = _t([[1.0, -1.0]]).expand(6, -1).contiguous()
    rank3, _ = solver._backward_pass(net, expand_bounds_dict(bounds, 3), c, eta_state=eta)
    dense, _ = solver._backward_pass(net, expand_bounds_dict(bounds, 3, materialize=True), c, eta_state=eta)
    torch.testing.assert_close(rank3, dense)


def test_expand_bounds_dict_external_shape_preserved() -> None:
    net, _ = _make_test_net()
    bounds = _compute_bounds(net, batch_size=2)
    solver = DualSolver(DualTF())
    spec = OutputSpec(kind=OutKind.RANGE, lb=_t([-1.0, -1.0]), ub=_t([1.0, 1.0]))
    result = solver.evaluate_spec(net, bounds, spec)
    assert result.margins.shape == (2, 4)


def test_expand_bounds_dict_materialized_matches_rank3_values() -> None:
    net, _ = _make_test_net()
    bounds = _compute_bounds(net, batch_size=2)
    rank3 = expand_bounds_dict(bounds, 3)
    dense = expand_bounds_dict(bounds, 3, materialize=True)
    torch.testing.assert_close(rank3[2].lb.reshape(-1, rank3[2].lb.shape[-1]), dense[2].lb)
    torch.testing.assert_close(rank3[2].ub.reshape(-1, rank3[2].ub.shape[-1]), dense[2].ub)


def test_evaluate_spec_materialize_flag_parity() -> None:
    net, _ = _make_test_net()
    bounds = _compute_bounds(net, batch_size=2)
    solver = DualSolver(DualTF())
    spec = OutputSpec(kind=OutKind.RANGE, lb=_t([-1.0, -1.0]), ub=_t([1.0, 1.0]))
    rank3 = solver.evaluate_spec(net, bounds, spec, materialize=False)
    dense = solver.evaluate_spec(net, bounds, spec, materialize=True)
    torch.testing.assert_close(rank3.min_slack, dense.min_slack)
    torch.testing.assert_close(rank3.certified, dense.certified)


def test_range_spec_singleton_target_broadcast() -> None:
    net, _ = _make_test_net()
    bounds = _compute_bounds(net, batch_size=2)
    solver = DualSolver(DualTF())
    spec = OutputSpec(kind=OutKind.MARGIN_ROBUST, y_true=torch.tensor([0], device=get_default_device()), margin=torch.tensor([0.1], dtype=get_default_dtype(), device=get_default_device()))
    result = solver.evaluate_spec(net, bounds, spec, num_classes=2)
    assert result.margins.shape == (2, 2)
    assert torch.isfinite(result.margins).all()


def test_prepare_eta_state_none_passthrough() -> None:
    solver = DualSolver(DualTF())
    assert solver._prepare_eta_state(None, 4) is None
