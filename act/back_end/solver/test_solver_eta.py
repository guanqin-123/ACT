# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportAttributeAccessIssue=false

from __future__ import annotations

from typing import Tuple, cast

import pytest
import torch

from act.back_end.bab.eta import EtaState
from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf import DualTF, compute_forward_bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_dual import DualSolver
from act.front_end.specs import OutKind, OutputSpec
from act.util.device_manager import get_default_device, get_default_dtype, initialize_device


@pytest.fixture(scope="module", autouse=True)
def _cpu_float32() -> None:
    initialize_device("cpu", "float32")


def _t(data: object) -> torch.Tensor:
    return torch.tensor(data, dtype=get_default_dtype(), device=get_default_device())


def _clone_eta(eta: EtaState) -> EtaState:
    return EtaState(
        val={lid: value.clone() for lid, value in eta.val.items()},
        sign={lid: value.clone() for lid, value in eta.sign.items()},
        point={lid: value.clone() for lid, value in eta.point.items()},
    )


def _make_test_net(output_dim: int = 1) -> Tuple[Net, int]:
    in_vars = [0, 1]
    hidden_vars = [10, 11, 12]
    relu_vars = [20, 21, 22]
    out_vars = list(range(30, 30 + output_dim))

    hidden_weight = _t([
        [1.0, 0.0],
        [0.5, -0.5],
        [0.0, 0.0],
    ])
    hidden_bias = _t([0.0, 0.0, 0.0])

    if output_dim == 1:
        out_weight = _t([[-1.0, 0.0, 0.0]])
        out_bias = _t([0.0])
    elif output_dim == 2:
        out_weight = _t([
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        out_bias = _t([0.0, 0.0])
    else:
        raise ValueError(f"Unsupported output_dim={output_dim}")

    layers = [
        Layer(
            0,
            LayerKind.INPUT.value,
            {
                "shape": [2],
                "dtype": "float32",
                "num_classes": 1,
                "value_range": [0.0, 1.0],
            },
            in_vars,
            in_vars,
        ),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, in_vars, in_vars),
        Layer(
            2,
            LayerKind.DENSE.value,
            {
                "in_features": 2,
                "out_features": 3,
                "weight": hidden_weight,
                "bias": hidden_bias,
            },
            in_vars,
            hidden_vars,
        ),
        Layer(3, LayerKind.RELU.value, {}, hidden_vars, relu_vars),
        Layer(
            4,
            LayerKind.DENSE.value,
            {
                "in_features": 3,
                "out_features": output_dim,
                "weight": out_weight,
                "bias": out_bias,
            },
            relu_vars,
            out_vars,
        ),
        Layer(5, LayerKind.ASSERT.value, {"kind": "RANGE"}, out_vars, out_vars),
    ]
    net = Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )
    return net, 2


def _compute_bounds(net: Net, batch_size: int) -> dict[int, Bounds]:
    lb = _t([[-1.0, -1.0]]).expand(batch_size, -1).clone()
    ub = _t([[1.0, 1.0]]).expand(batch_size, -1).clone()
    return compute_forward_bounds(net, lb, ub)


def _make_eta(
    batch_size: int,
    pre_lid: int,
    sign_value: float,
    init_value: float = 0.0,
) -> EtaState:
    val = {pre_lid: torch.full((batch_size, 3), init_value, device=get_default_device(), dtype=get_default_dtype())}
    sign = {pre_lid: torch.zeros((batch_size, 3), device=get_default_device(), dtype=get_default_dtype())}
    sign[pre_lid][:, 0] = sign_value
    point = {pre_lid: torch.zeros((batch_size, 3), device=get_default_device(), dtype=get_default_dtype())}
    return EtaState(val=val, sign=sign, point=point)


def test_compute_bound_fast_path_bitidentical() -> None:
    net, _pre_lid = _make_test_net(output_dim=1)
    bounds = _compute_bounds(net, batch_size=1)
    c = _t([[1.0]])
    solver = DualSolver(DualTF())

    actual_obj, actual_sce = cast(
        Tuple[torch.Tensor, torch.Tensor | None],
        solver.compute_bound(net, bounds, c, return_sce=True),
    )
    expected_obj, expected_sce = cast(
        Tuple[torch.Tensor, torch.Tensor | None],
        solver._compute_bound_direct(net, bounds, c, return_sce=True),
    )

    assert torch.equal(actual_obj, expected_obj)
    assert expected_sce is not None and actual_sce is not None
    assert torch.equal(actual_sce, expected_sce)


def test_compute_bound_eta_empty_fast_path() -> None:
    net, pre_lid = _make_test_net(output_dim=1)
    bounds = _compute_bounds(net, batch_size=1)
    c = _t([[1.0]])
    solver = DualSolver(DualTF())
    eta = _make_eta(batch_size=1, pre_lid=pre_lid, sign_value=0.0)
    solver.set_eta(eta)

    actual = cast(torch.Tensor, solver.compute_bound(net, bounds, c))
    expected = cast(torch.Tensor, solver._compute_bound_direct(net, bounds, c))

    assert solver._eta_state is not None and solver._eta_state.fast_path_skip()
    assert torch.equal(actual, expected)


def test_compute_bound_eta_large_value_tightens() -> None:
    net, pre_lid = _make_test_net(output_dim=1)
    bounds = _compute_bounds(net, batch_size=1)
    c = _t([[1.0]])
    eta = _make_eta(batch_size=1, pre_lid=pre_lid, sign_value=1.0)

    solver_one = DualSolver(DualTF())
    solver_one.eta_iters = 1
    solver_one.set_eta(_clone_eta(eta))
    obj_one = cast(torch.Tensor, solver_one.compute_bound(net, bounds, c))

    solver_twenty = DualSolver(DualTF())
    solver_twenty.eta_iters = 20
    solver_twenty.set_eta(_clone_eta(eta))
    obj_twenty = cast(torch.Tensor, solver_twenty.compute_bound(net, bounds, c))

    assert obj_twenty.item() > obj_one.item() + 1e-4


def test_compute_bound_eta_non_negative_after_step() -> None:
    net, pre_lid = _make_test_net(output_dim=1)
    bounds = _compute_bounds(net, batch_size=1)
    c = _t([[1.0]])
    solver = DualSolver(DualTF())
    solver.eta_iters = 1
    solver.set_eta(_make_eta(batch_size=1, pre_lid=pre_lid, sign_value=1.0, init_value=-0.25))

    _ = solver.compute_bound(net, bounds, c)

    assert solver._eta_state is not None
    assert torch.all(solver._eta_state.val[pre_lid] >= 0)


def test_evaluate_spec_eta_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    net, pre_lid = _make_test_net(output_dim=2)
    bounds = _compute_bounds(net, batch_size=3)
    solver = DualSolver(DualTF())
    solver.set_eta(_make_eta(batch_size=3, pre_lid=pre_lid, sign_value=1.0))
    spec = OutputSpec(kind=OutKind.RANGE, lb=_t([-1.0, -1.0]), ub=_t([1.0, 1.0]))

    seen: dict[str, object] = {}
    original_compute_bound = solver.compute_bound

    def wrapped_compute_bound(
        net_arg: Net,
        bounds_arg: dict[int, Bounds],
        c_arg: torch.Tensor,
        return_sce: bool = False,
        enable_grad: bool = False,
        **kwargs,
    ):
        seen["rows"] = int(c_arg.shape[0])
        assert solver._eta_state is not None
        seen["eta_shape"] = tuple(solver._eta_state.val[pre_lid].shape)
        return original_compute_bound(
            net_arg,
            bounds_arg,
            c_arg,
            return_sce=return_sce,
            enable_grad=enable_grad,
            **kwargs,
        )

    monkeypatch.setattr(solver, "compute_bound", wrapped_compute_bound)
    result = solver.evaluate_spec(net, bounds, spec)

    assert result.margins.shape == (3, 4)
    assert seen["rows"] == 12
    assert seen["eta_shape"] == (12, 3)
    assert solver._eta_state is not None
    assert solver._eta_state.val[pre_lid].shape == (3, 3)


def test_evaluate_spec_eta_restores_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    net, pre_lid = _make_test_net(output_dim=2)
    bounds = _compute_bounds(net, batch_size=3)
    solver = DualSolver(DualTF())
    solver.set_eta(_make_eta(batch_size=3, pre_lid=pre_lid, sign_value=1.0))
    spec = OutputSpec(kind=OutKind.RANGE, lb=_t([-1.0, -1.0]), ub=_t([1.0, 1.0]))

    def boom_compute_bound(
        _net_arg: Net,
        _bounds_arg: dict[int, Bounds],
        _c_arg: torch.Tensor,
        return_sce: bool = False,
        enable_grad: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del return_sce, enable_grad, kwargs
        assert solver._eta_state is not None
        assert solver._eta_state.val[pre_lid].shape == (12, 3)
        raise RuntimeError("boom")

    monkeypatch.setattr(solver, "compute_bound", boom_compute_bound)

    with pytest.raises(RuntimeError, match="boom"):
        solver.evaluate_spec(net, bounds, spec)

    assert solver._eta_state is not None
    assert solver._eta_state.val[pre_lid].shape == (3, 3)


def test_set_clear_eta_lifecycle() -> None:
    _net, pre_lid = _make_test_net(output_dim=1)
    solver = DualSolver(DualTF())

    solver.set_eta(_make_eta(batch_size=2, pre_lid=pre_lid, sign_value=1.0))
    assert solver._eta_state is not None

    solver.clear_eta()

    assert solver._eta_state is None
