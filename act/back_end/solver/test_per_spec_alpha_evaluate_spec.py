# pyright: reportMissingImports=false, reportPrivateUsage=false

from __future__ import annotations

import pytest
import torch

from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf import DualTF, compute_forward_bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.alpha_state import AlphaState
from act.back_end.solver.solver_dual import DualSolver
from act.front_end.specs import OutKind, OutputSpec
from act.util.device_manager import get_default_device, get_default_dtype, initialize_device


@pytest.fixture(scope="module", autouse=True)
def _cpu_float32() -> None:
    initialize_device("cpu", "float32")


def _t(data: object) -> torch.Tensor:
    return torch.tensor(data, dtype=get_default_dtype(), device=get_default_device())


def _make_test_net() -> Net:
    in_vars = [0, 1]
    hidden_vars = [10, 11, 12]
    relu_vars = [20, 21, 22]
    out_vars = [30, 31]
    layers = [
        Layer(0, LayerKind.INPUT.value, {"shape": (2,), "dtype": "float32", "num_classes": 1, "value_range": (0.0, 1.0)}, in_vars, in_vars),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, in_vars, in_vars),
        Layer(2, LayerKind.DENSE.value, {"in_features": 2, "out_features": 3, "weight": _t([[1.0, 0.0], [0.5, -0.5], [0.2, 0.3]]), "bias": _t([0.0, 0.0, 0.1])}, in_vars, hidden_vars),
        Layer(3, LayerKind.RELU.value, {}, hidden_vars, relu_vars),
        Layer(4, LayerKind.DENSE.value, {"in_features": 3, "out_features": 2, "weight": _t([[-1.0, 0.0, 0.2], [0.0, 1.0, -0.1]]), "bias": _t([0.0, 0.0])}, relu_vars, out_vars),
        Layer(5, LayerKind.ASSERT.value, {"kind": "RANGE"}, out_vars, out_vars),
    ]
    return Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )


def _bounds(net: Net, B: int = 2) -> dict[int, Bounds]:
    lb = _t([[-1.0, -1.0]]).expand(B, -1).contiguous()
    ub = _t([[1.0, 1.0]]).expand(B, -1).contiguous()
    return compute_forward_bounds(net, lb, ub)


def _multi_spec() -> OutputSpec:
    return OutputSpec(kind=OutKind.RANGE, lb=_t([-1.0, -1.0]), ub=_t([1.0, 1.0]))


def test_dual_solver_alpha_per_spec_default_off_returns_legacy_alphastate() -> None:
    net = _make_test_net()
    bounds = _bounds(net, B=2)
    spec = _multi_spec()
    solver = DualSolver(DualTF())
    solver.eta_iters = 1
    assert solver.alpha_per_spec is False

    result = solver.evaluate_spec(net, bounds, spec)

    assert result.out_alphas is not None
    assert result.out_alphas.per_spec is False
    relu_alphas = result.out_alphas.for_start_node(AlphaState.FINAL_SID)
    assert relu_alphas, "expected non-empty α dict for ReLU layer"
    sample_tensor = next(iter(relu_alphas.values()))
    assert sample_tensor.dim() == 2
    assert sample_tensor.shape[0] == 2


def test_dual_solver_alpha_per_spec_on_returns_per_spec_alphastate() -> None:
    net = _make_test_net()
    bounds = _bounds(net, B=2)
    spec = _multi_spec()
    solver = DualSolver(DualTF())
    solver.eta_iters = 1
    solver.alpha_per_spec = True

    result = solver.evaluate_spec(net, bounds, spec)

    assert result.out_alphas is not None
    assert result.out_alphas.per_spec is True
    relu_alphas = result.out_alphas.for_start_node(AlphaState.FINAL_SID)
    assert relu_alphas, "expected non-empty α dict for ReLU layer"
    sample_tensor = next(iter(relu_alphas.values()))
    assert sample_tensor.dim() == 3
    B = 2
    M = result.margins.shape[1]
    assert sample_tensor.shape[0] == B
    assert sample_tensor.shape[1] == M
    assert sample_tensor.shape[2] == 3


def test_dual_solver_alpha_per_spec_warm_start_roundtrip() -> None:
    net = _make_test_net()
    bounds = _bounds(net, B=1)
    spec = _multi_spec()

    solver = DualSolver(DualTF())
    solver.eta_iters = 5
    solver.alpha_per_spec = True

    result1 = solver.evaluate_spec(net, bounds, spec)
    assert result1.out_alphas is not None
    assert result1.out_alphas.per_spec is True

    solver.set_alphas(result1.out_alphas)
    assert solver._alpha_state.per_spec is True

    result2 = solver.evaluate_spec(net, bounds, spec)
    assert result2.out_alphas is not None
    assert result2.out_alphas.per_spec is True

    relu_alphas_2 = result2.out_alphas.for_start_node(AlphaState.FINAL_SID)
    sample_tensor_2 = next(iter(relu_alphas_2.values()))
    M = result2.margins.shape[1]
    assert sample_tensor_2.shape == (1, M, 3)
