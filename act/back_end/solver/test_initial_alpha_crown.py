# pyright: reportMissingImports=false

from __future__ import annotations

import act.back_end.bab.bab as bab_module

import pytest
import torch

from act.back_end.bab.bab import verify_bab
from act.back_end.config import BaBConfig
from act.back_end.core import Layer, Net
from act.back_end.dual_tf import DualTF, compute_forward_bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.solver._initial_alpha_crown import enumerate_intermediate_start_nodes, optimize_initial_intermediate_bounds
from act.back_end.solver.solver_dual import DualSolver
from act.back_end.solver.solver_interval import TorchLPSolver
from act.front_end.specs import OutKind
from act.util.device_manager import get_default_device, get_default_dtype, initialize_device
from act.util.stats import VerifyStatus


def setup_module() -> None:
    initialize_device("cpu", "float64")


def _t(data: object) -> torch.Tensor:
    return torch.tensor(data, dtype=get_default_dtype(), device=get_default_device())


def _sample_box(lb: torch.Tensor, ub: torch.Tensor, *, count: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=get_default_device()).manual_seed(seed)
    return lb + torch.rand((count, lb.shape[-1]), generator=gen, dtype=lb.dtype, device=lb.device) * (ub - lb)


def _make_deep_relu_net() -> tuple[Net, list[int], torch.Tensor, torch.Tensor]:
    dims = [3, 5, 5, 5, 5, 5, 2]
    gen = torch.Generator(device=get_default_device()).manual_seed(0)
    input_vars = list(range(dims[0]))
    layers = [
        Layer(0, LayerKind.INPUT.value, {"shape": (dims[0],), "dtype": "float64", "num_classes": 1, "value_range": (0.0, 1.0)}, input_vars, input_vars),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, input_vars, input_vars),
    ]
    preds: dict[int, list[int]] = {0: [], 1: [0]}
    succs: dict[int, list[int]] = {0: [1], 1: []}
    prev_vars = input_vars
    next_var = 10
    layer_id = 2
    for idx, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        dense_out = list(range(next_var, next_var + dout))
        next_var += dout
        weight = torch.randn((dout, din), generator=gen, dtype=get_default_dtype(), device=get_default_device())
        bias = torch.randn((dout,), generator=gen, dtype=get_default_dtype(), device=get_default_device()) * 0.4
        layers.append(Layer(layer_id, LayerKind.DENSE.value, {"in_features": din, "out_features": dout, "weight": weight, "bias": bias}, prev_vars, dense_out))
        preds[layer_id] = [layer_id - 1]
        succs[layer_id - 1] = [layer_id]
        prev_vars = dense_out
        if idx < len(dims) - 2:
            relu_out = list(range(next_var, next_var + dout))
            next_var += dout
            layers.append(Layer(layer_id + 1, LayerKind.RELU.value, {}, dense_out, relu_out))
            preds[layer_id + 1] = [layer_id]
            succs[layer_id] = [layer_id + 1]
            prev_vars = relu_out
            layer_id += 2
        else:
            layer_id += 1
    layers.append(Layer(layer_id, LayerKind.ASSERT.value, {"kind": "RANGE"}, prev_vars, prev_vars))
    preds[layer_id] = [layer_id - 1]
    succs[layer_id - 1] = [layer_id]
    succs[layer_id] = []
    net = Net(layers=layers, preds=preds, succs=succs)
    input_lb = torch.full((1, dims[0]), -1.0, dtype=get_default_dtype(), device=get_default_device())
    input_ub = torch.full((1, dims[0]), 1.0, dtype=get_default_dtype(), device=get_default_device())
    return net, enumerate_intermediate_start_nodes(net), input_lb, input_ub


def _forward_mlp_preacts(net: Net, x: torch.Tensor) -> dict[int, torch.Tensor]:
    y = x
    preacts: dict[int, torch.Tensor] = {}
    for layer in net.layers:
        kind = layer.kind.upper()
        if kind in {LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value, LayerKind.ASSERT.value}:
            continue
        if kind == LayerKind.DENSE.value:
            weight = layer.get_tensor("weight")
            bias = layer.get_tensor("bias")
            assert weight is not None and bias is not None
            y = y @ weight.T + bias
            preacts[layer.id] = y
        elif kind == LayerKind.RELU.value:
            y = torch.relu(y)
        else:
            raise AssertionError(f"Unexpected layer kind in helper: {kind}")
    return preacts


def _make_verify_relu_net() -> Net:
    in_vars = [0, 1]
    z = [10, 11]
    a = [20, 21]
    out = [30]
    return Net(
        layers=[
            Layer(0, LayerKind.INPUT.value, {"shape": (2,), "dtype": "float64", "num_classes": 1, "value_range": (0.0, 1.0)}, in_vars, in_vars),
            Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX", "lb": _t([-1.0, -2.0]), "ub": _t([2.0, 1.0])}, in_vars, in_vars),
            Layer(2, LayerKind.DENSE.value, {"in_features": 2, "out_features": 2, "weight": _t([[1.0, -0.5], [0.3, 0.8]]), "bias": _t([0.1, -0.2])}, in_vars, z),
            Layer(3, LayerKind.RELU.value, {}, z, a),
            Layer(4, LayerKind.DENSE.value, {"in_features": 2, "out_features": 1, "weight": _t([[0.2, 0.1]]), "bias": _t([0.0])}, a, out),
            Layer(5, LayerKind.ASSERT.value, {"kind": OutKind.LINEAR_LE, "c": _t([1.0]), "d": _t([10.0])}, out, out),
        ],
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )


def test_initial_intermediate_bounds_tighten_or_equal() -> None:
    net, start_nodes, input_lb, input_ub = _make_deep_relu_net()
    bounds = compute_forward_bounds(net, input_lb, input_ub, post_activation=False)
    new_bounds, _alpha_state = optimize_initial_intermediate_bounds(
        net,
        bounds,
        alpha_iters=20,
        lr_alpha=0.5,
    )
    tighten_ratios: list[float] = []
    for sid_int in start_nodes:
        old = bounds[sid_int]
        new = new_bounds[sid_int]
        assert torch.all(new.lb >= old.lb - 1e-7)
        assert torch.all(new.ub <= old.ub + 1e-7)
        old_width = (old.ub - old.lb).reshape(-1)
        new_width = (new.ub - new.lb).reshape(-1)
        tighten = ((old_width - new_width) / old_width.clamp(min=1e-12)).median().item()
        tighten_ratios.append(tighten)
    assert torch.tensor(tighten_ratios).median().item() > 0.0


def test_initial_intermediate_bounds_sound_on_samples() -> None:
    net, start_nodes, input_lb, input_ub = _make_deep_relu_net()
    bounds = compute_forward_bounds(net, input_lb, input_ub, post_activation=False)
    new_bounds, _alpha_state = optimize_initial_intermediate_bounds(
        net,
        bounds,
        alpha_iters=20,
        lr_alpha=0.5,
    )
    samples = _sample_box(input_lb[0], input_ub[0], count=100, seed=11)
    preacts = _forward_mlp_preacts(net, samples)
    for sid_int in start_nodes:
        actual = preacts[sid_int]
        lower = new_bounds[sid_int].lb[0].unsqueeze(0)
        upper = new_bounds[sid_int].ub[0].unsqueeze(0)
        assert torch.all(lower <= actual + 1e-7)
        assert torch.all(upper >= actual - 1e-7)


def test_initial_intermediate_bounds_disabled_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    net = _make_verify_relu_net()
    out_bounds: dict[int, dict[str, torch.Tensor]] = {}
    called = {"value": False}

    def _should_not_run(*_args: object, **_kwargs: object) -> None:
        called["value"] = True
        raise AssertionError("optimize_initial_intermediate_bounds should not be called")

    monkeypatch.setattr(bab_module, "optimize_initial_intermediate_bounds", _should_not_run)
    result = verify_bab(
        net,
        TorchLPSolver(),
        config=BaBConfig(max_depth=0, max_nodes=1, branching_method="babsr", bounding_method="bfs", subproblem_batch_size=1, alpha_split_objective=False),
        dual_solver=DualSolver(DualTF()),
        out_bounds_dict=out_bounds,
    )
    assert result.status == VerifyStatus.CERTIFIED
    assert called["value"] is False
    torch.testing.assert_close(out_bounds[2]["lb"], _t([[-1.4, -2.1]]))
    torch.testing.assert_close(out_bounds[2]["ub"], _t([[3.1, 1.2]]))
    assert enumerate_intermediate_start_nodes(net) == [2]
