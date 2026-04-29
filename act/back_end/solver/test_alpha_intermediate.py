# pyright: reportMissingImports=false

from __future__ import annotations

from typing import cast

import torch

from act.back_end.config import BaBConfig
from act.back_end.core import Layer, Net
from act.back_end.dual_tf import DualTF, compute_forward_bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.solver._backward_truncated import backward_truncated_lb, backward_truncated_objective, backward_truncated_ub
from act.back_end.solver._initial_alpha_crown import (
    enumerate_intermediate_start_nodes,
    optimize_initial_intermediate_bounds,
)
from act.back_end.solver.alpha_state import AlphaState
from act.back_end.solver.solver_dual import DualSolver
from act.back_end.solver.spec_batching import build_spec_batch
from act.front_end.specs import OutKind, OutputSpec
from act.util.device_manager import get_default_device, get_default_dtype, initialize_device


def setup_module() -> None:
    initialize_device("cpu", "float64")


def _t(data: object) -> torch.Tensor:
    return torch.tensor(data, dtype=get_default_dtype(), device=get_default_device())


def _sample_box(lb: torch.Tensor, ub: torch.Tensor, *, count: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=get_default_device()).manual_seed(seed)
    return lb + torch.rand((count, lb.shape[-1]), generator=gen, dtype=lb.dtype, device=lb.device) * (ub - lb)


def _make_three_layer_relu_net() -> tuple[Net, list[int]]:
    in_vars = [0, 1]
    z1 = [10, 11, 12]
    a1 = [20, 21, 22]
    z2 = [30, 31]
    a2 = [40, 41]
    out = [50, 51]
    layers = [
        Layer(0, LayerKind.INPUT.value, {"shape": (2,), "dtype": "float64", "num_classes": 1, "value_range": (0.0, 1.0)}, in_vars, in_vars),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, in_vars, in_vars),
        Layer(2, LayerKind.DENSE.value, {"in_features": 2, "out_features": 3, "weight": _t([[1.0, -0.8], [0.6, 0.7], [-1.1, 0.9]]), "bias": _t([0.1, -0.2, 0.05])}, in_vars, z1),
        Layer(3, LayerKind.RELU.value, {}, z1, a1),
        Layer(4, LayerKind.DENSE.value, {"in_features": 3, "out_features": 2, "weight": _t([[1.2, -0.7, 0.4], [-0.9, 1.1, -0.6]]), "bias": _t([0.0, 0.15])}, a1, z2),
        Layer(5, LayerKind.RELU.value, {}, z2, a2),
        Layer(6, LayerKind.DENSE.value, {"in_features": 2, "out_features": 2, "weight": _t([[0.7, -0.4], [-0.5, 0.9]]), "bias": _t([0.1, -0.2])}, a2, out),
        Layer(7, LayerKind.ASSERT.value, {"kind": "RANGE"}, out, out),
    ]
    net = Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4], 6: [5], 7: [6]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6], 6: [7], 7: []},
    )
    return net, [2, 4]


def _make_deep_relu_net() -> tuple[Net, list[int], torch.Tensor, torch.Tensor]:
    in_vars = [0, 1]
    z1 = [10, 11, 12]
    a1 = [20, 21, 22]
    z2 = [30, 31, 32]
    a2 = [40, 41, 42]
    z3 = [50, 51]
    a3 = [60, 61]
    out = [70, 71]
    layers = [
        Layer(0, LayerKind.INPUT.value, {"shape": (2,), "dtype": "float64", "num_classes": 1, "value_range": (0.0, 1.0)}, in_vars, in_vars),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, in_vars, in_vars),
        Layer(2, LayerKind.DENSE.value, {"in_features": 2, "out_features": 3, "weight": _t([[1.0, -0.8], [0.6, 0.7], [-1.1, 0.9]]), "bias": _t([0.1, -0.2, 0.05])}, in_vars, z1),
        Layer(3, LayerKind.RELU.value, {}, z1, a1),
        Layer(4, LayerKind.DENSE.value, {"in_features": 3, "out_features": 3, "weight": _t([[1.2, -0.7, 0.4], [-0.9, 1.1, -0.6], [0.5, -1.0, 0.8]]), "bias": _t([0.0, 0.15, -0.1])}, a1, z2),
        Layer(5, LayerKind.RELU.value, {}, z2, a2),
        Layer(6, LayerKind.DENSE.value, {"in_features": 3, "out_features": 2, "weight": _t([[1.0, -0.8, 0.6], [-1.2, 0.9, -0.7]]), "bias": _t([0.05, -0.05])}, a2, z3),
        Layer(7, LayerKind.RELU.value, {}, z3, a3),
        Layer(8, LayerKind.DENSE.value, {"in_features": 2, "out_features": 2, "weight": _t([[0.7, -0.4], [-0.5, 0.9]]), "bias": _t([0.1, -0.2])}, a3, out),
        Layer(9, LayerKind.ASSERT.value, {"kind": "RANGE"}, out, out),
    ]
    net = Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4], 6: [5], 7: [6], 8: [7], 9: [8]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6], 6: [7], 7: [8], 8: [9], 9: []},
    )
    input_lb = _t([[-1.0, -0.8]])
    input_ub = _t([[0.9, 1.1]])
    return net, [2, 4, 6], input_lb, input_ub


def _make_conv_relu_dense_net() -> tuple[Net, torch.Tensor, torch.Tensor]:
    input_vars = list(range(16))
    conv1_out = list(range(100, 116))
    relu1_out = list(range(200, 216))
    conv2_out = list(range(300, 316))
    relu2_out = list(range(400, 416))
    out = [500, 501]
    conv1_w = _t([[[[0.2, -0.1, 0.0], [0.1, 0.3, -0.2], [0.0, 0.2, 0.1]]]])
    conv2_w = _t([[[[-0.3, 0.2, 0.1], [0.0, 0.1, 0.2], [0.2, -0.1, 0.3]]]])
    dense_w = _t([[0.1, -0.2, 0.15, -0.05, 0.03, 0.04, -0.1, 0.07, 0.02, 0.05, -0.06, 0.08, 0.09, -0.03, 0.11, -0.04],
                  [-0.07, 0.12, -0.03, 0.06, -0.08, 0.09, 0.05, -0.02, 0.1, -0.11, 0.04, 0.03, -0.06, 0.07, -0.01, 0.02]])
    net = Net(
        layers=[
            Layer(0, LayerKind.INPUT.value, {"shape": (1, 4, 4), "dtype": "float64", "num_classes": 1, "value_range": (0.0, 1.0)}, input_vars, input_vars),
            Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, input_vars, input_vars),
            Layer(2, LayerKind.CONV2D.value, {"in_channels": 1, "out_channels": 1, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "weight": conv1_w, "bias": _t([0.02]), "input_shape": (1, 1, 4, 4), "output_shape": (1, 1, 4, 4)}, input_vars, conv1_out),
            Layer(3, LayerKind.RELU.value, {}, conv1_out, relu1_out),
            Layer(4, LayerKind.CONV2D.value, {"in_channels": 1, "out_channels": 1, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "weight": conv2_w, "bias": _t([-0.03]), "input_shape": (1, 1, 4, 4), "output_shape": (1, 1, 4, 4)}, relu1_out, conv2_out),
            Layer(5, LayerKind.RELU.value, {}, conv2_out, relu2_out),
            Layer(6, LayerKind.FLATTEN.value, {"start_dim": 1}, relu2_out, relu2_out),
            Layer(7, LayerKind.DENSE.value, {"in_features": 16, "out_features": 2, "weight": dense_w, "bias": _t([0.05, -0.04])}, relu2_out, out),
            Layer(8, LayerKind.ASSERT.value, {"kind": "RANGE"}, out, out),
        ],
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4], 6: [5], 7: [6], 8: [7]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6], 6: [7], 7: [8], 8: []},
    )
    lb = _t([[-0.2] * 16])
    ub = _t([[0.3] * 16])
    return net, lb, ub


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


def test_intermediate_start_nodes_enumerated() -> None:
    net, expected = _make_three_layer_relu_net()
    assert enumerate_intermediate_start_nodes(net) == expected


def test_intermediate_alpha_receives_nonzero_grad() -> None:
    net, lb, ub = _make_conv_relu_dense_net()
    bounds = compute_forward_bounds(net, lb, ub, post_activation=False)
    config = BaBConfig(alpha_split_objective=True, alpha_iters=1, lr_alpha=0.5)
    _new_bounds, alpha_state = optimize_initial_intermediate_bounds(
        net,
        bounds,
        alpha_iters=config.alpha_iters,
        lr_alpha=config.lr_alpha,
    )
    found = False
    for sid_int in alpha_state.start_nodes:
        for layer in net.layers:
            tensor = alpha_state.get(layer.id, sid_int)
            if sid_int == AlphaState.FINAL_SID or tensor is None or tensor.grad is None:
                continue
            if tensor.grad.abs().sum().item() > 0:
                found = True
                break
        if found:
            break
    assert found


def test_intermediate_bound_lb_le_concrete() -> None:
    net, start_nodes, input_lb, input_ub = _make_deep_relu_net()
    bounds = compute_forward_bounds(net, input_lb, input_ub, post_activation=False)
    _new_bounds, alpha_state = optimize_initial_intermediate_bounds(
        net,
        bounds,
        alpha_iters=5,
        lr_alpha=0.5,
    )
    samples = _sample_box(input_lb[0], input_ub[0], count=100, seed=7)
    preacts = _forward_mlp_preacts(net, samples)
    for sid_int in start_nodes:
        lb_sid = backward_truncated_lb(net, bounds, sid_int, alpha_state)
        ub_sid = backward_truncated_ub(net, bounds, sid_int, alpha_state)
        actual = preacts[sid_int]
        assert torch.all(lb_sid[0].unsqueeze(0) <= actual + 1e-7)
        assert torch.all(ub_sid[0].unsqueeze(0) >= actual - 1e-7)


def test_intermediate_loss_zero_when_disabled() -> None:
    net, _start_nodes, input_lb, input_ub = _make_deep_relu_net()
    bounds = compute_forward_bounds(net, input_lb, input_ub, post_activation=False)
    config = BaBConfig(alpha_split_objective=False)
    new_bounds, alpha_state = optimize_initial_intermediate_bounds(
        net,
        bounds,
        alpha_iters=0 if not config.alpha_split_objective else config.alpha_iters,
        lr_alpha=config.lr_alpha,
    )
    assert alpha_state.is_empty()
    for lid, old in bounds.items():
        torch.testing.assert_close(new_bounds[lid].lb, old.lb)
        torch.testing.assert_close(new_bounds[lid].ub, old.ub)


def test_intermediate_backward_truncated_matches_full() -> None:
    net, _start_nodes, input_lb, input_ub = _make_deep_relu_net()
    bounds = compute_forward_bounds(net, input_lb, input_ub, post_activation=False)
    solver = DualSolver(DualTF())
    result = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, AlphaState | None, object],
        solver.compute_robust_bound(net, bounds, y_true=0, num_classes=2, n_iters=0),
    )
    assert isinstance(result, tuple)
    spec = build_spec_batch(
        OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=torch.tensor([0], device=get_default_device(), dtype=torch.long)),
        B=1,
        n_out=2,
        num_classes=2,
        device=get_default_device(),
        dtype=get_default_dtype(),
    )
    final_sid = net.preds[net.layers[-1].id][0]
    truncated = backward_truncated_objective(net, bounds, final_sid, spec.C, AlphaState()).reshape(1, spec.M)
    torch.testing.assert_close(truncated, result[2], rtol=1e-5, atol=1e-7)


def test_chunked_objective_rows_match_unchunked() -> None:
    net, start_nodes, input_lb, input_ub = _make_deep_relu_net()
    bounds = compute_forward_bounds(net, input_lb, input_ub, post_activation=False)
    _, alpha_state = optimize_initial_intermediate_bounds(
        net, bounds, alpha_iters=3, lr_alpha=0.5,
    )
    for sid_int in start_nodes:
        baseline_lb = backward_truncated_lb(net, bounds, sid_int, alpha_state)
        baseline_ub = backward_truncated_ub(net, bounds, sid_int, alpha_state)
        for chunk in (1, 2, 3, 5):
            lb_chunked = backward_truncated_lb(
                net, bounds, sid_int, alpha_state, objective_chunk_size=chunk,
            )
            ub_chunked = backward_truncated_ub(
                net, bounds, sid_int, alpha_state, objective_chunk_size=chunk,
            )
            torch.testing.assert_close(lb_chunked, baseline_lb, rtol=1e-7, atol=1e-9)
            torch.testing.assert_close(ub_chunked, baseline_ub, rtol=1e-7, atol=1e-9)


def test_chunk_size_default_off_unchunked() -> None:
    net, start_nodes, input_lb, input_ub = _make_deep_relu_net()
    bounds = compute_forward_bounds(net, input_lb, input_ub, post_activation=False)
    alpha_state = AlphaState()
    sid_int = start_nodes[0]
    lb_default = backward_truncated_lb(net, bounds, sid_int, alpha_state)
    lb_explicit_none = backward_truncated_lb(net, bounds, sid_int, alpha_state, objective_chunk_size=None)
    torch.testing.assert_close(lb_default, lb_explicit_none, rtol=0.0, atol=0.0)
