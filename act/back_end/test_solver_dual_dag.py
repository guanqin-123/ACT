#===- act/back_end/test_solver_dual_dag.py - DAG-aware dual backward tests ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnusedFunction=false, reportUntypedFunctionDecorator=false

"""DAG-aware soundness tests for DualSolver.compute_bound() and compute_forward_bounds()."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf import DualTF, compute_forward_bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_dual import DualSolver
from act.pipeline.verification.act2torch import ACTToTorch
from act.util.device_manager import get_default_device, get_default_dtype, initialize_device


@pytest.fixture(scope="module", autouse=True)
def _cpu_float32() -> None:
    """Run these tests on CPU/float32 for determinism."""
    initialize_device("cpu", "float32")


def _t(data: object) -> torch.Tensor:
    """Create a tensor on the configured default device and dtype."""
    return torch.tensor(data, dtype=get_default_dtype(), device=get_default_device())


def _bound_value(solver: DualSolver, net: Net, bounds: dict[int, Bounds], c: torch.Tensor) -> float:
    """Extract a scalar lower bound from the singleton dual solve path."""
    bound = cast(torch.Tensor, solver.compute_bound(net, bounds, c, return_sce=False))
    return float(bound.item())


def _make_linear_residual_net() -> tuple[Net, dict[str, Layer]]:
    """Build INPUT -> INPUT_SPEC -> DENSE -> ADD(input_spec, dense) -> ASSERT."""
    weight = _t([
        [0.5, -0.3, 0.1],
        [0.2, 0.4, -0.2],
        [-0.1, 0.3, 0.5],
    ])
    bias = _t([0.1, -0.2, 0.15])
    layers = {
        "input": Layer(0, LayerKind.INPUT.value, {"shape": [3], "dtype": "float32", "num_classes": 1, "value_range": [0.0, 1.0]}, [0, 1, 2], [0, 1, 2]),
        "input_spec": Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, [0, 1, 2], [0, 1, 2]),
        "dense": Layer(2, LayerKind.DENSE.value, {"in_features": 3, "out_features": 3, "weight": weight, "bias": bias}, [0, 1, 2], [10, 11, 12]),
        "add": Layer(3, LayerKind.ADD.value, {}, [0, 1, 2, 10, 11, 12], [20, 21, 22]),
        "assert": Layer(4, LayerKind.ASSERT.value, {"kind": "RANGE"}, [20, 21, 22], [20, 21, 22]),
    }
    net = Net(
        layers=list(layers.values()),
        preds={0: [], 1: [0], 2: [1], 3: [1, 2], 4: [3]},
        succs={0: [1], 1: [2, 3], 2: [3], 3: [4], 4: []},
    )
    return net, layers


def _make_relu_cancellation_net() -> tuple[Net, dict[str, Layer]]:
    """Build a fan-out cancellation net that requires accumulate-then-clamp."""
    w_a = _t([[1.0, 0.0], [0.0, 1.0]])
    b_a = _t([0.0, 0.0])
    w_b = _t([[1.0, 0.0], [0.0, 0.0]])
    b_b = _t([0.0, 0.0])
    w_c = _t([[-1.0, 0.0], [0.0, 0.0]])
    b_c = _t([0.0, 0.0])
    relu_out = [20, 21]
    layers = {
        "input": Layer(0, LayerKind.INPUT.value, {"shape": [2], "dtype": "float32", "num_classes": 1, "value_range": [0.0, 1.0]}, [0, 1], [0, 1]),
        "input_spec": Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, [0, 1], [0, 1]),
        "dense_a": Layer(2, LayerKind.DENSE.value, {"in_features": 2, "out_features": 2, "weight": w_a, "bias": b_a}, [0, 1], [10, 11]),
        "relu": Layer(3, LayerKind.RELU.value, {}, [10, 11], relu_out),
        "dense_b": Layer(4, LayerKind.DENSE.value, {"in_features": 2, "out_features": 2, "weight": w_b, "bias": b_b}, relu_out, [30, 31]),
        "dense_c": Layer(5, LayerKind.DENSE.value, {"in_features": 2, "out_features": 2, "weight": w_c, "bias": b_c}, relu_out, [32, 33]),
        "add": Layer(6, LayerKind.ADD.value, {}, [30, 31, 32, 33], [40, 41]),
        "assert": Layer(7, LayerKind.ASSERT.value, {"kind": "RANGE"}, [40, 41], [40, 41]),
    }
    net = Net(
        layers=list(layers.values()),
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [3], 6: [4, 5], 7: [6]},
        succs={0: [1], 1: [2], 2: [3], 3: [4, 5], 4: [6], 5: [6], 6: [7], 7: []},
    )
    return net, layers


def _make_resnet_block_net() -> tuple[Net, dict[str, Layer]]:
    """Build INPUT -> INPUT_SPEC -> DENSE_A -> RELU -> DENSE_B -> ADD -> ASSERT."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        w_a = torch.randn(3, 3, dtype=get_default_dtype(), device=get_default_device()) * 0.25
        b_a = torch.randn(3, dtype=get_default_dtype(), device=get_default_device()) * 0.15
        torch.manual_seed(43)
        w_b = torch.randn(3, 3, dtype=get_default_dtype(), device=get_default_device()) * 0.25
        b_b = torch.randn(3, dtype=get_default_dtype(), device=get_default_device()) * 0.15
    layers = {
        "input": Layer(0, LayerKind.INPUT.value, {"shape": [3], "dtype": "float32", "num_classes": 1, "value_range": [0.0, 1.0]}, [0, 1, 2], [0, 1, 2]),
        "input_spec": Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, [0, 1, 2], [0, 1, 2]),
        "dense_a": Layer(2, LayerKind.DENSE.value, {"in_features": 3, "out_features": 3, "weight": w_a, "bias": b_a}, [0, 1, 2], [10, 11, 12]),
        "relu": Layer(3, LayerKind.RELU.value, {}, [10, 11, 12], [20, 21, 22]),
        "dense_b": Layer(4, LayerKind.DENSE.value, {"in_features": 3, "out_features": 3, "weight": w_b, "bias": b_b}, [20, 21, 22], [30, 31, 32]),
        "add": Layer(5, LayerKind.ADD.value, {}, [0, 1, 2, 30, 31, 32], [40, 41, 42]),
        "assert": Layer(6, LayerKind.ASSERT.value, {"kind": "RANGE"}, [40, 41, 42], [40, 41, 42]),
    }
    net = Net(
        layers=list(layers.values()),
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [1, 4], 6: [5]},
        succs={0: [1], 1: [2, 5], 2: [3], 3: [4], 4: [5], 5: [6], 6: []},
    )
    return net, layers


def _make_permuted_layers_net() -> tuple[Net, dict[str, Layer]]:
    """Build the ResNet block with a non-topological body order."""
    net, layers = _make_resnet_block_net()
    permuted = [
        layers["input"],
        layers["input_spec"],
        layers["add"],
        layers["relu"],
        layers["dense_b"],
        layers["dense_a"],
        layers["assert"],
    ]
    return Net(layers=permuted, preds=net.preds, succs=net.succs), layers


def test_exact_linear_residual() -> None:
    """y = Wx + x + b (no ReLU): dual bound must be a SOUND lower bound on analytic min over input box."""
    torch.manual_seed(0)
    net, layers = _make_linear_residual_net()
    lb = _t([-1.0, -1.0, -1.0])
    ub = _t([1.0, 1.0, 1.0])
    bounds = compute_forward_bounds(net, lb.unsqueeze(0), ub.unsqueeze(0))
    c = _t([[1.0, 0.5, -0.3]])
    dual_bound = _bound_value(DualSolver(DualTF()), net, bounds, c)

    W = cast(torch.Tensor, layers["dense"].params["weight"])
    b = cast(torch.Tensor, layers["dense"].params["bias"])
    W_plus_I = W + torch.eye(3, dtype=W.dtype, device=W.device)
    coef = W_plus_I.T @ c.squeeze(0)
    c_dot_b = (c.squeeze(0) * b).sum().item()
    analytic = c_dot_b + (lb * coef.clamp(min=0)).sum().item() + (ub * coef.clamp(max=0)).sum().item()

    assert dual_bound <= analytic + 1e-5, (
        f"Unsound linear residual bound: dual={dual_bound:.6f} > analytic={analytic:.6f} + tol"
    )
    print(f"\n  linear residual: dual={dual_bound:.6f}  analytic={analytic:.6f}  gap={analytic - dual_bound:.6f}")


def test_relu_cancellation() -> None:
    """ReLU fan-out to +coef / -coef branches merged by ADD should cancel exactly."""
    torch.manual_seed(0)
    net, _layers = _make_relu_cancellation_net()
    lb = _t([-1.0, -1.0])
    ub = _t([1.0, 1.0])
    bounds = compute_forward_bounds(net, lb.unsqueeze(0), ub.unsqueeze(0))
    dual_bound = _bound_value(DualSolver(DualTF()), net, bounds, _t([[1.0, 0.0]]))
    analytic = 0.0

    assert dual_bound >= analytic - 1e-4, (
        f"ReLU cancellation bound too loose: dual={dual_bound:.6f}, "
        f"expected ≥ {analytic - 1e-4:.6f} (accumulate-then-clamp must capture shared-neuron cancellation)"
    )
    assert dual_bound <= analytic + 1e-5, (
        f"Unsound ReLU cancellation bound: dual={dual_bound:.6f} > analytic={analytic:.6f}"
    )
    print(f"\n  relu cancellation: dual={dual_bound:.6f}  analytic={analytic:.6f}")


def test_resnet_sampled_soundness() -> None:
    """For a ResNet block, sample N=500 random inputs; assert dual_bound ≤ min(c·f(x))."""
    torch.manual_seed(0)
    net, _layers = _make_resnet_block_net()
    lb_1d = _t([-0.5, -0.5, -0.5])
    ub_1d = _t([0.5, 0.5, 0.5])
    model = ACTToTorch(net).run()
    model.eval()
    c_1d = _t([1.0, -0.5, 0.3])
    N = 500
    samples = lb_1d + torch.rand(N, 3, dtype=get_default_dtype(), device=get_default_device()) * (ub_1d - lb_1d)

    with torch.no_grad():
        outputs = model(samples)
        if isinstance(outputs, dict):
            outputs = outputs.get("output", list(outputs.values())[0])
        if outputs.dim() > 2:
            outputs = outputs.flatten(start_dim=1)

    concrete = (outputs * c_1d).sum(dim=-1)
    sampled_min = concrete.min().item()
    bounds = compute_forward_bounds(net, lb_1d.unsqueeze(0), ub_1d.unsqueeze(0))
    dual_bound = _bound_value(DualSolver(DualTF()), net, bounds, c_1d.unsqueeze(0))

    assert dual_bound <= sampled_min + 1e-4, (
        f"Unsound ResNet bound: dual={dual_bound:.6f} > sampled_min={sampled_min:.6f} over N={N} samples"
    )
    print(f"\n  resnet sampled: dual={dual_bound:.6f}  sampled_min={sampled_min:.6f}  N={N}")


def test_batched_equivalence() -> None:
    """B=3 batched compute_bound equals stacking 3 singleton runs."""
    torch.manual_seed(0)
    net, _layers = _make_resnet_block_net()
    boxes = [
        (_t([-0.5, -0.5, -0.5]), _t([0.5, 0.5, 0.5])),
        (_t([-0.3, -0.4, -0.2]), _t([0.4, 0.3, 0.5])),
        (_t([-0.1, -0.6, 0.0]), _t([0.2, 0.1, 0.7])),
    ]
    cs = [_t([1.0, -0.5, 0.3]), _t([0.2, 0.8, -0.1]), _t([-0.3, 0.4, 0.7])]
    solver = DualSolver(DualTF())

    singleton_bounds = []
    for (lb, ub), c_i in zip(boxes, cs):
        bounds_i = compute_forward_bounds(net, lb.unsqueeze(0), ub.unsqueeze(0))
        singleton_bounds.append(_bound_value(solver, net, bounds_i, c_i.unsqueeze(0)))

    lb_batched = torch.stack([lb for lb, _ in boxes])
    ub_batched = torch.stack([ub for _, ub in boxes])
    c_batched = torch.stack(cs)
    bounds = compute_forward_bounds(net, lb_batched, ub_batched)
    batched_bounds = cast(torch.Tensor, solver.compute_bound(net, bounds, c_batched, return_sce=False))

    for i, sb in enumerate(singleton_bounds):
        bb = batched_bounds[i].item()
        assert abs(bb - sb) < 1e-4, f"Batched[{i}] != singleton[{i}]: batched={bb:.6f} vs singleton={sb:.6f}"


def test_topo_robustness() -> None:
    """Permuting net.layers into non-topo order still produces identical bounds."""
    torch.manual_seed(0)
    net_sorted, _layers = _make_resnet_block_net()
    net_permuted, _ = _make_permuted_layers_net()
    sorted_ids = [L.id for L in net_sorted.layers]
    permuted_ids = [L.id for L in net_permuted.layers]
    assert sorted(sorted_ids) == sorted(permuted_ids), "permuted net has different layer IDs"
    assert sorted_ids != permuted_ids, "permuted net is actually sorted"

    lb = _t([-0.5, -0.5, -0.5])
    ub = _t([0.5, 0.5, 0.5])
    c = _t([[1.0, -0.5, 0.3]])
    solver = DualSolver(DualTF())

    bounds_sorted = compute_forward_bounds(net_sorted, lb.unsqueeze(0), ub.unsqueeze(0))
    bound_sorted = _bound_value(solver, net_sorted, bounds_sorted, c)
    bounds_permuted = compute_forward_bounds(net_permuted, lb.unsqueeze(0), ub.unsqueeze(0))
    bound_permuted = _bound_value(solver, net_permuted, bounds_permuted, c)

    assert abs(bound_sorted - bound_permuted) < 1e-5, (
        f"Topo permutation affected result: sorted={bound_sorted:.6f}, permuted={bound_permuted:.6f}"
    )


def test_forward_dag_checks() -> None:
    """Forward DAG: batched regression + bounds are finite and shape-correct on ResNet."""
    torch.manual_seed(0)
    net, _ = _make_resnet_block_net()
    B = 2
    lb = torch.full((B, 3), -0.5, dtype=get_default_dtype(), device=get_default_device())
    ub = torch.full((B, 3), 0.5, dtype=get_default_dtype(), device=get_default_device())
    bounds = compute_forward_bounds(net, lb, ub)
    for lid, b in bounds.items():
        assert b.lb.dim() >= 2, f"layer {lid}: lb not batched"
        assert b.lb.shape[0] == B, f"layer {lid}: batch dim {b.lb.shape[0]} != {B}"
        assert torch.isfinite(b.lb).all(), f"layer {lid}: non-finite lb"
        assert torch.isfinite(b.ub).all(), f"layer {lid}: non-finite ub"
        assert (b.lb <= b.ub).all(), f"layer {lid}: lb > ub"

    net_lin, layers_lin = _make_linear_residual_net()
    lb_lin = _t([[-1.0, -1.0, -1.0]])
    ub_lin = _t([[1.0, 1.0, 1.0]])
    bounds_lin = compute_forward_bounds(net_lin, lb_lin, ub_lin)
    add_bounds = bounds_lin[layers_lin["add"].id]
    assert add_bounds.lb.shape == (1, 3), f"ADD bounds shape {add_bounds.lb.shape} != (1, 3)"
    assert torch.isfinite(add_bounds.lb).all() and torch.isfinite(add_bounds.ub).all()
