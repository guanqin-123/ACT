#===- act/back_end/test_forward_bounds.py - Batched forward bounds tests ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnusedFunction=false, reportUntypedFunctionDecorator=false

"""Contract tests for the batched-native compute_forward_bounds() API."""

from __future__ import annotations

import pytest
import torch

from act.back_end.core import Layer, Net
from act.back_end.dual_tf import compute_forward_bounds
from act.back_end.layer_schema import LayerKind
from act.util.device_manager import get_default_device, get_default_dtype, initialize_device


@pytest.fixture(scope="module", autouse=True)
def _cpu_float32() -> None:
    """Run these contract tests on a stable CPU/float32 configuration."""
    initialize_device("cpu", "float32")


def _t(data: object) -> torch.Tensor:
    """Create a tensor on the configured default device and dtype."""
    return torch.tensor(data, dtype=get_default_dtype(), device=get_default_device())


def _box_dense(weight: torch.Tensor, bias: torch.Tensor | None, lb: torch.Tensor, ub: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-aware interval propagation for a dense layer."""
    w_pos = weight.clamp(min=0)
    w_neg = weight.clamp(max=0)
    out_lb = lb @ w_pos.transpose(0, 1) + ub @ w_neg.transpose(0, 1)
    out_ub = ub @ w_pos.transpose(0, 1) + lb @ w_neg.transpose(0, 1)
    if bias is not None:
        out_lb = out_lb + bias
        out_ub = out_ub + bias
    return out_lb, out_ub


def _make_mlp() -> tuple[Net, dict[str, Layer]]:
    """Build INPUT → INPUT_SPEC → DENSE → RELU → DENSE → ASSERT."""
    w1 = _t([
        [0.8, -0.3, 0.2, 0.1],
        [-0.2, 0.6, 0.3, -0.4],
        [0.5, 0.1, -0.7, 0.2],
        [0.0, 0.4, 0.2, 0.3],
        [-0.6, 0.2, 0.1, 0.5],
        [0.3, -0.5, 0.4, 0.2],
    ])
    b1 = _t([0.1, -0.2, 0.0, 0.2, -0.1, 0.05])
    w2 = _t([[0.4, -0.1, 0.3, 0.2, -0.5, 0.6], [-0.3, 0.5, 0.1, -0.2, 0.4, 0.2]])
    b2 = _t([0.0, 0.15])
    layers = {
        "input": Layer(0, LayerKind.INPUT.value, {"shape": [4], "dtype": "float32", "num_classes": 1, "value_range": [0.0, 1.0]}, [0, 1, 2, 3], [0, 1, 2, 3]),
        "input_spec": Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, [0, 1, 2, 3], [0, 1, 2, 3]),
        "dense1": Layer(2, LayerKind.DENSE.value, {"in_features": 4, "out_features": 6, "weight": w1, "bias": b1}, [0, 1, 2, 3], [10, 11, 12, 13, 14, 15]),
        "relu1": Layer(3, LayerKind.RELU.value, {}, [10, 11, 12, 13, 14, 15], [10, 11, 12, 13, 14, 15]),
        "dense2": Layer(4, LayerKind.DENSE.value, {"in_features": 6, "out_features": 2, "weight": w2, "bias": b2}, [10, 11, 12, 13, 14, 15], [20, 21]),
        "assert": Layer(5, LayerKind.ASSERT.value, {"kind": "RANGE"}, [20, 21], [20, 21]),
    }
    net = Net(
        layers=list(layers.values()),
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )
    return net, layers


def _make_cnn() -> tuple[Net, dict[str, Layer]]:
    """Build INPUT → INPUT_SPEC → CONV2D → RELU → FLATTEN → DENSE → ASSERT."""
    conv_w = _t([
        [[[0.2, -0.1, 0.0], [0.1, 0.3, -0.2], [0.0, 0.2, 0.1]]],
        [[[-0.3, 0.2, 0.1], [0.0, 0.1, 0.2], [0.2, -0.1, 0.3]]],
    ])
    conv_b = _t([0.05, -0.1])
    dense_w = torch.arange(64, dtype=get_default_dtype(), device=get_default_device()).reshape(2, 32) / 100.0 - 0.2
    dense_b = _t([0.1, -0.05])
    layers = {
        "input": Layer(10, LayerKind.INPUT.value, {"shape": [1, 4, 4], "dtype": "float32", "num_classes": 1, "value_range": [0.0, 1.0]}, list(range(16)), list(range(16))),
        "input_spec": Layer(11, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, list(range(16)), list(range(16))),
        "conv": Layer(12, LayerKind.CONV2D.value, {"in_channels": 1, "out_channels": 2, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "weight": conv_w, "bias": conv_b, "input_shape": (1, 1, 4, 4)}, list(range(16)), list(range(100, 132))),
        "relu": Layer(13, LayerKind.RELU.value, {}, list(range(100, 132)), list(range(100, 132))),
        "flatten": Layer(14, LayerKind.FLATTEN.value, {"start_dim": 1}, list(range(100, 132)), list(range(100, 132))),
        "dense": Layer(15, LayerKind.DENSE.value, {"in_features": 32, "out_features": 2, "weight": dense_w, "bias": dense_b}, list(range(100, 132)), [200, 201]),
        "assert": Layer(16, LayerKind.ASSERT.value, {"kind": "RANGE"}, [200, 201], [200, 201]),
    }
    net = Net(
        layers=list(layers.values()),
        preds={10: [], 11: [10], 12: [11], 13: [12], 14: [13], 15: [14], 16: [15]},
        succs={10: [11], 11: [12], 12: [13], 13: [14], 14: [15], 15: [16], 16: []},
    )
    return net, layers


def _make_add_net() -> tuple[Net, dict[str, Layer]]:
    """Build INPUT → INPUT_SPEC → DENSE_A, then ADD(INPUT_SPEC, DENSE_A) → ASSERT."""
    weight = _t([[0.5, -0.2, 0.3, 0.1], [-0.4, 0.6, 0.2, -0.1], [0.1, 0.2, -0.5, 0.4], [0.3, 0.0, 0.2, -0.6]])
    bias = _t([0.1, -0.15, 0.05, 0.2])
    layers = {
        "input": Layer(20, LayerKind.INPUT.value, {"shape": [4], "dtype": "float32", "num_classes": 1, "value_range": [0.0, 1.0]}, [0, 1, 2, 3], [0, 1, 2, 3]),
        "input_spec": Layer(21, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, [0, 1, 2, 3], [0, 1, 2, 3]),
        "dense_a": Layer(22, LayerKind.DENSE.value, {"in_features": 4, "out_features": 4, "weight": weight, "bias": bias}, [0, 1, 2, 3], [30, 31, 32, 33]),
        "add": Layer(23, LayerKind.ADD.value, {}, [0, 1, 2, 3, 30, 31, 32, 33], [40, 41, 42, 43]),
        "assert": Layer(24, LayerKind.ASSERT.value, {"kind": "RANGE"}, [40, 41, 42, 43], [40, 41, 42, 43]),
    }
    net = Net(
        layers=list(layers.values()),
        preds={20: [], 21: [20], 22: [21], 23: [21, 22], 24: [23]},
        succs={20: [21], 21: [22, 23], 22: [23], 23: [24], 24: []},
    )
    return net, layers


def test_singleton_compat_mlp() -> None:
    """A singleton input box should still produce batched [1, *] outputs."""
    net, layers = _make_mlp()
    bounds = compute_forward_bounds(net, _t([-0.2, 0.1, -0.4, 0.0]), _t([0.3, 0.5, 0.2, 0.4]))
    for layer in net.layers[1:]:
        assert bounds[layer.id].lb.dim() == 2, f"layer {layer.kind} should expose batched lower bounds for singleton input"
        assert bounds[layer.id].ub.dim() == 2, f"layer {layer.kind} should expose batched upper bounds for singleton input"
    assert bounds[layers["dense1"].id].lb.shape == (1, 6), "first dense layer should return shape (1, 6)"
    assert bounds[layers["dense2"].id].lb.shape == (1, 2), "second dense layer should return shape (1, 2)"
    assert bool(torch.isfinite(bounds[layers["dense2"].id].lb).all()), "final lower bounds must be finite"
    assert bool(torch.isfinite(bounds[layers["dense2"].id].ub).all()), "final upper bounds must be finite"


def test_batched_equals_stacked_mlp() -> None:
    """Batched propagation must match stacking per-instance singleton runs."""
    net, _layers = _make_mlp()
    boxes = [(_t([-0.4, 0.1, -0.1, 0.0]), _t([0.0, 0.4, 0.2, 0.3])), (_t([0.1, -0.5, 0.2, -0.2]), _t([0.5, -0.1, 0.6, 0.1])), (_t([-0.2, -0.1, -0.3, 0.4]), _t([0.2, 0.3, 0.1, 0.8]))]
    singles = [compute_forward_bounds(net, lb.unsqueeze(0), ub.unsqueeze(0)) for lb, ub in boxes]
    batched = compute_forward_bounds(net, torch.stack([lb for lb, _ in boxes]), torch.stack([ub for _, ub in boxes]))
    for lid in batched:
        for i in range(len(boxes)):
            torch.testing.assert_close(batched[lid].lb[i], singles[i][lid].lb[0], rtol=1e-5, atol=1e-6, msg=f"layer {lid} lower bounds should equal stacked singleton result for batch index {i}")
            torch.testing.assert_close(batched[lid].ub[i], singles[i][lid].ub[0], rtol=1e-5, atol=1e-6, msg=f"layer {lid} upper bounds should equal stacked singleton result for batch index {i}")


def test_shape_invariant() -> None:
    """Every returned Bounds entry should preserve batch as the leading dimension."""
    for batch_size in [1, 2, 4]:
        net, _layers = _make_mlp()
        lb = torch.linspace(-0.6, 0.1, steps=batch_size * 4, dtype=get_default_dtype(), device=get_default_device()).reshape(batch_size, 4)
        ub = lb + 0.5
        bounds = compute_forward_bounds(net, lb, ub)
        for lid, layer_bounds in bounds.items():
            assert layer_bounds.lb.dim() >= 2, f"layer {lid} lower bounds should keep batch rank for B={batch_size}"
            assert layer_bounds.lb.shape[0] == batch_size, f"layer {lid} lower bounds should keep leading batch dimension for B={batch_size}"
            assert layer_bounds.lb.shape == layer_bounds.ub.shape, f"layer {lid} lower/upper bounds should have identical shapes"
            assert bool((layer_bounds.lb <= layer_bounds.ub).all()), f"layer {lid} must satisfy lb <= ub elementwise"


def test_conv2d_batched() -> None:
    """Conv2D bounds must preserve the leading batch dim; feature dims are flat (C*H*W)."""
    net, layers = _make_cnn()
    for batch_size in [2, 1]:
        lb = torch.linspace(-0.3, 0.2, steps=batch_size * 16, dtype=get_default_dtype(), device=get_default_device()).reshape(batch_size, 1, 4, 4)
        ub = lb + 0.4
        bounds = compute_forward_bounds(net, lb, ub)
        # LinearBound stores bounds as [B, out_dim] with out_dim=C*H*W flat
        # (the refactor does not reshape spatial structure back into [B,C,H,W]).
        conv_lb = bounds[layers["conv"].id].lb
        conv_ub = bounds[layers["conv"].id].ub
        assert conv_lb.dim() >= 2, f"conv lower bounds must preserve batch dim (got dim={conv_lb.dim()})"
        assert conv_lb.shape[0] == batch_size, f"conv first dim must be batch_size={batch_size}, got {conv_lb.shape[0]}"
        assert conv_lb.flatten(start_dim=1).shape == (batch_size, 32), f"conv feature volume must flatten to 32=C*H*W, got {conv_lb.flatten(start_dim=1).shape}"
        assert conv_lb.shape == conv_ub.shape, "conv lb/ub shapes must match"
        assert bool((conv_lb <= conv_ub).all()), "conv bounds must satisfy lb <= ub"
        final_lb = bounds[layers["dense"].id].lb
        assert final_lb.shape == (batch_size, 2), f"final dense layer must return ({batch_size}, 2), got {final_lb.shape}"


def test_relu_alpha_broadcast() -> None:
    """Per-batch ReLU relaxations must not collapse distinct batches into one alpha choice."""
    weight = _t([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.3, -0.3]])
    bias = _t([0.2, 0.2, 0.2, 0.2])
    layers = [
        Layer(30, LayerKind.INPUT.value, {"shape": [2], "dtype": "float32", "num_classes": 1, "value_range": [0.0, 1.0]}, [0, 1], [0, 1]),
        Layer(31, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, [0, 1], [0, 1]),
        Layer(32, LayerKind.DENSE.value, {"in_features": 2, "out_features": 4, "weight": weight, "bias": bias}, [0, 1], [10, 11, 12, 13]),
        Layer(33, LayerKind.RELU.value, {}, [10, 11, 12, 13], [10, 11, 12, 13]),
        Layer(34, LayerKind.ASSERT.value, {"kind": "RANGE"}, [10, 11, 12, 13], [10, 11, 12, 13]),
    ]
    net = Net(layers=layers, preds={30: [], 31: [30], 32: [31], 33: [32], 34: [33]}, succs={30: [31], 31: [32], 32: [33], 33: [34], 34: []})
    lb = _t([[0.0, 0.0], [-1.0, -1.0]])
    ub = _t([[0.1, 0.1], [1.0, 1.0]])
    bounds = compute_forward_bounds(net, lb, ub, post_activation=True)
    relu_bounds = bounds[33]
    assert relu_bounds.lb[0].shape == (4,), "ReLU batch element 0 should preserve feature shape (4,)"
    assert relu_bounds.lb[1].shape == (4,), "ReLU batch element 1 should preserve feature shape (4,)"
    assert bool((relu_bounds.lb[0] >= 0).all()), "post-activation ReLU lower bounds must be non-negative for batch 0"
    assert not torch.equal(relu_bounds.ub[0], relu_bounds.ub[1]), "ReLU upper bounds should differ across batches when pre-activation boxes differ"


def test_add_dual_track_identity() -> None:
    """ADD should remain batch-shaped, finite, and consistent for shared-frame predecessors."""
    net, layers = _make_add_net()
    lb = _t([[-0.4, 0.1, -0.2, 0.0], [0.2, -0.3, 0.1, -0.5]])
    ub = _t([[0.2, 0.6, 0.3, 0.4], [0.7, 0.2, 0.5, -0.1]])
    bounds = compute_forward_bounds(net, lb, ub)
    dense = layers["dense_a"]
    expected_dense_lb, expected_dense_ub = _box_dense(dense.params["weight"], dense.params["bias"], lb, ub)
    expected_add_lb = lb + expected_dense_lb
    expected_add_ub = ub + expected_dense_ub
    add_bounds = bounds[layers["add"].id]
    assert add_bounds.lb.shape == (2, 4), "ADD lower bounds should preserve shape (B, 4)"
    assert bool(torch.isfinite(add_bounds.lb).all() and torch.isfinite(add_bounds.ub).all()), "ADD bounds must be finite"
    assert bool((add_bounds.lb <= add_bounds.ub).all()), "ADD bounds must satisfy lb <= ub"
    assert bool((add_bounds.lb >= expected_add_lb - 1e-6).all()), "ADD lower bounds should be at least as tight as interval lower bounds"
    assert bool((add_bounds.ub <= expected_add_ub + 1e-6).all()), "ADD upper bounds should be at least as tight as interval upper bounds"
