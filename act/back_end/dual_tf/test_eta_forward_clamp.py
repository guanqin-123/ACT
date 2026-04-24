#===- act/back_end/dual_tf/test_eta_forward_clamp.py - eta clamp tests ===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnusedFunction=false, reportUntypedFunctionDecorator=false

"""Tests for the eta pre-activation clamp spliced into compute_forward_bounds.

Covers:
  (a) bit-identity when eta_state is None
  (b) bit-identity when eta_state.fast_path_skip() is True (all signs == 0)
  (c) inactive clamp  (sign = +1, z <= point)  -> ReLU output == 0
  (d) active   clamp  (sign = -1, z >= point)  -> pre-activation lb >= point
  (e) smooth  clamp on TANH pre-activation upper bound
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import pytest
import torch

from act.back_end.bab.eta import EtaState
from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf import compute_forward_bounds
from act.back_end.layer_schema import LayerKind
from act.util.device_manager import get_default_device, get_default_dtype, initialize_device


@pytest.fixture(scope="module", autouse=True)
def _cpu_float32() -> None:
    initialize_device("cpu", "float32")


def _t(data: object) -> torch.Tensor:
    return torch.tensor(data, dtype=get_default_dtype(), device=get_default_device())


def _make_clamp_mlp() -> tuple[Net, dict[str, Layer], int]:
    """INPUT → INPUT_SPEC → DENSE(2→2, identity) → RELU → DENSE(2→1) → ASSERT.

    Identity weights on the hidden layer let us reason about neuron `j` of
    the hidden DENSE directly from input `j`. Returns (net, named_layers,
    hidden_dense_id) for test setup.
    """
    w1 = _t([[1.0, 0.0], [0.0, 1.0]])
    b1 = _t([0.0, 0.0])
    w2 = _t([[1.0, 1.0]])
    b2 = _t([0.0])
    layers = {
        "input":      Layer(0, LayerKind.INPUT.value,      {"shape": [2], "dtype": "float32", "num_classes": 1, "value_range": [-1.0, 1.0]}, [0, 1], [0, 1]),
        "input_spec": Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, [0, 1], [0, 1]),
        "dense1":     Layer(2, LayerKind.DENSE.value,      {"in_features": 2, "out_features": 2, "weight": w1, "bias": b1}, [0, 1], [10, 11]),
        "relu":       Layer(3, LayerKind.RELU.value,       {}, [10, 11], [10, 11]),
        "dense2":     Layer(4, LayerKind.DENSE.value,      {"in_features": 2, "out_features": 1, "weight": w2, "bias": b2}, [10, 11], [20]),
        "assert":     Layer(5, LayerKind.ASSERT.value,     {"kind": "RANGE"}, [20], [20]),
    }
    net = Net(
        layers=list(layers.values()),
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )
    return net, layers, layers["dense1"].id


def _make_tanh_mlp() -> tuple[Net, dict[str, Layer], int]:
    """INPUT → INPUT_SPEC → DENSE(2→2, identity) → TANH → ASSERT."""
    w1 = _t([[1.0, 0.0], [0.0, 1.0]])
    b1 = _t([0.0, 0.0])
    layers = {
        "input":      Layer(100, LayerKind.INPUT.value,      {"shape": [2], "dtype": "float32", "num_classes": 1, "value_range": [-1.0, 1.0]}, [0, 1], [0, 1]),
        "input_spec": Layer(101, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, [0, 1], [0, 1]),
        "dense":      Layer(102, LayerKind.DENSE.value,      {"in_features": 2, "out_features": 2, "weight": w1, "bias": b1}, [0, 1], [10, 11]),
        "tanh":       Layer(103, LayerKind.TANH.value,       {}, [10, 11], [10, 11]),
        "assert":     Layer(104, LayerKind.ASSERT.value,     {"kind": "RANGE"}, [10, 11], [10, 11]),
    }
    net = Net(
        layers=list(layers.values()),
        preds={100: [], 101: [100], 102: [101], 103: [102], 104: [103]},
        succs={100: [101], 101: [102], 102: [103], 103: [104], 104: []},
    )
    return net, layers, layers["dense"].id


def _zero_eta(dense_id: int, B: int, width: int) -> EtaState:
    shape = (B, width)
    return EtaState(
        val={dense_id: torch.zeros(shape, dtype=get_default_dtype(), device=get_default_device())},
        sign={dense_id: torch.zeros(shape, dtype=get_default_dtype(), device=get_default_device())},
        point={dense_id: torch.zeros(shape, dtype=get_default_dtype(), device=get_default_device())},
    )


def _assert_bounds_equal(a_bounds: dict[int, Bounds], b_bounds: dict[int, Bounds],
                         layer_ids: Iterable[int]) -> None:
    for lid in layer_ids:
        assert torch.equal(a_bounds[lid].lb, b_bounds[lid].lb), f"layer {lid} lb differs"
        assert torch.equal(a_bounds[lid].ub, b_bounds[lid].ub), f"layer {lid} ub differs"


def test_compute_forward_bounds_eta_passthrough() -> None:
    net, _layers, _dense_id = _make_clamp_mlp()
    lb = _t([[-1.0, -1.0]])
    ub = _t([[1.0, 1.0]])
    baseline = compute_forward_bounds(net, lb, ub)
    with_none = compute_forward_bounds(net, lb, ub, eta_state=None)
    assert set(baseline.keys()) == set(with_none.keys())
    _assert_bounds_equal(baseline, with_none, baseline.keys())


def test_compute_forward_bounds_eta_empty() -> None:
    net, _layers, dense_id = _make_clamp_mlp()
    lb = _t([[-1.0, -1.0]])
    ub = _t([[1.0, 1.0]])
    eta = _zero_eta(dense_id, B=1, width=2)
    assert eta.fast_path_skip(), "zero-sign eta_state must enable the fast path"
    baseline = compute_forward_bounds(net, lb, ub)
    with_empty = compute_forward_bounds(net, lb, ub, eta_state=eta)
    _assert_bounds_equal(baseline, with_empty, baseline.keys())


def test_forward_clamp_inactive() -> None:
    net, layers, dense_id = _make_clamp_mlp()
    lb = _t([[-1.0, -1.0]])
    ub = _t([[1.0, 1.0]])

    eta = _zero_eta(dense_id, B=1, width=2)
    eta.sign[dense_id][0, 0] = 1.0
    eta.point[dense_id][0, 0] = 0.0

    bounds = compute_forward_bounds(net, lb, ub, post_activation=True, eta_state=eta)

    dense_lb = bounds[dense_id].lb[0]
    dense_ub = bounds[dense_id].ub[0]
    assert dense_ub[0].item() <= 0.0 + 1e-7, f"DENSE pre-activation ub[0] must be clamped to 0, got {dense_ub[0].item()}"
    assert dense_ub[1].item() == pytest.approx(1.0, abs=1e-6), "untouched neuron 1 must retain ub=1"
    assert dense_lb[0].item() == pytest.approx(-1.0, abs=1e-6), "clamp must leave lb untouched (sign>0)"

    relu_lb = bounds[layers["relu"].id].lb[0]
    relu_ub = bounds[layers["relu"].id].ub[0]
    assert relu_ub[0].item() <= 0.0 + 1e-7, f"ReLU post-activation ub[0] must be 0 when z<=0, got {relu_ub[0].item()}"
    assert relu_lb[0].item() >= -1e-7, "ReLU post-activation lb always non-negative"
    assert relu_ub[1].item() == pytest.approx(1.0, abs=1e-6), "untouched neuron 1 post-activation retains ub=1"


def test_forward_clamp_active() -> None:
    net, layers, dense_id = _make_clamp_mlp()
    lb = _t([[-1.0, -1.0]])
    ub = _t([[1.0, 1.0]])

    baseline = compute_forward_bounds(net, lb, ub, post_activation=True)
    baseline_dense_lb0 = baseline[dense_id].lb[0, 0].item()
    assert baseline_dense_lb0 == pytest.approx(-1.0, abs=1e-6), "sanity: unclamped DENSE lb[0] is -1"

    eta = _zero_eta(dense_id, B=1, width=2)
    eta.sign[dense_id][0, 0] = -1.0
    eta.point[dense_id][0, 0] = 0.0

    bounds = compute_forward_bounds(net, lb, ub, post_activation=True, eta_state=eta)

    dense_lb = bounds[dense_id].lb[0]
    dense_ub = bounds[dense_id].ub[0]
    assert dense_lb[0].item() >= 0.0 - 1e-7, f"DENSE pre-activation lb[0] must be clamped to >= 0, got {dense_lb[0].item()}"
    assert dense_lb[1].item() == pytest.approx(-1.0, abs=1e-6), "untouched neuron 1 must retain lb=-1"
    assert dense_ub[0].item() == pytest.approx(1.0, abs=1e-6), "clamp must leave ub untouched (sign<0)"
    assert dense_lb[0].item() > baseline_dense_lb0, "active clamp must strictly raise pre-activation lb"

    relu_lb = bounds[layers["relu"].id].lb[0]
    assert relu_lb[0].item() >= 0.0 - 1e-7, "ReLU post-activation lb must be non-negative"


def test_forward_clamp_smooth_tanh() -> None:
    net, layers, dense_id = _make_tanh_mlp()
    lb = _t([[-1.0, -1.0]])
    ub = _t([[1.0, 1.0]])

    baseline = compute_forward_bounds(net, lb, ub, post_activation=True)
    baseline_tanh_ub0 = baseline[layers["tanh"].id].ub[0, 0].item()
    assert baseline_tanh_ub0 == pytest.approx(math.tanh(1.0), abs=5e-5), \
        f"sanity: unclamped TANH ub[0] ≈ tanh(1) but got {baseline_tanh_ub0}"

    midpoint = 0.0
    eta = _zero_eta(dense_id, B=1, width=2)
    eta.sign[dense_id][0, 0] = 1.0
    eta.point[dense_id][0, 0] = midpoint

    bounds = compute_forward_bounds(net, lb, ub, post_activation=True, eta_state=eta)

    dense_ub = bounds[dense_id].ub[0]
    assert dense_ub[0].item() <= midpoint + 1e-7, f"DENSE pre-activation ub[0] must be clamped to {midpoint}, got {dense_ub[0].item()}"
    assert dense_ub[1].item() == pytest.approx(1.0, abs=1e-6), "untouched neuron 1 must retain pre-activation ub=1"

    tanh_ub0 = bounds[layers["tanh"].id].ub[0, 0].item()
    assert tanh_ub0 < baseline_tanh_ub0 - 1e-4, \
        f"clamped TANH ub[0]={tanh_ub0} must be strictly tighter than baseline {baseline_tanh_ub0}"
    assert tanh_ub0 <= math.tanh(midpoint) + 5e-5, \
        f"TANH post-activation ub must respect tanh(midpoint)={math.tanh(midpoint)}, got {tanh_ub0}"
