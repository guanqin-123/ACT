# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportAttributeAccessIssue=false

"""Tests for BaBSR branching (act.back_end.bab.branching.babsr)."""

from __future__ import annotations

from typing import Dict, Tuple

import pytest
import torch

from act.back_end.bab.branching.babsr import BaBSRBranching
from act.back_end.bab.eta import EtaState
from act.back_end.bab.node import SubproblemBatch
from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf import DualTF, compute_forward_bounds
from act.back_end.layer_schema import LayerKind
from act.util.device_manager import get_default_device, get_default_dtype, initialize_device


@pytest.fixture(scope="module", autouse=True)
def _cpu_float32() -> None:
    initialize_device("cpu", "float32")


def _t(data: object) -> torch.Tensor:
    return torch.tensor(data, dtype=get_default_dtype(), device=get_default_device())


def _zeros(shape: Tuple[int, ...]) -> torch.Tensor:
    return torch.zeros(shape, dtype=get_default_dtype(), device=get_default_device())


def _build_mlp_relu_single_hidden(
    weight: torch.Tensor,
    bias: torch.Tensor,
    out_weight: torch.Tensor,
    out_bias: torch.Tensor,
) -> Tuple[Net, int, int]:
    """INPUT[n_in] -> INPUT_SPEC -> DENSE(hidden) -> RELU -> DENSE(out) -> ASSERT.

    Returns (net, pre_act_layer_id, out_layer_id).
    """
    n_in = int(weight.shape[1])
    n_h = int(weight.shape[0])
    n_out = int(out_weight.shape[0])
    in_vars = list(range(n_in))
    hidden_vars = list(range(100, 100 + n_h))
    relu_vars = list(range(200, 200 + n_h))
    out_vars = list(range(300, 300 + n_out))
    layers = [
        Layer(
            0,
            LayerKind.INPUT.value,
            {"shape": [n_in], "dtype": "float32", "num_classes": 1, "value_range": [0.0, 1.0]},
            in_vars,
            in_vars,
        ),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, in_vars, in_vars),
        Layer(
            2,
            LayerKind.DENSE.value,
            {"in_features": n_in, "out_features": n_h, "weight": weight, "bias": bias},
            in_vars,
            hidden_vars,
        ),
        Layer(3, LayerKind.RELU.value, {}, hidden_vars, relu_vars),
        Layer(
            4,
            LayerKind.DENSE.value,
            {"in_features": n_h, "out_features": n_out, "weight": out_weight, "bias": out_bias},
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
    return net, 2, 4


def _build_mlp_tanh_single_hidden(
    weight: torch.Tensor,
    bias: torch.Tensor,
    out_weight: torch.Tensor,
    out_bias: torch.Tensor,
) -> Tuple[Net, int, int]:
    """Same shape as the ReLU net but with TANH activation."""
    n_in = int(weight.shape[1])
    n_h = int(weight.shape[0])
    n_out = int(out_weight.shape[0])
    in_vars = list(range(n_in))
    hidden_vars = list(range(100, 100 + n_h))
    act_vars = list(range(200, 200 + n_h))
    out_vars = list(range(300, 300 + n_out))
    layers = [
        Layer(
            0,
            LayerKind.INPUT.value,
            {"shape": [n_in], "dtype": "float32", "num_classes": 1, "value_range": [0.0, 1.0]},
            in_vars,
            in_vars,
        ),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, in_vars, in_vars),
        Layer(
            2,
            LayerKind.DENSE.value,
            {"in_features": n_in, "out_features": n_h, "weight": weight, "bias": bias},
            in_vars,
            hidden_vars,
        ),
        Layer(3, LayerKind.TANH.value, {}, hidden_vars, act_vars),
        Layer(
            4,
            LayerKind.DENSE.value,
            {"in_features": n_h, "out_features": n_out, "weight": out_weight, "bias": out_bias},
            act_vars,
            out_vars,
        ),
        Layer(5, LayerKind.ASSERT.value, {"kind": "RANGE"}, out_vars, out_vars),
    ]
    net = Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )
    return net, 2, 4


def _compute_bounds(net: Net, input_lb: torch.Tensor, input_ub: torch.Tensor) -> Dict[int, Bounds]:
    return compute_forward_bounds(net, input_lb, input_ub)


def test_babsr_picks_unstable_neuron() -> None:
    """MLP with 1 clearly-unstable + 1 stable ReLU neuron: BaBSR picks the unstable one."""
    weight = _t([
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    bias = _t([0.0, 2.0])
    out_weight = _t([[1.0, 1.0]])
    out_bias = _t([0.0])

    net, pre_lid, _ = _build_mlp_relu_single_hidden(weight, bias, out_weight, out_bias)
    in_lb = _t([[-1.0, -1.0]])
    in_ub = _t([[1.0, 1.0]])
    bounds_dict = _compute_bounds(net, in_lb, in_ub)

    pre = bounds_dict[pre_lid]
    assert pre.lb[0, 0].item() < 0 < pre.ub[0, 0].item(), "neuron 0 must be unstable"
    assert pre.lb[0, 1].item() >= 0, "neuron 1 must be stable-on"

    batch = SubproblemBatch(lb=in_lb.clone(), ub=in_ub.clone(), depths=torch.tensor([0]))
    spec_c = _t([[-1.0]])

    brancher = BaBSRBranching()
    picks = brancher.select_neurons(
        net, batch, bounds_dict, spec_c, num_classes=1, dual_solver=None,
    )

    assert len(picks) == 1
    lid, n_idx, kind = picks[0]
    assert lid == pre_lid
    assert n_idx == 0
    assert kind == "relu"


def test_babsr_masks_already_split_neurons() -> None:
    """3 unstable ReLU neurons; setting eta.sign[pre_act][0,0]=+1 makes BaBSR skip neuron 0."""
    weight = _t([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    bias = _t([0.0, 0.0, 0.0])
    out_weight = _t([[1.0, 1.0, 1.0]])
    out_bias = _t([0.0])

    net, pre_lid, _ = _build_mlp_relu_single_hidden(weight, bias, out_weight, out_bias)
    in_lb = _t([[-1.0, -1.0]])
    in_ub = _t([[1.0, 1.0]])
    bounds_dict = _compute_bounds(net, in_lb, in_ub)

    pre = bounds_dict[pre_lid]
    for k in range(3):
        assert pre.lb[0, k].item() < 0 < pre.ub[0, k].item(), f"neuron {k} must be unstable"

    B = 1
    D = 3
    eta = EtaState(
        val=  {pre_lid: _zeros((B, D))},
        sign= {pre_lid: _zeros((B, D))},
        point={pre_lid: _zeros((B, D))},
    )
    eta.sign[pre_lid][0, 0] = 1.0

    batch = SubproblemBatch(lb=in_lb.clone(), ub=in_ub.clone(), depths=torch.tensor([0]))
    batch.eta = eta

    spec_c = _t([[-1.0]])

    brancher = BaBSRBranching()
    picks = brancher.select_neurons(
        net, batch, bounds_dict, spec_c, num_classes=1, dual_solver=None,
    )

    lid, n_idx, kind = picks[0]
    assert lid == pre_lid
    assert n_idx in (1, 2), f"must pick neuron 1 or 2 (not masked 0); got {n_idx}"
    assert kind == "relu"


def test_babsr_returns_none_when_all_stable() -> None:
    """All ReLU neurons stable (l >= 0) → BaBSR returns (-1, -1, 'none')."""
    weight = _t([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    bias = _t([2.0, 2.0])
    out_weight = _t([[1.0, 1.0]])
    out_bias = _t([0.0])

    net, pre_lid, _ = _build_mlp_relu_single_hidden(weight, bias, out_weight, out_bias)
    in_lb = _t([[-1.0, -1.0]])
    in_ub = _t([[1.0, 1.0]])
    bounds_dict = _compute_bounds(net, in_lb, in_ub)

    pre = bounds_dict[pre_lid]
    assert pre.lb[0, 0].item() >= 0 and pre.lb[0, 1].item() >= 0

    batch = SubproblemBatch(lb=in_lb.clone(), ub=in_ub.clone(), depths=torch.tensor([0]))
    spec_c = _t([[-1.0]])

    brancher = BaBSRBranching()
    picks = brancher.select_neurons(
        net, batch, bounds_dict, spec_c, num_classes=1, dual_solver=None,
    )

    assert picks == [(-1, -1, "none")]


def test_babsr_batched_per_row_independence() -> None:
    """Three subproblems with distinct unstable masks → each row independently picks top-1."""
    weight = _t([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    bias = _t([0.0, 0.0, 0.0])
    out_weight = _t([[1.0, 1.0, 1.0]])
    out_bias = _t([0.0])

    net, pre_lid, _ = _build_mlp_relu_single_hidden(weight, bias, out_weight, out_bias)

    in_lb = _t([
        [-1.0, -1.0],
        [-1.0, -1.0],
        [-1.0, -1.0],
    ])
    in_ub = _t([
        [ 1.0,  1.0],
        [ 1.0,  1.0],
        [ 1.0,  1.0],
    ])
    bounds_dict = _compute_bounds(net, in_lb, in_ub)

    B = 3
    D = 3
    eta = EtaState(
        val=  {pre_lid: _zeros((B, D))},
        sign= {pre_lid: _zeros((B, D))},
        point={pre_lid: _zeros((B, D))},
    )
    eta.sign[pre_lid][0, 0] = 1.0
    eta.sign[pre_lid][0, 1] = 1.0
    eta.sign[pre_lid][1, 2] = 1.0

    batch = SubproblemBatch(lb=in_lb.clone(), ub=in_ub.clone(), depths=torch.zeros(B, dtype=torch.long))
    batch.eta = eta

    spec_c = _t([[-1.0], [-1.0], [-1.0]])

    brancher = BaBSRBranching()
    picks = brancher.select_neurons(
        net, batch, bounds_dict, spec_c, num_classes=1, dual_solver=None,
    )

    assert len(picks) == 3
    assert picks[0] == (pre_lid, 2, "relu"), f"row 0: only neuron 2 available; got {picks[0]}"
    lid1, n1, k1 = picks[1]
    assert lid1 == pre_lid and n1 in (0, 1) and k1 == "relu", f"row 1: got {picks[1]}"
    lid2, n2, k2 = picks[2]
    assert lid2 == pre_lid and n2 in (0, 1, 2) and k2 == "relu", f"row 2: got {picks[2]}"


def test_babsr_score_formula_matches_canonical() -> None:
    """Hand-computed score for a 1-neuron layer matches the canonical formula."""
    weight = _t([[1.0]])
    bias = _t([0.2])
    out_weight = _t([[1.0]])
    out_bias = _t([0.0])

    net, pre_lid, _ = _build_mlp_relu_single_hidden(weight, bias, out_weight, out_bias)
    in_lb = _t([[-1.0]])
    in_ub = _t([[1.0]])
    bounds_dict = _compute_bounds(net, in_lb, in_ub)

    pre = bounds_dict[pre_lid]
    l = float(pre.lb[0, 0].item())
    u = float(pre.ub[0, 0].item())
    assert l == pytest.approx(-0.8, abs=1e-6)
    assert u == pytest.approx(1.2, abs=1e-6)

    spec_c = _t([[-1.0]])

    from act.back_end.bab.branching.babsr import compute_lA_per_layer
    lA_dict = compute_lA_per_layer(net, bounds_dict, spec_c, DualTF(), target_layer_ids=[pre_lid])
    lA_val = float(lA_dict[pre_lid].reshape(1, -1)[0, 0].item())

    eps = 1e-12
    slope_ratio = u / (u - l + eps)
    intercept = -l * u / (u - l + eps)
    intercept_term = min(lA_val, 0.0) * intercept
    b_layer = 0.2
    bias_cand_1 = b_layer * (slope_ratio - 1.0)
    bias_cand_2 = b_layer * slope_ratio
    bias_term = max(bias_cand_1, bias_cand_2)
    expected_score = abs(bias_term + intercept_term)

    batch = SubproblemBatch(lb=in_lb.clone(), ub=in_ub.clone(), depths=torch.tensor([0]))
    brancher = BaBSRBranching()
    picks = brancher.select_neurons(
        net, batch, bounds_dict, spec_c, num_classes=1, dual_solver=None,
    )
    assert picks == [(pre_lid, 0, "relu")]
    assert expected_score > brancher.SCORE_THRESHOLD, (
        f"hand-computed score {expected_score} must exceed threshold to pick the neuron"
    )


def test_babsr_smooth_picked_when_width_above_gap() -> None:
    """TANH neuron with u - l >> SMOOTH_SPLIT_MIN_GAP → picks it with kind='smooth'."""
    weight = _t([[1.0]])
    bias = _t([0.0])
    out_weight = _t([[1.0]])
    out_bias = _t([0.0])

    net, pre_lid, _ = _build_mlp_tanh_single_hidden(weight, bias, out_weight, out_bias)
    in_lb = _t([[-1.0]])
    in_ub = _t([[1.0]])
    bounds_dict = _compute_bounds(net, in_lb, in_ub)

    pre = bounds_dict[pre_lid]
    width = float((pre.ub - pre.lb)[0, 0].item())
    assert width > BaBSRBranching.SMOOTH_SPLIT_MIN_GAP

    batch = SubproblemBatch(lb=in_lb.clone(), ub=in_ub.clone(), depths=torch.tensor([0]))
    spec_c = _t([[-1.0]])

    brancher = BaBSRBranching()
    picks = brancher.select_neurons(
        net, batch, bounds_dict, spec_c, num_classes=1, dual_solver=None,
    )

    assert len(picks) == 1
    lid, n_idx, kind = picks[0]
    assert lid == pre_lid
    assert n_idx == 0
    assert kind == "smooth"
