from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from act.back_end.bab.branching.babsr import (
    BaBSRBranching,
    _lA_to_score_tensor,
    compute_lA_per_layer,
)
from act.back_end.bab.node import SubproblemBatch
from act.back_end.bounds_dispatch import get_conv_mode, set_conv_mode
from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf import DualTF, compute_forward_bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.patches import Patches, build_identity_patches
from act.util.device_manager import initialize_device


@pytest.fixture(scope="module", autouse=True)
def _cpu_float64() -> None:
    initialize_device("cpu", "float64")


@contextmanager
def _conv_mode(mode: str):
    prev_mode = get_conv_mode()
    set_conv_mode(mode)
    try:
        yield
    finally:
        set_conv_mode(prev_mode)


def _build_conv_relu_conv_net() -> tuple[Net, int, int]:
    input_vars = list(range(16))
    conv1_out = list(range(1000, 1016))
    relu_out = list(range(2000, 2016))
    conv2_out = list(range(3000, 3016))
    conv_params = {
        "in_channels": 1,
        "out_channels": 1,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "input_shape": (1, 1, 4, 4),
        "output_shape": (1, 1, 4, 4),
    }
    layers = [
        Layer(
            0,
            LayerKind.INPUT.value,
            {"shape": (1, 1, 4, 4), "dtype": "float64", "num_classes": 16, "value_range": (0.0, 1.0)},
            input_vars,
            input_vars,
        ),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, input_vars, input_vars),
        Layer(
            2,
            LayerKind.CONV2D.value,
            {
                **conv_params,
                "weight": torch.ones((1, 1, 1, 1), dtype=torch.float64),
                "bias": None,
            },
            input_vars,
            conv1_out,
        ),
        Layer(3, LayerKind.RELU.value, {}, conv1_out, relu_out),
        Layer(
            4,
            LayerKind.CONV2D.value,
            {
                **conv_params,
                "weight": torch.ones((1, 1, 1, 1), dtype=torch.float64),
                "bias": None,
            },
            relu_out,
            conv2_out,
        ),
        Layer(5, LayerKind.ASSERT.value, {"kind": "RANGE"}, conv2_out, conv2_out),
    ]
    net = Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )
    return net, 2, 4


def _input_box() -> tuple[torch.Tensor, torch.Tensor]:
    lb = torch.tensor(
        [[[[-0.7, -0.2, 0.2, -0.4], [-0.1, -0.6, 0.3, 0.1], [-0.5, 0.0, -0.3, 0.2], [0.1, -0.4, 0.4, -0.2]]]],
        dtype=torch.float64,
    )
    ub = torch.tensor(
        [[[[0.4, 0.5, 0.8, 0.6], [0.7, 0.2, 0.9, 0.5], [0.1, 0.6, 0.4, 0.8], [0.5, 0.3, 0.9, 0.7]]]],
        dtype=torch.float64,
    )
    return lb, ub


def _spec_c(index: int) -> torch.Tensor:
    c = torch.zeros((1, 16), dtype=torch.float64)
    c[0, index] = -1.0
    return c


def _score_tensor_for_mode(net: Net, pre_lid: int, spec_c: torch.Tensor, mode: str) -> torch.Tensor:
    lb, ub = _input_box()
    with _conv_mode(mode):
        bounds = compute_forward_bounds(net, lb, ub)
        lA = compute_lA_per_layer(net, bounds, spec_c, DualTF(), target_layer_ids=[pre_lid])
        return _lA_to_score_tensor(pre_lid, lA[pre_lid], bounds[pre_lid], net.by_id[pre_lid], 1)


def test_babsr_score_patches_vs_matrix_parity() -> None:
    net, pre_lid, _ = _build_conv_relu_conv_net()
    spec_c = _spec_c(5)
    matrix = _score_tensor_for_mode(net, pre_lid, spec_c, "matrix")
    patches = _score_tensor_for_mode(net, pre_lid, spec_c, "patches")
    torch.testing.assert_close(patches, matrix, rtol=1e-5, atol=1e-7)


def test_babsr_no_patches_to_matrix_call(monkeypatch: pytest.MonkeyPatch) -> None:
    net, _pre_lid, _ = _build_conv_relu_conv_net()
    lb, ub = _input_box()
    batch = SubproblemBatch(
        lb=lb.reshape(1, -1),
        ub=ub.reshape(1, -1),
        depths=torch.tensor([0]),
    )
    with _conv_mode("patches"):
        bounds = compute_forward_bounds(net, lb, ub)
        spec_c = _spec_c(5)

        def _forbidden(*_args, **_kwargs):
            raise AssertionError("patches_to_matrix should not run during BaBSR patches scoring")

        monkeypatch.setattr("act.back_end.patches.patches_to_matrix", _forbidden)
        picks = BaBSRBranching().select_neurons(net, batch, bounds, spec_c, num_classes=16)
    assert picks[0][0] == 2


def test_babsr_score_identity_patches() -> None:
    pre_layer = Layer(
        7,
        LayerKind.CONV2D.value,
        {
            "weight": torch.ones((1, 1, 1, 1), dtype=torch.float64),
            "in_channels": 1,
            "out_channels": 1,
            "kernel_size": 1,
            "output_shape": (1, 1, 2, 2),
            "bias": torch.tensor([0.2], dtype=torch.float64),
        },
        [],
        [],
    )
    bounds = Bounds(
        lb=torch.tensor([[-0.8, -0.4, -0.2, -0.1]], dtype=torch.float64),
        ub=torch.tensor([[0.4, 0.7, 0.5, 0.9]], dtype=torch.float64),
    )
    patches = build_identity_patches(1, 1, 2, 2, dtype=torch.float64, device=torch.device("cpu"))
    score_patches = _lA_to_score_tensor(7, patches, bounds, pre_layer, 1)
    score_matrix = torch.ones_like(score_patches)
    denom = (bounds.ub - bounds.lb).clamp(min=1e-12)
    slope_ratio = bounds.ub / denom
    bias = torch.full_like(bounds.lb, 0.2)
    bias_score_patches = torch.maximum(bias * (slope_ratio - 1.0), bias * slope_ratio)
    bias_score_matrix = torch.maximum(bias * (slope_ratio - 1.0), bias * slope_ratio)
    assert torch.all(bias_score_patches > 0)
    torch.testing.assert_close(score_patches, score_matrix)
    torch.testing.assert_close(bias_score_patches, bias_score_matrix)


def test_babsr_picks_same_neuron_as_matrix_path() -> None:
    net, _pre_lid, _ = _build_conv_relu_conv_net()
    lb, ub = _input_box()
    batch = SubproblemBatch(
        lb=lb.reshape(1, -1),
        ub=ub.reshape(1, -1),
        depths=torch.tensor([0]),
    )
    spec_c = _spec_c(5)
    with _conv_mode("matrix"):
        matrix_bounds = compute_forward_bounds(net, lb, ub)
        matrix_pick = BaBSRBranching().select_neurons(net, batch, matrix_bounds, spec_c, num_classes=16)
    with _conv_mode("patches"):
        patches_bounds = compute_forward_bounds(net, lb, ub)
        patches_pick = BaBSRBranching().select_neurons(net, batch, patches_bounds, spec_c, num_classes=16)
    assert patches_pick == matrix_pick
