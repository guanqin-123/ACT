from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportArgumentType=false, reportCallIssue=false, reportOptionalSubscript=false, reportIndexIssue=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportMissingImports=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false

from contextlib import contextmanager
from pathlib import Path
import sys
from typing import cast

import pytest
import torch
import torch.nn.functional as F

from act.back_end.bab.branching.babsr import BaBSRBranching, compute_lA_per_layer
from act.back_end.bab.node import SubproblemBatch
from act.back_end.bounds_dispatch import (
    dispatch_conv_forward,
    get_conv_mode,
    reset_conv_materialization_count,
    set_conv_mode,
)
from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf import compute_forward_bounds
from act.back_end.dual_tf.tf_cnn import backward_conv2d, forward_conv2d
from act.back_end.dual_tf.tf_cnn_patches import (
    _is_identity_conv,
    backward_conv2d_patches,
    forward_conv2d_patches,
    linear_form_patches_to_tensor,
    linear_form_tensor_to_patches,
)
from act.back_end.dual_tf.tf_forward import Frame, LinearBound
from act.back_end.dual_tf.dual_tf import DualTF
from act.back_end.layer_schema import LayerKind
from act.back_end.patches import Patches, compute_patches_stride_padding
from act.util.device_manager import initialize_device


@pytest.fixture(scope="module", autouse=True)
def _cpu_float64() -> None:
    initialize_device("cpu", "float64")


@contextmanager
def _conv_mode(mode: str):
    previous = get_conv_mode()
    set_conv_mode(mode)
    try:
        yield
    finally:
        set_conv_mode(previous)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _make_input_box(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> tuple[Bounds, Frame, torch.Tensor, torch.Tensor]:
    gen = _generator(seed)
    center = torch.randn((batch_size, channels, height, width), generator=gen, dtype=dtype)
    radius = 0.05 + 0.2 * torch.rand((batch_size, channels, height, width), generator=gen, dtype=dtype)
    lb_4d = center - radius
    ub_4d = center + radius
    return (
        Bounds(lb=lb_4d.reshape(batch_size, -1), ub=ub_4d.reshape(batch_size, -1)),
        (lb_4d.reshape(batch_size, -1), ub_4d.reshape(batch_size, -1)),
        lb_4d,
        ub_4d,
    )


def _identity_linear_bound(batch_size: int, input_dim: int, *, dtype: torch.dtype) -> LinearBound:
    eye = torch.eye(input_dim, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()
    zeros = torch.zeros((batch_size, input_dim), dtype=dtype)
    return LinearBound(A_lb=eye, b_lb=zeros, A_ub=eye.clone(), b_ub=zeros.clone())


def _identity_patches(batch_size: int, channels: int, height: int, width: int) -> Patches:
    return Patches(
        identity=1,
        shape=(channels, batch_size, height, width, channels, 1, 1),
        input_shape=(batch_size, channels, height, width),
        output_shape=(batch_size, channels, height, width),
    )


def _make_conv_layer(
    *,
    layer_id: int,
    in_c: int,
    out_c: int,
    kernel: int,
    stride: int,
    padding: int,
    height: int,
    width: int,
    seed: int,
    bias: bool = True,
    zero_bias: bool = False,
    dtype: torch.dtype = torch.float64,
) -> Layer:
    gen = _generator(seed)
    weight = torch.randn((out_c, in_c, kernel, kernel), generator=gen, dtype=dtype)
    bias_value = None
    if bias:
        bias_value = torch.zeros((out_c,), dtype=dtype) if zero_bias else torch.randn((out_c,), generator=gen, dtype=dtype)
    dummy = torch.zeros((1, in_c, height, width), dtype=dtype)
    out_h, out_w = F.conv2d(dummy, weight, bias=None, stride=stride, padding=padding).shape[-2:]
    return Layer(
        id=layer_id,
        kind=LayerKind.CONV2D.value,
        params={
            "in_channels": in_c,
            "out_channels": out_c,
            "kernel_size": kernel,
            "stride": stride,
            "padding": padding,
            "dilation": 1,
            "groups": 1,
            "input_shape": (1, in_c, height, width),
            "output_shape": (1, out_c, out_h, out_w),
            "weight": weight,
            "bias": bias_value,
        },
        in_vars=[],
        out_vars=[],
    )


def _make_identity_conv(channels: int, height: int, width: int, *, layer_id: int = 9) -> Layer:
    weight = torch.eye(channels, dtype=torch.float64).view(channels, channels, 1, 1)
    bias = torch.zeros((channels,), dtype=torch.float64)
    return Layer(
        id=layer_id,
        kind=LayerKind.CONV2D.value,
        params={
            "in_channels": channels,
            "out_channels": channels,
            "kernel_size": 1,
            "stride": 1,
            "padding": 0,
            "dilation": 1,
            "groups": 1,
            "input_shape": (1, channels, height, width),
            "output_shape": (1, channels, height, width),
            "weight": weight,
            "bias": bias,
        },
        in_vars=[],
        out_vars=[],
    )


def _run_forward_matrix(layer: Layer, bounds: Bounds, frame: Frame) -> tuple[Bounds, Bounds, LinearBound, Frame]:
    batch_size, input_dim = bounds.lb.shape
    lin = _identity_linear_bound(batch_size, input_dim, dtype=bounds.lb.dtype)
    return forward_conv2d(
        layer,
        [bounds],
        [lin],
        [frame],
        [0],
        False,
        torch.device("cpu"),
        bounds.lb.dtype,
    )


def _run_forward_patches(
    layer: Layer,
    bounds: Bounds,
    frame: Frame,
    patches: Patches | None = None,
) -> tuple[Bounds, Bounds, LinearBound | Patches, Frame]:
    return forward_conv2d_patches(
        layer,
        [bounds],
        [patches or _identity_patches(bounds.lb.shape[0], layer.params["in_channels"], layer.params["input_shape"][2], layer.params["input_shape"][3])],
        [frame],
        [0],
        False,
        torch.device("cpu"),
        bounds.lb.dtype,
    )


def _flatten_output_shape(layer: Layer) -> int:
    _, out_c, out_h, out_w = layer.params["output_shape"]
    return int(out_c * out_h * out_w)


def _sample_inputs(lb_4d: torch.Tensor, ub_4d: torch.Tensor, *, seed: int, count: int = 100) -> torch.Tensor:
    gen = _generator(seed)
    return lb_4d.unsqueeze(0) + torch.rand((count, *lb_4d.shape), generator=gen, dtype=lb_4d.dtype) * (ub_4d - lb_4d).unsqueeze(0)


def _conv_output(layer: Layer, x: torch.Tensor) -> torch.Tensor:
    return F.conv2d(
        x,
        layer.params["weight"],
        layer.params["bias"],
        stride=layer.params["stride"],
        padding=layer.params["padding"],
    )


def _assert_samples_within_bounds(
    lower: torch.Tensor,
    upper: torch.Tensor,
    actual: torch.Tensor,
    *,
    context: str,
) -> None:
    actual_flat = actual.reshape(actual.shape[0], -1)
    upper_violation = actual_flat - upper
    lower_violation = lower - actual_flat
    if torch.any(upper_violation > 1e-7):
        sample_idx, feat_idx = (upper_violation > 1e-7).nonzero(as_tuple=False)[0].tolist()
        raise AssertionError(
            f"{context}: upper violation at sample={sample_idx} feature={feat_idx} by {upper_violation[sample_idx, feat_idx].item():.3e}"
        )
    if torch.any(lower_violation > 1e-7):
        sample_idx, feat_idx = (lower_violation > 1e-7).nonzero(as_tuple=False)[0].tolist()
        raise AssertionError(
            f"{context}: lower violation at sample={sample_idx} feature={feat_idx} by {lower_violation[sample_idx, feat_idx].item():.3e}"
        )


def _make_bounds_dict(layer: Layer, batch_size: int) -> dict[int, Bounds]:
    _, out_c, out_h, out_w = layer.params["output_shape"]
    out_dim = out_c * out_h * out_w
    zeros = torch.zeros((batch_size, out_dim), dtype=layer.params["weight"].dtype)
    return {layer.id: Bounds(lb=zeros.clone(), ub=zeros.clone())}


def _single_conv_relu_net(seed: int = 0) -> tuple[Net, int]:
    conv = _make_conv_layer(layer_id=2, in_c=1, out_c=1, kernel=3, stride=1, padding=1, height=4, width=4, seed=seed, zero_bias=True)
    input_vars = list(range(16))
    conv_out = list(range(100, 116))
    relu_out = list(range(200, 216))
    net = Net(
        layers=[
            Layer(0, LayerKind.INPUT.value, {"shape": (1, 1, 4, 4), "dtype": "float64", "num_classes": 16, "value_range": (-1.0, 1.0)}, input_vars, input_vars),
            Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, input_vars, input_vars),
            Layer(2, conv.kind, conv.params, input_vars, conv_out),
            Layer(3, LayerKind.RELU.value, {}, conv_out, relu_out),
            Layer(4, LayerKind.ASSERT.value, {"kind": "RANGE"}, relu_out, relu_out),
        ],
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: []},
    )
    return net, 2


def _conv_dense_relu_net(seed: int = 0) -> tuple[Net, int, int]:
    conv = _make_conv_layer(layer_id=2, in_c=1, out_c=1, kernel=3, stride=1, padding=1, height=4, width=4, seed=seed + 1, zero_bias=True)
    dense_weight = torch.randn((4, 16), generator=_generator(seed + 2), dtype=torch.float64)
    dense_bias = torch.randn((4,), generator=_generator(seed + 3), dtype=torch.float64)
    input_vars = list(range(16))
    conv_out = list(range(100, 116))
    relu1_out = list(range(200, 216))
    dense_out = list(range(300, 304))
    relu2_out = list(range(400, 404))
    net = Net(
        layers=[
            Layer(0, LayerKind.INPUT.value, {"shape": (1, 1, 4, 4), "dtype": "float64", "num_classes": 4, "value_range": (-1.0, 1.0)}, input_vars, input_vars),
            Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, input_vars, input_vars),
            Layer(2, conv.kind, conv.params, input_vars, conv_out),
            Layer(3, LayerKind.RELU.value, {}, conv_out, relu1_out),
            Layer(4, LayerKind.DENSE.value, {"in_features": 16, "out_features": 4, "weight": dense_weight, "bias": dense_bias}, relu1_out, dense_out),
            Layer(5, LayerKind.RELU.value, {}, dense_out, relu2_out),
            Layer(6, LayerKind.ASSERT.value, {"kind": "RANGE"}, relu2_out, relu2_out),
        ],
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4], 6: [5]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6], 6: []},
    )
    return net, 2, 4


def _score_picks(net: Net, lb_4d: torch.Tensor, ub_4d: torch.Tensor, *, mode: str):
    batch_size = lb_4d.shape[0]
    reset_conv_materialization_count()
    with _conv_mode(mode):
        bounds = compute_forward_bounds(net, lb_4d.reshape(batch_size, -1), ub_4d.reshape(batch_size, -1))
        batch = SubproblemBatch(lb=lb_4d.reshape(batch_size, -1), ub=ub_4d.reshape(batch_size, -1), depths=torch.zeros(batch_size, dtype=torch.long))
        spec_c = torch.randn((batch_size, bounds[net.layers[-2].id].lb.shape[1]), generator=_generator(9), dtype=lb_4d.dtype)
        brancher = BaBSRBranching()
        return brancher.select_neurons(net, batch, bounds, spec_c, num_classes=spec_c.shape[1], dual_solver=None), bounds, spec_c


def test_forward_conv2d_patches_stride1_pad0_basic() -> None:
    layer = _make_conv_layer(layer_id=10, in_c=3, out_c=4, kernel=3, stride=1, padding=0, height=8, width=8, seed=1)
    bounds, frame, _lb_4d, _ub_4d = _make_input_box(2, 3, 8, 8, seed=2)
    patches = _run_forward_patches(layer, bounds, frame)
    matrix = _run_forward_matrix(layer, bounds, frame)
    torch.testing.assert_close(patches[0].lb, matrix[0].lb, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(patches[0].ub, matrix[0].ub, rtol=1e-5, atol=1e-7)


def test_forward_conv2d_patches_stride2_pad1() -> None:
    layer = _make_conv_layer(layer_id=11, in_c=3, out_c=4, kernel=3, stride=2, padding=1, height=8, width=8, seed=3)
    bounds, frame, _lb_4d, _ub_4d = _make_input_box(2, 3, 8, 8, seed=4)
    patches = _run_forward_patches(layer, bounds, frame)
    matrix = _run_forward_matrix(layer, bounds, frame)
    torch.testing.assert_close(patches[0].lb, matrix[0].lb, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(patches[0].ub, matrix[0].ub, rtol=1e-5, atol=1e-7)


def test_forward_conv2d_patches_preserves_patches_shape() -> None:
    layer = _make_conv_layer(layer_id=12, in_c=2, out_c=3, kernel=3, stride=1, padding=1, height=6, width=6, seed=5)
    bounds, frame, _lb_4d, _ub_4d = _make_input_box(2, 2, 6, 6, seed=6)
    _stored, _out, lin, _frame = _run_forward_patches(layer, bounds, frame)
    assert isinstance(lin, Patches)
    assert lin.patches is not None
    assert lin.patches.shape[:4] == (3, 2, 6, 6)


def test_forward_conv2d_patches_kernel_fusion() -> None:
    layer1 = _make_conv_layer(layer_id=13, in_c=3, out_c=3, kernel=3, stride=1, padding=1, height=8, width=8, seed=7, zero_bias=True)
    layer2 = _make_conv_layer(layer_id=14, in_c=3, out_c=4, kernel=3, stride=2, padding=1, height=8, width=8, seed=8)
    layer1.params["weight"] = cast(torch.Tensor, layer1.params["weight"]).abs()
    layer2.params["weight"] = cast(torch.Tensor, layer2.params["weight"]).abs()
    bounds, frame, _lb_4d, _ub_4d = _make_input_box(2, 3, 8, 8, seed=9)

    first_matrix = _run_forward_matrix(layer1, bounds, frame)
    second_matrix = forward_conv2d(
        layer2,
        [first_matrix[1]],
        [first_matrix[2]],
        [frame],
        [0],
        False,
        torch.device("cpu"),
        bounds.lb.dtype,
    )

    first_patches = _run_forward_patches(layer1, bounds, frame)
    assert isinstance(first_patches[2], Patches)
    second_patches = forward_conv2d_patches(
        layer2,
        [first_patches[1]],
        [first_patches[2]],
        [frame],
        [0],
        False,
        torch.device("cpu"),
        bounds.lb.dtype,
    )
    assert isinstance(second_patches[2], Patches)
    expected_stride, expected_padding = compute_patches_stride_padding((1, 1), (1, 1), (2, 2), (1, 1))
    assert second_patches[2].stride == expected_stride
    assert second_patches[2].padding == expected_padding
    torch.testing.assert_close(second_patches[0].lb, second_matrix[0].lb, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(second_patches[0].ub, second_matrix[0].ub, rtol=1e-5, atol=1e-7)


def test_forward_conv2d_patches_alpha_broadcast() -> None:
    layer = _make_conv_layer(layer_id=15, in_c=2, out_c=2, kernel=3, stride=1, padding=1, height=5, width=5, seed=10)
    bounds, frame, _lb_4d, _ub_4d = _make_input_box(2, 2, 5, 5, seed=11)
    bounds_hetero = Bounds(lb=torch.cat([bounds.lb[:1], (bounds.lb[1:] - 0.3)], dim=0), ub=torch.cat([bounds.ub[:1], (bounds.ub[1:] + 0.4)], dim=0))
    frame_hetero = (bounds_hetero.lb, bounds_hetero.ub)
    out = _run_forward_patches(layer, bounds_hetero, frame_hetero)
    assert not torch.allclose(out[0].lb[0], out[0].lb[1])


def test_forward_conv2d_patches_mixed_input_linearbound_warns_and_falls_back(caplog: pytest.LogCaptureFixture) -> None:
    layer = _make_conv_layer(layer_id=17, in_c=2, out_c=3, kernel=3, stride=1, padding=1, height=5, width=5, seed=15)
    bounds, frame, _lb_4d, _ub_4d = _make_input_box(2, 2, 5, 5, seed=16)
    lin = _identity_linear_bound(2, bounds.lb.shape[1], dtype=bounds.lb.dtype)
    with caplog.at_level("WARNING"):
        patches = forward_conv2d_patches(layer, [bounds], [lin], [frame], [0], False, torch.device("cpu"), bounds.lb.dtype)
    matrix = forward_conv2d(layer, [bounds], [lin], [frame], [0], False, torch.device("cpu"), bounds.lb.dtype)
    assert any("falling back to matrix path" in record.message for record in caplog.records)
    torch.testing.assert_close(patches[0].lb, matrix[0].lb)
    torch.testing.assert_close(patches[0].ub, matrix[0].ub)


def test_forward_conv2d_patches_multi_batch() -> None:
    layer = _make_conv_layer(layer_id=18, in_c=3, out_c=5, kernel=3, stride=1, padding=1, height=6, width=6, seed=17)
    bounds, frame, _lb_4d, _ub_4d = _make_input_box(3, 3, 6, 6, seed=18)
    out = _run_forward_patches(layer, bounds, frame)
    assert isinstance(out[2], Patches)
    assert out[0].lb.shape[0] == 3
    assert out[0].ub.shape[0] == 3


def test_forward_conv2d_patches_float64() -> None:
    layer = _make_conv_layer(layer_id=19, in_c=3, out_c=4, kernel=5, stride=1, padding=0, height=8, width=8, seed=19)
    bounds, frame, _lb_4d, _ub_4d = _make_input_box(2, 3, 8, 8, seed=20)
    patches = _run_forward_patches(layer, bounds, frame)
    matrix = _run_forward_matrix(layer, bounds, frame)
    torch.testing.assert_close(patches[0].lb, matrix[0].lb, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(patches[0].ub, matrix[0].ub, rtol=1e-10, atol=1e-12)


def test_backward_conv2d_patches_stride1_pad0() -> None:
    layer = _make_conv_layer(layer_id=20, in_c=2, out_c=3, kernel=3, stride=1, padding=0, height=6, width=6, seed=21)
    out_dim = _flatten_output_shape(layer)
    nu = torch.randn((2, out_dim), generator=_generator(22), dtype=torch.float64)
    bounds_dict = _make_bounds_dict(layer, 2)
    patches = backward_conv2d_patches(layer, nu, bounds_dict, [0])
    matrix = backward_conv2d(layer, nu, bounds_dict, [0])
    torch.testing.assert_close(patches[0][0], matrix[0][0])
    torch.testing.assert_close(patches[1], matrix[1])


def test_backward_conv2d_patches_stride2_pad1() -> None:
    layer = _make_conv_layer(layer_id=21, in_c=2, out_c=3, kernel=3, stride=2, padding=1, height=8, width=8, seed=23)
    out_dim = _flatten_output_shape(layer)
    nu = torch.randn((2, out_dim), generator=_generator(24), dtype=torch.float64)
    bounds_dict = _make_bounds_dict(layer, 2)
    patches = backward_conv2d_patches(layer, nu, bounds_dict, [0])
    matrix = backward_conv2d(layer, nu, bounds_dict, [0])
    torch.testing.assert_close(patches[0][0], matrix[0][0])
    torch.testing.assert_close(patches[1], matrix[1])


def test_backward_conv2d_patches_nu_accumulation_float64() -> None:
    layer = _make_conv_layer(layer_id=22, in_c=2, out_c=3, kernel=3, stride=1, padding=1, height=5, width=5, seed=25)
    coeffs = torch.randn((2, _flatten_output_shape(layer)), generator=_generator(26), dtype=torch.float64)
    nu_patches = linear_form_tensor_to_patches(coeffs, layer.params["output_shape"])
    pred_nus, contrib = backward_conv2d_patches(layer, nu_patches, _make_bounds_dict(layer, 2), [0])
    matrix = backward_conv2d(layer, coeffs, _make_bounds_dict(layer, 2), [0])
    assert isinstance(pred_nus[0], Patches)
    pred_tensor = linear_form_patches_to_tensor(pred_nus[0], layer.params["input_shape"])
    torch.testing.assert_close(pred_tensor, matrix[0][0], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(contrib, matrix[1], rtol=1e-10, atol=1e-12)


def test_backward_conv2d_patches_identity_passthrough() -> None:
    layer = _make_identity_conv(3, 5, 5, layer_id=23)
    coeffs = torch.randn((2, _flatten_output_shape(layer)), generator=_generator(27), dtype=torch.float64)
    nu_patches = linear_form_tensor_to_patches(coeffs, layer.params["output_shape"])
    pred_nus, contrib = backward_conv2d_patches(layer, nu_patches, _make_bounds_dict(layer, 2), [0])
    assert isinstance(pred_nus[0], Patches)
    torch.testing.assert_close(linear_form_patches_to_tensor(pred_nus[0], layer.params["input_shape"]), coeffs)
    torch.testing.assert_close(contrib, torch.zeros_like(contrib))


def test_backward_conv2d_patches_preserves_patches_nu() -> None:
    layer = _make_conv_layer(layer_id=24, in_c=2, out_c=2, kernel=3, stride=1, padding=1, height=5, width=5, seed=28)
    coeffs = torch.randn((2, _flatten_output_shape(layer)), generator=_generator(29), dtype=torch.float64)
    nu_patches = linear_form_tensor_to_patches(coeffs, layer.params["output_shape"])
    pred_nus, _contrib = backward_conv2d_patches(layer, nu_patches, _make_bounds_dict(layer, 2), [0])
    assert isinstance(pred_nus[0], Patches)
    assert pred_nus[0].patches is not None


def test_backward_conv2d_patches_multi_preds() -> None:
    layer = _make_conv_layer(layer_id=25, in_c=2, out_c=2, kernel=3, stride=1, padding=1, height=5, width=5, seed=30)
    coeffs_a = torch.randn((2, _flatten_output_shape(layer)), generator=_generator(31), dtype=torch.float64)
    coeffs_b = torch.randn((2, _flatten_output_shape(layer)), generator=_generator(32), dtype=torch.float64)
    out_a = backward_conv2d_patches(layer, linear_form_tensor_to_patches(coeffs_a, layer.params["output_shape"]), _make_bounds_dict(layer, 2), [0])
    out_b = backward_conv2d_patches(layer, linear_form_tensor_to_patches(coeffs_b, layer.params["output_shape"]), _make_bounds_dict(layer, 2), [0])
    merged = linear_form_patches_to_tensor(out_a[0][0], layer.params["input_shape"]) + linear_form_patches_to_tensor(out_b[0][0], layer.params["input_shape"])
    matrix = backward_conv2d(layer, coeffs_a + coeffs_b, _make_bounds_dict(layer, 2), [0])
    torch.testing.assert_close(merged, matrix[0][0], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(out_a[1] + out_b[1], matrix[1], rtol=1e-10, atol=1e-12)


def test_backward_conv2d_patches_soundness_sampled() -> None:
    layer = _make_conv_layer(layer_id=26, in_c=2, out_c=3, kernel=3, stride=1, padding=1, height=5, width=5, seed=33)
    bounds, _frame, lb_4d, ub_4d = _make_input_box(2, 2, 5, 5, seed=34)
    coeffs = torch.randn((2, _flatten_output_shape(layer)), generator=_generator(35), dtype=torch.float64)
    nu_patches = linear_form_tensor_to_patches(coeffs, layer.params["output_shape"])
    pred_nus, contrib = backward_conv2d_patches(layer, nu_patches, _make_bounds_dict(layer, 2), [0])
    pred_coeffs = linear_form_patches_to_tensor(pred_nus[0], layer.params["input_shape"])
    lb_obj = (pred_coeffs.clamp(min=0) * bounds.lb).sum(dim=-1) + (pred_coeffs.clamp(max=0) * bounds.ub).sum(dim=-1) - contrib
    ub_obj = (pred_coeffs.clamp(min=0) * bounds.ub).sum(dim=-1) + (pred_coeffs.clamp(max=0) * bounds.lb).sum(dim=-1) - contrib
    samples = _sample_inputs(lb_4d, ub_4d, seed=36)
    for sample_idx, sample in enumerate(samples):
        actual = (_conv_output(layer, sample).reshape(sample.shape[0], -1) * coeffs).sum(dim=-1)
        upper_violation = actual - ub_obj
        lower_violation = lb_obj - actual
        if torch.any(upper_violation > 1e-7):
            batch_idx = int((upper_violation > 1e-7).nonzero(as_tuple=False)[0].item())
            raise AssertionError(
                f"backward soundness upper violation sample={sample_idx} batch={batch_idx} by {upper_violation[batch_idx].item():.3e}"
            )
        if torch.any(lower_violation > 1e-7):
            batch_idx = int((lower_violation > 1e-7).nonzero(as_tuple=False)[0].item())
            raise AssertionError(
                f"backward soundness lower violation sample={sample_idx} batch={batch_idx} by {lower_violation[batch_idx].item():.3e}"
            )


def test_backward_conv2d_patches_matches_abcrown() -> None:
    repo_root = Path("/data1/guanqin/newACT/alpha-beta-CROWN/auto_LiRPA")
    sys.path.insert(0, str(repo_root))
    try:
        from auto_LiRPA.operators.convolution import BoundConv
        from auto_LiRPA.patches import Patches as RefPatches

        layer = _make_conv_layer(layer_id=27, in_c=2, out_c=3, kernel=3, stride=1, padding=1, height=5, width=5, seed=37, zero_bias=True)
        coeffs = torch.randn((2, _flatten_output_shape(layer)), generator=_generator(38), dtype=torch.float64)
        nu_patches = linear_form_tensor_to_patches(coeffs, layer.params["output_shape"])
        ours, _contrib = backward_conv2d_patches(layer, nu_patches, _make_bounds_dict(layer, 2), [0])
        assert isinstance(ours[0], Patches)

        class _Node:
            def __init__(self, lower: torch.Tensor):
                self.lower = lower

        class _FakeConv:
            conv_dim = 2
            relu_followed = False
            has_bias = False
            stride = (1, 1)
            padding = (1, 1)
            dilation = (1, 1)
            groups = 1
            input_shape = layer.params["input_shape"]
            output_shape = layer.params["output_shape"]

            @staticmethod
            def is_input_perturbed(_idx: int) -> bool:
                return False

        ref_patches = RefPatches(
            patches=nu_patches.patches.clone(),
            stride=nu_patches.stride,
            padding=nu_patches.padding,
            shape=nu_patches.shape,
            identity=nu_patches.identity,
            unstable_idx=nu_patches.unstable_idx,
            output_shape=nu_patches.output_shape,
            input_shape=nu_patches.input_shape,
            inserted_zeros=nu_patches.inserted_zeros,
            output_padding=nu_patches.output_padding,
        )
        result, _lbias, _ubias = BoundConv.bound_backward(
            _FakeConv(),
            ref_patches,
            None,
            _Node(torch.zeros(1, dtype=torch.float64)),
            _Node(layer.params["weight"]),
        )
        ref_out = result[0][0]
        ours_tensor = linear_form_patches_to_tensor(ours[0], layer.params["input_shape"])
        if isinstance(ref_out, torch.Tensor):
            ref_tensor = ref_out.squeeze(0).reshape(ref_out.shape[1], -1)
        else:
            ref_tensor = ref_out.to_matrix(layer.params["input_shape"])[:, 0, :]
        torch.testing.assert_close(ours_tensor, ref_tensor, rtol=1e-5, atol=1e-7)
    finally:
        if sys.path and sys.path[0] == str(repo_root):
            sys.path.pop(0)


def test_identity_conv_forward_passthrough() -> None:
    layer = _make_identity_conv(2, 5, 5, layer_id=28)
    bounds, frame, _lb_4d, _ub_4d = _make_input_box(2, 2, 5, 5, seed=39)
    random_coeffs = torch.randn((2, 2 * 5 * 5), generator=_generator(40), dtype=torch.float64)
    in_patches = linear_form_tensor_to_patches(random_coeffs, layer.params["input_shape"])
    _stored, _out, lin, _frame = _run_forward_patches(layer, bounds, frame, in_patches)
    assert isinstance(lin, Patches)
    torch.testing.assert_close(linear_form_patches_to_tensor(lin, layer.params["input_shape"]), random_coeffs)


def test_identity_conv_backward_passthrough() -> None:
    layer = _make_identity_conv(2, 5, 5, layer_id=29)
    coeffs = torch.randn((2, 2 * 5 * 5), generator=_generator(41), dtype=torch.float64)
    nu = linear_form_tensor_to_patches(coeffs, layer.params["output_shape"])
    pred_nus, contrib = backward_conv2d_patches(layer, nu, _make_bounds_dict(layer, 2), [0])
    assert isinstance(pred_nus[0], Patches)
    torch.testing.assert_close(linear_form_patches_to_tensor(pred_nus[0], layer.params["input_shape"]), coeffs)
    torch.testing.assert_close(contrib, torch.zeros_like(contrib))


def test_identity_patches_shape_inference() -> None:
    layer = _make_identity_conv(4, 7, 7, layer_id=30)
    bounds, frame, _lb_4d, _ub_4d = _make_input_box(2, 4, 7, 7, seed=42)
    _stored, _out, lin, _frame = _run_forward_patches(layer, bounds, frame, _identity_patches(2, 4, 7, 7))
    assert isinstance(lin, Patches)
    assert lin.shape == (4, 2, 7, 7, 4, 1, 1)


def test_mixed_mode_patches_mode_reseeds_identity_linearbound_input(caplog: pytest.LogCaptureFixture) -> None:
    layer = _make_conv_layer(layer_id=31, in_c=2, out_c=3, kernel=3, stride=1, padding=1, height=5, width=5, seed=43)
    bounds, frame, _lb_4d, _ub_4d = _make_input_box(2, 2, 5, 5, seed=44)
    lin = _identity_linear_bound(2, bounds.lb.shape[1], dtype=bounds.lb.dtype)
    with _conv_mode("patches"), caplog.at_level("WARNING"):
        dispatched = dispatch_conv_forward(layer, [bounds], [lin], [frame], [0], False, torch.device("cpu"), bounds.lb.dtype)
    matrix = forward_conv2d(layer, [bounds], [lin], [frame], [0], False, torch.device("cpu"), bounds.lb.dtype)
    assert not any("LinearBound input" in record.message for record in caplog.records)
    torch.testing.assert_close(dispatched[0].lb, matrix[0].lb)
    torch.testing.assert_close(dispatched[0].ub, matrix[0].ub)


def test_mixed_mode_matrix_mode_with_patches_input_to_matrix_conversion(caplog: pytest.LogCaptureFixture) -> None:
    layer = _make_conv_layer(layer_id=32, in_c=2, out_c=3, kernel=3, stride=1, padding=1, height=5, width=5, seed=45)
    bounds, frame, _lb_4d, _ub_4d = _make_input_box(2, 2, 5, 5, seed=46)
    with _conv_mode("matrix"), caplog.at_level("WARNING"):
        dispatched = dispatch_conv_forward(layer, [bounds], [_identity_patches(2, 2, 5, 5)], [frame], [0], False, torch.device("cpu"), bounds.lb.dtype)
    matrix = _run_forward_matrix(layer, bounds, frame)
    assert any("converting to dense LinearBound" in record.message for record in caplog.records)
    torch.testing.assert_close(dispatched[0].lb, matrix[0].lb)
    torch.testing.assert_close(dispatched[0].ub, matrix[0].ub)


def test_mixed_mode_preserves_soundness() -> None:
    layer = _make_conv_layer(layer_id=33, in_c=2, out_c=3, kernel=3, stride=1, padding=1, height=5, width=5, seed=47)
    bounds, frame, lb_4d, ub_4d = _make_input_box(2, 2, 5, 5, seed=48)
    lin = _identity_linear_bound(2, bounds.lb.shape[1], dtype=bounds.lb.dtype)
    with _conv_mode("patches"):
        out = dispatch_conv_forward(layer, [bounds], [lin], [frame], [0], False, torch.device("cpu"), bounds.lb.dtype)
    samples = _sample_inputs(lb_4d, ub_4d, seed=49, count=32)
    actual = torch.cat([_conv_output(layer, sample).reshape(sample.shape[0], -1) for sample in samples], dim=0)
    lower = out[0].lb.repeat(samples.shape[0], 1)
    upper = out[0].ub.repeat(samples.shape[0], 1)
    _assert_samples_within_bounds(lower, upper, actual, context="mixed-mode soundness")


def test_babsr_lA_patches_preserved() -> None:
    net, conv_id = _single_conv_relu_net(seed=50)
    bounds, _, lb_4d, ub_4d = _make_input_box(2, 1, 4, 4, seed=51)
    with _conv_mode("patches"):
        bound_dict = compute_forward_bounds(net, lb_4d.reshape(2, -1), ub_4d.reshape(2, -1))
        spec_c = torch.randn((2, 16), generator=_generator(52), dtype=torch.float64)
        lA = compute_lA_per_layer(net, bound_dict, spec_c, DualTF(), target_layer_ids=[conv_id])
    assert isinstance(lA[conv_id], Patches)


def test_babsr_lA_patches_parity() -> None:
    net, _conv_id = _single_conv_relu_net(seed=53)
    _bounds, _, lb_4d, ub_4d = _make_input_box(2, 1, 4, 4, seed=54)
    patches_picks, _patches_bounds, _patches_c = _score_picks(net, lb_4d, ub_4d, mode="patches")
    matrix_picks, _matrix_bounds, _matrix_c = _score_picks(net, lb_4d, ub_4d, mode="matrix")
    assert patches_picks == matrix_picks


def test_babsr_instrumentation_no_silent_densify(caplog: pytest.LogCaptureFixture) -> None:
    net, _conv_id = _single_conv_relu_net(seed=55)
    _bounds, _, lb_4d, ub_4d = _make_input_box(2, 1, 4, 4, seed=56)
    with caplog.at_level("WARNING"):
        _score_picks(net, lb_4d, ub_4d, mode="patches")
    assert not any("materializing patches lA" in record.message for record in caplog.records)
    caplog.clear()
    with caplog.at_level("WARNING"):
        _score_picks(net, lb_4d, ub_4d, mode="matrix")
    assert not any("materializing patches lA" in record.message for record in caplog.records)


def test_babsr_mixed_layer_stack() -> None:
    net, conv_id, dense_id = _conv_dense_relu_net(seed=57)
    _bounds, _, lb_4d, ub_4d = _make_input_box(2, 1, 4, 4, seed=58)
    with _conv_mode("patches"):
        bound_dict = compute_forward_bounds(net, lb_4d.reshape(2, -1), ub_4d.reshape(2, -1))
        spec_c = torch.randn((2, 4), generator=_generator(59), dtype=torch.float64)
        lA = compute_lA_per_layer(net, bound_dict, spec_c, DualTF(), target_layer_ids=[conv_id, dense_id])
    assert isinstance(lA[conv_id], Patches)
    assert isinstance(lA[dense_id], torch.Tensor)


SOUNDNESS_CONFIGS = [
    (1, 0, 3, 4, 3, 8, 8, 60),
    (1, 1, 3, 4, 3, 8, 8, 61),
    (2, 0, 3, 4, 3, 8, 8, 62),
    (2, 1, 3, 4, 3, 8, 8, 63),
    (1, 0, 3, 3, 1, 8, 8, 64),
]


@pytest.mark.parametrize("stride,padding,in_c,out_c,kernel,height,width,seed", SOUNDNESS_CONFIGS)
def test_conv2d_patches_soundness_sampled(
    stride: int,
    padding: int,
    in_c: int,
    out_c: int,
    kernel: int,
    height: int,
    width: int,
    seed: int,
) -> None:
    layer = _make_identity_conv(in_c, height, width, layer_id=34) if kernel == 1 and in_c == out_c and padding == 0 else _make_conv_layer(
        layer_id=34,
        in_c=in_c,
        out_c=out_c,
        kernel=kernel,
        stride=stride,
        padding=padding,
        height=height,
        width=width,
        seed=seed,
    )
    bounds, frame, lb_4d, ub_4d = _make_input_box(2, in_c, height, width, seed=seed + 100)
    out = _run_forward_patches(layer, bounds, frame)
    samples = _sample_inputs(lb_4d, ub_4d, seed=seed + 200)
    actual = torch.cat([_conv_output(layer, sample).reshape(sample.shape[0], -1) for sample in samples], dim=0)
    lower = out[0].lb.repeat(samples.shape[0], 1)
    upper = out[0].ub.repeat(samples.shape[0], 1)
    _assert_samples_within_bounds(lower, upper, actual, context=f"soundness stride={stride} pad={padding}")


PARITY_CONFIGS = [
    (1, 0, 1, 3, 70),
    (1, 0, 3, 4, 71),
    (1, 0, 5, 5, 72),
    (1, 0, 7, 6, 73),
    (1, 0, 3, 7, 74),
    (1, 1, 3, 3, 75),
    (1, 1, 3, 4, 76),
    (1, 1, 3, 5, 77),
    (1, 1, 3, 6, 78),
    (1, 1, 3, 7, 79),
    (2, 0, 1, 3, 80),
    (2, 0, 3, 4, 81),
    (2, 0, 3, 5, 82),
    (2, 0, 5, 6, 83),
    (2, 0, 3, 7, 84),
    (2, 1, 3, 3, 85),
    (2, 1, 3, 4, 86),
    (2, 1, 3, 5, 87),
    (2, 1, 5, 6, 88),
    (2, 1, 3, 7, 89),
]


@pytest.mark.parametrize("stride,padding,kernel,out_c,seed", PARITY_CONFIGS)
def test_conv2d_patches_matrix_parity(
    stride: int,
    padding: int,
    kernel: int,
    out_c: int,
    seed: int,
) -> None:
    layer = _make_conv_layer(layer_id=35, in_c=3, out_c=out_c, kernel=kernel, stride=stride, padding=padding, height=8, width=8, seed=seed)
    bounds, frame, _lb_4d, _ub_4d = _make_input_box(2, 3, 8, 8, seed=seed + 100)
    patches = _run_forward_patches(layer, bounds, frame)
    matrix = _run_forward_matrix(layer, bounds, frame)
    torch.testing.assert_close(patches[0].lb, matrix[0].lb, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(patches[0].ub, matrix[0].ub, rtol=1e-5, atol=1e-7)


def test_dense_matrix_to_patches_perf_no_python_loop_blowup() -> None:
    # Regression for the seven-deep Python-loop hang in
    # _dense_matrix_to_patches on CIFAR-100 ResNet-medium first conv
    # (B=16, out_c=16, OH=OW=32, in_c=3, kh=kw=3 -> ~7M scalar CUDA
    # dispatches, ~10 minutes wall-time before fix). The vectorized
    # path completes in <1s on CPU; we set the budget at 5s to allow
    # for slow CI runners.
    from act.back_end.dual_tf.tf_cnn_patches import _dense_matrix_to_patches
    import time

    torch.manual_seed(0)
    batch_size = 16
    in_c, in_h, in_w = 3, 32, 32
    out_c, k = 16, 3
    out_h, out_w = 32, 32
    matrix = torch.randn(batch_size, out_c * out_h * out_w, in_c * in_h * in_w, dtype=torch.float64)

    t0 = time.time()
    pieces = _dense_matrix_to_patches(
        matrix,
        (batch_size, in_c, in_h, in_w),
        out_c,
        (out_h, out_w),
        (k, k),
        1,
        1,
    )
    elapsed = time.time() - t0
    assert pieces.shape == (out_c, batch_size, out_h, out_w, in_c, k, k), pieces.shape
    assert elapsed < 5.0, f"perf regression: dense->patches took {elapsed:.2f}s (was ~600s pre-fix)"
