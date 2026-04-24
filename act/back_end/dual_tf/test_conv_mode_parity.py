from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportCallIssue=false, reportOptionalIterable=false, reportAttributeAccessIssue=false, reportReturnType=false, reportGeneralTypeIssues=false, reportOptionalMemberAccess=false

from contextlib import contextmanager
from typing import cast

import pytest
import torch
import torch.nn.functional as F

from act.back_end.bounds_dispatch import get_conv_mode, set_conv_mode
from act.back_end.core import Layer, Net
from act.back_end.dual_tf import DualTF, compute_forward_bounds
from act.back_end.dual_tf.tf_cnn_patches import (
    linear_form_patches_to_tensor,
    linear_form_tensor_to_patches,
)
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_dual import DualSolver
from act.util.device_manager import initialize_device


@pytest.fixture(scope="module", autouse=True)
def _cpu_float32() -> None:
    initialize_device("cpu", "float32")


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
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = _generator(seed)
    center = torch.randn((batch_size, channels, height, width), generator=gen)
    radius = 0.05 + 0.15 * torch.rand((batch_size, channels, height, width), generator=gen)
    return center - radius, center + radius


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
) -> Layer:
    gen = _generator(seed)
    weight = torch.randn((out_c, in_c, kernel, kernel), generator=gen)
    bias = torch.randn((out_c,), generator=gen)
    dummy = torch.zeros((1, in_c, height, width), dtype=weight.dtype)
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
            "bias": bias,
        },
        in_vars=[],
        out_vars=[],
    )


def _build_single_conv_net(
    *,
    stride: int,
    padding: int,
    in_c: int,
    out_c: int,
    kernel: int,
    height: int,
    width: int,
    seed: int,
) -> tuple[Net, int]:
    input_vars = list(range(in_c * height * width))
    conv_layer = _make_conv_layer(
        layer_id=2,
        in_c=in_c,
        out_c=out_c,
        kernel=kernel,
        stride=stride,
        padding=padding,
        height=height,
        width=width,
        seed=seed,
    )
    out_shape_param = conv_layer.params["output_shape"]
    assert isinstance(out_shape_param, tuple)
    out_shape = tuple(int(dim) for dim in out_shape_param)
    _, _, out_h, out_w = out_shape
    conv_out = list(range(1000, 1000 + out_c * out_h * out_w))
    layers = [
        Layer(
            0,
            LayerKind.INPUT.value,
            {"shape": (1, in_c, height, width), "dtype": "float32", "num_classes": out_c * out_h * out_w, "value_range": (0.0, 1.0)},
            input_vars,
            input_vars,
        ),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, input_vars, input_vars),
        Layer(conv_layer.id, conv_layer.kind, conv_layer.params, input_vars, conv_out),
        Layer(3, LayerKind.ASSERT.value, {"kind": "RANGE"}, conv_out, conv_out),
    ]
    net = Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2]},
        succs={0: [1], 1: [2], 2: [3], 3: []},
    )
    return net, 2


@pytest.fixture
def relu_conv_net() -> Net:
    _ = torch.manual_seed(0)
    input_vars = list(range(3 * 8 * 8))
    conv1_out = list(range(1000, 1000 + 8 * 6 * 6))
    relu_out = list(range(2000, 2000 + 8 * 6 * 6))
    conv2_out = list(range(3000, 3000 + 4 * 4 * 4))
    layers = [
        Layer(
            0,
            LayerKind.INPUT.value,
            {"shape": (1, 3, 8, 8), "dtype": "float32", "num_classes": 4, "value_range": (0.0, 1.0)},
            input_vars,
            input_vars,
        ),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, input_vars, input_vars),
        Layer(
            2,
            LayerKind.CONV2D.value,
            {
                "in_channels": 3,
                "out_channels": 8,
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "groups": 1,
                "input_shape": (1, 3, 8, 8),
                "output_shape": (1, 8, 6, 6),
                "weight": torch.randn(8, 3, 3, 3),
                "bias": torch.randn(8),
            },
            input_vars,
            conv1_out,
        ),
        Layer(3, LayerKind.RELU.value, {}, conv1_out, relu_out),
        Layer(
            4,
            LayerKind.CONV2D.value,
            {
                "in_channels": 8,
                "out_channels": 4,
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "dilation": 1,
                "groups": 1,
                "input_shape": (1, 8, 6, 6),
                "output_shape": (1, 4, 4, 4),
                "weight": torch.randn(4, 8, 3, 3),
                "bias": torch.randn(4),
            },
            relu_out,
            conv2_out,
        ),
        Layer(5, LayerKind.ASSERT.value, {"kind": "RANGE"}, conv2_out, conv2_out),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]}
    succs = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []}
    return Net(layers=layers, preds=preds, succs=succs)


CONV_CONFIGS = [
    (1, 0, 3, 8, 3, 8, 8),
    (1, 1, 3, 8, 3, 8, 8),
    (2, 0, 3, 16, 3, 16, 16),
    (2, 1, 3, 16, 3, 16, 16),
    (1, 0, 1, 4, 5, 10, 10),
    (1, 0, 8, 8, 3, 6, 6),
    (1, 1, 8, 8, 3, 4, 4),
    (2, 1, 8, 8, 3, 32, 32),
    (1, 0, 3, 12, 5, 16, 16),
    (2, 1, 3, 12, 5, 32, 32),
]


def _run_forward(net: Net, lb: torch.Tensor, ub: torch.Tensor, *, mode: str):
    with _conv_mode(mode):
        return compute_forward_bounds(net, lb, ub, post_activation=True)


def _run_dual_bound(net: Net, lb: torch.Tensor, ub: torch.Tensor, c: torch.Tensor, *, mode: str) -> torch.Tensor:
    with _conv_mode(mode):
        bounds = compute_forward_bounds(net, lb, ub, post_activation=True)
        result = DualSolver(DualTF()).compute_bound(net, bounds, c)
        assert isinstance(result, torch.Tensor)
        return result


def _eval_relu_conv_net(net: Net, x: torch.Tensor) -> torch.Tensor:
    conv1 = F.conv2d(
        x,
        cast(torch.Tensor, net.by_id[2].params["weight"]),
        cast(torch.Tensor | None, net.by_id[2].params["bias"]),
    )
    relu = F.relu(conv1)
    conv2 = F.conv2d(
        relu,
        cast(torch.Tensor, net.by_id[4].params["weight"]),
        cast(torch.Tensor | None, net.by_id[4].params["bias"]),
    )
    return conv2.reshape(x.shape[0], -1)


def _build_resnet_basic_block(seed: int = 11) -> Net:
    gen = _generator(seed)
    input_vars = list(range(3 * 8 * 8))
    conv1_out = list(range(1000, 1000 + 3 * 8 * 8))
    relu_out = list(range(3000, 3000 + 3 * 8 * 8))
    conv2_out = list(range(4000, 4000 + 3 * 8 * 8))
    add_out = list(range(6000, 6000 + 3 * 8 * 8))
    layers = [
        Layer(0, LayerKind.INPUT.value, {"shape": (1, 3, 8, 8), "dtype": "float32", "num_classes": 3 * 8 * 8, "value_range": (0.0, 1.0)}, input_vars, input_vars),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, input_vars, input_vars),
        Layer(
            2,
            LayerKind.CONV2D.value,
            {
                "in_channels": 3,
                "out_channels": 3,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "groups": 1,
                "input_shape": (1, 3, 8, 8),
                "output_shape": (1, 3, 8, 8),
                "weight": torch.randn((3, 3, 3, 3), generator=gen),
                "bias": torch.randn((3,), generator=gen),
            },
            input_vars,
            conv1_out,
        ),
        Layer(3, LayerKind.RELU.value, {}, conv1_out, relu_out),
        Layer(
            4,
            LayerKind.CONV2D.value,
            {
                "in_channels": 3,
                "out_channels": 3,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 1,
                "groups": 1,
                "input_shape": (1, 3, 8, 8),
                "output_shape": (1, 3, 8, 8),
                "weight": torch.randn((3, 3, 3, 3), generator=gen),
                "bias": torch.randn((3,), generator=gen),
            },
            relu_out,
            conv2_out,
        ),
        Layer(5, LayerKind.ADD.value, {}, conv2_out, add_out),
        Layer(6, LayerKind.ASSERT.value, {"kind": "RANGE"}, add_out, add_out),
    ]
    net = Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4, 1], 6: [5]},
        succs={0: [1], 1: [2, 5], 2: [3], 3: [4], 4: [5], 5: [6], 6: []},
    )
    return net


def _eval_resnet_basic_block(net: Net, x: torch.Tensor) -> torch.Tensor:
    conv1 = F.conv2d(
        x,
        cast(torch.Tensor, net.by_id[2].params["weight"]),
        cast(torch.Tensor | None, net.by_id[2].params["bias"]),
        stride=1,
        padding=1,
    )
    relu = F.relu(conv1)
    conv2 = F.conv2d(
        relu,
        cast(torch.Tensor, net.by_id[4].params["weight"]),
        cast(torch.Tensor | None, net.by_id[4].params["bias"]),
        stride=1,
        padding=1,
    )
    out = conv2 + x
    return out.reshape(x.shape[0], -1)


def test_conv_patches_forward_parity() -> None:
    for idx, (stride, padding, in_c, out_c, kernel, h, w) in enumerate(CONV_CONFIGS):
        net, conv_id = _build_single_conv_net(
            stride=stride,
            padding=padding,
            in_c=in_c,
            out_c=out_c,
            kernel=kernel,
            height=h,
            width=w,
            seed=100 + idx,
        )
        lb, ub = _make_input_box(2, in_c, h, w, seed=200 + idx)
        patches = _run_forward(net, lb, ub, mode="patches")
        matrix = _run_forward(net, lb, ub, mode="matrix")
        torch.testing.assert_close(patches[conv_id].lb, matrix[conv_id].lb, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(patches[conv_id].ub, matrix[conv_id].ub, rtol=1e-5, atol=1e-7)


def test_conv_patches_backward_parity() -> None:
    for idx, (stride, padding, in_c, out_c, kernel, h, w) in enumerate(CONV_CONFIGS):
        net, conv_id = _build_single_conv_net(
            stride=stride,
            padding=padding,
            in_c=in_c,
            out_c=out_c,
            kernel=kernel,
            height=h,
            width=w,
            seed=300 + idx,
        )
        lb, ub = _make_input_box(2, in_c, h, w, seed=400 + idx)
        output_shape = cast(tuple[int, int, int, int], net.by_id[conv_id].params["output_shape"])
        out_dim = int(torch.tensor(output_shape).prod().item())
        c = torch.randn((2, out_dim), generator=_generator(500 + idx))
        patches = _run_dual_bound(net, lb, ub, c, mode="patches")
        matrix = _run_dual_bound(net, lb, ub, c, mode="matrix")
        torch.testing.assert_close(patches, matrix, rtol=1e-5, atol=1e-7)


def test_patches_to_matrix_roundtrip() -> None:
    coeffs = torch.randn((2, 3 * 4 * 4), generator=_generator(1234))
    patches = linear_form_tensor_to_patches(coeffs, (2, 3, 4, 4))
    round_trip = linear_form_patches_to_tensor(patches, (2, 3, 4, 4))
    torch.testing.assert_close(round_trip, coeffs)


def test_conv_chain_parity(relu_conv_net: Net) -> None:
    lb, ub = _make_input_box(2, 3, 8, 8, seed=700)
    patches = _run_forward(relu_conv_net, lb, ub, mode="patches")
    matrix = _run_forward(relu_conv_net, lb, ub, mode="matrix")
    torch.testing.assert_close(patches[5].lb, matrix[5].lb, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(patches[5].ub, matrix[5].ub, rtol=1e-5, atol=1e-7)


def test_resnet_basic_block_parity() -> None:
    net = _build_resnet_basic_block()
    lb, ub = _make_input_box(2, 3, 8, 8, seed=800)
    patches = _run_forward(net, lb, ub, mode="patches")
    matrix = _run_forward(net, lb, ub, mode="matrix")
    torch.testing.assert_close(patches[6].lb, matrix[6].lb, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(patches[6].ub, matrix[6].ub, rtol=1e-5, atol=1e-7)


def test_soundness_concrete_samples_matrix() -> None:
    net, conv_id = _build_single_conv_net(stride=1, padding=1, in_c=3, out_c=4, kernel=3, height=8, width=8, seed=900)
    lb, ub = _make_input_box(2, 3, 8, 8, seed=901)
    bounds = _run_forward(net, lb, ub, mode="matrix")
    samples = lb + torch.rand((100, *lb.shape), generator=_generator(902)) * (ub - lb)
    layer = net.by_id[conv_id]
    lower = bounds[conv_id].lb
    upper = bounds[conv_id].ub
    for sample_idx, sample in enumerate(samples):
        actual = F.conv2d(
            sample,
            cast(torch.Tensor, layer.params["weight"]),
            cast(torch.Tensor | None, layer.params["bias"]),
            stride=1,
            padding=1,
        ).reshape(sample.shape[0], -1)
        if not torch.all(actual <= upper + 1e-6):
            raise AssertionError(f"matrix soundness upper violation at sample {sample_idx}")
        if not torch.all(actual >= lower - 1e-6):
            raise AssertionError(f"matrix soundness lower violation at sample {sample_idx}")


def test_soundness_concrete_samples_patches() -> None:
    net, conv_id = _build_single_conv_net(stride=1, padding=1, in_c=3, out_c=4, kernel=3, height=8, width=8, seed=910)
    lb, ub = _make_input_box(2, 3, 8, 8, seed=911)
    bounds = _run_forward(net, lb, ub, mode="patches")
    samples = lb + torch.rand((100, *lb.shape), generator=_generator(912)) * (ub - lb)
    layer = net.by_id[conv_id]
    lower = bounds[conv_id].lb
    upper = bounds[conv_id].ub
    for sample_idx, sample in enumerate(samples):
        actual = F.conv2d(
            sample,
            cast(torch.Tensor, layer.params["weight"]),
            cast(torch.Tensor | None, layer.params["bias"]),
            stride=1,
            padding=1,
        ).reshape(sample.shape[0], -1)
        if not torch.all(actual <= upper + 1e-6):
            raise AssertionError(f"patches soundness upper violation at sample {sample_idx}")
        if not torch.all(actual >= lower - 1e-6):
            raise AssertionError(f"patches soundness lower violation at sample {sample_idx}")
