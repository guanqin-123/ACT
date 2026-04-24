from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F

from act.back_end.bounds_dispatch import (
    get_conv_materialization_count,
    get_conv_mode,
    get_strict_patches,
    reset_conv_materialization_count,
    set_conv_mode,
    set_strict_patches,
)
from act.back_end.core import Layer, Net
from act.back_end.dual_tf import DualTF, compute_forward_bounds
from act.back_end.dual_tf.tf_forward import LinearBound
from act.back_end.layer_schema import LayerKind
from act.back_end.patches import Patches
from act.util.device_manager import initialize_device


@pytest.fixture(scope="module", autouse=True)
def _cpu_float64() -> None:
    initialize_device("cpu", "float64")


@contextmanager
def _conv_mode(mode: str, *, strict_patches: bool = False):
    prev_mode = get_conv_mode()
    prev_strict = get_strict_patches()
    set_conv_mode(mode)
    set_strict_patches(strict_patches)
    try:
        yield
    finally:
        set_conv_mode(prev_mode)
        set_strict_patches(prev_strict)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _make_input_box(*, batch_size: int, channels: int, height: int, width: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = _generator(seed)
    center = torch.randn((batch_size, channels, height, width), generator=gen, dtype=torch.float64)
    radius = 0.1 + 0.2 * torch.rand((batch_size, channels, height, width), generator=gen, dtype=torch.float64)
    return center - radius, center + radius


def _make_conv_layer(
    *,
    layer_id: int,
    in_c: int,
    out_c: int,
    height: int,
    width: int,
    seed: int,
    bias: bool = False,
) -> Layer:
    gen = _generator(seed)
    weight = torch.randn((out_c, in_c, 3, 3), generator=gen, dtype=torch.float64)
    bias_tensor = (
        torch.randn((out_c,), generator=gen, dtype=torch.float64)
        if bias
        else None
    )
    dummy = torch.zeros((1, in_c, height, width), dtype=torch.float64)
    out_h, out_w = F.conv2d(dummy, weight, None, stride=1, padding=1).shape[-2:]
    return Layer(
        id=layer_id,
        kind=LayerKind.CONV2D.value,
        params={
            "in_channels": in_c,
            "out_channels": out_c,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "groups": 1,
            "input_shape": (1, in_c, height, width),
            "output_shape": (1, out_c, out_h, out_w),
            "weight": weight,
            "bias": bias_tensor,
        },
        in_vars=[],
        out_vars=[],
    )


def _build_conv_net(*, with_second_conv: bool) -> tuple[Net, int]:
    input_vars = list(range(1 * 4 * 4))
    conv1_out = list(range(1000, 1000 + 1 * 4 * 4))
    layers = [
        Layer(
            0,
            LayerKind.INPUT.value,
            {"shape": (1, 1, 4, 4), "dtype": "float64", "num_classes": 1, "value_range": (0.0, 1.0)},
            input_vars,
            input_vars,
        ),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, input_vars, input_vars),
        _make_conv_layer(layer_id=2, in_c=1, out_c=1, height=4, width=4, seed=1, bias=False),
    ]
    preds = {0: [], 1: [0], 2: [1]}
    succs = {0: [1], 1: [2], 2: []}
    final_id = 2
    if with_second_conv:
        conv2_out = list(range(2000, 2000 + 1 * 4 * 4))
        layers.append(_make_conv_layer(layer_id=3, in_c=1, out_c=1, height=4, width=4, seed=2, bias=False))
        layers.append(Layer(4, LayerKind.ASSERT.value, {"kind": "RANGE"}, conv2_out, conv2_out))
        layers[2] = Layer(2, LayerKind.CONV2D.value, layers[2].params, input_vars, conv1_out)
        layers[3] = Layer(3, LayerKind.CONV2D.value, layers[3].params, conv1_out, conv2_out)
        preds.update({3: [2], 4: [3]})
        succs.update({2: [3], 3: [4], 4: []})
        final_id = 3
    else:
        layers.append(Layer(3, LayerKind.ASSERT.value, {"kind": "RANGE"}, conv1_out, conv1_out))
        layers[2] = Layer(2, LayerKind.CONV2D.value, layers[2].params, input_vars, conv1_out)
        preds.update({3: [2]})
        succs.update({2: [3], 3: []})
    return Net(layers=layers, preds=preds, succs=succs), final_id


def _build_dense_net() -> Net:
    input_vars = list(range(4))
    dense_out = list(range(100, 104))
    weight = torch.tensor(
        [[1.0, 0.5, -0.2, 0.3], [-0.4, 0.1, 0.7, -0.2], [0.2, -0.3, 0.4, 0.5], [0.6, 0.2, -0.1, 0.8]],
        dtype=torch.float64,
    )
    bias = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float64)
    return Net(
        layers=[
            Layer(
                0,
                LayerKind.INPUT.value,
                {"shape": (4,), "dtype": "float64", "num_classes": 1, "value_range": (0.0, 1.0)},
                input_vars,
                input_vars,
            ),
            Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, input_vars, input_vars),
            Layer(
                2,
                LayerKind.DENSE.value,
                {"weight": weight, "bias": bias, "in_features": 4, "out_features": 4},
                input_vars,
                dense_out,
            ),
            Layer(3, LayerKind.ASSERT.value, {"kind": "RANGE"}, dense_out, dense_out),
        ],
        preds={0: [], 1: [0], 2: [1], 3: [2]},
        succs={0: [1], 1: [2], 2: [3], 3: []},
    )


def _build_conv_relu_conv_net() -> tuple[Net, int]:
    input_vars = list(range(1 * 4 * 4))
    conv1_out = list(range(1000, 1000 + 1 * 4 * 4))
    relu_out = list(range(2000, 2000 + 1 * 4 * 4))
    conv2_out = list(range(3000, 3000 + 1 * 4 * 4))
    conv1 = _make_conv_layer(layer_id=2, in_c=1, out_c=1, height=4, width=4, seed=21, bias=False)
    conv2 = _make_conv_layer(layer_id=4, in_c=1, out_c=1, height=4, width=4, seed=22, bias=False)
    return (
        Net(
            layers=[
                Layer(
                    0,
                    LayerKind.INPUT.value,
                    {"shape": (1, 1, 4, 4), "dtype": "float64", "num_classes": 1, "value_range": (0.0, 1.0)},
                    input_vars,
                    input_vars,
                ),
                Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, input_vars, input_vars),
                Layer(2, LayerKind.CONV2D.value, conv1.params, input_vars, conv1_out),
                Layer(3, LayerKind.RELU.value, {}, conv1_out, relu_out),
                Layer(4, LayerKind.CONV2D.value, conv2.params, relu_out, conv2_out),
                Layer(5, LayerKind.ASSERT.value, {"kind": "RANGE"}, conv2_out, conv2_out),
            ],
            preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
            succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
        ),
        4,
    )


def test_seed_patches_when_first_layer_is_conv(monkeypatch: pytest.MonkeyPatch) -> None:
    net, _ = _build_conv_net(with_second_conv=False)
    lb, ub = _make_input_box(batch_size=2, channels=1, height=4, width=4, seed=10)
    seen_types: list[str] = []

    from act.back_end.dual_tf import tf_cnn_patches

    original = tf_cnn_patches.forward_conv2d_patches

    def wrapped(*args, **kwargs):
        seen_types.append(type(args[2][0]).__name__)
        return original(*args, **kwargs)

    monkeypatch.setattr(tf_cnn_patches, "forward_conv2d_patches", wrapped)
    with _conv_mode("patches"):
        compute_forward_bounds(net, lb, ub)
    assert seen_types == [Patches.__name__]


def test_seed_matrix_when_first_layer_is_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    net = _build_dense_net()
    lb = torch.tensor([[-1.0, -0.5, 0.2, -0.1]], dtype=torch.float64)
    ub = torch.tensor([[0.5, 0.4, 0.9, 0.7]], dtype=torch.float64)
    seen_types: list[str] = []

    original = DualTF._FORWARD_REGISTRY[LayerKind.DENSE.value]

    def wrapped(*args, **kwargs):
        seen_types.append(type(args[2][0]).__name__)
        return original(*args, **kwargs)

    monkeypatch.setitem(DualTF._FORWARD_REGISTRY, LayerKind.DENSE.value, wrapped)
    with _conv_mode("patches"):
        compute_forward_bounds(net, lb, ub)
    assert seen_types == [LinearBound.__name__]


def test_strict_patches_raises_on_dense_first() -> None:
    net = _build_dense_net()
    lb = torch.tensor([[-1.0, -0.5, 0.2, -0.1]], dtype=torch.float64)
    ub = torch.tensor([[0.5, 0.4, 0.9, 0.7]], dtype=torch.float64)
    with _conv_mode("patches", strict_patches=True), pytest.raises(
        RuntimeError,
        match="strict_patches: network does not start with Conv2d — cannot seed Patches",
    ):
        compute_forward_bounds(net, lb, ub)


def test_patches_propagate_through_conv_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    net, _ = _build_conv_net(with_second_conv=True)
    lb, ub = _make_input_box(batch_size=2, channels=1, height=4, width=4, seed=11)
    seen_types: list[str] = []

    from act.back_end.dual_tf import tf_cnn_patches

    original = tf_cnn_patches.forward_conv2d_patches

    def wrapped(*args, **kwargs):
        seen_types.append(type(args[2][0]).__name__)
        return original(*args, **kwargs)

    monkeypatch.setattr(tf_cnn_patches, "forward_conv2d_patches", wrapped)
    reset_conv_materialization_count()
    with _conv_mode("patches"):
        compute_forward_bounds(net, lb, ub)
    assert seen_types == [Patches.__name__, Patches.__name__]
    assert get_conv_materialization_count() == 0


def test_parity_matrix_vs_patches_seeded() -> None:
    net, final_id = _build_conv_net(with_second_conv=True)
    lb, ub = _make_input_box(batch_size=2, channels=1, height=4, width=4, seed=12)
    with _conv_mode("matrix"):
        matrix = compute_forward_bounds(net, lb, ub)
    with _conv_mode("patches"):
        patches = compute_forward_bounds(net, lb, ub)
    torch.testing.assert_close(patches[final_id].lb, matrix[final_id].lb, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(patches[final_id].ub, matrix[final_id].ub, rtol=1e-5, atol=1e-7)


def test_identity_resets_reseed_patches_before_next_conv(monkeypatch: pytest.MonkeyPatch) -> None:
    net, _ = _build_conv_relu_conv_net()
    lb, ub = _make_input_box(batch_size=2, channels=1, height=4, width=4, seed=13)
    seen_types: list[str] = []

    from act.back_end.dual_tf import tf_cnn_patches

    original = tf_cnn_patches.forward_conv2d_patches

    def wrapped(*args, **kwargs):
        seen_types.append(type(args[2][0]).__name__)
        return original(*args, **kwargs)

    monkeypatch.setattr(tf_cnn_patches, "forward_conv2d_patches", wrapped)
    reset_conv_materialization_count()
    with _conv_mode("patches"):
        compute_forward_bounds(net, lb, ub)
    assert seen_types == [Patches.__name__, Patches.__name__]
    assert get_conv_materialization_count() == 0
