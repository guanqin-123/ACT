from __future__ import annotations

from contextlib import contextmanager
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F

from act.back_end.bounds_dispatch import dispatch_conv_forward, get_conv_mode, set_conv_mode
from act.back_end.core import Bounds, Layer
from act.back_end.dual_tf.tf_forward import Frame, LinearBound, forward_add
from act.back_end.dual_tf.tf_mlp import forward_bn, forward_relu
from act.back_end.layer_schema import LayerKind
from act.back_end.patches import Patches
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


def _bounds_image() -> Bounds:
    return Bounds(
        lb=torch.tensor([[-0.8, -0.5, -0.2, 0.1, -0.7, -0.3, 0.0, 0.2, -0.4, -0.1, 0.2, 0.5, -0.2, 0.1, 0.4, 0.7]], dtype=torch.float64),
        ub=torch.tensor([[0.1, 0.4, 0.7, 1.0, 0.2, 0.5, 0.9, 1.2, 0.3, 0.7, 1.0, 1.3, 0.4, 0.8, 1.2, 1.5]], dtype=torch.float64),
    )


def _frame(box: Bounds) -> Frame:
    return box.lb.clone(), box.ub.clone()


def _identity_patches() -> Patches:
    pieces = torch.zeros((1, 1, 4, 4, 1, 1, 1), dtype=torch.float64)
    pieces[0, :, :, :, 0, 0, 0] = 1.0
    return Patches(
        patches=pieces,
        stride=1,
        padding=0,
        shape=tuple(int(dim) for dim in pieces.shape),
        input_shape=(1, 1, 4, 4),
        output_shape=(1, 1, 4, 4),
    )


def _conv(layer_id: int, *, bias: float | None) -> Layer:
    return Layer(
        id=layer_id,
        kind=LayerKind.CONV2D.value,
        params={
            "weight": torch.tensor([[[[0.2, -0.1, 0.0], [0.1, 0.3, -0.2], [0.0, -0.1, 0.2]]]], dtype=torch.float64),
            "bias": None if bias is None else torch.tensor([bias], dtype=torch.float64),
            "in_channels": 1,
            "out_channels": 1,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "groups": 1,
            "input_shape": (1, 1, 4, 4),
            "output_shape": (1, 1, 4, 4),
        },
        in_vars=[],
        out_vars=[],
    )


def _weight(layer: Layer) -> torch.Tensor:
    return cast(torch.Tensor, layer.params["weight"])


def _bias(layer: Layer) -> torch.Tensor | None:
    return cast(torch.Tensor | None, layer.params["bias"])


def _sample(box: Bounds, *, seed: int, count: int = 100) -> torch.Tensor:
    lb = box.lb.view(1, 1, 4, 4)
    ub = box.ub.view(1, 1, 4, 4)
    gen = torch.Generator().manual_seed(seed)
    return lb.unsqueeze(0) + torch.rand((count, *lb.shape), generator=gen, dtype=lb.dtype) * (ub - lb).unsqueeze(0)


def _assert_sound(bounds: Bounds, actual: torch.Tensor) -> None:
    flat = actual.reshape(actual.shape[0], -1)
    assert torch.all(flat <= bounds.ub + 1e-7)
    assert torch.all(flat >= bounds.lb - 1e-7)


def test_conv_bias_then_conv_chain_soundness() -> None:
    box = _bounds_image()
    conv1 = _conv(1, bias=0.1)
    conv2 = _conv(2, bias=None)
    with _conv_mode("patches"):
        first = dispatch_conv_forward(conv1, [box], [_identity_patches()], [_frame(box)], [0], False, torch.device("cpu"), torch.float64)
        assert isinstance(first[2], LinearBound)
        second = dispatch_conv_forward(conv2, [first[1]], [cast(LinearBound, first[2])], [first[3]], [0], False, torch.device("cpu"), torch.float64)
    samples = _sample(box, seed=1).squeeze(1)
    actual = F.conv2d(F.conv2d(samples, _weight(conv1), _bias(conv1), stride=1, padding=1), _weight(conv2), None, stride=1, padding=1)
    _assert_sound(second[1], actual)


def test_conv_relu_conv_chain_soundness() -> None:
    box = _bounds_image()
    conv1 = _conv(3, bias=None)
    conv2 = _conv(4, bias=None)
    relu = Layer(5, LayerKind.RELU.value, {}, [], [])
    with _conv_mode("patches"):
        first = dispatch_conv_forward(conv1, [box], [_identity_patches()], [_frame(box)], [0], False, torch.device("cpu"), torch.float64)
        relu_out = forward_relu(relu, [first[1]], [cast(LinearBound, first[2])], [first[3]], [0], False, torch.device("cpu"), torch.float64)
        assert isinstance(relu_out[2], LinearBound)
        second = dispatch_conv_forward(conv2, [relu_out[1]], [cast(LinearBound, relu_out[2])], [relu_out[3]], [0], False, torch.device("cpu"), torch.float64)
    samples = _sample(box, seed=2).squeeze(1)
    actual = F.conv2d(F.relu(F.conv2d(samples, _weight(conv1), None, stride=1, padding=1)), _weight(conv2), None, stride=1, padding=1)
    _assert_sound(second[1], actual)


def test_conv_bn_conv_chain_soundness() -> None:
    box = _bounds_image()
    conv1 = _conv(6, bias=None)
    conv2 = _conv(7, bias=None)
    bn = cast(
        Any,
        type(
            "BnStub",
            (),
            {
                "id": 8,
                "kind": LayerKind.BN.value,
                "params": {
                    "A": torch.full((16,), 1.5, dtype=torch.float64),
                    "c": torch.linspace(-0.2, 0.2, steps=16, dtype=torch.float64),
                },
            },
        )(),
    )
    with _conv_mode("patches"):
        first = dispatch_conv_forward(conv1, [box], [_identity_patches()], [_frame(box)], [0], False, torch.device("cpu"), torch.float64)
        bn_out = forward_bn(bn, [first[1]], [cast(LinearBound, first[2])], [first[3]], [0], False, torch.device("cpu"), torch.float64)
        assert isinstance(bn_out[2], LinearBound)
        second = dispatch_conv_forward(conv2, [bn_out[1]], [cast(LinearBound, bn_out[2])], [bn_out[3]], [0], False, torch.device("cpu"), torch.float64)
    samples = _sample(box, seed=3).squeeze(1)
    conv_actual = F.conv2d(samples, _weight(conv1), None, stride=1, padding=1)
    bn_actual = (conv_actual.reshape(conv_actual.shape[0], -1) * bn.params["A"] + bn.params["c"]).reshape_as(conv_actual)
    actual = F.conv2d(bn_actual, _weight(conv2), None, stride=1, padding=1)
    _assert_sound(second[1], actual)


def test_add_skip_conv_chain_soundness() -> None:
    box = _bounds_image()
    conv = _conv(9, bias=None)
    add = Layer(10, LayerKind.ADD.value, params={}, in_vars=[], out_vars=[])
    with _conv_mode("patches"):
        added = forward_add(add, [box, box], [_identity_patches(), _identity_patches()], [_frame(box), _frame(box)], [0, 1], False, torch.device("cpu"), torch.float64)
        assert isinstance(added[2], LinearBound)
        out = dispatch_conv_forward(conv, [added[1]], [cast(LinearBound, added[2])], [added[3]], [0], False, torch.device("cpu"), torch.float64)
    samples = _sample(box, seed=4).squeeze(1)
    actual = F.conv2d(samples + samples, _weight(conv), None, stride=1, padding=1)
    _assert_sound(out[1], actual)
