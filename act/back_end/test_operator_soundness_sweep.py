from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, cast

import pytest
import torch
import torch.nn.functional as F

from act.back_end.bounds_dispatch import dispatch_conv_forward, get_conv_mode, set_conv_mode
from act.back_end.core import Bounds, Layer
from act.back_end.dual_tf.dual_tf import DualTF, _FORWARD_STUBS
from act.back_end.dual_tf.tf_cnn import forward_avgpool2d, forward_maxpool2d
from act.back_end.dual_tf.tf_forward import Frame, LinearBound, forward_add, forward_concat
from act.back_end.dual_tf.tf_mlp import (
    forward_bias,
    forward_bn,
    forward_dense,
    forward_lrelu,
    forward_relu,
    forward_reshape,
    forward_scale,
)
from act.back_end.dual_tf.tf_smooth import forward_sigmoid, forward_tanh
from act.back_end.layer_schema import LayerKind
from act.back_end.patches import Patches
from act.util.device_manager import initialize_device


RTOL = 1e-5
ATOL = 1e-7
_WRAPPER_KINDS = {
    LayerKind.INPUT.value,
    LayerKind.INPUT_SPEC.value,
    LayerKind.ASSERT.value,
    LayerKind.TRANSPOSE.value,
    LayerKind.SQUEEZE.value,
    LayerKind.UNSQUEEZE.value,
}


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


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _identity_linear_bound(batch: int, dim: int) -> LinearBound:
    eye = torch.eye(dim, dtype=torch.float64).unsqueeze(0).expand(batch, -1, -1).clone()
    zeros = torch.zeros((batch, dim), dtype=torch.float64)
    return LinearBound(A_lb=eye, b_lb=zeros, A_ub=eye.clone(), b_ub=zeros.clone())


def _identity_patches(batch: int, channels: int, height: int, width: int) -> Patches:
    pieces = torch.zeros((channels, batch, height, width, channels, 1, 1), dtype=torch.float64)
    diag = torch.arange(channels)
    pieces[diag, :, :, :, diag, 0, 0] = 1.0
    return Patches(
        patches=pieces,
        stride=1,
        padding=0,
        shape=tuple(int(dim) for dim in pieces.shape),
        input_shape=(batch, channels, height, width),
        output_shape=(batch, channels, height, width),
    )


def _sample_from_box(bounds: Bounds, shape: tuple[int, ...] | None, *, seed: int, count: int = 100) -> torch.Tensor:
    lb = bounds.lb if shape is None else bounds.lb.view(*shape)
    ub = bounds.ub if shape is None else bounds.ub.view(*shape)
    return lb.unsqueeze(0) + torch.rand((count, *lb.shape), generator=_gen(seed), dtype=lb.dtype) * (ub - lb).unsqueeze(0)


def _flatten_samples(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1)


@dataclass(frozen=True)
class OperatorCase:
    kind: str
    layer: Any
    handler: Callable[..., tuple[Bounds, Bounds, object, tuple[torch.Tensor, torch.Tensor]]]
    parent_boxes: list[Bounds]
    parent_frames: list[Frame]
    parent_shapes: list[tuple[int, ...] | None]
    parent_lins_factory: Callable[[str], list[LinearBound | Patches]]
    apply: Callable[[list[torch.Tensor]], torch.Tensor]
    needs_conv_dispatch: bool = False


VECTOR_BOX = Bounds(
    lb=torch.tensor([[-0.8, -0.4, 0.1, -0.2]], dtype=torch.float64),
    ub=torch.tensor([[0.3, 0.9, 0.7, 0.6]], dtype=torch.float64),
)
VECTOR_FRAME: Frame = (VECTOR_BOX.lb.clone(), VECTOR_BOX.ub.clone())

IMAGE_BOX = Bounds(
    lb=torch.tensor([[-0.8, -0.5, -0.2, 0.1, -0.7, -0.3, 0.0, 0.2, -0.4, -0.1, 0.2, 0.5, -0.2, 0.1, 0.4, 0.7]], dtype=torch.float64),
    ub=torch.tensor([[0.1, 0.4, 0.7, 1.0, 0.2, 0.5, 0.9, 1.2, 0.3, 0.7, 1.0, 1.3, 0.4, 0.8, 1.2, 1.5]], dtype=torch.float64),
)
IMAGE_FRAME: Frame = (IMAGE_BOX.lb.clone(), IMAGE_BOX.ub.clone())

OTHER_BOX = Bounds(
    lb=torch.tensor([[-0.3, -0.1, -0.6, 0.2]], dtype=torch.float64),
    ub=torch.tensor([[0.6, 0.8, 0.2, 0.9]], dtype=torch.float64),
)
OTHER_FRAME: Frame = (OTHER_BOX.lb.clone(), OTHER_BOX.ub.clone())

CONV_WEIGHT = torch.tensor(
    [[[[0.2, -0.1, 0.0], [0.1, 0.3, -0.2], [0.0, -0.1, 0.2]]]],
    dtype=torch.float64,
)
CONV_BIAS = torch.tensor([0.05], dtype=torch.float64)


def _vector_or_patch_lins(mode: str) -> list[LinearBound | Patches]:
    if mode == "patches":
        return [_identity_patches(1, 1, 2, 2)]
    return [_identity_linear_bound(1, 4)]


def _image_or_patch_lins(mode: str) -> list[LinearBound | Patches]:
    if mode == "patches":
        return [_identity_patches(1, 1, 4, 4)]
    return [_identity_linear_bound(1, 16)]


def _add_or_concat_lins(mode: str) -> list[LinearBound | Patches]:
    if mode == "patches":
        return [_identity_patches(1, 1, 2, 2), _identity_patches(1, 1, 2, 2)]
    return [_identity_linear_bound(1, 4), _identity_linear_bound(1, 4)]


CASES: list[OperatorCase] = [
    OperatorCase(
        kind=LayerKind.DENSE.value,
        layer=Layer(10, LayerKind.DENSE.value, {"weight": torch.tensor([[0.4, -0.2, 0.1, 0.5], [-0.3, 0.6, 0.2, -0.1]], dtype=torch.float64), "bias": torch.tensor([0.1, -0.2], dtype=torch.float64), "in_features": 4, "out_features": 2}, [], []),
        handler=forward_dense,
        parent_boxes=[VECTOR_BOX],
        parent_frames=[VECTOR_FRAME],
        parent_shapes=[None],
        parent_lins_factory=lambda _mode: [_identity_linear_bound(1, 4)],
        apply=lambda xs: F.linear(xs[0], cast(torch.Tensor, cast(Layer, CASES[0].layer).params["weight"]), cast(torch.Tensor, cast(Layer, CASES[0].layer).params["bias"])),
    ),
    OperatorCase(
        kind=LayerKind.BIAS.value,
        layer=Layer(11, LayerKind.BIAS.value, {"c": torch.tensor([0.2, -0.1, 0.3, -0.2], dtype=torch.float64)}, [], []),
        handler=forward_bias,
        parent_boxes=[VECTOR_BOX],
        parent_frames=[VECTOR_FRAME],
        parent_shapes=[None],
        parent_lins_factory=lambda _mode: [_identity_linear_bound(1, 4)],
        apply=lambda xs: xs[0] + cast(torch.Tensor, cast(Layer, CASES[1].layer).params["c"]),
    ),
    OperatorCase(
        kind=LayerKind.SCALE.value,
        layer=Layer(12, LayerKind.SCALE.value, {"a": torch.tensor([1.2, -0.5, 0.8, 1.5], dtype=torch.float64)}, [], []),
        handler=forward_scale,
        parent_boxes=[VECTOR_BOX],
        parent_frames=[VECTOR_FRAME],
        parent_shapes=[None],
        parent_lins_factory=lambda _mode: [_identity_linear_bound(1, 4)],
        apply=lambda xs: xs[0] * cast(torch.Tensor, cast(Layer, CASES[2].layer).params["a"]),
    ),
    OperatorCase(
        kind=LayerKind.BN.value,
        layer=type("BnStub", (), {"id": 13, "kind": LayerKind.BN.value, "params": {"A": torch.tensor([1.4, 0.7, 1.1, 0.9], dtype=torch.float64), "c": torch.tensor([0.2, -0.1, 0.1, -0.2], dtype=torch.float64)}})(),
        handler=forward_bn,
        parent_boxes=[VECTOR_BOX],
        parent_frames=[VECTOR_FRAME],
        parent_shapes=[(1, 2, 2)],
        parent_lins_factory=_vector_or_patch_lins,
        apply=lambda xs: xs[0] * cast(torch.Tensor, cast(Layer, CASES[3].layer).params["A"]).view(1, 1, 2, 2) + cast(torch.Tensor, cast(Layer, CASES[3].layer).params["c"]).view(1, 1, 2, 2),
    ),
    OperatorCase(
        kind=LayerKind.RELU.value,
        layer=Layer(14, LayerKind.RELU.value, {}, [], []),
        handler=forward_relu,
        parent_boxes=[VECTOR_BOX],
        parent_frames=[VECTOR_FRAME],
        parent_shapes=[(1, 2, 2)],
        parent_lins_factory=_vector_or_patch_lins,
        apply=lambda xs: F.relu(xs[0]),
    ),
    OperatorCase(
        kind=LayerKind.LRELU.value,
        layer=type("LReluStub", (), {"id": 15, "kind": LayerKind.LRELU.value, "params": {"alpha": 0.1}})(),
        handler=forward_lrelu,
        parent_boxes=[VECTOR_BOX],
        parent_frames=[VECTOR_FRAME],
        parent_shapes=[None],
        parent_lins_factory=lambda _mode: [_identity_linear_bound(1, 4)],
        apply=lambda xs: F.leaky_relu(xs[0], negative_slope=0.1),
    ),
    OperatorCase(
        kind="LEAKY_RELU",
        layer=type("LeakyReluStub", (), {"id": 16, "kind": "LEAKY_RELU", "params": {"alpha": 0.2}})(),
        handler=forward_lrelu,
        parent_boxes=[VECTOR_BOX],
        parent_frames=[VECTOR_FRAME],
        parent_shapes=[None],
        parent_lins_factory=lambda _mode: [_identity_linear_bound(1, 4)],
        apply=lambda xs: F.leaky_relu(xs[0], negative_slope=0.2),
    ),
    OperatorCase(
        kind=LayerKind.SIGMOID.value,
        layer=Layer(17, LayerKind.SIGMOID.value, {}, [], []),
        handler=forward_sigmoid,
        parent_boxes=[VECTOR_BOX],
        parent_frames=[VECTOR_FRAME],
        parent_shapes=[None],
        parent_lins_factory=lambda _mode: [_identity_linear_bound(1, 4)],
        apply=lambda xs: torch.sigmoid(xs[0]),
    ),
    OperatorCase(
        kind=LayerKind.TANH.value,
        layer=Layer(18, LayerKind.TANH.value, {}, [], []),
        handler=forward_tanh,
        parent_boxes=[VECTOR_BOX],
        parent_frames=[VECTOR_FRAME],
        parent_shapes=[None],
        parent_lins_factory=lambda _mode: [_identity_linear_bound(1, 4)],
        apply=lambda xs: torch.tanh(xs[0]),
    ),
    OperatorCase(
        kind=LayerKind.CONV2D.value,
        layer=Layer(19, LayerKind.CONV2D.value, {"weight": CONV_WEIGHT, "bias": CONV_BIAS, "in_channels": 1, "out_channels": 1, "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "input_shape": (1, 1, 4, 4), "output_shape": (1, 1, 4, 4)}, [], []),
        handler=dispatch_conv_forward,
        parent_boxes=[IMAGE_BOX],
        parent_frames=[IMAGE_FRAME],
        parent_shapes=[(1, 4, 4)],
        parent_lins_factory=_image_or_patch_lins,
        apply=lambda xs: F.conv2d(xs[0], CONV_WEIGHT, CONV_BIAS, stride=1, padding=1),
        needs_conv_dispatch=True,
    ),
    OperatorCase(
        kind=LayerKind.MAXPOOL2D.value,
        layer=Layer(20, LayerKind.MAXPOOL2D.value, {"kernel_size": 2, "stride": 2, "padding": 0, "input_shape": (1, 1, 4, 4), "output_shape": (1, 1, 2, 2)}, [], []),
        handler=forward_maxpool2d,
        parent_boxes=[IMAGE_BOX],
        parent_frames=[IMAGE_FRAME],
        parent_shapes=[(1, 4, 4)],
        parent_lins_factory=_image_or_patch_lins,
        apply=lambda xs: F.max_pool2d(xs[0], kernel_size=2, stride=2, padding=0),
    ),
    OperatorCase(
        kind=LayerKind.AVGPOOL2D.value,
        layer=Layer(21, LayerKind.AVGPOOL2D.value, {"kernel_size": 2, "stride": 2, "padding": 0, "input_shape": (1, 1, 4, 4), "output_shape": (1, 1, 2, 2)}, [], []),
        handler=forward_avgpool2d,
        parent_boxes=[IMAGE_BOX],
        parent_frames=[IMAGE_FRAME],
        parent_shapes=[(1, 4, 4)],
        parent_lins_factory=_image_or_patch_lins,
        apply=lambda xs: F.avg_pool2d(xs[0], kernel_size=2, stride=2, padding=0),
    ),
    OperatorCase(
        kind=LayerKind.ADD.value,
        layer=Layer(22, LayerKind.ADD.value, {"bias": torch.tensor([0.1, -0.2, 0.3, -0.1], dtype=torch.float64)}, [], []),
        handler=forward_add,
        parent_boxes=[VECTOR_BOX, OTHER_BOX],
        parent_frames=[VECTOR_FRAME, OTHER_FRAME],
        parent_shapes=[(1, 2, 2), (1, 2, 2)],
        parent_lins_factory=_add_or_concat_lins,
        apply=lambda xs: xs[0] + xs[1] + cast(torch.Tensor, cast(Layer, CASES[12].layer).params["bias"]).view(1, 1, 2, 2),
    ),
    OperatorCase(
        kind=LayerKind.CONCAT.value,
        layer=Layer(23, LayerKind.CONCAT.value, {"concat_dim": 1}, [], []),
        handler=forward_concat,
        parent_boxes=[VECTOR_BOX, OTHER_BOX],
        parent_frames=[VECTOR_FRAME, OTHER_FRAME],
        parent_shapes=[(1, 2, 2), (1, 2, 2)],
        parent_lins_factory=_add_or_concat_lins,
        apply=lambda xs: torch.cat(xs, dim=1),
    ),
    OperatorCase(
        kind=LayerKind.FLATTEN.value,
        layer=Layer(24, LayerKind.FLATTEN.value, {}, [], []),
        handler=forward_reshape,
        parent_boxes=[IMAGE_BOX],
        parent_frames=[IMAGE_FRAME],
        parent_shapes=[(1, 4, 4)],
        parent_lins_factory=_image_or_patch_lins,
        apply=lambda xs: xs[0].reshape(xs[0].shape[0], -1),
    ),
    OperatorCase(
        kind=LayerKind.RESHAPE.value,
        layer=Layer(25, LayerKind.RESHAPE.value, {"target_shape": (1, 16)}, [], []),
        handler=forward_reshape,
        parent_boxes=[IMAGE_BOX],
        parent_frames=[IMAGE_FRAME],
        parent_shapes=[(1, 4, 4)],
        parent_lins_factory=_image_or_patch_lins,
        apply=lambda xs: xs[0].reshape(xs[0].shape[0], -1),
    ),
]


def _covered_registry_kinds() -> set[str]:
    stub_kinds = {
        kind
        for kind, handler in DualTF._FORWARD_REGISTRY.items()
        if handler in _FORWARD_STUBS
    }
    return set(DualTF._FORWARD_REGISTRY) - stub_kinds - _WRAPPER_KINDS


def _run_case(case: OperatorCase, mode: str) -> Bounds:
    parent_lins = case.parent_lins_factory(mode)
    args = (
        case.layer,
        case.parent_boxes,
        parent_lins,
        case.parent_frames,
        list(range(len(case.parent_boxes))),
        False,
        torch.device("cpu"),
        torch.float64,
    )
    if case.needs_conv_dispatch:
        with _conv_mode(mode):
            return cast(Bounds, case.handler(*args)[1])
    return cast(Bounds, case.handler(*args)[1])


def test_operator_soundness_sweep_registry_coverage() -> None:
    assert {case.kind for case in CASES} == _covered_registry_kinds()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.kind)
@pytest.mark.parametrize("mode", ["matrix", "patches"])
def test_operator_soundness_sweep(case: OperatorCase, mode: str) -> None:
    out = _run_case(case, mode)
    samples = [
        _sample_from_box(box, shape, seed=100 + index)
        for index, (box, shape) in enumerate(zip(case.parent_boxes, case.parent_shapes, strict=True))
    ]
    actual = _flatten_samples(case.apply(samples))
    torch.testing.assert_close(
        actual,
        actual.clamp(min=out.lb[0], max=out.ub[0]),
        rtol=RTOL,
        atol=ATOL,
    )
    assert torch.all(actual <= out.ub[0] + ATOL)
    assert torch.all(actual >= out.lb[0] - ATOL)
