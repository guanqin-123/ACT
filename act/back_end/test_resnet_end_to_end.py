from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportCallIssue=false, reportOptionalMemberAccess=false, reportAttributeAccessIssue=false, reportPrivateUsage=false, reportUnusedFunction=false, reportUnknownVariableType=false

from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import pytest
import torch
import torch.nn.functional as F

from act.back_end.bab.bab import verify_bab
from act.back_end.bounds_dispatch import (
    dispatch_add_forward,
    dispatch_bn_forward,
    dispatch_conv_forward,
    get_conv_materialization_count,
    get_conv_mode,
    get_strict_patches,
    reset_conv_materialization_count,
    set_conv_mode,
    set_strict_patches,
)
from act.back_end.config import BaBConfig
from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf import DualTF, compute_forward_bounds
from act.back_end.dual_tf.tf_forward import LinearBound
from act.back_end.dual_tf.tf_mlp import forward_relu
from act.back_end.layer_schema import LayerKind
from act.back_end.patches import Patches
from act.back_end.solver.solver_dual import DualSolver
from act.back_end.solver.solver_interval import TorchLPSolver
from act.front_end.specs import OutKind, OutputSpec
from act.util.device_manager import initialize_device
from scripts.bab_clamp_diagnostic import load_model_and_instance


KNOWN_3995_VNNLIB = "CIFAR100_resnet_medium_prop_idx_3995_sidx_7978_eps_0.0039.vnnlib"


@pytest.fixture(scope="module", autouse=True)
def _cpu_float64() -> None:
    initialize_device("cpu", "float64")


@contextmanager
def _conv_mode(mode: str, *, strict_patches: bool = False):
    previous_mode = get_conv_mode()
    previous_strict = get_strict_patches()
    set_conv_mode(mode)
    set_strict_patches(strict_patches)
    try:
        yield
    finally:
        set_conv_mode(previous_mode)
        set_strict_patches(previous_strict)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _make_input_box(
    *,
    batch_size: int = 2,
    channels: int = 3,
    height: int = 8,
    width: int = 8,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = _generator(seed)
    center = torch.randn((batch_size, channels, height, width), generator=gen, dtype=dtype)
    radius = 0.03 + 0.07 * torch.rand((batch_size, channels, height, width), generator=gen, dtype=dtype)
    return center - radius, center + radius


def _identity_patches(batch_size: int, channels: int, height: int, width: int) -> Patches:
    pieces = torch.zeros((channels, batch_size, height, width, channels, 1, 1), dtype=torch.float64)
    diag = torch.arange(channels)
    pieces[diag, :, :, :, diag, 0, 0] = 1.0
    return Patches(
        patches=pieces,
        stride=1,
        padding=0,
        shape=tuple(pieces.shape),
        input_shape=(batch_size, channels, height, width),
        output_shape=(batch_size, channels, height, width),
    )


def _sample_inputs(lb_4d: torch.Tensor, ub_4d: torch.Tensor, *, seed: int, count: int = 100) -> torch.Tensor:
    gen = _generator(seed)
    return lb_4d.unsqueeze(0) + torch.rand((count, *lb_4d.shape), generator=gen, dtype=lb_4d.dtype) * (ub_4d - lb_4d).unsqueeze(0)


def _assert_samples_within_bounds(
    lower: torch.Tensor,
    upper: torch.Tensor,
    actual: torch.Tensor,
    *,
    context: str,
) -> None:
    actual_flat = actual.reshape(actual.shape[0], -1)
    if torch.any(actual_flat > upper + 1e-7):
        sample_idx, feat_idx = (actual_flat > upper + 1e-7).nonzero(as_tuple=False)[0].tolist()
        raise AssertionError(f"{context}: upper violation at sample={sample_idx} feature={feat_idx}")
    if torch.any(actual_flat < lower - 1e-7):
        sample_idx, feat_idx = (actual_flat < lower - 1e-7).nonzero(as_tuple=False)[0].tolist()
        raise AssertionError(f"{context}: lower violation at sample={sample_idx} feature={feat_idx}")


def _new_vars(cursor: int, count: int) -> tuple[list[int], int]:
    return list(range(cursor, cursor + count)), cursor + count


def _make_conv_layer(
    *,
    layer_id: int,
    seed: int,
    dtype: torch.dtype,
    in_vars: list[int],
    out_vars: list[int],
    channels: int = 3,
    height: int = 8,
    width: int = 8,
) -> Layer:
    gen = _generator(seed)
    return Layer(
        id=layer_id,
        kind=LayerKind.CONV2D.value,
        params={
            "in_channels": channels,
            "out_channels": channels,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "groups": 1,
            "input_shape": (1, channels, height, width),
            "output_shape": (1, channels, height, width),
            "weight": torch.randn((channels, channels, 3, 3), generator=gen, dtype=dtype) / 4.0,
            "bias": torch.randn((channels,), generator=gen, dtype=dtype) / 10.0,
        },
        in_vars=in_vars,
        out_vars=out_vars,
    )


def _make_scale_layer(
    *,
    layer_id: int,
    seed: int,
    dtype: torch.dtype,
    vars_in: list[int],
    vars_out: list[int],
    channels: int = 3,
    height: int = 8,
    width: int = 8,
) -> Layer:
    gen = _generator(seed)
    size = channels * height * width
    scale = 0.8 + 0.4 * torch.rand((size,), generator=gen, dtype=dtype)
    return Layer(
        id=layer_id,
        kind=LayerKind.SCALE.value,
        params={"a": scale},
        in_vars=vars_in,
        out_vars=vars_out,
    )


def _make_bias_layer(
    *,
    layer_id: int,
    seed: int,
    dtype: torch.dtype,
    vars_in: list[int],
    vars_out: list[int],
    channels: int = 3,
    height: int = 8,
    width: int = 8,
) -> Layer:
    gen = _generator(seed)
    size = channels * height * width
    bias = 0.1 * torch.randn((size,), generator=gen, dtype=dtype)
    return Layer(
        id=layer_id,
        kind=LayerKind.BIAS.value,
        params={"c": bias},
        in_vars=vars_in,
        out_vars=vars_out,
    )


def _make_dense_layer(
    *,
    layer_id: int,
    seed: int,
    dtype: torch.dtype,
    in_vars: list[int],
    out_vars: list[int],
) -> Layer:
    gen = _generator(seed)
    return Layer(
        id=layer_id,
        kind=LayerKind.DENSE.value,
        params={
            "in_features": 48,
            "out_features": 10,
            "weight": torch.randn((10, 48), generator=gen, dtype=dtype) / 4.0,
            "bias": torch.randn((10,), generator=gen, dtype=dtype) / 10.0,
        },
        in_vars=in_vars,
        out_vars=out_vars,
    )


def _apply_bn(layer: Layer, x: torch.Tensor) -> torch.Tensor:
    if layer.kind == LayerKind.SCALE.value:
        scale = cast(torch.Tensor, layer.params["a"]).reshape(1, *x.shape[1:])
        return scale * x
    bias = cast(torch.Tensor, layer.params["c"]).reshape(1, *x.shape[1:])
    return x + bias


def _apply_conv(layer: Layer, x: torch.Tensor) -> torch.Tensor:
    return F.conv2d(
        x,
        cast(torch.Tensor, layer.params["weight"]),
        cast(torch.Tensor | None, layer.params["bias"]),
        stride=cast(int, layer.params["stride"]),
        padding=cast(int, layer.params["padding"]),
    )


@dataclass(frozen=True)
class BlockIds:
    conv1: int
    scale1: int
    bias1: int
    relu1: int
    conv2: int
    scale2: int
    bias2: int
    add: int
    relu2: int


@dataclass(frozen=True)
class ResNetFixture:
    net: Net
    blocks: list[BlockIds]
    pool_id: int
    flatten_id: int
    dense_id: int
    assert_id: int
    compare_layer_ids: list[int]


def _build_resnet_fixture(*, num_blocks: int, seed: int, dtype: torch.dtype = torch.float64) -> ResNetFixture:
    channels, height, width = 3, 8, 8
    feature_dim = channels * height * width
    cursor = 1000
    input_vars = list(range(feature_dim))
    layers = [
        Layer(
            0,
            LayerKind.INPUT.value,
            {"shape": (1, channels, height, width), "dtype": "float64", "num_classes": 10, "value_range": (-5.0, 5.0)},
            input_vars,
            input_vars,
        ),
        Layer(1, LayerKind.INPUT_SPEC.value, {"kind": "BOX"}, input_vars, input_vars),
    ]
    preds: dict[int, list[int]] = {0: [], 1: [0]}
    succs: dict[int, list[int]] = {0: [1], 1: []}
    compare_layer_ids: list[int] = []
    blocks: list[BlockIds] = []

    prev_pred = 1
    prev_vars = input_vars
    next_layer_id = 2
    for block_idx in range(num_blocks):
        conv1_vars, cursor = _new_vars(cursor, feature_dim)
        scale1_vars, cursor = _new_vars(cursor, feature_dim)
        bias1_vars, cursor = _new_vars(cursor, feature_dim)
        relu1_vars, cursor = _new_vars(cursor, feature_dim)
        conv2_vars, cursor = _new_vars(cursor, feature_dim)
        scale2_vars, cursor = _new_vars(cursor, feature_dim)
        bias2_vars, cursor = _new_vars(cursor, feature_dim)
        add_vars, cursor = _new_vars(cursor, feature_dim)
        relu2_vars, cursor = _new_vars(cursor, feature_dim)

        block = BlockIds(
            conv1=next_layer_id,
            scale1=next_layer_id + 1,
            bias1=next_layer_id + 2,
            relu1=next_layer_id + 3,
            conv2=next_layer_id + 4,
            scale2=next_layer_id + 5,
            bias2=next_layer_id + 6,
            add=next_layer_id + 7,
            relu2=next_layer_id + 8,
        )
        next_layer_id += 9

        conv1 = _make_conv_layer(layer_id=block.conv1, seed=seed + 10 * block_idx + 1, dtype=dtype, in_vars=prev_vars, out_vars=conv1_vars)
        scale1 = _make_scale_layer(layer_id=block.scale1, seed=seed + 10 * block_idx + 2, dtype=dtype, vars_in=conv1_vars, vars_out=scale1_vars)
        bias1 = _make_bias_layer(layer_id=block.bias1, seed=seed + 10 * block_idx + 3, dtype=dtype, vars_in=scale1_vars, vars_out=bias1_vars)
        relu1 = Layer(block.relu1, LayerKind.RELU.value, {}, bias1_vars, relu1_vars)
        conv2 = _make_conv_layer(layer_id=block.conv2, seed=seed + 10 * block_idx + 3, dtype=dtype, in_vars=relu1_vars, out_vars=conv2_vars)
        scale2 = _make_scale_layer(layer_id=block.scale2, seed=seed + 10 * block_idx + 4, dtype=dtype, vars_in=conv2_vars, vars_out=scale2_vars)
        bias2 = _make_bias_layer(layer_id=block.bias2, seed=seed + 10 * block_idx + 5, dtype=dtype, vars_in=scale2_vars, vars_out=bias2_vars)
        add = Layer(block.add, LayerKind.ADD.value, {}, bias2_vars, add_vars)
        relu2 = Layer(block.relu2, LayerKind.RELU.value, {}, add_vars, relu2_vars)
        layers.extend([conv1, scale1, bias1, relu1, conv2, scale2, bias2, add, relu2])

        preds[block.conv1] = [prev_pred]
        succs.setdefault(prev_pred, []).append(block.conv1)
        succs[block.conv1] = [block.scale1]
        preds[block.scale1] = [block.conv1]
        succs[block.scale1] = [block.bias1]
        preds[block.bias1] = [block.scale1]
        succs[block.bias1] = [block.relu1]
        preds[block.relu1] = [block.bias1]
        succs[block.relu1] = [block.conv2]
        preds[block.conv2] = [block.relu1]
        succs[block.conv2] = [block.scale2]
        preds[block.scale2] = [block.conv2]
        succs[block.scale2] = [block.bias2]
        preds[block.bias2] = [block.scale2]
        succs[block.bias2] = [block.add]
        preds[block.add] = [block.bias2, prev_pred]
        succs.setdefault(prev_pred, []).append(block.add)
        succs[block.add] = [block.relu2]
        preds[block.relu2] = [block.add]
        succs[block.relu2] = []

        compare_layer_ids.extend([block.conv1, block.scale1, block.bias1, block.relu1, block.conv2, block.scale2, block.bias2, block.add, block.relu2])
        blocks.append(block)
        prev_pred = block.relu2
        prev_vars = relu2_vars

    pool_vars, cursor = _new_vars(cursor, 3 * 4 * 4)
    flat_vars, cursor = _new_vars(cursor, 48)
    dense_vars, cursor = _new_vars(cursor, 10)
    pool_id = next_layer_id
    flatten_id = next_layer_id + 1
    dense_id = next_layer_id + 2
    assert_id = next_layer_id + 3

    layers.extend(
        [
            Layer(
                pool_id,
                LayerKind.AVGPOOL2D.value,
                {"kernel_size": 2, "stride": 2, "padding": 0, "input_shape": (1, 3, 8, 8), "output_shape": (1, 3, 4, 4)},
                prev_vars,
                pool_vars,
            ),
            Layer(flatten_id, LayerKind.FLATTEN.value, {}, pool_vars, flat_vars),
            _make_dense_layer(layer_id=dense_id, seed=seed + 999, dtype=dtype, in_vars=flat_vars, out_vars=dense_vars),
            Layer(assert_id, LayerKind.ASSERT.value, {"kind": OutKind.RANGE}, dense_vars, dense_vars),
        ]
    )
    preds[pool_id] = [prev_pred]
    succs.setdefault(prev_pred, []).append(pool_id)
    succs[pool_id] = [flatten_id]
    preds[flatten_id] = [pool_id]
    succs[flatten_id] = [dense_id]
    preds[dense_id] = [flatten_id]
    succs[dense_id] = [assert_id]
    preds[assert_id] = [dense_id]
    succs[assert_id] = []
    compare_layer_ids.extend([pool_id, flatten_id, dense_id, assert_id])

    return ResNetFixture(
        net=Net(layers=layers, preds=preds, succs=succs),
        blocks=blocks,
        pool_id=pool_id,
        flatten_id=flatten_id,
        dense_id=dense_id,
        assert_id=assert_id,
        compare_layer_ids=compare_layer_ids,
    )


def _evaluate_blocks(fixture: ResNetFixture, x: torch.Tensor, *, num_blocks: int) -> torch.Tensor:
    out = x
    for block in fixture.blocks[:num_blocks]:
        residual = out
        out = _apply_conv(fixture.net.by_id[block.conv1], out)
        out = _apply_bn(fixture.net.by_id[block.scale1], out)
        out = _apply_bn(fixture.net.by_id[block.bias1], out)
        out = F.relu(out)
        out = _apply_conv(fixture.net.by_id[block.conv2], out)
        out = _apply_bn(fixture.net.by_id[block.scale2], out)
        out = _apply_bn(fixture.net.by_id[block.bias2], out)
        out = F.relu(out + residual)
    return out


def _evaluate_full_network(fixture: ResNetFixture, x: torch.Tensor) -> torch.Tensor:
    out = _evaluate_blocks(fixture, x, num_blocks=len(fixture.blocks))
    out = F.avg_pool2d(out, kernel_size=2, stride=2)
    out = out.reshape(out.shape[0], -1)
    dense = fixture.net.by_id[fixture.dense_id]
    return F.linear(out, dense.params["weight"], dense.params["bias"])


def _make_bn_stub(scale_layer: Layer, bias_layer: Layer) -> object:
    return type(
        "BNStub",
        (),
        {
            "id": scale_layer.id,
            "kind": LayerKind.BN.value,
            "params": {
                "A": scale_layer.params["a"],
                "c": bias_layer.params["c"],
            },
        },
    )()


def _output_spec(batch_size: int, dtype: torch.dtype) -> OutputSpec:
    y_true = torch.tensor([idx % 10 for idx in range(batch_size)], dtype=torch.int64)
    margin = torch.tensor([0.0], dtype=dtype)
    return OutputSpec(kind=OutKind.MARGIN_ROBUST, y_true=y_true, margin=margin)


def _run_forward(fixture: ResNetFixture, lb_4d: torch.Tensor, ub_4d: torch.Tensor, *, mode: str) -> dict[int, Bounds]:
    with _conv_mode(mode):
        return compute_forward_bounds(fixture.net, lb_4d, ub_4d, post_activation=True)


def _run_margins(fixture: ResNetFixture, bounds_dict: dict[int, Bounds], *, batch_size: int) -> torch.Tensor:
    spec = _output_spec(batch_size, bounds_dict[fixture.dense_id].lb.dtype)
    logits = bounds_dict[fixture.dense_id]
    y_true = torch.as_tensor(spec.y_true, dtype=torch.int64).reshape(-1)
    lb_true = logits.lb[torch.arange(batch_size), y_true]
    margins = lb_true.unsqueeze(1) - logits.ub
    margins[torch.arange(batch_size), y_true] = 0.0
    return margins


def test_resnet_block_parity_matrix_vs_patches() -> None:
    fixture = _build_resnet_fixture(num_blocks=3, seed=11)
    lb_4d, ub_4d = _make_input_box(seed=101)
    matrix = _run_forward(fixture, lb_4d, ub_4d, mode="matrix")
    patches = _run_forward(fixture, lb_4d, ub_4d, mode="patches")
    for layer_id in fixture.compare_layer_ids:
        torch.testing.assert_close(matrix[layer_id].lb, patches[layer_id].lb, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(matrix[layer_id].ub, patches[layer_id].ub, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(
        _run_margins(fixture, matrix, batch_size=lb_4d.shape[0]),
        _run_margins(fixture, patches, batch_size=lb_4d.shape[0]),
        rtol=1e-5,
        atol=1e-7,
    )


def test_resnet_block_parity_single_block() -> None:
    fixture = _build_resnet_fixture(num_blocks=1, seed=21)
    lb_4d, ub_4d = _make_input_box(seed=202)
    matrix = _run_forward(fixture, lb_4d, ub_4d, mode="matrix")
    patches = _run_forward(fixture, lb_4d, ub_4d, mode="patches")
    for layer_id in fixture.compare_layer_ids:
        torch.testing.assert_close(matrix[layer_id].lb, patches[layer_id].lb, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(matrix[layer_id].ub, patches[layer_id].ub, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(
        _run_margins(fixture, matrix, batch_size=lb_4d.shape[0]),
        _run_margins(fixture, patches, batch_size=lb_4d.shape[0]),
        rtol=1e-5,
        atol=1e-7,
    )


def test_resnet_block_parity_identity_skip_reseeds_patches_end_to_end(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fixture = _build_resnet_fixture(num_blocks=1, seed=31)
    block = fixture.blocks[0]
    lb_4d, ub_4d = _make_input_box(seed=303)
    box = Bounds(lb=lb_4d.reshape(lb_4d.shape[0], -1), ub=ub_4d.reshape(ub_4d.shape[0], -1))
    frame = (box.lb, box.ub)
    patches_in = _identity_patches(lb_4d.shape[0], 3, 8, 8)
    bn1 = _make_bn_stub(fixture.net.by_id[block.scale1], fixture.net.by_id[block.bias1])
    bn2 = _make_bn_stub(fixture.net.by_id[block.scale2], fixture.net.by_id[block.bias2])
    reset_conv_materialization_count()
    with caplog.at_level("WARNING"):
        with _conv_mode("patches", strict_patches=False):
            conv1 = dispatch_conv_forward(fixture.net.by_id[block.conv1], [box], [patches_in], [frame], [1], False, torch.device("cpu"), torch.float64)
            bn1_out = dispatch_bn_forward(bn1, [conv1[1]], [conv1[2]], [conv1[3]], [block.conv1], False, torch.device("cpu"), torch.float64)
            relu1 = forward_relu(fixture.net.by_id[block.relu1], [bn1_out[1]], [bn1_out[2]], [bn1_out[3]], [block.scale1], False, torch.device("cpu"), torch.float64)
            conv2 = dispatch_conv_forward(fixture.net.by_id[block.conv2], [relu1[1]], [relu1[2]], [relu1[3]], [block.relu1], False, torch.device("cpu"), torch.float64)
            bn2_out = dispatch_bn_forward(bn2, [conv2[1]], [conv2[2]], [conv2[3]], [block.conv2], False, torch.device("cpu"), torch.float64)
            added = dispatch_add_forward(fixture.net.by_id[block.add], [bn2_out[1], box], [bn2_out[2], patches_in], [bn2_out[3], frame], [block.scale2, 1], False, torch.device("cpu"), torch.float64)
    assert isinstance(conv1[2], LinearBound)
    assert isinstance(relu1[2], LinearBound)
    assert isinstance(conv2[2], LinearBound)
    assert isinstance(added[2], LinearBound)
    assert get_conv_materialization_count() == 0
    assert not any(
        "mixed Patches+LinearBound" in record.message or "falling back to matrix path" in record.message
        for record in caplog.records
    )


def test_resnet_block_soundness_single_block() -> None:
    fixture = _build_resnet_fixture(num_blocks=1, seed=41)
    lb_4d, ub_4d = _make_input_box(seed=404)
    bounds = _run_forward(fixture, lb_4d, ub_4d, mode="patches")
    samples = _sample_inputs(lb_4d, ub_4d, seed=405)
    lower = bounds[fixture.blocks[0].relu2].lb
    upper = bounds[fixture.blocks[0].relu2].ub
    for sample_idx, sample in enumerate(samples):
        actual = _evaluate_blocks(fixture, sample, num_blocks=1)
        _assert_samples_within_bounds(lower, upper, actual, context=f"single_block[{sample_idx}]")


def test_resnet_block_soundness_three_blocks() -> None:
    fixture = _build_resnet_fixture(num_blocks=3, seed=51)
    lb_4d, ub_4d = _make_input_box(seed=505)
    bounds = _run_forward(fixture, lb_4d, ub_4d, mode="patches")
    samples = _sample_inputs(lb_4d, ub_4d, seed=506)
    lower = bounds[fixture.dense_id].lb
    upper = bounds[fixture.dense_id].ub
    for sample_idx, sample in enumerate(samples):
        actual = _evaluate_full_network(fixture, sample)
        _assert_samples_within_bounds(lower, upper, actual, context=f"three_blocks[{sample_idx}]")


def test_resnet_block_soundness_patches_vs_matrix_same_box() -> None:
    fixture = _build_resnet_fixture(num_blocks=3, seed=61)
    lb_4d, ub_4d = _make_input_box(seed=606)
    matrix = _run_forward(fixture, lb_4d, ub_4d, mode="matrix")
    patches = _run_forward(fixture, lb_4d, ub_4d, mode="patches")
    samples = _sample_inputs(lb_4d, ub_4d, seed=607)
    for sample_idx, sample in enumerate(samples):
        actual = _evaluate_full_network(fixture, sample)
        _assert_samples_within_bounds(matrix[fixture.dense_id].lb, matrix[fixture.dense_id].ub, actual, context=f"matrix[{sample_idx}]")
        _assert_samples_within_bounds(patches[fixture.dense_id].lb, patches[fixture.dense_id].ub, actual, context=f"patches[{sample_idx}]")


def test_resnet_end_to_end_no_unsound_verdict() -> None:
    initialize_device("cpu", "float32")
    try:
        net, _vnnlib_name = load_model_and_instance(instance_idx=0, vnnlib_name=KNOWN_3995_VNNLIB)
        config = BaBConfig(
            branching_method="babsr",
            bounding_method="bfs",
            subproblem_batch_size=4,
            eta_iters=2,
            lr_eta=0.05,
            max_nodes=2,
            max_depth=2,
            verbose=False,
        )
        with _conv_mode("patches"):
            result = verify_bab(
                net,
                solver=TorchLPSolver(),
                dual_solver=DualSolver(DualTF()),
                config=config,
                time_budget_s=1.0,
            )
        assert result.status.name in {"CERTIFIED", "UNKNOWN", "TIMEOUT"}
    finally:
        initialize_device("cpu", "float64")
