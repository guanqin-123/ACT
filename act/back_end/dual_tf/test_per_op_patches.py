# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false, reportArgumentType=false, reportOperatorIssue=false

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F

from act.back_end.core import Bounds, Layer
from act.back_end.dual_tf.tf_cnn import forward_avgpool2d, forward_maxpool2d
from act.back_end.dual_tf.tf_cnn_patches import linear_form_patches_to_tensor, linear_form_tensor_to_patches
from act.back_end.dual_tf.tf_forward import Frame, LinearBound, forward_add, forward_concat
from act.back_end.dual_tf.tf_mlp import (
    backward_bn,
    backward_dense,
    backward_relu,
    dual_bn_backward,
    dual_dense_backward,
    dual_relu_backward,
    forward_bn,
    forward_relu,
)
from act.back_end.layer_schema import LayerKind
from act.back_end.patches import Patches
from act.util.device_manager import initialize_device


@pytest.fixture(scope="module", autouse=True)
def _cpu_float64() -> None:
    initialize_device("cpu", "float64")


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _bounds(lb: list[list[float]], ub: list[list[float]]) -> Bounds:
    return Bounds(lb=torch.tensor(lb, dtype=torch.float64), ub=torch.tensor(ub, dtype=torch.float64))


def _identity_linear_bound(batch: int, dim: int) -> LinearBound:
    eye = torch.eye(dim, dtype=torch.float64).unsqueeze(0).expand(batch, -1, -1).clone()
    zeros = torch.zeros((batch, dim), dtype=torch.float64)
    return LinearBound(A_lb=eye, b_lb=zeros, A_ub=eye.clone(), b_ub=zeros.clone())


def _feature_identity_patches(batch: int, channels: int, height: int, width: int) -> Patches:
    pieces = torch.zeros((channels, batch, height, width, channels, 1, 1), dtype=torch.float64)
    diag = torch.arange(channels)
    pieces[diag, :, :, :, diag, 0, 0] = 1.0
    return Patches(
        patches=pieces,
        stride=1,
        padding=0,
        shape=tuple(pieces.shape),
        input_shape=(batch, channels, height, width),
        output_shape=(batch, channels, height, width),
    )


def _random_patches(batch: int, channels: int, height: int, width: int, *, seed: int) -> Patches:
    pieces = torch.randn((channels, batch, height, width, channels, 1, 1), generator=_gen(seed), dtype=torch.float64)
    return Patches(
        patches=pieces,
        stride=1,
        padding=0,
        shape=tuple(pieces.shape),
        input_shape=(batch, channels, height, width),
        output_shape=(batch, channels, height, width),
    )


def _relu_layer(alpha: torch.Tensor | None = None) -> Any:
    params: dict[str, Any] = {}
    if alpha is not None:
        params["alpha"] = alpha
    return type("ReluStub", (), {"id": 10, "kind": LayerKind.RELU.value, "params": params})()


def _bn_layer(scale: torch.Tensor, bias: torch.Tensor) -> Any:
    return type("BnStub", (), {"id": 11, "kind": LayerKind.BN.value, "params": {"A": scale, "c": bias}})()


def _dense_layer(weight: torch.Tensor, bias: torch.Tensor | None = None, **params: Any) -> Any:
    merged = {
        "weight": weight,
        "bias": bias,
        "in_features": int(weight.shape[1]),
        "out_features": int(weight.shape[0]),
        **params,
    }
    return type("DenseStub", (), {"id": 12, "kind": LayerKind.DENSE.value, "params": merged})()


def _pool_layer(kind: str) -> Layer:
    return Layer(
        id=13,
        kind=kind,
        params={"kernel_size": 2, "stride": 2, "padding": 0, "input_shape": (1, 1, 4, 4), "output_shape": (1, 1, 2, 2)},
        in_vars=[],
        out_vars=[],
    )


def _frame_from_bounds(box: Bounds) -> Frame:
    return box.lb.clone(), box.ub.clone()


def _sample_box(bounds: Bounds, shape: tuple[int, ...], *, seed: int, count: int = 32) -> torch.Tensor:
    lb = bounds.lb.view(shape)
    ub = bounds.ub.view(shape)
    return lb.unsqueeze(0) + torch.rand((count, *lb.shape), generator=_gen(seed), dtype=lb.dtype) * (ub - lb).unsqueeze(0)


def _assert_objective_sound(pred_coeff: torch.Tensor, contrib: torch.Tensor, bounds: Bounds, actual: torch.Tensor, *, tol: float = 1e-7) -> None:
    lb_obj = (pred_coeff.clamp(min=0) * bounds.lb).sum(dim=-1) + (pred_coeff.clamp(max=0) * bounds.ub).sum(dim=-1) - contrib
    ub_obj = (pred_coeff.clamp(min=0) * bounds.ub).sum(dim=-1) + (pred_coeff.clamp(max=0) * bounds.lb).sum(dim=-1) - contrib
    assert torch.all(actual <= ub_obj + tol)
    assert torch.all(actual >= lb_obj - tol)


def test_forward_relu_patches_preserves_patches() -> None:
    box = _bounds([[-1.0, -0.2, 0.1, -0.5]], [[0.6, 0.4, 0.9, 0.3]])
    alpha = torch.tensor([[[[0.2, 0.6], [1.0, 0.0]]]], dtype=torch.float64)
    layer = _relu_layer(alpha)
    patches = _feature_identity_patches(1, 1, 2, 2)
    _stored, out, lin, _frame = forward_relu(layer, [box], [patches], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    assert isinstance(lin, Patches)
    torch.testing.assert_close(out.lb, box.lb.clamp(min=0))
    torch.testing.assert_close(out.ub, box.ub.clamp(min=0))


def test_forward_relu_patches_alpha_broadcasts_over_batch() -> None:
    box = _bounds([[-1.0, -0.2, 0.1, -0.5], [-0.4, 0.2, -0.1, 0.3]], [[0.6, 0.4, 0.9, 0.3], [0.7, 0.8, 0.5, 0.9]])
    alpha = torch.tensor([[[[0.25, 0.75], [0.4, 0.9]]]], dtype=torch.float64)
    patches = _feature_identity_patches(2, 1, 2, 2)
    _stored, _out, lin, _frame = forward_relu(_relu_layer(alpha), [box], [patches], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    dense = cast(Patches, lin).to_matrix((2, 1, 2, 2))
    assert dense.shape == (2, 4, 4)
    torch.testing.assert_close(torch.diag(dense[0]), torch.tensor([0.25, 0.75, 1.0, 0.9], dtype=torch.float64))
    torch.testing.assert_close(torch.diag(dense[1]), torch.tensor([0.25, 1.0, 0.4, 1.0], dtype=torch.float64))


def test_forward_relu_patches_stable_on_off_masks() -> None:
    box = _bounds([[0.1, -1.0, 0.2, -0.3]], [[0.5, -0.2, 0.4, -0.1]])
    patches = _feature_identity_patches(1, 1, 2, 2)
    _stored, _out, lin, _frame = forward_relu(_relu_layer(), [box], [patches], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    dense = cast(Patches, lin).to_matrix((1, 1, 2, 2))[0]
    expected = torch.diag(torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(dense, expected)


def test_forward_relu_patches_post_activation_stays_patches() -> None:
    box = _bounds([[-1.0, 0.2, -0.3, 0.1]], [[0.4, 0.8, 0.5, 0.9]])
    patches = _feature_identity_patches(1, 1, 2, 2)
    _stored, _out, lin, _frame = forward_relu(_relu_layer(), [box], [patches], [_frame_from_bounds(box)], [0], True, torch.device("cpu"), torch.float64)
    assert isinstance(lin, Patches)


def test_forward_relu_patches_soundness_sampled() -> None:
    box = _bounds([[-1.0, -0.2, 0.1, -0.5]], [[0.6, 0.4, 0.9, 0.3]])
    patches = _feature_identity_patches(1, 1, 2, 2)
    _stored, out, _lin, _frame = forward_relu(_relu_layer(), [box], [patches], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    samples = _sample_box(box, (1, 1, 2, 2), seed=1)
    actual = F.relu(samples).reshape(samples.shape[0], -1)
    assert torch.all(actual <= out.ub + 1e-7)
    assert torch.all(actual >= out.lb - 1e-7)


def test_backward_relu_patches_matches_matrix_path() -> None:
    coeffs = torch.tensor([[1.0, -0.5, 0.2, -0.3]], dtype=torch.float64)
    bounds = _bounds([[-1.0, -0.2, 0.1, -0.5]], [[0.6, 0.4, 0.9, 0.3]])
    nu = linear_form_tensor_to_patches(coeffs, (1, 1, 2, 2))
    pred_nus, contrib = backward_relu(_relu_layer(), nu, {10: bounds}, [0])
    expected_nu, expected_contrib = dual_relu_backward(coeffs, bounds)
    torch.testing.assert_close(linear_form_patches_to_tensor(cast(Patches, pred_nus[0]), (1, 1, 2, 2)), expected_nu)
    torch.testing.assert_close(contrib, expected_contrib)


def test_backward_relu_patches_stable_on_passthrough() -> None:
    coeffs = torch.tensor([[1.0, 2.0, -3.0, 4.0]], dtype=torch.float64)
    bounds = _bounds([[0.1, 0.2, 0.3, 0.4]], [[0.5, 0.6, 0.7, 0.8]])
    pred_nus, _contrib = backward_relu(_relu_layer(), linear_form_tensor_to_patches(coeffs, (1, 1, 2, 2)), {10: bounds}, [0])
    torch.testing.assert_close(linear_form_patches_to_tensor(cast(Patches, pred_nus[0]), (1, 1, 2, 2)), coeffs)


def test_backward_relu_patches_stable_off_zero() -> None:
    coeffs = torch.tensor([[1.0, 2.0, -3.0, 4.0]], dtype=torch.float64)
    bounds = _bounds([[-0.5, -0.6, -0.7, -0.8]], [[-0.1, -0.2, -0.3, -0.4]])
    pred_nus, _contrib = backward_relu(_relu_layer(), linear_form_tensor_to_patches(coeffs, (1, 1, 2, 2)), {10: bounds}, [0])
    torch.testing.assert_close(linear_form_patches_to_tensor(cast(Patches, pred_nus[0]), (1, 1, 2, 2)), torch.zeros_like(coeffs))


def test_backward_relu_patches_ambiguous_uses_upper_slope() -> None:
    coeffs = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
    bounds = _bounds([[-1.0, -1.0, -1.0, -1.0]], [[1.0, 3.0, 1.0, 3.0]])
    pred_nus, _contrib = backward_relu(_relu_layer(), linear_form_tensor_to_patches(coeffs, (1, 1, 2, 2)), {10: bounds}, [0])
    expected, _ = dual_relu_backward(coeffs, bounds)
    torch.testing.assert_close(linear_form_patches_to_tensor(cast(Patches, pred_nus[0]), (1, 1, 2, 2)), expected)


def test_backward_relu_patches_soundness_sampled() -> None:
    coeffs = torch.tensor([[0.6, -1.2, 0.5, 0.3]], dtype=torch.float64)
    bounds = _bounds([[-1.0, -0.2, 0.1, -0.5]], [[0.6, 0.4, 0.9, 0.3]])
    pred_nus, contrib = backward_relu(_relu_layer(), linear_form_tensor_to_patches(coeffs, (1, 1, 2, 2)), {10: bounds}, [0])
    pred_coeff = linear_form_patches_to_tensor(cast(Patches, pred_nus[0]), (1, 1, 2, 2))
    samples = _sample_box(bounds, (1, 1, 2, 2), seed=2)
    actual = (F.relu(samples).reshape(samples.shape[0], -1) * coeffs).sum(dim=-1)
    _assert_objective_sound(pred_coeff, contrib, bounds, actual)


def test_forward_add_both_patches_preserves_patches() -> None:
    boxes = [_bounds([[0.0, 0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6, 0.7]]), _bounds([[0.1, 0.0, 0.3, 0.2]], [[0.5, 0.4, 0.7, 0.6]])]
    patches = [_random_patches(1, 1, 2, 2, seed=3), _random_patches(1, 1, 2, 2, seed=4)]
    layer = Layer(id=20, kind=LayerKind.ADD.value, params={"bias": torch.tensor([0.1, -0.1, 0.2, -0.2], dtype=torch.float64)}, in_vars=[], out_vars=[])
    _stored, out, lin, _frame = forward_add(layer, boxes, patches, [_frame_from_bounds(boxes[0]), _frame_from_bounds(boxes[0])], [0, 1], False, torch.device("cpu"), torch.float64)
    assert isinstance(lin, Patches)
    expected = patches[0].to_matrix((1, 1, 2, 2)) + patches[1].to_matrix((1, 1, 2, 2))
    torch.testing.assert_close(cast(Patches, lin).to_matrix((1, 1, 2, 2)), expected)
    torch.testing.assert_close(out.lb, boxes[0].lb + boxes[1].lb + layer.params["bias"])


def test_forward_concat_both_patches_preserves_patches() -> None:
    boxes = [_bounds([[0.0, 0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6, 0.7]]), _bounds([[0.1, 0.0, 0.3, 0.2]], [[0.5, 0.4, 0.7, 0.6]])]
    patches = [_random_patches(1, 1, 2, 2, seed=5), _random_patches(1, 1, 2, 2, seed=6)]
    layer = Layer(id=21, kind=LayerKind.CONCAT.value, params={"concat_dim": 1}, in_vars=[], out_vars=[])
    _stored, out, lin, _frame = forward_concat(layer, boxes, patches, [_frame_from_bounds(boxes[0]), _frame_from_bounds(boxes[0])], [0, 1], False, torch.device("cpu"), torch.float64)
    assert isinstance(lin, Patches)
    expected = torch.cat([patches[0].to_matrix((1, 1, 2, 2)), patches[1].to_matrix((1, 1, 2, 2))], dim=1)
    torch.testing.assert_close(cast(Patches, lin).to_matrix((1, 1, 2, 2)), expected)
    torch.testing.assert_close(out.lb, torch.cat([boxes[0].lb, boxes[1].lb], dim=1))


def test_forward_add_mixed_warns_and_materializes(caplog: pytest.LogCaptureFixture) -> None:
    box = _bounds([[0.0, 0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6, 0.7]])
    patch = _random_patches(1, 1, 2, 2, seed=7)
    lin = _identity_linear_bound(1, 4)
    layer = Layer(id=22, kind=LayerKind.ADD.value, params={}, in_vars=[], out_vars=[])
    with caplog.at_level("WARNING"):
        _stored, _out, lin_out, _frame = forward_add(layer, [box, box], [patch, lin], [_frame_from_bounds(box), _frame_from_bounds(box)], [0, 1], False, torch.device("cpu"), torch.float64)
    assert any("mixed Patches+LinearBound" in record.message for record in caplog.records)
    assert isinstance(lin_out, LinearBound)


def test_forward_concat_mixed_warns_and_materializes(caplog: pytest.LogCaptureFixture) -> None:
    box = _bounds([[0.0, 0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6, 0.7]])
    patch = _random_patches(1, 1, 2, 2, seed=8)
    lin = _identity_linear_bound(1, 4)
    layer = Layer(id=23, kind=LayerKind.CONCAT.value, params={"concat_dim": 1}, in_vars=[], out_vars=[])
    with caplog.at_level("WARNING"):
        _stored, _out, lin_out, _frame = forward_concat(layer, [box, box], [patch, lin], [_frame_from_bounds(box), _frame_from_bounds(box)], [0, 1], False, torch.device("cpu"), torch.float64)
    assert any("mixed Patches+LinearBound" in record.message for record in caplog.records)
    assert isinstance(lin_out, LinearBound)


def test_forward_add_identity_plus_patches() -> None:
    box = _bounds([[0.0, 0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6, 0.7]])
    identity = _feature_identity_patches(1, 1, 2, 2)
    patch = _random_patches(1, 1, 2, 2, seed=9)
    layer = Layer(id=24, kind=LayerKind.ADD.value, params={}, in_vars=[], out_vars=[])
    _stored, _out, lin, _frame = forward_add(layer, [box, box], [identity, patch], [_frame_from_bounds(box), _frame_from_bounds(box)], [0, 1], False, torch.device("cpu"), torch.float64)
    expected = identity.to_matrix((1, 1, 2, 2)) + patch.to_matrix((1, 1, 2, 2))
    torch.testing.assert_close(cast(Patches, lin).to_matrix((1, 1, 2, 2)), expected)


def test_forward_add_soundness_preserved() -> None:
    boxes = [_bounds([[-0.2, -0.1, 0.0, 0.1]], [[0.3, 0.4, 0.5, 0.6]]), _bounds([[-0.1, 0.0, -0.2, 0.1]], [[0.2, 0.3, 0.4, 0.5]])]
    bias = torch.tensor([0.1, -0.2, 0.3, -0.1], dtype=torch.float64)
    layer = Layer(id=25, kind=LayerKind.ADD.value, params={"bias": bias}, in_vars=[], out_vars=[])
    _stored, out, _lin, _frame = forward_add(layer, boxes, [_random_patches(1, 1, 2, 2, seed=10), _random_patches(1, 1, 2, 2, seed=11)], [_frame_from_bounds(boxes[0]), _frame_from_bounds(boxes[0])], [0, 1], False, torch.device("cpu"), torch.float64)
    samples_a = _sample_box(boxes[0], (1, 1, 2, 2), seed=3)
    samples_b = _sample_box(boxes[1], (1, 1, 2, 2), seed=4)
    actual = (samples_a + samples_b + bias.view(1, 1, 1, 2, 2)).reshape(samples_a.shape[0], -1)
    assert torch.all(actual <= out.ub + 1e-7)
    assert torch.all(actual >= out.lb - 1e-7)


def test_forward_bn_patches_path_scales_coefficients() -> None:
    box = _bounds([[-1.0, -0.5, 0.2, 0.1]], [[0.5, 0.6, 0.9, 0.8]])
    scale = torch.tensor([2.0], dtype=torch.float64)
    bias = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float64)
    patches = _feature_identity_patches(1, 1, 2, 2)
    _stored, _out, lin, _frame = forward_bn(_bn_layer(scale, bias), [box], [patches], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    expected = 2.0 * torch.eye(4, dtype=torch.float64).unsqueeze(0)
    torch.testing.assert_close(cast(Patches, lin).to_matrix((1, 1, 2, 2)), expected)


def test_backward_bn_patches_path_matches_matrix() -> None:
    coeffs = torch.tensor([[1.0, -0.5, 0.2, -0.3]], dtype=torch.float64)
    box = _bounds([[-1.0, -0.5, 0.2, 0.1]], [[0.5, 0.6, 0.9, 0.8]])
    layer = _bn_layer(torch.tensor([1.5], dtype=torch.float64), torch.tensor([0.2, 0.2, 0.2, 0.2], dtype=torch.float64))
    pred_nus, contrib = backward_bn(layer, linear_form_tensor_to_patches(coeffs, (1, 1, 2, 2)), {11: box}, [0])
    expected_nu, expected_contrib = dual_bn_backward(coeffs, layer.params["A"], layer.params["c"])
    torch.testing.assert_close(linear_form_patches_to_tensor(cast(Patches, pred_nus[0]), (1, 1, 2, 2)), expected_nu)
    torch.testing.assert_close(contrib, expected_contrib)


def test_bn_stats_broadcasting_over_channels() -> None:
    box = Bounds(lb=torch.tensor([[-1.0, -0.5, 0.2, 0.1, -0.2, 0.3, -0.4, 0.5]], dtype=torch.float64), ub=torch.tensor([[0.5, 0.6, 0.9, 0.8, 0.4, 0.9, 0.3, 1.0]], dtype=torch.float64))
    patches = _feature_identity_patches(1, 2, 2, 2)
    layer = _bn_layer(torch.tensor([2.0, 0.5], dtype=torch.float64), torch.zeros(8, dtype=torch.float64))
    _stored, _out, lin, _frame = forward_bn(layer, [box], [patches], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    dense = cast(Patches, lin).to_matrix((1, 2, 2, 2))[0]
    torch.testing.assert_close(torch.diag(dense), torch.tensor([2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5], dtype=torch.float64))


def test_forward_bn_patches_parity_with_matrix_path() -> None:
    box = _bounds([[-1.0, -0.5, 0.2, 0.1]], [[0.5, 0.6, 0.9, 0.8]])
    layer = _bn_layer(torch.tensor([1.25], dtype=torch.float64), torch.tensor([0.1, -0.1, 0.2, -0.2], dtype=torch.float64))
    patches = _feature_identity_patches(1, 1, 2, 2)
    patch_out = forward_bn(layer, [box], [patches], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    matrix_out = forward_bn(layer, [box], [_identity_linear_bound(1, 4)], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    torch.testing.assert_close(patch_out[1].lb, matrix_out[1].lb)
    torch.testing.assert_close(patch_out[1].ub, matrix_out[1].ub)


def test_forward_bn_patches_soundness_sampled() -> None:
    box = _bounds([[-1.0, -0.5, 0.2, 0.1]], [[0.5, 0.6, 0.9, 0.8]])
    layer = _bn_layer(torch.tensor([1.5], dtype=torch.float64), torch.tensor([0.1, -0.1, 0.2, -0.2], dtype=torch.float64))
    _stored, out, _lin, _frame = forward_bn(layer, [box], [_feature_identity_patches(1, 1, 2, 2)], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    samples = _sample_box(box, (1, 1, 2, 2), seed=5)
    actual = (1.5 * samples + layer.params["c"].view(1, 1, 2, 2)).reshape(samples.shape[0], -1)
    assert torch.all(actual <= out.ub + 1e-7)
    assert torch.all(actual >= out.lb - 1e-7)


def test_backward_dense_materializes_patches_to_matrix() -> None:
    coeffs = torch.tensor([[1.0, -0.5, 0.2, -0.3]], dtype=torch.float64)
    weight = torch.randn((4, 6), generator=_gen(12), dtype=torch.float64)
    bias = torch.randn((4,), generator=_gen(13), dtype=torch.float64)
    pred_nus, contrib = backward_dense(_dense_layer(weight, bias), linear_form_tensor_to_patches(coeffs, (1, 1, 2, 2)), {12: Bounds(lb=torch.zeros_like(coeffs), ub=torch.ones_like(coeffs))}, [0])
    expected_nu, expected_contrib = dual_dense_backward(coeffs, weight, bias)
    torch.testing.assert_close(pred_nus[0], expected_nu)
    torch.testing.assert_close(contrib, expected_contrib)


def test_backward_dense_matrix_passthrough_unchanged() -> None:
    coeffs = torch.tensor([[1.0, -0.5, 0.2, -0.3]], dtype=torch.float64)
    weight = torch.randn((4, 6), generator=_gen(14), dtype=torch.float64)
    bias = torch.randn((4,), generator=_gen(15), dtype=torch.float64)
    pred_nus, contrib = backward_dense(_dense_layer(weight, bias), coeffs, {12: Bounds(lb=torch.zeros_like(coeffs), ub=torch.ones_like(coeffs))}, [0])
    expected_nu, expected_contrib = dual_dense_backward(coeffs, weight, bias)
    torch.testing.assert_close(pred_nus[0], expected_nu)
    torch.testing.assert_close(contrib, expected_contrib)


def test_backward_dense_correctness_after_materialization() -> None:
    coeffs = torch.tensor([[0.2, 0.3, -0.4, 0.5]], dtype=torch.float64)
    weight = torch.randn((4, 5), generator=_gen(16), dtype=torch.float64)
    pred_nus, _contrib = backward_dense(_dense_layer(weight, None), linear_form_tensor_to_patches(coeffs, (1, 1, 2, 2)), {12: Bounds(lb=torch.zeros_like(coeffs), ub=torch.ones_like(coeffs))}, [0])
    expected_nu, _ = dual_dense_backward(coeffs, weight, None)
    torch.testing.assert_close(pred_nus[0], expected_nu)


def test_backward_dense_soundness_preserved_across_boundary() -> None:
    coeffs = torch.tensor([[0.6, -0.2, 0.5, -0.1]], dtype=torch.float64)
    x_bounds = _bounds([[-1.0, -0.5, 0.2, -0.2, 0.1, 0.0]], [[0.5, 0.6, 0.9, 0.8, 0.7, 0.4]])
    weight = torch.randn((4, 6), generator=_gen(17), dtype=torch.float64)
    bias = torch.randn((4,), generator=_gen(18), dtype=torch.float64)
    pred_nus, contrib = backward_dense(_dense_layer(weight, bias), linear_form_tensor_to_patches(coeffs, (1, 1, 2, 2)), {12: Bounds(lb=torch.zeros_like(coeffs), ub=torch.ones_like(coeffs))}, [0])
    samples = _sample_box(x_bounds, (1, 1, 1, 6), seed=6).reshape(-1, 6)
    actual = (samples @ weight.t() + bias) @ coeffs[0]
    _assert_objective_sound(cast(torch.Tensor, pred_nus[0]), contrib, x_bounds, actual)


def test_backward_dense_materialized_plus_tensor_matches_sum() -> None:
    coeffs_a = torch.tensor([[1.0, -0.5, 0.2, -0.3]], dtype=torch.float64)
    coeffs_b = torch.tensor([[0.4, 0.1, -0.2, 0.7]], dtype=torch.float64)
    weight = torch.randn((4, 6), generator=_gen(19), dtype=torch.float64)
    bias = torch.randn((4,), generator=_gen(20), dtype=torch.float64)
    pred_a, contrib_a = backward_dense(_dense_layer(weight, bias), linear_form_tensor_to_patches(coeffs_a, (1, 1, 2, 2)), {12: Bounds(lb=torch.zeros_like(coeffs_a), ub=torch.ones_like(coeffs_a))}, [0])
    pred_b, contrib_b = backward_dense(_dense_layer(weight, bias), coeffs_b, {12: Bounds(lb=torch.zeros_like(coeffs_b), ub=torch.ones_like(coeffs_b))}, [0])
    pred_sum, contrib_sum = backward_dense(_dense_layer(weight, bias), coeffs_a + coeffs_b, {12: Bounds(lb=torch.zeros_like(coeffs_a), ub=torch.ones_like(coeffs_a))}, [0])
    torch.testing.assert_close(pred_a[0] + pred_b[0], pred_sum[0])
    torch.testing.assert_close(contrib_a + contrib_b, contrib_sum)


def test_backward_dense_input_shape_inference_uses_nu_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    coeffs = torch.tensor([[1.0, -0.5, 0.2, -0.3]], dtype=torch.float64)
    nu = linear_form_tensor_to_patches(coeffs, (1, 1, 2, 2))
    called: dict[str, tuple[int, ...]] = {}
    original = Patches.to_matrix

    def _spy(self: Patches, input_shape: tuple[int, ...]) -> torch.Tensor:
        called["shape"] = input_shape
        return original(self, input_shape)

    monkeypatch.setattr(Patches, "to_matrix", _spy)
    weight = torch.randn((4, 6), generator=_gen(21), dtype=torch.float64)
    backward_dense(_dense_layer(weight, None), nu, {12: Bounds(lb=torch.zeros_like(coeffs), ub=torch.ones_like(coeffs))}, [0])
    assert called["shape"] == (1, 1, 2, 2)


def test_backward_dense_float64_parity_through_materialization() -> None:
    coeffs = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
    weight = torch.randn((4, 3), generator=_gen(22), dtype=torch.float64)
    pred_nus, contrib = backward_dense(_dense_layer(weight, None), linear_form_tensor_to_patches(coeffs, (1, 1, 2, 2)), {12: Bounds(lb=torch.zeros_like(coeffs), ub=torch.ones_like(coeffs))}, [0])
    expected_nu, expected_contrib = dual_dense_backward(coeffs, weight, None)
    torch.testing.assert_close(pred_nus[0], expected_nu, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(contrib, expected_contrib, rtol=1e-10, atol=1e-12)


def test_forward_maxpool_materializes_patches_and_matches_matrix(caplog: pytest.LogCaptureFixture) -> None:
    box = _bounds([[-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    patches = _feature_identity_patches(1, 1, 4, 4)
    layer = _pool_layer(LayerKind.MAXPOOL2D.value)
    with caplog.at_level("WARNING"):
        patch_out = forward_maxpool2d(layer, [box], [patches], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    matrix_out = forward_maxpool2d(layer, [box], [_identity_linear_bound(1, 16)], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    assert any("materializes Patches" in record.message for record in caplog.records)
    torch.testing.assert_close(patch_out[1].lb, matrix_out[1].lb)
    torch.testing.assert_close(patch_out[1].ub, matrix_out[1].ub)


def test_forward_avgpool_materializes_patches_and_matches_matrix(caplog: pytest.LogCaptureFixture) -> None:
    box = _bounds([[-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    patches = _feature_identity_patches(1, 1, 4, 4)
    layer = _pool_layer(LayerKind.AVGPOOL2D.value)
    with caplog.at_level("WARNING"):
        patch_out = forward_avgpool2d(layer, [box], [patches], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    matrix_out = forward_avgpool2d(layer, [box], [_identity_linear_bound(1, 16)], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    assert any("materializes Patches" in record.message for record in caplog.records)
    torch.testing.assert_close(patch_out[1].lb, matrix_out[1].lb)
    torch.testing.assert_close(patch_out[1].ub, matrix_out[1].ub)


def test_forward_maxpool_warning_emitted_once(caplog: pytest.LogCaptureFixture) -> None:
    box = _bounds([[-0.4] * 16], [[0.6] * 16])
    with caplog.at_level("WARNING"):
        forward_maxpool2d(_pool_layer(LayerKind.MAXPOOL2D.value), [box], [_feature_identity_patches(1, 1, 4, 4)], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    assert sum("forward_maxpool2d: pool materializes Patches" in record.message for record in caplog.records) == 1


def test_forward_pool_parity_vs_matrix_path() -> None:
    box = _bounds([[-0.4] * 16], [[0.6] * 16])
    max_patch = forward_maxpool2d(_pool_layer(LayerKind.MAXPOOL2D.value), [box], [_feature_identity_patches(1, 1, 4, 4)], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    max_matrix = forward_maxpool2d(_pool_layer(LayerKind.MAXPOOL2D.value), [box], [_identity_linear_bound(1, 16)], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    avg_patch = forward_avgpool2d(_pool_layer(LayerKind.AVGPOOL2D.value), [box], [_feature_identity_patches(1, 1, 4, 4)], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    avg_matrix = forward_avgpool2d(_pool_layer(LayerKind.AVGPOOL2D.value), [box], [_identity_linear_bound(1, 16)], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    torch.testing.assert_close(max_patch[1].lb, max_matrix[1].lb)
    torch.testing.assert_close(avg_patch[1].ub, avg_matrix[1].ub)


def test_forward_avgpool_soundness_sampled() -> None:
    box = _bounds([[-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    out = forward_avgpool2d(_pool_layer(LayerKind.AVGPOOL2D.value), [box], [_feature_identity_patches(1, 1, 4, 4)], [_frame_from_bounds(box)], [0], False, torch.device("cpu"), torch.float64)
    samples = _sample_box(box, (1, 1, 4, 4), seed=7)
    actual = F.avg_pool2d(samples.squeeze(1), 2, 2).reshape(samples.shape[0], -1)
    assert torch.all(actual <= out[1].ub + 1e-7)
    assert torch.all(actual >= out[1].lb - 1e-7)
