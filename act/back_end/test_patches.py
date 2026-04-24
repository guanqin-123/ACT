from __future__ import annotations

from typing import Any, cast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from act.back_end.config import BackendConfig
from act.back_end.patches import (
    Patches,
    compute_patches_stride_padding,
    deterministic_patches_ctx,
    inplace_unfold,
    insert_zeros,
    is_deterministic_patches,
    patches_to_matrix,
    unify_shape,
)


def _explicit_conv_matrix(
    weight: torch.Tensor,
    input_shape: tuple[int, int, int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> torch.Tensor:
    batch_size, in_channels, in_h, in_w = input_shape
    basis = torch.eye(
        in_channels * in_h * in_w,
        dtype=weight.dtype,
        device=weight.device,
    ).reshape(in_channels * in_h * in_w, in_channels, in_h, in_w)
    outputs = F.conv2d(basis, weight, stride=stride, padding=padding)
    return outputs.reshape(in_channels * in_h * in_w, -1).transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1).clone()


def _conv_weight_to_patches(
    weight: torch.Tensor,
    batch_size: int,
    input_hw: tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> Patches:
    dummy = torch.zeros((batch_size, weight.shape[1], *input_hw), dtype=weight.dtype)
    out_h, out_w = F.conv2d(dummy, weight, stride=stride, padding=padding).shape[-2:]
    pieces = weight[:, None, None, None, :, :, :].expand(
        weight.shape[0],
        batch_size,
        out_h,
        out_w,
        weight.shape[1],
        weight.shape[2],
        weight.shape[3],
    ).clone()
    return Patches(
        patches=pieces,
        stride=stride,
        padding=padding,
        shape=tuple(pieces.shape),
    )


def _matrix_to_patches(
    matrix: torch.Tensor,
    input_shape: tuple[int, int, int, int],
    out_channels: int,
    output_hw: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> Patches:
    _, in_channels, in_h, in_w = input_shape
    out_h, out_w = output_hw
    k_h, k_w = kernel_size
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    pieces = matrix.new_zeros((out_channels, input_shape[0], out_h, out_w, in_channels, k_h, k_w))
    flat = matrix[0]
    for out_c in range(out_channels):
        for oh in range(out_h):
            for ow in range(out_w):
                row_idx = out_c * (out_h * out_w) + oh * out_w + ow
                top = oh * stride_h - pad_h
                left = ow * stride_w - pad_w
                for in_c in range(in_channels):
                    for kh in range(k_h):
                        in_y = top + kh
                        if 0 <= in_y < in_h:
                            for kw in range(k_w):
                                in_x = left + kw
                                if 0 <= in_x < in_w:
                                    col_idx = in_c * (in_h * in_w) + in_y * in_w + in_x
                                    pieces[out_c, :, oh, ow, in_c, kh, kw] = flat[row_idx, col_idx]
    return Patches(patches=pieces, stride=stride, padding=padding, shape=tuple(pieces.shape))


def _base_config_yaml() -> dict[str, Any]:
    with open(Path(__file__).with_name("config.yaml")) as handle:
        return cast(dict[str, Any], yaml.safe_load(handle))


def test_patches_to_matrix_calls_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    patches = Patches(identity=1)
    expected = torch.ones(1, 1, 1)
    called: dict[str, tuple[int, ...]] = {}

    def _fake_helper(obj: Patches, input_shape: tuple[int, ...]) -> torch.Tensor:
        called["shape"] = input_shape
        assert obj is patches
        return expected

    monkeypatch.setattr("act.back_end.patches.patches_to_matrix", _fake_helper)
    result = patches.to_matrix((1, 1, 1, 1))
    assert called["shape"] == (1, 1, 1, 1)
    assert result is expected


def test_patches_clone_deep_copy() -> None:
    pieces = torch.arange(36.0).reshape(1, 1, 2, 2, 1, 3, 3)
    unstable_idx = (torch.tensor([0, 1]), torch.tensor([1, 0]))
    patches = Patches(
        patches=pieces,
        stride=(2, 1),
        padding=(1, 0),
        shape=tuple(pieces.shape),
        unstable_idx=unstable_idx,
        output_shape=(1, 1, 2, 2),
        input_shape=(1, 1, 4, 4),
    )
    cloned = patches.clone()
    assert patches.patches is not None
    patches.patches[0, 0, 0, 0, 0, 0, 0] = -9.0
    unstable_idx[0][0] = 99
    assert cloned is not patches
    assert cloned.patches is not patches.patches
    assert cloned.unstable_idx is not patches.unstable_idx
    assert cloned.patches is not None
    assert cloned.unstable_idx is not None
    assert cloned.patches[0, 0, 0, 0, 0, 0, 0].item() == 0.0
    assert cloned.unstable_idx[0][0].item() == 0


def test_patches_detach_no_grad() -> None:
    pieces = torch.randn(1, 1, 1, 1, 1, 3, 3, requires_grad=True)
    unstable = (torch.randn(2, requires_grad=True), torch.randn(2, requires_grad=True))
    detached = Patches(patches=pieces, unstable_idx=unstable, shape=tuple(pieces.shape)).detach()
    assert detached.patches is not None
    assert detached.patches.requires_grad is False
    assert detached.unstable_idx is not None
    assert all(not idx.requires_grad for idx in detached.unstable_idx)


def test_patches_eq_same_fields() -> None:
    pieces = torch.arange(36.0).reshape(1, 1, 2, 2, 1, 3, 3)
    left = Patches(patches=pieces.clone(), stride=2, padding=1, shape=tuple(pieces.shape))
    right = Patches(patches=pieces.clone(), stride=2, padding=1, shape=tuple(pieces.shape))
    assert left == right


def test_patches_eq_different_patches_tensor() -> None:
    pieces = torch.arange(36.0).reshape(1, 1, 2, 2, 1, 3, 3)
    left = Patches(patches=pieces.clone(), shape=tuple(pieces.shape))
    right = Patches(patches=pieces.clone(), shape=tuple(pieces.shape))
    assert right.patches is not None
    right.patches[0, 0, 0, 0, 0, 0, 0] += 1.0
    assert left != right


def test_patches_to_matrix_stride1_padding0() -> None:
    weight = torch.arange(9.0, dtype=torch.float64).reshape(1, 1, 3, 3)
    patches = _conv_weight_to_patches(weight, batch_size=1, input_hw=(4, 4), stride=1, padding=0)
    result = patches_to_matrix(patches, (1, 1, 4, 4))
    expected = _explicit_conv_matrix(weight, (1, 1, 4, 4), stride=1, padding=0)
    torch.testing.assert_close(result, expected)


def test_patches_to_matrix_stride2_padding1() -> None:
    weight = torch.arange(36.0, dtype=torch.float64).reshape(2, 2, 3, 3)
    patches = _conv_weight_to_patches(weight, batch_size=1, input_hw=(5, 5), stride=2, padding=1)
    result = patches_to_matrix(patches, (1, 2, 5, 5))
    expected = _explicit_conv_matrix(weight, (1, 2, 5, 5), stride=2, padding=1)
    torch.testing.assert_close(result, expected)


def test_patches_to_matrix_identity() -> None:
    result = patches_to_matrix(Patches(identity=1), (2, 1, 2, 3))
    expected = torch.eye(6, dtype=result.dtype).unsqueeze(0).expand(2, -1, -1).clone()
    torch.testing.assert_close(result, expected)


def test_patches_to_matrix_round_trip() -> None:
    weight = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float64)
    original = _conv_weight_to_patches(weight, batch_size=1, input_hw=(3, 3), stride=1, padding=0)
    dense = patches_to_matrix(original, (1, 1, 3, 3))
    rebuilt = _matrix_to_patches(
        dense,
        input_shape=(1, 1, 3, 3),
        out_channels=1,
        output_hw=(2, 2),
        kernel_size=(2, 2),
        stride=1,
        padding=0,
    )
    round_trip = patches_to_matrix(rebuilt, (1, 1, 3, 3))
    torch.testing.assert_close(round_trip, dense)


def test_patches_to_matrix_float64_precision() -> None:
    weight = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]], dtype=torch.float64)
    patches = _conv_weight_to_patches(weight, batch_size=1, input_hw=(4, 4), stride=2, padding=1)
    result = patches_to_matrix(patches, (1, 1, 4, 4))
    expected = _explicit_conv_matrix(weight, (1, 1, 4, 4), stride=2, padding=1)
    torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-12)


def test_patches_to_matrix_raises_on_none() -> None:
    with pytest.raises(ValueError, match="patches tensor is None"):
        patches_to_matrix(Patches(), (1, 1, 2, 2))


def test_insert_zeros_zero_is_clone() -> None:
    pieces = torch.ones(1, 1, 1, 1, 1, 3, 3)
    patches = Patches(patches=pieces, shape=tuple(pieces.shape))
    result = insert_zeros(patches, 0)
    assert result == patches
    assert result is not patches
    assert result.patches is not patches.patches


def test_insert_zeros_positive_raises() -> None:
    with pytest.raises(NotImplementedError, match="dilation > 1"):
        insert_zeros(Patches(identity=1), 1)


def test_insert_zeros_returns_patches_type() -> None:
    result = insert_zeros(Patches(identity=1, shape=(1,)), 0)
    assert isinstance(result, Patches)


def test_insert_zeros_clone_independence() -> None:
    pieces = torch.zeros(1, 1, 1, 1, 1, 3, 3)
    patches = Patches(patches=pieces, shape=tuple(pieces.shape))
    result = insert_zeros(patches, 0)
    assert patches.patches is not None
    assert result.patches is not None
    patches.patches[0, 0, 0, 0, 0, 0, 0] = 5.0
    assert result.patches[0, 0, 0, 0, 0, 0, 0].item() == 0.0


@pytest.mark.parametrize(
    ("prev_stride", "prev_padding", "cur_stride", "cur_padding", "expected_stride", "expected_padding"),
    [
        (1, 0, 1, 0, (1, 1), (0, 0)),
        (1, 1, 1, 1, (1, 1), (2, 2)),
        (2, 0, 1, 0, (2, 2), (0, 0)),
        (2, 1, 1, 1, (2, 2), (2, 2)),
        (1, 0, 2, 1, (2, 2), (1, 1)),
        ((2, 1), (1, 0), (1, 2), (0, 1), (2, 2), (1, 1)),
    ],
)
def test_compute_patches_stride_padding_formula_cases(
    prev_stride: int | tuple[int, int],
    prev_padding: int | tuple[int, int],
    cur_stride: int | tuple[int, int],
    cur_padding: int | tuple[int, int],
    expected_stride: tuple[int, int],
    expected_padding: tuple[int, int],
) -> None:
    stride, padding = compute_patches_stride_padding(
        prev_stride,
        prev_padding,
        cur_stride,
        cur_padding,
    )
    assert stride == expected_stride
    assert padding == expected_padding


def test_compute_patches_stride_padding_prev_inserted_zeros_raises() -> None:
    with pytest.raises(NotImplementedError, match="inserted_zeros > 0"):
        compute_patches_stride_padding(1, 0, 1, 0, prev_inserted_zeros=1)


def test_compute_patches_stride_padding_matches_manual_unfold_case() -> None:
    image = torch.arange(1.0, 17.0).reshape(1, 1, 4, 4)
    first = inplace_unfold(image, kernel_size=2, stride=2, padding=1)
    second = inplace_unfold(first.reshape(1, 4, 3, 3), kernel_size=1, stride=1, padding=1)
    stride, padding = compute_patches_stride_padding(2, 1, 1, 1)
    fused = inplace_unfold(image, kernel_size=2, stride=stride, padding=padding[0])
    assert second.shape[-1] == 25
    assert fused.shape[-1] == 16
    assert stride == (2, 2)
    assert padding == (2, 2)


def test_unify_shape_same_shape_returns_clone() -> None:
    pieces = torch.randn(1, 1, 2, 2, 1, 3, 3)
    patches = Patches(patches=pieces, shape=tuple(pieces.shape))
    result = unify_shape(patches, tuple(pieces.shape))
    assert result == patches
    assert result is not patches
    assert result.patches is not patches.patches


def test_unify_shape_identity_reshape() -> None:
    patches = Patches(identity=1, shape=(1, 1, 1, 1, 1, 1, 1))
    result = unify_shape(patches, (2, 1, 3, 3, 1, 1, 1))
    assert result.is_identity
    assert result.shape == (2, 1, 3, 3, 1, 1, 1)


def test_unify_shape_view_expand() -> None:
    pieces = torch.randn(1, 1, 1, 2, 1, 3, 3)
    patches = Patches(patches=pieces, shape=tuple(pieces.shape))
    result = unify_shape(patches, (1, 1, 4, 2, 1, 3, 3))
    assert result.patches is not None
    assert result.patches.shape == (1, 1, 4, 2, 1, 3, 3)
    assert 0 in result.patches.stride()


def test_unify_shape_repeat_expand() -> None:
    view_source = torch.randn(1, 1, 1, 2, 1, 3, 3)
    patches = Patches(patches=view_source, shape=tuple(view_source.shape))
    repeated_source = torch.randn(1, 1, 1, 2, 1, 3, 1).expand(1, 1, 1, 2, 1, 3, 3)
    repeated = unify_shape(
        Patches(patches=repeated_source, shape=tuple(repeated_source.shape)),
        (1, 1, 4, 2, 1, 3, 3),
    )
    result = unify_shape(patches, (1, 1, 4, 2, 1, 3, 3))
    assert result.patches is not None
    assert repeated.patches is not None
    assert result.patches.shape == (1, 1, 4, 2, 1, 3, 3)
    assert 0 in result.patches.stride()
    assert all(stride != 0 for stride in repeated.patches.stride())


def test_unify_shape_incompatible_channel_axis_raises() -> None:
    pieces = torch.randn(1, 1, 2, 2, 1, 3, 3)
    with pytest.raises(ValueError, match="Channel dimensions"):
        unify_shape(Patches(patches=pieces, shape=tuple(pieces.shape)), (2, 1, 2, 2, 2, 3, 3))


def test_inplace_unfold_f_unfold_hot_path(monkeypatch: pytest.MonkeyPatch) -> None:
    tensor = torch.arange(1.0, 17.0).reshape(1, 1, 4, 4)
    called = {"as_strided": False}
    original = torch.as_strided

    def _spy(*args, **kwargs):
        called["as_strided"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "as_strided", _spy)
    result = inplace_unfold(tensor, kernel_size=2, stride=1, padding=0)
    assert result.shape == (1, 4, 9)
    assert called["as_strided"] is False


def test_inplace_unfold_shape_correctness() -> None:
    tensor = torch.randn(2, 3, 5, 5)
    result = inplace_unfold(tensor, kernel_size=3, stride=2, padding=1)
    assert result.shape == (2, 27, 9)


def test_inplace_unfold_value_correctness() -> None:
    tensor = torch.arange(1.0, 17.0).reshape(1, 1, 4, 4)
    result = inplace_unfold(tensor, kernel_size=2, stride=1, padding=0)
    expected = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0],
                [2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0],
                [5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 13.0, 14.0, 15.0],
                [6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0],
            ]
        ]
    )
    torch.testing.assert_close(result, expected)


def test_inplace_unfold_ctx_mgr_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    tensor = torch.arange(1.0, 17.0).reshape(1, 1, 4, 4)
    called = {"as_strided": False}
    original = torch.as_strided

    def _spy(*args, **kwargs):
        called["as_strided"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "as_strided", _spy)
    with deterministic_patches_ctx(deterministic=False):
        result = inplace_unfold(tensor, kernel_size=2, stride=1, padding=0)
    assert called["as_strided"] is True
    torch.testing.assert_close(result, F.unfold(tensor, kernel_size=2, stride=1, padding=0))


def test_deterministic_patches_ctx_toggles_flag() -> None:
    assert is_deterministic_patches() is True
    with deterministic_patches_ctx(deterministic=False):
        assert is_deterministic_patches() is False
    assert is_deterministic_patches() is True


def test_deterministic_patches_ctx_nested_restore() -> None:
    assert is_deterministic_patches() is True
    with deterministic_patches_ctx(deterministic=False):
        assert is_deterministic_patches() is False
        with deterministic_patches_ctx(deterministic=True):
            assert is_deterministic_patches() is True
        assert is_deterministic_patches() is False
    assert is_deterministic_patches() is True


def test_config_default_patches() -> None:
    config = BackendConfig.from_yaml()
    assert config.conv_mode == "patches"


def test_config_conv_mode_patches_allowed(tmp_path: Path) -> None:
    config_yaml = _base_config_yaml()
    backend = cast(dict[str, Any], config_yaml["backend"])
    backend["conv_mode"] = "patches"
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as handle:
        yaml.safe_dump(config_yaml, handle, sort_keys=False)
    config = BackendConfig.from_yaml(config_path)
    assert config.conv_mode == "patches"


def test_config_conv_mode_invalid_raises(tmp_path: Path) -> None:
    config_yaml = _base_config_yaml()
    backend = cast(dict[str, Any], config_yaml["backend"])
    backend["conv_mode"] = "foo"
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as handle:
        yaml.safe_dump(config_yaml, handle, sort_keys=False)
    with pytest.raises(ValueError, match="Invalid conv_mode"):
        BackendConfig.from_yaml(config_path)
