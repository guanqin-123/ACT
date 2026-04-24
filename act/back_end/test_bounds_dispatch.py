from __future__ import annotations

# pyright: reportPrivateUsage=false, reportPrivateLocalImportUsage=false, reportUnknownMemberType=false, reportMissingTypeArgument=false, reportUnannotatedClassAttribute=false, reportUnusedFunction=false, reportUnusedCallResult=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownVariableType=false, reportImplicitStringConcatenation=false

from pathlib import Path

import pytest
import torch

from act.back_end.bab.eta import EtaState
from act.back_end.bounds_dispatch import (
    dispatch_add_forward,
    dispatch_bn_forward,
    dispatch_conv_forward,
    dispatch_pool_forward,
    dispatch_relu_backward,
    expand_rank3,
    is_rank3_view,
    materialize_if_needed,
)
from act.back_end.core import Bounds, Layer
from act.back_end.dual_tf.tf_cnn import (
    forward_avgpool2d,
    forward_conv2d,
    forward_maxpool2d,
)
from act.back_end.dual_tf.tf_forward import Frame, LinearBound, forward_add
from act.back_end.dual_tf.tf_mlp import backward_relu, forward_bn
from act.back_end.layer_schema import LayerKind
from act.back_end.patches import Patches
from act.util.device_manager import initialize_device
from scripts import benchmark_cifar100_baseline as benchmark
from scripts.bab_clamp_diagnostic import (
    resolve_vnnlib_row_index,
)


KNOWN_VNNLIB = "CIFAR100_resnet_medium_prop_idx_3995_sidx_7978_eps_0.0039.vnnlib"


@pytest.fixture(scope="module", autouse=True)
def _cpu_float32() -> None:
    initialize_device("cpu", "float32")


def _t(data: object) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float32)


def _storage_elems(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().nbytes() // tensor.element_size()


def _assert_linear_bound_close(actual: LinearBound, expected: LinearBound) -> None:
    torch.testing.assert_close(actual.A_lb, expected.A_lb)
    torch.testing.assert_close(actual.b_lb, expected.b_lb)
    torch.testing.assert_close(actual.A_ub, expected.A_ub)
    torch.testing.assert_close(actual.b_ub, expected.b_ub)


def _assert_forward_tuple_close(
    actual: tuple[Bounds, Bounds, LinearBound, Frame],
    expected: tuple[Bounds, Bounds, LinearBound, Frame],
) -> None:
    actual_stored, actual_out, actual_lin, actual_frame = actual
    expected_stored, expected_out, expected_lin, expected_frame = expected
    torch.testing.assert_close(actual_stored.lb, expected_stored.lb)
    torch.testing.assert_close(actual_stored.ub, expected_stored.ub)
    torch.testing.assert_close(actual_out.lb, expected_out.lb)
    torch.testing.assert_close(actual_out.ub, expected_out.ub)
    _assert_linear_bound_close(actual_lin, expected_lin)
    torch.testing.assert_close(actual_frame[0], expected_frame[0])
    torch.testing.assert_close(actual_frame[1], expected_frame[1])


def _make_linear_bound(batch: int = 2, out_dim: int = 4, input_dim: int = 3) -> LinearBound:
    A_lb = torch.arange(batch * out_dim * input_dim, dtype=torch.float32).reshape(batch, out_dim, input_dim) / 10.0
    A_ub = A_lb + 0.5
    b_lb = torch.arange(batch * out_dim, dtype=torch.float32).reshape(batch, out_dim) / 20.0
    b_ub = b_lb + 0.25
    return LinearBound(A_lb=A_lb, b_lb=b_lb, A_ub=A_ub, b_ub=b_ub)


def _make_conv_case(*, stride: int = 1, padding: int = 1) -> tuple[Layer, list[Bounds], list[LinearBound], list[tuple[torch.Tensor, torch.Tensor]], list[int], bool, torch.device, torch.dtype]:
    layer = Layer(
        id=12,
        kind=LayerKind.CONV2D.value,
        params={
            "weight": torch.arange(18, dtype=torch.float32).reshape(2, 1, 3, 3) / 50.0,
            "bias": _t([0.1, -0.2]),
            "in_channels": 1,
            "out_channels": 2,
            "kernel_size": 3,
            "stride": stride,
            "padding": padding,
            "dilation": 1,
            "groups": 1,
            "input_shape": (1, 1, 4, 4),
            "output_shape": (1, 2, 4, 4) if stride == 1 and padding == 1 else (1, 2, 2, 2),
        },
        in_vars=[],
        out_vars=[],
    )
    current_dim = 16
    parent_box = Bounds(
        lb=torch.linspace(-0.3, 0.1, steps=2 * current_dim, dtype=torch.float32).reshape(2, current_dim),
        ub=torch.linspace(0.2, 0.6, steps=2 * current_dim, dtype=torch.float32).reshape(2, current_dim),
    )
    parent_lin = _make_linear_bound(batch=2, out_dim=current_dim, input_dim=3)
    parent_frame = (
        _t([[-0.2, 0.1, 0.0], [0.0, -0.1, 0.2]]),
        _t([[0.3, 0.4, 0.5], [0.4, 0.3, 0.6]]),
    )
    return layer, [parent_box], [parent_lin], [parent_frame], [0], False, torch.device("cpu"), torch.float32


def _make_bn_case() -> tuple[object, list[Bounds], list[LinearBound], list[tuple[torch.Tensor, torch.Tensor]], list[int], bool, torch.device, torch.dtype]:
    layer = type(
        "BNStub",
        (),
        {
            "id": 21,
            "kind": LayerKind.BN.value,
            "params": {
                "A": _t([1.2, 0.8, 1.1, 0.9]),
                "c": _t([0.1, -0.1, 0.05, 0.2]),
            },
        },
    )()
    parent_box = Bounds(lb=_t([[-1.0, -0.2, 0.1, -0.5], [0.0, -0.1, -0.2, 0.2]]), ub=_t([[0.5, 0.8, 1.0, 0.3], [0.7, 0.6, 0.2, 0.9]]))
    parent_lin = _make_linear_bound(batch=2, out_dim=4, input_dim=3)
    parent_frame = (
        _t([[-0.3, 0.0, -0.2], [0.1, -0.4, 0.2]]),
        _t([[0.4, 0.6, 0.5], [0.7, 0.2, 0.8]]),
    )
    return layer, [parent_box], [parent_lin], [parent_frame], [0], False, torch.device("cpu"), torch.float32


def _make_add_case() -> tuple[Layer, list[Bounds], list[LinearBound], list[tuple[torch.Tensor, torch.Tensor]], list[int], bool, torch.device, torch.dtype]:
    layer = Layer(
        id=30,
        kind=LayerKind.ADD.value,
        params={"bias": _t([0.1, -0.1, 0.05, 0.0])},
        in_vars=[],
        out_vars=[],
    )
    parent_boxes = [
        Bounds(lb=_t([[-0.5, -0.3, 0.0, 0.2], [0.1, -0.4, -0.2, 0.0]]), ub=_t([[0.5, 0.3, 0.4, 0.6], [0.4, 0.1, 0.3, 0.5]])),
        Bounds(lb=_t([[-0.2, 0.0, -0.1, -0.3], [-0.1, -0.2, 0.0, -0.1]]), ub=_t([[0.3, 0.5, 0.2, 0.1], [0.2, 0.4, 0.5, 0.3]])),
    ]
    parent_lins = [_make_linear_bound(batch=2, out_dim=4, input_dim=3), _make_linear_bound(batch=2, out_dim=4, input_dim=3)]
    shared_frame = (
        _t([[-0.2, 0.0, -0.1], [0.1, -0.2, 0.0]]),
        _t([[0.4, 0.6, 0.5], [0.6, 0.2, 0.4]]),
    )
    return layer, parent_boxes, parent_lins, [shared_frame, shared_frame], [0, 1], False, torch.device("cpu"), torch.float32


def _make_pool_case(kind: str) -> tuple[Layer, list[Bounds], list[LinearBound], list[tuple[torch.Tensor, torch.Tensor]], list[int], bool, torch.device, torch.dtype]:
    layer = Layer(
        id=41,
        kind=kind,
        params={
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
            "input_shape": (1, 1, 4, 4),
            "output_shape": (1, 1, 2, 2),
        },
        in_vars=[],
        out_vars=[],
    )
    parent_box = Bounds(
        lb=torch.linspace(-0.4, 0.2, steps=32, dtype=torch.float32).reshape(2, 16),
        ub=torch.linspace(0.1, 0.8, steps=32, dtype=torch.float32).reshape(2, 16),
    )
    parent_lin = _make_linear_bound(batch=2, out_dim=16, input_dim=3)
    parent_frame = (
        _t([[-0.2, 0.0, -0.1], [0.1, -0.2, 0.0]]),
        _t([[0.4, 0.6, 0.5], [0.6, 0.2, 0.4]]),
    )
    return layer, [parent_box], [parent_lin], [parent_frame], [0], False, torch.device("cpu"), torch.float32


def test_is_rank3_view_true_for_stride_zero_rank3() -> None:
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    view = base.unsqueeze(0).expand(4, -1, -1)
    assert is_rank3_view(view) is True


def test_is_rank3_view_false_for_rank3_non_stride_zero() -> None:
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    assert is_rank3_view(tensor) is False


def test_is_rank3_view_false_for_rank2() -> None:
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert is_rank3_view(tensor) is False


def test_is_rank3_view_false_for_non_tensor() -> None:
    assert is_rank3_view("not-a-tensor") is False


def test_expand_rank3_stride_zero_spec_axis() -> None:
    expanded = expand_rank3({1: _make_linear_bound()}, M=5)[1]
    assert expanded.A_lb.shape == (2, 5, 4, 3)
    assert expanded.A_lb.stride(-3) == 0


def test_expand_rank3_m1_identity_like() -> None:
    bounds = {1: _make_linear_bound()}
    expanded = expand_rank3(bounds, M=1)
    _assert_linear_bound_close(expanded[1], bounds[1])
    assert expanded[1].A_lb.shape == bounds[1].A_lb.shape


def test_expand_rank3_large_m_does_not_materialize() -> None:
    expanded = expand_rank3({1: _make_linear_bound()}, M=99)[1]
    assert _storage_elems(expanded.A_lb) < expanded.A_lb.numel()


def test_expand_rank3_raises_on_eta_state() -> None:
    eta = EtaState(
        val={1: torch.ones(2, 3)},
        sign={1: torch.zeros(2, 3)},
        point={1: torch.zeros(2, 3)},
    )
    with pytest.raises(NotImplementedError, match="EtaState"):
        _ = expand_rank3(eta, M=2)


def test_expand_rank3_multi_layer_bounds_dict() -> None:
    expanded = expand_rank3({1: _make_linear_bound(), 2: _make_linear_bound(out_dim=5)}, M=3)
    assert set(expanded) == {1, 2}
    assert expanded[2].A_lb.shape == (2, 3, 5, 3)


def test_expand_rank3_preserves_linear_bound_structure() -> None:
    expanded = expand_rank3({1: _make_linear_bound()}, M=4)[1]
    assert isinstance(expanded, LinearBound)
    assert hasattr(expanded, "A_lb")
    assert hasattr(expanded, "A_ub")
    assert hasattr(expanded, "b_lb")
    assert hasattr(expanded, "b_ub")


def test_materialize_if_needed_on_rank3_view_produces_contiguous_tensor() -> None:
    expanded = expand_rank3({1: _make_linear_bound()}, M=4)[1]
    materialized = materialize_if_needed(expanded)
    assert materialized.A_lb.is_contiguous()
    assert materialized.b_lb.is_contiguous()
    assert all(stride != 0 for stride in materialized.A_lb.stride())


def test_materialize_if_needed_passthrough_on_regular_linear_bound() -> None:
    bounds = _make_linear_bound()
    assert materialize_if_needed(bounds) is bounds


def test_materialize_if_needed_no_allocation_if_already_contiguous() -> None:
    bounds = _make_linear_bound()
    same = materialize_if_needed(bounds)
    assert same.A_lb.data_ptr() == bounds.A_lb.data_ptr()
    assert same.b_lb.data_ptr() == bounds.b_lb.data_ptr()


def test_dispatch_conv_forward_matrix_matches_direct_supported_shape() -> None:
    args = _make_conv_case()
    direct = forward_conv2d(*args)
    dispatched = dispatch_conv_forward(*args)
    _assert_forward_tuple_close(dispatched, direct)


def test_dispatch_conv_forward_matrix_matches_direct_stride_two() -> None:
    args = _make_conv_case(stride=2, padding=1)
    direct = forward_conv2d(*args)
    dispatched = dispatch_conv_forward(*args)
    _assert_forward_tuple_close(dispatched, direct)


def test_dispatch_conv_forward_patches_raises() -> None:
    layer, parent_boxes, _parent_lins, parent_frames, preds, post_activation, device, dtype = _make_conv_case()
    with pytest.raises(NotImplementedError, match="Filled in W3"):
        dispatch_conv_forward(
            layer,
            parent_boxes,
            [Patches()],
            parent_frames,
            preds,
            post_activation,
            device,
            dtype,
        )


def test_dispatch_conv_forward_identity_patches_raises() -> None:
    layer, parent_boxes, _parent_lins, parent_frames, preds, post_activation, device, dtype = _make_conv_case()
    with pytest.raises(NotImplementedError, match="Filled in W3"):
        dispatch_conv_forward(
            layer,
            parent_boxes,
            [Patches(identity=1)],
            parent_frames,
            preds,
            post_activation,
            device,
            dtype,
        )


def test_dispatch_relu_backward_matrix_matches_direct_and_patches_raise() -> None:
    layer = Layer(id=51, kind=LayerKind.RELU.value, params={}, in_vars=[], out_vars=[])
    nu = _t([[1.0, -0.5, 0.2], [-0.3, 0.4, -0.1]])
    bounds_dict = {51: Bounds(lb=_t([[-1.0, -0.5, 0.0], [-0.2, -0.1, -0.3]]), ub=_t([[0.5, 1.0, 0.8], [0.7, 0.4, 0.2]]))}
    direct = backward_relu(layer, nu, bounds_dict, [0])
    dispatched = dispatch_relu_backward(layer, nu, bounds_dict, [0])
    torch.testing.assert_close(dispatched[0][0], direct[0][0])
    torch.testing.assert_close(dispatched[1], direct[1])
    with pytest.raises(NotImplementedError, match="Filled in W3/W4"):
        dispatch_relu_backward(layer, Patches(identity=1), bounds_dict, [0])


def test_dispatch_bn_forward_matrix_matches_direct_and_patches_raise() -> None:
    args = _make_bn_case()
    direct = forward_bn(*args)
    dispatched = dispatch_bn_forward(*args)
    _assert_forward_tuple_close(dispatched, direct)
    layer, parent_boxes, _parent_lins, parent_frames, preds, post_activation, device, dtype = args
    with pytest.raises(NotImplementedError, match="Filled in W3/W4"):
        dispatch_bn_forward(
            layer,
            parent_boxes,
            [Patches(identity=1)],
            parent_frames,
            preds,
            post_activation,
            device,
            dtype,
        )


def test_dispatch_add_forward_matrix_matches_direct_and_patches_raise() -> None:
    args = _make_add_case()
    direct = forward_add(*args)
    dispatched = dispatch_add_forward(*args)
    _assert_forward_tuple_close(dispatched, direct)
    layer, parent_boxes, parent_lins, parent_frames, preds, post_activation, device, dtype = args
    with pytest.raises(NotImplementedError, match="Filled in W3/W4"):
        dispatch_add_forward(
            layer,
            parent_boxes,
            [parent_lins[0], Patches(identity=1)],
            parent_frames,
            preds,
            post_activation,
            device,
            dtype,
        )


def test_dispatch_pool_forward_max_matrix_matches_direct_and_patches_raise() -> None:
    args = _make_pool_case(LayerKind.MAXPOOL2D.value)
    direct = forward_maxpool2d(*args)
    dispatched = dispatch_pool_forward(*args)
    _assert_forward_tuple_close(dispatched, direct)
    layer, parent_boxes, _parent_lins, parent_frames, preds, post_activation, device, dtype = args
    with pytest.raises(NotImplementedError, match="Filled in W3/W4"):
        dispatch_pool_forward(
            layer,
            parent_boxes,
            [Patches(identity=1)],
            parent_frames,
            preds,
            post_activation,
            device,
            dtype,
        )


def test_dispatch_pool_forward_avg_matrix_matches_direct_and_patches_raise() -> None:
    args = _make_pool_case(LayerKind.AVGPOOL2D.value)
    direct = forward_avgpool2d(*args)
    dispatched = dispatch_pool_forward(*args)
    _assert_forward_tuple_close(dispatched, direct)
    layer, parent_boxes, _parent_lins, parent_frames, preds, post_activation, device, dtype = args
    with pytest.raises(NotImplementedError, match="Filled in W3/W4"):
        dispatch_pool_forward(
            layer,
            parent_boxes,
            [Patches(identity=1)],
            parent_frames,
            preds,
            post_activation,
            device,
            dtype,
        )


def test_cli_feature_a_only_flag_parses() -> None:
    args = benchmark._parse_args(["--feature-a-only"])
    assert args.feature_a_only is True


def test_cli_measure_vram_emits_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class _DummyStatus:
        name = "UNKNOWN"

    class _DummyResult:
        status = _DummyStatus()
        metadata = {"nodes": 7, "ces_attempted": 0}

    monkeypatch.setattr(benchmark, "initialize_device", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark, "load_model_and_instance", lambda **_kwargs: (object(), KNOWN_VNNLIB))
    monkeypatch.setattr(benchmark, "verify_bab", lambda *args, **kwargs: _DummyResult())
    monkeypatch.setattr(benchmark, "_peak_vram_gb", lambda _device: 1.25)

    exit_code = benchmark.main(["--measure-vram", "--output", str(tmp_path / "out.json")])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "peak_conv_layer_vram_mb" in captured
    assert "peak_conv_layer_vram_gb" in captured
    assert "peak_total_vram_gb" in captured


def test_cli_assert_no_densify_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = benchmark.main(["--assert-no-densify"])
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "ALL_CONV_LAYERS_PATCHES=None" in captured


def test_cli_all_flags_in_help() -> None:
    help_text = benchmark._build_parser().format_help()
    for flag in [
        "--feature-a-only",
        "--measure-vram",
        "--assert-no-densify",
        "--all-instances",
        "--all-configs",
        "--output",
        "--vnnlib-name",
        "--device",
    ]:
        assert flag in help_text


def test_vnnlib_name_resolves_to_row_index() -> None:
    assert resolve_vnnlib_row_index(KNOWN_VNNLIB) == 3


def test_vnnlib_name_not_found_raises() -> None:
    with pytest.raises(ValueError, match="Unknown CIFAR100_resnet_medium vnnlib_name"):
        resolve_vnnlib_row_index("missing-property.vnnlib")


def test_instance_and_vnnlib_name_vnnlib_wins(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        resolved = resolve_vnnlib_row_index(KNOWN_VNNLIB, instance_idx=1)
    assert resolved == 3
    assert "vnnlib_name wins" in caplog.text


def test_device_auto_prefers_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
    args = benchmark._parse_args(["--device", "auto"])
    assert args.device == "cuda"
