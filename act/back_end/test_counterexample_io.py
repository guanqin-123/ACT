#===- act/back_end/test_counterexample_io.py - CE I/O Tests (RED) --------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===----------------------------------------------------------------------===#
#
# Purpose:
#   Failing unit tests (TDD RED phase) for the not-yet-implemented module
#   `act.back_end.counterexample_io`. Covers the 13 enumerated test cases from
#   `.sisyphus/plans/save_counterexample_vnnlib.md` section
#   "NEW TESTS: act/back_end/test_counterexample_io.py":
#
#     Format (3):    byte-exact VNN-LIB, row-major flatten, dtype preservation.
#     Validation (3): true-CE, spurious, agreement with check_violation_at_point.
#     Save/path (4): timestamped dir, both files written, double-call, roundtrip.
#     Default dir (1): get_pipeline_log_dir is used (no I/O performed).
#     Combo (2):     validate_and_save returns (True, Path) / (False, None).
#
#   The target module `counterexample_io` does NOT yet exist; each test performs
#   a local import inside the function body, which allows pytest collection to
#   succeed (this file parses cleanly) while the tests themselves fail with
#   ModuleNotFoundError. Wave 2 (T2.1) will implement the module and flip these
#   tests to GREEN.
#
#===----------------------------------------------------------------------===#

from __future__ import annotations

"""Regression tests for the VNN-LIB counterexample I/O module
(``act.back_end.counterexample_io``). Covers the public API
(``validate_counterexample``, ``save_counterexample``, ``validate_and_save``),
internal formatting / path helpers, and round-trip parsing using the same
``X_i`` / ``Y_j`` regexes as the project's vnnlib_loader."""

import re
from pathlib import Path

import pytest
import torch

from act.back_end.bab.bab import check_violation_at_point_batched
from act.back_end.core import Layer, Net
from act.back_end.layer_schema import LayerKind
from act.front_end.specs import OutKind
from act.util.device_manager import (
    get_default_device,
    get_default_dtype,
    initialize_device,
)


# ---------------------------------------------------------------------------
# Test helpers (adapted from act/back_end/bab/test_bab_batched.py)
# ---------------------------------------------------------------------------


def _t(data: object, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Create a tensor on the default device, float64 by default for this file."""
    return torch.tensor(
        data,
        dtype=dtype if dtype is not None else get_default_dtype(),
        device=get_default_device(),
    )


def _make_relu_net(
    hidden_bias: float = 0.0,
    out_weight: float = 1.0,
    out_bias: float = 0.0,
    threshold: float = 0.1,
) -> Net:
    """1-input, 1-output ReLU net with LINEAR_LE ASSERT (y <= threshold).

    Mirrors the `_make_relu_net` fixture in `test_bab_batched.py`. For
    ``hidden_bias=0.0, out_weight=1.0, out_bias=0.0, threshold=0.1`` the concrete
    forward is ``y = max(0, x)`` over the BOX ``x in [-1, 1]``. Thus:

        * x = 0.5  -> y = 0.5, violates y <= 0.1 (true CE candidate).
        * x = -0.5 -> y = 0.0, satisfies y <= 0.1 (spurious CE).
    """
    in_vars = [0]
    z = [10]
    a = [20]
    out = [30]
    layers = [
        Layer(
            0,
            LayerKind.INPUT.value,
            {"shape": [1], "dtype": "float64", "num_classes": 1,
             "value_range": [-1.0, 1.0]},
            in_vars,
            in_vars,
        ),
        Layer(
            1,
            LayerKind.INPUT_SPEC.value,
            {"kind": "BOX", "lb": _t([-1.0]), "ub": _t([1.0])},
            in_vars,
            in_vars,
        ),
        Layer(
            2,
            LayerKind.DENSE.value,
            {"in_features": 1, "out_features": 1,
             "weight": _t([[1.0]]), "bias": _t([hidden_bias])},
            in_vars,
            z,
        ),
        Layer(3, LayerKind.RELU.value, {}, z, a),
        Layer(
            4,
            LayerKind.DENSE.value,
            {"in_features": 1, "out_features": 1,
             "weight": _t([[out_weight]]), "bias": _t([out_bias])},
            a,
            out,
        ),
        Layer(
            5,
            LayerKind.ASSERT.value,
            {"kind": OutKind.LINEAR_LE, "c": _t([1.0]), "d": _t([threshold])},
            out,
            out,
        ),
    ]
    return Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )


def _patch_log_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect the CE I/O module's ``get_pipeline_log_dir`` to ``tmp_path``.

    Kept in one place so every save-test uses the identical monkeypatch target,
    and so the intent ("never touch the real pipeline log dir in tests") is
    obvious at the call site.
    """
    monkeypatch.setattr(
        "act.back_end.counterexample_io.get_pipeline_log_dir",
        lambda: str(tmp_path),
    )


# ---------------------------------------------------------------------------
# Module-level setup: float64 default so byte-exact ``repr`` of 0.1 is '0.1'.
# ---------------------------------------------------------------------------


def setup_module() -> None:
    """Initialize device as CPU + float64 for deterministic format output.

    float64 is required for test #1 (byte-exact ``(Y_0 0.1)``) because
    ``repr(float(torch.tensor(0.1, dtype=torch.float32).item()))`` is
    ``'0.10000000149011612'`` (float32 rounding) whereas in float64 it is the
    short form ``'0.1'``.
    """
    initialize_device("cpu", "float64")


# ---------------------------------------------------------------------------
# Format tests (3)
# ---------------------------------------------------------------------------


def test_format_vnnlib_simple_2x3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Byte-exact VNN-LIB format for a [2-input, 3-output] counterexample.

    The saved ``cex.counterexample`` file must equal the VNN-COMP sat body
    string exactly (no trailing newline, two-space indent, ``repr(float(v))``
    for values).
    """
    from act.back_end.counterexample_io import save_counterexample

    _patch_log_dir(monkeypatch, tmp_path)
    net = _make_relu_net()
    ce_input = torch.tensor([0.5, -1.0], dtype=torch.float64)
    model_output = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

    out_dir = save_counterexample(
        net,
        ce_input,
        model_output,
        Path("demo_net.json"),
        {"status": "FALSIFIED", "verifier_mode": "verify_once",
         "property_kind": "LINEAR_LE"},
    )

    content = (out_dir / "cex.counterexample").read_text()
    expected = (
        "sat\n"
        "(\n"
        "  (X_0 0.5)\n"
        "  (X_1 -1.0)\n"
        "  (Y_0 0.1)\n"
        "  (Y_1 0.2)\n"
        "  (Y_2 0.3)\n"
        ")"
    )
    assert content == expected, (
        f"VNN-LIB format mismatch.\n"
        f"--- expected ({len(expected)} bytes) ---\n{expected!r}\n"
        f"--- got ({len(content)} bytes) ---\n{content!r}"
    )


def test_format_flatten_rowmajor_cifar_3x32x32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A [3,32,32] tensor flattens row-major to 3072 X variables in order.

    ``torch.arange(3072).view(3,32,32)`` puts index ``i`` at flat position ``i``
    under row-major (C-order). The saved file must then list
    ``(X_0 0.0)`` ... ``(X_3071 3071.0)`` in order, extractable via the
    project's ``_X_RE`` regex.
    """
    from act.back_end.counterexample_io import save_counterexample

    _patch_log_dir(monkeypatch, tmp_path)
    net = _make_relu_net()
    ce_input = torch.arange(3 * 32 * 32, dtype=torch.float32).view(3, 32, 32)
    model_output = torch.zeros(10, dtype=torch.float32)

    out_dir = save_counterexample(
        net,
        ce_input,
        model_output,
        Path("cifar_net.json"),
        {"status": "FALSIFIED"},
    )

    content = (out_dir / "cex.counterexample").read_text()
    x_pairs = re.findall(r"\(X_(\d+)\s+([^)]+)\)", content)

    assert len(x_pairs) == 3072, (
        f"Expected 3072 X variables, got {len(x_pairs)}"
    )
    for i, (idx_str, val_str) in enumerate(x_pairs):
        assert int(idx_str) == i, (
            f"Out-of-order X variable at position {i}: got X_{idx_str}"
        )
        assert float(val_str) == float(i), (
            f"X_{i} value mismatch: got {val_str!r}, expected {float(i)!r}"
        )


def test_format_preserves_float32_vs_float64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dtype precision survives into the VNN-LIB output via ``repr(float(v))``.

    For the value 0.1 the float32 cast produces the Python float
    ``0.10000000149011612`` (its exact float32 bit pattern) whereas float64
    keeps the short form ``0.1``. The two saved files must therefore differ.
    """
    from act.back_end.counterexample_io import save_counterexample

    _patch_log_dir(monkeypatch, tmp_path)
    net = _make_relu_net()

    f32_in = torch.tensor([0.1], dtype=torch.float32)
    f32_out = torch.tensor([0.2], dtype=torch.float32)
    f64_in = torch.tensor([0.1], dtype=torch.float64)
    f64_out = torch.tensor([0.2], dtype=torch.float64)

    d32 = save_counterexample(net, f32_in, f32_out, Path("n32.json"),
                              {"status": "FALSIFIED"})
    d64 = save_counterexample(net, f64_in, f64_out, Path("n64.json"),
                              {"status": "FALSIFIED"})

    content32 = (d32 / "cex.counterexample").read_text()
    content64 = (d64 / "cex.counterexample").read_text()

    assert content32 != content64, (
        "Expected float32 / float64 outputs to differ in precision, "
        "but both files are identical."
    )
    # float64 uses the short repr
    assert "(X_0 0.1)" in content64, (
        f"Expected short float64 repr '(X_0 0.1)' in:\n{content64}"
    )
    # float32 value 0.1 is actually 0.10000000149011612 as a Python float
    assert "0.10000000149011612" in content32, (
        f"Expected float32 long repr '0.10000000149011612' in:\n{content32}"
    )


# ---------------------------------------------------------------------------
# Validation tests (3)
# ---------------------------------------------------------------------------


def test_validate_ce_true_violation() -> None:
    """``validate_counterexample`` returns (True, output) for a violating input.

    Net: y = max(0, x) with assert y <= 0.1. At x = 0.5 -> y = 0.5, violates.
    """
    from act.back_end.counterexample_io import validate_counterexample

    net = _make_relu_net(hidden_bias=0.0, out_weight=1.0, out_bias=0.0, threshold=0.1)
    assert_layer = net.layers[-1]
    ce_input = _t([0.5])

    is_true, output = validate_counterexample(net, ce_input, assert_layer)

    assert is_true is True, f"Expected True for violating input, got {is_true}"
    assert output is not None, "Expected a concrete model output tensor, got None"
    assert isinstance(output, torch.Tensor), (
        f"Expected torch.Tensor output, got {type(output)}"
    )
    # y = max(0, 0.5) = 0.5; allow for broadcast/shape flexibility but value ~0.5.
    assert torch.allclose(
        output.reshape(-1)[:1], _t([0.5]), atol=1e-6
    ), f"Expected output ~[0.5], got {output}"


def test_validate_ce_spurious() -> None:
    """``validate_counterexample`` returns (False, output) for a safe input.

    Net: y = max(0, x) with assert y <= 0.1. At x = -0.5 -> y = 0.0, safe.
    """
    from act.back_end.counterexample_io import validate_counterexample

    net = _make_relu_net(hidden_bias=0.0, out_weight=1.0, out_bias=0.0, threshold=0.1)
    assert_layer = net.layers[-1]
    ce_input = _t([-0.5])

    is_true, output = validate_counterexample(net, ce_input, assert_layer)

    assert is_true is False, f"Expected False for safe input, got {is_true}"
    assert output is not None, "Expected a concrete model output tensor even on spurious"
    assert isinstance(output, torch.Tensor), (
        f"Expected torch.Tensor output, got {type(output)}"
    )
    assert torch.allclose(
        output.reshape(-1)[:1], _t([0.0]), atol=1e-6
    ), f"Expected output ~[0.0], got {output}"


def test_validate_consistency_with_check_violation_at_point() -> None:
    """For 5 random inputs in the BOX, ``validate_counterexample`` agrees with
    ``check_violation_at_point_batched`` on the boolean result (same oracle).
    """
    from act.back_end.counterexample_io import validate_counterexample

    torch.manual_seed(42)
    net = _make_relu_net(hidden_bias=0.0, out_weight=1.0, out_bias=0.0, threshold=0.1)
    assert_layer = net.layers[-1]

    for i in range(5):
        # Uniform in [-1, 1] — matches the BOX spec of the tiny net.
        x = torch.rand(1, dtype=get_default_dtype(), device=get_default_device()) * 2.0 - 1.0

        validated, _out = validate_counterexample(net, x, assert_layer)
        batched_mask = check_violation_at_point_batched(net, x.unsqueeze(0), assert_layer)
        batched = bool(batched_mask[0].item())

        assert bool(validated) == batched, (
            f"Disagreement at sample {i}: x={x.item():.6f}, "
            f"validate={validated}, batched={batched}"
        )


# ---------------------------------------------------------------------------
# Save / path tests (4)
# ---------------------------------------------------------------------------


def test_save_creates_per_network_timestamped_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``save_counterexample`` creates ``<stem>_YYYYMMDDTHHMMSS_ffffff/``.

    Uniqueness is guaranteed by microsecond-precision ISO timestamp so rapid
    double-calls (test #9) do not collide.
    """
    from act.back_end.counterexample_io import save_counterexample

    _patch_log_dir(monkeypatch, tmp_path)
    net = _make_relu_net()
    ce_input = torch.tensor([0.5], dtype=torch.float64)
    model_output = torch.tensor([0.5], dtype=torch.float64)

    out_dir = save_counterexample(
        net,
        ce_input,
        model_output,
        Path("my_net.json"),
        {"status": "FALSIFIED"},
    )

    assert out_dir.exists(), f"Expected {out_dir} to exist"
    assert out_dir.is_dir(), f"Expected {out_dir} to be a directory"
    assert re.fullmatch(r"my_net_\d{8}T\d{6}_\d{6}", out_dir.name), (
        f"Expected name matching '<stem>_<YYYYMMDDTHHMMSS_ffffff>', "
        f"got: {out_dir.name!r}"
    )


def test_save_produces_counterexample_and_json_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both ``cex.counterexample`` and ``metadata.json`` exist after save."""
    from act.back_end.counterexample_io import save_counterexample

    _patch_log_dir(monkeypatch, tmp_path)
    out_dir = save_counterexample(
        _make_relu_net(),
        torch.tensor([0.5], dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64),
        Path("my_net.json"),
        {"status": "FALSIFIED"},
    )

    counterexample_file = out_dir / "cex.counterexample"
    metadata_file = out_dir / "metadata.json"
    assert counterexample_file.exists(), f"Missing {counterexample_file}"
    assert metadata_file.exists(), f"Missing {metadata_file}"
    # Sanity: non-empty so we know they were actually written.
    assert counterexample_file.stat().st_size > 0, "cex.counterexample is empty"
    assert metadata_file.stat().st_size > 0, "metadata.json is empty"


def test_save_non_destructive_double_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two consecutive saves for the same network produce two distinct dirs
    (microsecond ISO timestamp guarantees uniqueness)."""
    from act.back_end.counterexample_io import save_counterexample

    _patch_log_dir(monkeypatch, tmp_path)
    net = _make_relu_net()
    ce = torch.tensor([0.5], dtype=torch.float64)
    out = torch.tensor([0.5], dtype=torch.float64)

    d1 = save_counterexample(net, ce, out, Path("my_net.json"),
                             {"status": "FALSIFIED"})
    d2 = save_counterexample(net, ce, out, Path("my_net.json"),
                             {"status": "FALSIFIED"})

    assert d1 != d2, f"Expected distinct timestamp dirs, got duplicate: {d1}"
    assert d1.exists() and d2.exists(), (
        f"Both directories should exist. d1.exists={d1.exists()}, "
        f"d2.exists={d2.exists()}"
    )
    # Neither call may overwrite the other's files.
    assert (d1 / "cex.counterexample").exists()
    assert (d2 / "cex.counterexample").exists()


def test_roundtrip_via_vnnlib_regex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Save a CE then re-parse with the project's VNN-LIB regexes and recover
    values within ``torch.allclose`` tolerance.

    Uses the same ``_X_RE`` / ``_Y_RE`` patterns as
    ``act.front_end.vnnlib_loader.vnnlib_parser`` (lines 326-327), ensuring the
    produced file is round-trip compatible with the existing parser.
    """
    from act.back_end.counterexample_io import save_counterexample

    _patch_log_dir(monkeypatch, tmp_path)
    torch.manual_seed(42)
    net = _make_relu_net()
    ce_input = torch.rand(5, dtype=torch.float64)
    model_output = torch.rand(3, dtype=torch.float64)

    out_dir = save_counterexample(
        net,
        ce_input,
        model_output,
        Path("roundtrip_net.json"),
        {"status": "FALSIFIED"},
    )
    content = (out_dir / "cex.counterexample").read_text()

    x_pairs = re.findall(r"\(X_(\d+)\s+([^)]+)\)", content)
    y_pairs = re.findall(r"\(Y_(\d+)\s+([^)]+)\)", content)

    assert len(x_pairs) == 5, f"Expected 5 X vars, got {len(x_pairs)}"
    assert len(y_pairs) == 3, f"Expected 3 Y vars, got {len(y_pairs)}"

    x_sorted = sorted(x_pairs, key=lambda p: int(p[0]))
    y_sorted = sorted(y_pairs, key=lambda p: int(p[0]))
    x_recovered = torch.tensor(
        [float(v) for _idx, v in x_sorted], dtype=torch.float64
    )
    y_recovered = torch.tensor(
        [float(v) for _idx, v in y_sorted], dtype=torch.float64
    )

    assert torch.allclose(x_recovered, ce_input.to(torch.float64), atol=1e-15), (
        f"X values did not round-trip. recovered={x_recovered}, "
        f"original={ce_input}"
    )
    assert torch.allclose(y_recovered, model_output.to(torch.float64), atol=1e-15), (
        f"Y values did not round-trip. recovered={y_recovered}, "
        f"original={model_output}"
    )


# ---------------------------------------------------------------------------
# Default directory test (1) — no real filesystem I/O
# ---------------------------------------------------------------------------


def test_save_uses_get_pipeline_log_dir_not_tmp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without monkeypatching, the default output dir lives under
    ``act/pipeline/log/counterexamples/``.

    We only inspect the returned path string; ``Path.mkdir`` is monkeypatched to
    a no-op so no real directory is created under the project tree.
    """
    from act.back_end.counterexample_io import _make_output_dir

    monkeypatch.setattr("pathlib.Path.mkdir", lambda self, *a, **kw: None)

    result = _make_output_dir(Path("some_network.json"))
    result_str = str(result).replace("\\", "/")

    assert "pipeline/log/counterexamples" in result_str, (
        f"Expected path under 'pipeline/log/counterexamples', got: {result_str!r}"
    )
    assert "/tmp/" not in result_str, (
        f"Default path must not fall back to /tmp/, got: {result_str!r}"
    )
    # Confirm no actual directory creation occurred (mkdir was a no-op).
    assert not result.exists(), (
        f"Directory {result} should NOT exist — Path.mkdir was mocked as no-op"
    )


# ---------------------------------------------------------------------------
# validate_and_save combo tests (2)
# ---------------------------------------------------------------------------


def test_validate_and_save_returns_none_on_spurious(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a spurious (non-violating) input ``validate_and_save`` returns
    ``(False, None)`` and writes NO file to disk."""
    from act.back_end.counterexample_io import validate_and_save

    _patch_log_dir(monkeypatch, tmp_path)
    net = _make_relu_net(hidden_bias=0.0, out_weight=1.0, out_bias=0.0, threshold=0.1)
    assert_layer = net.layers[-1]
    safe_input = _t([-0.5])  # y = 0.0, does not violate y <= 0.1

    saved, path = validate_and_save(
        net,
        safe_input,
        assert_layer,
        Path("some_net.json"),
        {"status": "FALSIFIED"},
    )

    assert saved is False, f"Expected saved=False for spurious, got {saved}"
    assert path is None, f"Expected path=None for spurious, got {path}"
    # tmp_path must remain empty (no counterexamples subtree, no files).
    leftover = list(tmp_path.rglob("*"))
    assert leftover == [], (
        f"No files should be written on spurious CE; found: {leftover}"
    )


def test_validate_and_save_returns_path_on_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a true-CE ``validate_and_save`` returns ``(True, Path)`` and the
    ``cex.counterexample`` file exists at the returned Path."""
    from act.back_end.counterexample_io import validate_and_save

    _patch_log_dir(monkeypatch, tmp_path)
    net = _make_relu_net(hidden_bias=0.0, out_weight=1.0, out_bias=0.0, threshold=0.1)
    assert_layer = net.layers[-1]
    violating_input = _t([0.5])  # y = 0.5, violates y <= 0.1

    saved, path = validate_and_save(
        net,
        violating_input,
        assert_layer,
        Path("violating_net.json"),
        {"status": "FALSIFIED"},
    )

    assert saved is True, f"Expected saved=True for true CE, got {saved}"
    assert path is not None, "Expected a Path for true CE, got None"
    assert isinstance(path, Path), f"Expected Path, got {type(path)}"
    assert path.exists(), f"Returned path {path} does not exist"
    assert (path / "cex.counterexample").exists(), (
        f"cex.counterexample missing under {path}"
    )
    assert (path / "metadata.json").exists(), (
        f"metadata.json missing under {path}"
    )
