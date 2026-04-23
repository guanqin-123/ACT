#===- act/back_end/counterexample_io.py - VNN-LIB CE I/O ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Persist FALSIFIED counterexamples (CEs) to disk in VNN-LIB / VNN-COMP
#   format. Public API:
#
#     * ``validate_counterexample(net, ce_input, assert_layer)`` — concrete
#       forward + per-OutKind violation check via the batched oracle in
#       ``bab.bab``. No file I/O.
#     * ``save_counterexample(net, ce_input, model_output, network_path,
#       status_metadata)`` — writes ``cex.counterexample`` (VNN-LIB sat body)
#       and a ``metadata.json`` sidecar under
#       ``<pipeline_log>/counterexamples/<stem>_<ISO8601_us>/``.
#     * ``validate_and_save(...)`` — combo helper: saves iff the CE is a true
#       concrete violation.
#
# Invariants:
#   - Only TRUE (concretely verified) CEs are ever written to disk.
#   - VNN-LIB output is byte-exact: ``sat`` header, two-space indent inside
#     parens, ``repr(float(v))`` for values (preserves dtype precision).
#   - Variable naming follows VNN-COMP convention: ``X_0..X_{N-1}`` for inputs
#     (row-major flatten), ``Y_0..Y_{M-1}`` for outputs.
#   - All paths resolve through ``path_config.get_pipeline_log_dir()`` —
#     nothing is hardcoded.
#   - Per-OutKind violation logic is NOT duplicated; we delegate to
#     ``bab.bab.check_violation_at_point_batched``.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

from act.back_end.core import Layer, Net
from act.util.device_manager import get_default_device, get_default_dtype
from act.util.path_config import get_pipeline_log_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _format_vnnlib_counterexample(
    input_flat: torch.Tensor, output_flat: torch.Tensor
) -> str:
    """Render flat input/output tensors as a byte-exact VNN-LIB ``sat`` body.

    Format: ``sat\\n(\\n  (X_i <repr(float(v))>)\\n  ...\\n  (Y_j <...>)\\n)``
    with two-space indent and no trailing newline. ``repr(float(v))``
    preserves float32 vs float64 precision (e.g. float32 ``0.1`` becomes
    ``'0.10000000149011612'``).
    """
    lines: list[str] = ["sat", "("]
    for i in range(int(input_flat.numel())):
        v = float(input_flat[i].item())
        lines.append(f"  (X_{i} {repr(v)})")
    for j in range(int(output_flat.numel())):
        v = float(output_flat[j].item())
        lines.append(f"  (Y_{j} {repr(v)})")
    lines.append(")")
    return "\n".join(lines)


def _get_git_sha() -> str:
    """Short git SHA of HEAD, or ``"unknown"`` on any failure (not a repo etc.)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _network_stem(network_path: Optional[Path]) -> str:
    """Filename stem of ``network_path`` or ``"unnamed_net"`` when ``None``."""
    if network_path is None:
        return "unnamed_net"
    return Path(network_path).stem


def _make_output_dir(network_path: Optional[Path]) -> Path:
    """Create ``<pipeline_log>/counterexamples/<stem>_<ISO_us>/`` and return it.

    Microsecond precision (``%Y%m%dT%H%M%S_%f``) guarantees uniqueness under
    rapid double-calls without needing a collision-counter suffix.
    """
    iso_ts = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    stem = _network_stem(network_path)
    dir_path = (
        Path(get_pipeline_log_dir()) / "counterexamples" / f"{stem}_{iso_ts}"
    )
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def _write_metadata_json(metadata: dict[str, Any], path: Path) -> None:
    """JSON-dump ``metadata`` to ``path``. ``default=str`` covers ``Path``,
    ``datetime``, ``torch.dtype``, numpy scalars etc. without extra code."""
    with path.open("w") as f:
        json.dump(metadata, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_counterexample(
    net: Net,
    ce_input: torch.Tensor,
    assert_layer: Layer,
) -> tuple[bool, Optional[torch.Tensor]]:
    """Concrete-inference validation of a candidate counterexample.

    Runs the real PyTorch model on ``ce_input`` and asks the batched
    per-OutKind oracle (``check_violation_at_point_batched`` in
    ``act.back_end.bab.bab``) whether the property is truly violated. This is
    the single source of truth for "true vs. spurious CE" across the code
    base — we never duplicate the ASSERT-kind switch-statement here.

    Args:
        net: ACT ``Net`` containing the model and the ASSERT layer.
        ce_input: Candidate input tensor (1-D or 2-D; auto-batched if 1-D).
        assert_layer: The target ASSERT layer to check violation against.

    Returns:
        ``(is_true_violation, model_output)``. ``model_output`` is the first
        row of the model forward on the (batched) input, or ``None`` if the
        model could not be inferred (e.g. conversion failure).
    """
    # Local import breaks a top-level circular dependency: ``bab.py`` imports
    # ``save_counterexample`` from this module at module load, so this module
    # MUST NOT import from ``bab.py`` at top-level. The local import inside
    # ``validate_counterexample`` defers the resolution until both modules are
    # fully loaded, making the runtime cycle safe.
    from act.back_end.bab.bab import (
        _run_model,
        check_violation_at_point_batched,
    )

    ce = ce_input.to(device=get_default_device(), dtype=get_default_dtype())
    ce_batch = ce.unsqueeze(0) if ce.dim() == 1 else ce

    model_output = _run_model(net, ce_batch)
    if model_output is None:
        return (False, None)

    violation_mask = check_violation_at_point_batched(net, ce_batch, assert_layer)
    is_true = bool(violation_mask[0].item())
    return (is_true, model_output[0])


def save_counterexample(
    net: Net,
    ce_input: torch.Tensor,
    model_output: torch.Tensor,
    network_path: Optional[Path],
    status_metadata: dict[str, Any],
) -> Path:
    """Persist a validated CE as VNN-LIB + JSON sidecar; return the new dir.

    Writes two files under ``<pipeline_log>/counterexamples/<stem>_<ISO_us>/``:

    * ``cex.counterexample`` — VNN-LIB sat body (see
      :func:`_format_vnnlib_counterexample`).
    * ``metadata.json`` — a rich sidecar with status, verifier mode, property
      kind, tensor shape/dtype, timestamp, network path, solver, time limit,
      git SHA, node count, spurious flag, plus any extra keys from
      ``status_metadata`` that are not explicitly reserved.

    This function does NOT re-validate — callers must ensure ``ce_input`` is
    a TRUE violation (use :func:`validate_and_save` for the combo).

    Args:
        net: ACT ``Net`` (used only for context; no analysis is run here).
        ce_input: The violating input tensor (any shape; flattened row-major).
        model_output: Concrete model output at ``ce_input`` (any shape).
        network_path: Optional source path of the network JSON (used to
            derive the output dir stem).
        status_metadata: Caller-supplied fields (``verifier_mode``,
            ``property_kind``, ``solver``, ``timelimit``, ``nodes``, and any
            extras to pass through).

    Returns:
        The absolute path of the freshly-created output directory.
    """
    input_flat = ce_input.detach().cpu().reshape(-1)
    output_flat = model_output.detach().cpu().reshape(-1)
    body = _format_vnnlib_counterexample(input_flat, output_flat)

    out_dir = _make_output_dir(network_path)
    (out_dir / "cex.counterexample").write_text(body)

    # Schema-owned keys MUST NOT be overwritten by caller-supplied extras.
    # If the caller passes any of these (e.g. solver-side ``status``), they are
    # silently dropped to preserve canonical metadata. Use a different key name
    # in extras (e.g. ``solver_status``) if you need to preserve such values.
    reserved = {
        "status",
        "verifier_mode",
        "property_kind",
        "ce_shape",
        "ce_dtype",
        "saved_at",
        "network_path",
        "solver",
        "timelimit",
        "git_sha",
        "nodes",
        "spurious_ce",
    }
    extras = {k: v for k, v in status_metadata.items() if k not in reserved}
    meta: dict[str, Any] = {
        "status": "FALSIFIED",
        "verifier_mode": status_metadata.get("verifier_mode", "unknown"),
        "property_kind": status_metadata.get("property_kind", "unknown"),
        "ce_shape": list(ce_input.shape),
        "ce_dtype": str(ce_input.dtype),
        "saved_at": datetime.now().isoformat(),
        "network_path": str(network_path) if network_path is not None else None,
        "solver": status_metadata.get("solver", "unknown"),
        "timelimit": status_metadata.get("timelimit"),
        "git_sha": _get_git_sha(),
        "nodes": status_metadata.get("nodes"),
        "spurious_ce": False,
        **extras,
    }
    _write_metadata_json(meta, out_dir / "metadata.json")
    logger.info("Counterexample saved to %s", out_dir)
    return out_dir


def validate_and_save(
    net: Net,
    ce_input: torch.Tensor,
    assert_layer: Layer,
    network_path: Optional[Path],
    status_metadata: dict[str, Any],
) -> tuple[bool, Optional[Path]]:
    """Validate a candidate CE and save iff it is a true concrete violation.

    On spurious CEs, logs a warning and returns ``(False, None)`` — no file
    is ever written for a spurious candidate (per the hard invariant).

    Args:
        net: ACT ``Net``.
        ce_input: Candidate violating input.
        assert_layer: ASSERT layer to validate against.
        network_path: Optional source path of the network JSON (passed
            through to :func:`save_counterexample`).
        status_metadata: Metadata dict to embed in the JSON sidecar.

    Returns:
        ``(True, output_dir)`` on a true CE that was saved; ``(False, None)``
        on a spurious or unverifiable candidate.
    """
    is_true, model_output = validate_counterexample(net, ce_input, assert_layer)
    if is_true and model_output is not None:
        path = save_counterexample(
            net, ce_input, model_output, network_path, status_metadata
        )
        return (True, path)

    logger.warning(
        "Spurious CE rejected (concrete inference shows no violation); not saved."
    )
    return (False, None)
