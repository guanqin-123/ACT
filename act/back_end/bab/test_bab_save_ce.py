#===- act/back_end/bab/test_bab_save_ce.py - CE save E2E tests ---------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

"""End-to-end integration tests for VNN-LIB counterexample saving on FALSIFIED."""

import json
import re
from pathlib import Path

import pytest
import torch

from act.back_end.bab.bab import verify_bab
from act.back_end.bab.test_bab_batched import _make_relu_net, _batched_cfg
from act.back_end.dual_tf.dual_tf import DualTF
from act.back_end.solver.solver_dual import DualSolver
from act.back_end.solver.solver_interval import TorchLPSolver
from act.util.stats import VerifyStatus


def test_verify_bab_falsified_saves_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """E2E: verify_bab with a known-falsifying net writes cex.counterexample + metadata.json."""
    # Monkeypatch the consumer's binding (counterexample_io's import of get_pipeline_log_dir)
    monkeypatch.setattr(
        "act.back_end.counterexample_io.get_pipeline_log_dir",
        lambda: str(tmp_path),
    )

    # Build the same falsifying net used by test_root_falsified
    net = _make_relu_net(hidden_bias=0.0, out_weight=1.0, out_bias=0.0, threshold=0.1)

    res = verify_bab(
        net,
        TorchLPSolver(),
        config=_batched_cfg(subproblem_batch_size=1, max_nodes=2, max_depth=0),
        dual_solver=DualSolver(DualTF()),
        network_path=Path("smoke_relu_falsify.json"),
    )

    # Status assertions
    assert res.status == VerifyStatus.FALSIFIED, f"expected FALSIFIED, got {res.status}"
    assert res.counterexample is not None, "counterexample tensor should be populated"
    assert "saved_to" in res.metadata, "metadata should contain 'saved_to'"

    saved_to = res.metadata["saved_to"]
    assert saved_to is not None, "saved_to should be populated (model_output computed)"

    saved_dir = Path(saved_to)
    assert saved_dir.exists(), f"saved directory does not exist: {saved_dir}"
    assert (saved_dir / "cex.counterexample").exists(), "cex.counterexample missing"
    assert (saved_dir / "metadata.json").exists(), "metadata.json missing"

    # Verify dir naming pattern: <stem>_<ISO8601_us>
    assert re.search(r"smoke_relu_falsify_\d{8}T\d{6}_\d{6}", str(saved_dir)), \
        f"unexpected dir name: {saved_dir.name}"

    # Verify content: starts with "sat\n"
    cex_text = (saved_dir / "cex.counterexample").read_text()
    assert cex_text.startswith("sat\n"), f"expected 'sat\\n' header, got: {cex_text[:20]!r}"

    # Verify X count matches input shape product
    input_size = int(torch.tensor(res.counterexample.shape).prod().item())
    xs = re.findall(r"\(X_(\d+)\s+([^)]+)\)", cex_text)
    assert len(xs) == input_size, f"X count mismatch: file has {len(xs)}, expected {input_size}"

    # Verify Y count >= 1 (at least one output)
    ys = re.findall(r"\(Y_(\d+)\s+([^)]+)\)", cex_text)
    assert len(ys) >= 1, f"Y count should be >=1, got {len(ys)}"

    # Round-trip: extracted X values match res.counterexample within tolerance
    extracted_xs = torch.tensor([float(v) for _, v in xs])
    flat_ce = res.counterexample.flatten()
    assert torch.allclose(extracted_xs, flat_ce, atol=1e-5), "X round-trip mismatch"

    # metadata.json sanity
    meta = json.loads((saved_dir / "metadata.json").read_text())
    for key in ("status", "verifier_mode", "property_kind", "ce_shape", "ce_dtype",
                "saved_at", "network_path", "solver", "git_sha", "nodes"):
        assert key in meta, f"metadata.json missing key: {key}"
    assert meta["status"] == "FALSIFIED"
    assert meta["verifier_mode"] == "verify_bab_batched"
    assert meta["solver"] == "DualSolver"


def test_verify_once_falsified_saves_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """E2E: verify_once true-save path produces canonical metadata.json schema.

    Regression test for a bug where ``**stats`` from the solver leaked into
    ``status_metadata`` and overwrote the canonical ``"status": "FALSIFIED"``
    field with the solver's raw ``SolveStatus.SAT`` value.

    ``verify_once`` with ``TorchLPSolver`` on the falsifying fixture produces a
    spurious LP-relaxation CE (correctly downgraded to UNKNOWN). To exercise
    the true-save branch deterministically without Gurobi, we monkeypatch the
    inner ``validate_counterexample`` to return ``(True, model_output)``,
    which forces ``validate_and_save`` to save the CE through the same
    metadata-building path used in production.
    """
    from act.back_end.verifier import verify_once
    from act.back_end import counterexample_io as cio

    monkeypatch.setattr(
        "act.back_end.counterexample_io.get_pipeline_log_dir",
        lambda: str(tmp_path),
    )

    real_validate = cio.validate_counterexample

    def force_true_validate(net, ce_input, assert_layer):
        _is_true, model_output = real_validate(net, ce_input, assert_layer)
        if model_output is None:
            model_output = torch.zeros(1, dtype=ce_input.dtype, device=ce_input.device)
        return (True, model_output)

    monkeypatch.setattr(
        "act.back_end.counterexample_io.validate_counterexample",
        force_true_validate,
    )

    net = _make_relu_net(hidden_bias=0.0, out_weight=1.0, out_bias=0.0, threshold=0.1)

    res = verify_once(net, TorchLPSolver(), timelimit=10.0, network_path=Path("once_save.json"))

    assert res.status == VerifyStatus.FALSIFIED, f"expected FALSIFIED, got {res.status}"
    assert res.counterexample is not None
    saved_to = res.metadata.get("saved_to")
    assert saved_to is not None, "verify_once must populate metadata['saved_to'] on FALSIFIED"

    saved_dir = Path(saved_to)
    assert saved_dir.exists(), f"saved directory does not exist: {saved_dir}"
    assert (saved_dir / "cex.counterexample").exists()
    assert (saved_dir / "metadata.json").exists()

    meta = json.loads((saved_dir / "metadata.json").read_text())
    assert meta["status"] == "FALSIFIED", \
        f"canonical status was overwritten by caller extras: got {meta['status']!r}"
    assert meta["verifier_mode"] == "verify_once", \
        f"verifier_mode mismatch: got {meta['verifier_mode']!r}"
    assert meta["solver"] == "TorchLPSolver"


def test_verify_once_spurious_downgrades_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """When validate_counterexample returns spurious, verify_once must downgrade to UNKNOWN and not save."""
    from act.back_end.verifier import verify_once
    from act.back_end.solver.solver_base import SolveStatus

    # Build the same falsifying net
    net = _make_relu_net(hidden_bias=0.0, out_weight=1.0, out_bias=0.0, threshold=0.1)

    # Monkeypatch validate_and_save to return (False, None) — simulating spurious CE
    def fake_validate_and_save(net, ce_input, assert_layer, network_path, status_metadata):
        return (False, None)

    monkeypatch.setattr(
        "act.back_end.counterexample_io.validate_and_save",
        fake_validate_and_save,
    )

    # Note: verify_once uses Gurobi or TorchLP; on the falsifying net the LP solver returns SAT
    res = verify_once(net, TorchLPSolver(), timelimit=10.0)

    # Spurious downgrade
    assert res.status == VerifyStatus.UNKNOWN, f"expected UNKNOWN, got {res.status}"
    assert res.metadata.get("spurious_ce") is True, f"expected spurious_ce=True in metadata"
    assert res.counterexample is None or "saved_to" not in res.metadata or res.metadata.get("saved_to") is None, \
        "should not have saved a spurious CE"


def test_double_falsify_creates_two_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two consecutive verify_bab runs should create two distinct timestamped directories."""
    monkeypatch.setattr(
        "act.back_end.counterexample_io.get_pipeline_log_dir",
        lambda: str(tmp_path),
    )

    net = _make_relu_net(hidden_bias=0.0, out_weight=1.0, out_bias=0.0, threshold=0.1)

    res1 = verify_bab(
        net, TorchLPSolver(),
        config=_batched_cfg(subproblem_batch_size=1, max_nodes=2, max_depth=0),
        dual_solver=DualSolver(DualTF()),
        network_path=Path("dup_test.json"),
    )

    # Brief sleep is NOT required — microsecond timestamp guarantees uniqueness
    res2 = verify_bab(
        net, TorchLPSolver(),
        config=_batched_cfg(subproblem_batch_size=1, max_nodes=2, max_depth=0),
        dual_solver=DualSolver(DualTF()),
        network_path=Path("dup_test.json"),
    )

    assert res1.status == VerifyStatus.FALSIFIED
    assert res2.status == VerifyStatus.FALSIFIED
    assert res1.metadata["saved_to"] != res2.metadata["saved_to"], "two runs should create distinct dirs"
    assert Path(res1.metadata["saved_to"]).exists()
    assert Path(res2.metadata["saved_to"]).exists()
