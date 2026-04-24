# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportArgumentType=false
# ===- act/back_end/dual_tf/test_shape_regression.py - Goldens regression -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
# ===---------------------------------------------------------------------===#
# Bit-identity goldens for the Phase 2 baseline bounds.
#
# Safety net for the shape-preserving dual-TF refactor (Waves 1-5): captures
# forward + backward bounds across every example net at both float32 and
# float64, stores them under .sisyphus/goldens/phase2_baseline_bounds.pt,
# and verifies in subsequent runs that results bit-match within absolute
# tolerance 1e-5 (float32) / 1e-12 (float64).
#
# Why this exists:
#   Stages 2-5 rewrite hot-path backward kernels (dual_relu_backward,
#   backward_add, dual_bn_backward, etc.) to preserve native tensor shape
#   end-to-end. Any numerical drift from the refactor would be silent under
#   the existing test suite (which checks correctness, not bit-identity).
#   This file's tests fail loudly on drift and gate every refactor commit.
#
# Semantics:
#   - First run (no goldens file): capture mode. Writes the file. Tests
#     still run comparison against the freshly-captured values (trivially
#     pass) so the "capture" and "verify" paths share the same assertion
#     code.
#   - Subsequent runs: verify mode. Every (net, dtype) pair must match.
#   - To recapture after a deliberate algorithmic change:
#         rm .sisyphus/goldens/phase2_baseline_bounds.pt
#     and re-run. The goldens file is NOT committed (lives under .sisyphus/).
#
# Coverage:
#   Every *.json net under act/back_end/examples/nets/ at {float32, float64}.
#   Forward: full bounds_dict (lb, ub per layer).
#   Backward: DualSolver.compute_bound(c) with a deterministic seeded c.
# ===---------------------------------------------------------------------===#

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch

from act.back_end.dual_tf import DualTF, compute_forward_bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.serialization.serialization import load_net_from_file
from act.back_end.solver.solver_dual import DualSolver
from act.util.device_manager import initialize_device
from act.util.path_config import get_project_root


_PROJECT_ROOT = Path(get_project_root())
GOLDENS_PATH = _PROJECT_ROOT / ".sisyphus" / "goldens" / "phase2_baseline_bounds.pt"
NETS_DIR = _PROJECT_ROOT / "act" / "back_end" / "examples" / "nets"

# Absolute tolerance per dtype. float64 is effectively bit-identity
# (any drift this small is IEEE-rounding from sum/mul reordering, safe).
# float32 allows a broader epsilon because kernels may fuse/reorder adds
# in ways that change the last mantissa bits.
ATOL: Dict[torch.dtype, float] = {torch.float32: 1e-5, torch.float64: 1e-12}
DTYPE_NAMES: Dict[torch.dtype, str] = {torch.float32: "float32", torch.float64: "float64"}


def _find_example_nets() -> List[Path]:
    if not NETS_DIR.exists():
        return []
    return sorted(NETS_DIR.glob("*.json"))


_NET_PATHS = _find_example_nets()


def _load_input_bounds(net) -> Tuple[torch.Tensor, torch.Tensor] | None:
    for L in net.layers:
        k = L.kind.upper() if isinstance(L.kind, str) else L.kind
        if k in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value):
            if "lb" in L.params and "ub" in L.params:
                return L.params["lb"], L.params["ub"]
    return None


def _last_non_assert_layer(net):
    for L in reversed(net.layers):
        k = L.kind.upper() if isinstance(L.kind, str) else L.kind
        if k != LayerKind.ASSERT.value:
            return L
    return None


def _make_deterministic_c(
    B: int, out_numel: int, dtype: torch.dtype, salt: int
) -> torch.Tensor:
    """Return a deterministic, seeded c of shape [B, out_numel].

    Seed is derived from B + out_numel + salt so every (net, dtype) pair
    gets a stable-but-distinct objective. Using two functionals (nc=2) so
    the test exercises the batched backward path non-trivially.
    """
    gen = torch.Generator(device="cpu").manual_seed(20260423 + salt + B * 997 + out_numel * 31)
    c_batched = torch.randn((B, 2, out_numel), generator=gen, dtype=dtype)
    return c_batched.reshape(B * 2, out_numel)


def _compute_signature(net_path: Path, dtype: torch.dtype) -> Dict[str, Any]:
    """Produce a deterministic forward+backward signature for one (net, dtype).

    Returns a dict with:
      - forward: Dict[layer_id, Dict['lb'|'ub', CPU Tensor]]
      - backward: CPU Tensor[B*nc] from DualSolver.compute_bound
      - skipped (str): if the net lacks input bounds or backward target
    """
    initialize_device("cpu", DTYPE_NAMES[dtype])

    net = load_net_from_file(str(net_path))
    bounds_in = _load_input_bounds(net)
    if bounds_in is None:
        return {"skipped": "no embedded INPUT bounds"}
    lb_in, ub_in = bounds_in

    fwd = compute_forward_bounds(net, lb_in, ub_in, post_activation=True)

    target = _last_non_assert_layer(net)
    if target is None or target.id not in fwd:
        return {"skipped": "no backward target layer"}

    tb = fwd[target.id]
    if tb.lb.dim() == 0:
        return {"skipped": "scalar target"}

    B = tb.lb.shape[0]
    out_numel = 1
    for d in tb.lb.shape[1:]:
        out_numel *= int(d)
    if out_numel == 0:
        return {"skipped": "empty target"}

    salt = int.from_bytes(net_path.name.encode(), "big") % (2**31)
    c = _make_deterministic_c(B, out_numel, dtype, salt)

    solver = DualSolver(DualTF(), n_iters=0)
    try:
        bound = solver.compute_bound(net, fwd, c)
    except (ValueError, RuntimeError) as exc:
        return {"skipped": f"compute_bound raised: {type(exc).__name__}: {exc}"}

    forward_ser: Dict[int, Dict[str, torch.Tensor]] = {}
    for lid, B_obj in fwd.items():
        forward_ser[lid] = {
            "lb": B_obj.lb.detach().cpu().clone(),
            "ub": B_obj.ub.detach().cpu().clone(),
        }
    backward_ser = bound.detach().cpu().clone()

    return {"forward": forward_ser, "backward": backward_ser}


def _signature_key(net_path: Path, dtype: torch.dtype) -> str:
    return f"{net_path.name}::{DTYPE_NAMES[dtype]}"


_GOLDENS_CACHE: Dict[str, Any] | None = None


def _get_or_create_goldens() -> Dict[str, Any]:
    """Load goldens from disk, or capture+save on first run.

    The captured dict is keyed by "<net_filename>::<dtype_name>", matching
    _signature_key. Values are either signature dicts or {"skipped": ...}.
    """
    global _GOLDENS_CACHE
    if _GOLDENS_CACHE is not None:
        return _GOLDENS_CACHE

    if GOLDENS_PATH.exists():
        _GOLDENS_CACHE = torch.load(GOLDENS_PATH, weights_only=False)
        return _GOLDENS_CACHE

    captured: Dict[str, Any] = {}
    for p in _NET_PATHS:
        for dt in (torch.float32, torch.float64):
            captured[_signature_key(p, dt)] = _compute_signature(p, dt)
    GOLDENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(captured, GOLDENS_PATH)
    _GOLDENS_CACHE = captured
    return captured


def _assert_close(
    current: Dict[str, Any], golden: Dict[str, Any], key: str, atol: float
) -> None:
    if "skipped" in golden:
        pytest.skip(f"{key}: golden skipped ({golden['skipped']})")
    if "skipped" in current:
        pytest.fail(
            f"{key}: current run was skipped ({current['skipped']}) but the "
            f"golden was captured. Possibly a regression in net loading or "
            f"input bounds extraction. Recapture if intentional."
        )

    # Forward
    golden_fwd = golden["forward"]
    current_fwd = current["forward"]
    missing = set(golden_fwd.keys()) - set(current_fwd.keys())
    extra = set(current_fwd.keys()) - set(golden_fwd.keys())
    assert not missing, f"{key}: layers missing from current run: {sorted(missing)}"
    assert not extra, f"{key}: unexpected new layers in current run: {sorted(extra)}"

    for lid in sorted(golden_fwd.keys()):
        lb_c, ub_c = current_fwd[lid]["lb"], current_fwd[lid]["ub"]
        lb_g, ub_g = golden_fwd[lid]["lb"], golden_fwd[lid]["ub"]
        assert lb_c.shape == lb_g.shape, (
            f"{key} forward lb shape drift at layer {lid}: "
            f"current={tuple(lb_c.shape)} golden={tuple(lb_g.shape)}"
        )
        assert ub_c.shape == ub_g.shape, (
            f"{key} forward ub shape drift at layer {lid}: "
            f"current={tuple(ub_c.shape)} golden={tuple(ub_g.shape)}"
        )
        diff_lb = float(torch.abs(lb_c - lb_g).max().item()) if lb_c.numel() > 0 else 0.0
        diff_ub = float(torch.abs(ub_c - ub_g).max().item()) if ub_c.numel() > 0 else 0.0
        assert diff_lb <= atol, (
            f"{key} forward lb drift at layer {lid}: max_diff={diff_lb:.3e} atol={atol}"
        )
        assert diff_ub <= atol, (
            f"{key} forward ub drift at layer {lid}: max_diff={diff_ub:.3e} atol={atol}"
        )

    bk_c = current["backward"]
    bk_g = golden["backward"]
    assert bk_c.shape == bk_g.shape, (
        f"{key} backward shape drift: current={tuple(bk_c.shape)} golden={tuple(bk_g.shape)}"
    )
    diff_bk = float(torch.abs(bk_c - bk_g).max().item()) if bk_c.numel() > 0 else 0.0
    assert diff_bk <= atol, (
        f"{key} backward drift: max_diff={diff_bk:.3e} atol={atol}"
    )


@pytest.fixture(scope="module", autouse=True)
def _ensure_goldens() -> None:
    _get_or_create_goldens()


@pytest.mark.skipif(not _NET_PATHS, reason="no example nets found; run `python -m act.back_end --generate` first")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=lambda d: DTYPE_NAMES[d])
@pytest.mark.parametrize("net_path", _NET_PATHS, ids=lambda p: p.name)
def test_bounds_bit_identity(net_path: Path, dtype: torch.dtype) -> None:
    goldens = _get_or_create_goldens()
    key = _signature_key(net_path, dtype)
    if key not in goldens:
        pytest.skip(f"{key}: no golden captured (stale goldens file?)")
    current = _compute_signature(net_path, dtype)
    _assert_close(current, goldens[key], key, ATOL[dtype])


def test_goldens_file_present() -> None:
    _get_or_create_goldens()
    assert GOLDENS_PATH.exists(), f"goldens file missing at {GOLDENS_PATH}"


def test_goldens_cover_all_nets() -> None:
    if not _NET_PATHS:
        pytest.skip("no example nets found")
    goldens = _get_or_create_goldens()
    missing: List[str] = []
    for p in _NET_PATHS:
        for dt in (torch.float32, torch.float64):
            key = _signature_key(p, dt)
            if key not in goldens:
                missing.append(key)
    assert not missing, (
        f"goldens missing for: {missing}. "
        f"Delete {GOLDENS_PATH} and re-run to recapture."
    )
