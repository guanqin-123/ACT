"""Verify CIFAR-100 ResNet-medium VNNLIB instances via BaB.

Regression harness for the patches roll-out. Run BEFORE a fix (baseline) and
AFTER a fix (validation); compare stdout + JSON to see whether the fix moved
the needle.

Instances (alpha-beta-CROWN ground truth):
    * ``3995`` (default) — CERTIFIED 43.5s, 516 nodes. Today's patches
      pipeline returns UNKNOWN at 126s; this is the perf gap.
    * ``8028`` — FALSIFIED ~0.3s. Exercises the CE validation path.

CLI::

    python -m scripts.verify_resnet_cifar100 --instance both --budget 60.0
    python -m scripts.verify_resnet_cifar100 --instance 3995 --conv-mode matrix
    python -m scripts.verify_resnet_cifar100 --instance 3995 --conv-mode patches_strict

Exits 0 iff every run's verdict is in its ``acceptable_statuses`` set.
"""
# pyright: reportExplicitAny=false, reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, override

import torch

from act.back_end.bab.bab import verify_bab
from act.back_end.bounds_dispatch import (
    get_conv_materialization_count,
    get_conv_mode,
    get_strict_patches,
    reset_conv_materialization_count,
    set_conv_mode,
    set_strict_patches,
)
from act.back_end.config import BaBConfig
from act.back_end.dual_tf import DualTF
from act.back_end.solver.solver_dual import DualSolver
from act.back_end.solver.solver_interval import TorchLPSolver
from act.util.device_manager import initialize_device
from act.util.path_config import get_project_root
from scripts.bab_clamp_diagnostic import load_model_and_instance


LOGGER = logging.getLogger("verify_resnet_cifar100")
LOGS_DIR = Path(get_project_root()) / "scripts" / "logs"

# `acceptable_statuses` encodes the soundness contract: CERTIFIED-known
# instances may return CERTIFIED/UNKNOWN/TIMEOUT but NEVER FALSIFIED
# (would indicate a sound-bound bug); FALSIFIED-known instances must
# return FALSIFIED (CE validated against the concrete model).
INSTANCE_REGISTRY: dict[str, dict[str, Any]] = {
    "3995": {
        "vnnlib_name": "CIFAR100_resnet_medium_prop_idx_3995_sidx_7978_eps_0.0039.vnnlib",
        "abcrown_status": "CERTIFIED",
        "abcrown_wall_s": 43.5,
        "abcrown_nodes": 516,
        "acceptable_statuses": ("CERTIFIED", "UNKNOWN", "TIMEOUT"),
    },
    "8028": {
        "vnnlib_name": "CIFAR100_resnet_medium_prop_idx_8028_sidx_5238_eps_0.0039.vnnlib",
        "abcrown_status": "FALSIFIED",
        "abcrown_wall_s": 0.3,
        "abcrown_nodes": 1,
        "acceptable_statuses": ("FALSIFIED",),
    },
}


@dataclass
class CliArgs:
    instance: str
    conv_mode: str
    alpha_split: bool
    device: str
    dtype: str
    batch: int
    eta_iters: int
    max_depth: int
    max_nodes: int | None
    budget: float
    verbose_bab: bool
    output: Path | None
    bound_trace: bool


@dataclass
class VerifyResult:
    instance: str
    vnnlib_name: str
    conv_mode: str
    strict_patches: bool
    status: str
    wall_time_s: float
    nodes: int | None
    peak_vram_gb: float | None
    conv_materializations: int
    warning_count: int
    expected_statuses: tuple[str, ...]
    abcrown_status: str
    abcrown_wall_s: float
    abcrown_nodes: int
    alpha_split: bool = False
    error: str | None = None
    warning_samples: list[str] = field(default_factory=list)
    bound_trace: dict[str, Any] | None = None
    intermediate_widths: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        if self.error is not None:
            return False
        return self.status in self.expected_statuses


class _WarningCounter(logging.Handler):
    def __init__(self, sample_cap: int = 5) -> None:
        super().__init__(level=logging.WARNING)
        self.count: int = 0
        self.samples: list[str] = []
        self._sample_cap: int = sample_cap

    @override
    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        if record.levelno < logging.WARNING:
            return
        self.count += 1
        if len(self.samples) < self._sample_cap:
            self.samples.append(record.getMessage())


@contextmanager
def _capture_warnings(loggers: list[str]) -> Iterator[_WarningCounter]:
    handler = _WarningCounter()
    attached: list[logging.Logger] = []
    for name in loggers:
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        attached.append(lg)
    try:
        yield handler
    finally:
        for lg in attached:
            lg.removeHandler(handler)


@contextmanager
def _conv_mode_context(conv_mode: str, strict_patches: bool) -> Iterator[None]:
    prev_mode = get_conv_mode()
    prev_strict = get_strict_patches()
    set_conv_mode(conv_mode)
    set_strict_patches(strict_patches)
    try:
        yield
    finally:
        set_conv_mode(prev_mode)
        set_strict_patches(prev_strict)


def _resolve_conv_mode_args(raw: str) -> tuple[str, bool]:
    mapping = {
        "matrix": ("matrix", False),
        "patches": ("patches", False),
        "patches_mixed": ("patches", False),
        "patches_strict": ("patches", True),
    }
    if raw not in mapping:
        raise ValueError(f"unknown --conv-mode={raw!r}; expected one of {list(mapping)}")
    return mapping[raw]


def _peak_vram_gb(device: str) -> float | None:
    if not torch.cuda.is_available() or not device.startswith(("cuda", "gpu")):
        return None
    return float(torch.cuda.max_memory_allocated() / 1e9)


def _reset_cuda_state(device: str) -> None:
    if torch.cuda.is_available() and device.startswith(("cuda", "gpu")):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _build_config(
    *,
    subproblem_batch_size: int,
    eta_iters: int,
    max_depth: int,
    max_nodes: int | None,
    verbose: bool,
    record_bound_trace: bool,
    alpha_split: bool,
) -> BaBConfig:
    defaults = BaBConfig()
    return BaBConfig(
        branching_method="babsr",
        bounding_method="bfs",
        subproblem_batch_size=subproblem_batch_size,
        eta_iters=eta_iters,
        alpha_split_objective=alpha_split,
        alpha_iters=defaults.alpha_iters,
        lr_alpha=defaults.lr_alpha,
        lr_eta=0.05,
        max_nodes=max_nodes,
        max_depth=max_depth,
        record_bound_trace=record_bound_trace,
        verbose=verbose,
    )


def _marshal_bound_trace(trace: Any) -> dict[str, Any]:
    if hasattr(trace, "to_dict"):
        return trace.to_dict()
    return {
        "min_slack_history": {str(k): v for k, v in getattr(trace, "min_slack_history", {}).items()},
        "iteration_history": {str(k): v for k, v in getattr(trace, "iteration_history", {}).items()},
        "parent": {str(k): v for k, v in getattr(trace, "parent", {}).items()},
        "depth": {str(k): v for k, v in getattr(trace, "depth", {}).items()},
        "adam_trajectory": {
            f"{sid},{bab_iter}": v
            for (sid, bab_iter), v in getattr(trace, "adam_trajectory", {}).items()
        },
    }


def verify_instance(
    instance_key: str,
    *,
    conv_mode_arg: str,
    device: str,
    subproblem_batch_size: int,
    eta_iters: int,
    max_depth: int,
    max_nodes: int | None,
    time_budget_s: float,
    verbose: bool,
    bound_trace: bool = False,
    alpha_split: bool = False,
) -> VerifyResult:
    if instance_key not in INSTANCE_REGISTRY:
        raise ValueError(f"unknown instance {instance_key!r}; expected one of {list(INSTANCE_REGISTRY)}")
    reg = INSTANCE_REGISTRY[instance_key]
    conv_mode, strict_patches = _resolve_conv_mode_args(conv_mode_arg)

    net, vnnlib_name = load_model_and_instance(vnnlib_name=reg["vnnlib_name"])

    _reset_cuda_state(device)
    reset_conv_materialization_count()

    config = _build_config(
        subproblem_batch_size=subproblem_batch_size,
        eta_iters=eta_iters,
        max_depth=max_depth,
        max_nodes=max_nodes,
        verbose=verbose,
        record_bound_trace=bound_trace,
        alpha_split=alpha_split,
    )
    trace = None
    out_bounds_dict: dict[int, dict[str, torch.Tensor]] = {}
    if bound_trace:
        from act.back_end.bab.trace import BoundTrace

        trace = BoundTrace()

    started = time.time()
    error: str | None = None
    status = "ERROR"
    nodes: int | None = None
    with (
        _conv_mode_context(conv_mode, strict_patches),
        _capture_warnings(
            [
                "act.back_end.bounds_dispatch",
                "act.back_end.bab.branching.babsr",
                "act.back_end.dual_tf.tf_forward",
                "act.back_end.dual_tf.tf_mlp",
                "act.back_end.dual_tf.tf_cnn",
            ]
        ) as warnings_handler,
    ):
        try:
            result = verify_bab(
                net,
                solver=TorchLPSolver(),
                dual_solver=DualSolver(DualTF()),
                config=config,
                time_budget_s=time_budget_s,
                trace=trace,
                out_bounds_dict=out_bounds_dict,
            )
            status = result.status.name
            nodes_meta = result.metadata.get("nodes")
            nodes = int(nodes_meta) if isinstance(nodes_meta, (int, float)) else None
        except Exception as exc:  # noqa: BLE001 — script harness; record exc verbatim
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("verify_bab raised on instance %s", instance_key)

    wall_time = time.time() - started
    return VerifyResult(
        instance=instance_key,
        vnnlib_name=vnnlib_name,
        conv_mode=conv_mode,
        strict_patches=strict_patches,
        alpha_split=alpha_split,
        status=status,
        wall_time_s=wall_time,
        nodes=nodes,
        peak_vram_gb=_peak_vram_gb(device),
        conv_materializations=get_conv_materialization_count(),
        warning_count=warnings_handler.count,
        warning_samples=list(warnings_handler.samples),
        expected_statuses=tuple(reg["acceptable_statuses"]),
        abcrown_status=str(reg["abcrown_status"]),
        abcrown_wall_s=float(reg["abcrown_wall_s"]),
        abcrown_nodes=int(reg["abcrown_nodes"]),
        error=error,
        bound_trace=_marshal_bound_trace(trace) if trace is not None else None,
        intermediate_widths={
            str(lid): {
                "lb": [float(x) for x in v["lb"].flatten()[:1000]],
                "ub": [float(x) for x in v["ub"].flatten()[:1000]],
                "n_total": int(v["lb"].numel()),
            }
            for lid, v in out_bounds_dict.items()
        }
        if bound_trace
        else None,
    )


def _format_row(r: VerifyResult) -> str:
    vram = f"{r.peak_vram_gb:.1f} GB" if r.peak_vram_gb is not None else "-"
    nodes = str(r.nodes) if r.nodes is not None else "-"
    mark = "OK" if r.passed else "FAIL"
    conv_tag = f"{r.conv_mode}{'+strict' if r.strict_patches else ''}{'+α-split' if r.alpha_split else ''}"
    return (
        f"  {mark:<4} "
        f"inst={r.instance:<6} "
        f"mode={conv_tag:<18} "
        f"status={r.status:<10} "
        f"wall={r.wall_time_s:6.2f}s "
        f"nodes={nodes:<5} "
        f"vram={vram:<9} "
        f"conv_mat={r.conv_materializations:<4} "
        f"warns={r.warning_count:<4} "
        f"(abcrown={r.abcrown_status} @ {r.abcrown_wall_s:.1f}s)"
    )


def _emit_summary(results: list[VerifyResult]) -> None:
    print("\n=== verify_resnet_cifar100 summary ===")
    for r in results:
        print(_format_row(r))
        if r.error is not None:
            print(f"       ERROR: {r.error}")
        if r.warning_samples and r.warning_count > 0:
            print(f"       sample warnings ({len(r.warning_samples)}/{r.warning_count}):")
            for line in r.warning_samples:
                print(f"         - {line[:140]}")
    n_pass = sum(1 for r in results if r.passed)
    print(f"\n{n_pass}/{len(results)} runs passed")


def _emit_json(results: list[VerifyResult], output_path: Path) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": [asdict(r) for r in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"JSON report: {output_path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify CIFAR-100 ResNet-medium instances via BaB (regression harness)."
    )
    parser.add_argument(
        "--instance",
        type=str,
        default="3995",
        choices=[*INSTANCE_REGISTRY.keys(), "both", "all"],
        help="which VNNLIB instance to verify (3995=CERTIFIED target, 8028=FALSIFIED).",
    )
    parser.add_argument(
        "--conv-mode",
        type=str,
        default="patches",
        choices=["matrix", "patches", "patches_mixed", "patches_strict"],
        help="how to handle Conv2d A/b representation in dual_tf.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "gpu"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--batch", type=int, default=16, help="subproblem_batch_size")
    parser.add_argument("--eta-iters", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="node cap; None = unlimited (stop on time_budget_s).",
    )
    parser.add_argument("--budget", type=float, default=60.0, help="wall-clock budget per run (s).")
    parser.add_argument(
        "--verbose-bab",
        action="store_true",
        help="emit per-iter BaB progress lines; off by default to keep the table readable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="path to JSON report; defaults to scripts/logs/verify_resnet_cifar100_<stamp>.json.",
    )
    parser.add_argument(
        "--bound-trace",
        action="store_true",
        help="capture per-iter Adam best-objective trajectory + final per-layer pre-activation widths",
    )
    parser.add_argument(
        "--alpha-split",
        action="store_true",
        help="enable Tier 3 pre-BaB α-CROWN intermediate-bound tightening",
    )
    return parser.parse_args(argv)


def _to_cli_args(args: argparse.Namespace) -> CliArgs:
    return CliArgs(
        instance=args.instance,
        conv_mode=args.conv_mode,
        alpha_split=args.alpha_split,
        device=args.device,
        dtype=args.dtype,
        batch=args.batch,
        eta_iters=args.eta_iters,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        budget=args.budget,
        verbose_bab=args.verbose_bab,
        output=args.output,
        bound_trace=args.bound_trace,
    )


def _instances_to_run(selector: str) -> list[str]:
    if selector in ("both", "all"):
        return list(INSTANCE_REGISTRY.keys())
    return [selector]


def main(argv: list[str] | None = None) -> int:
    args = _to_cli_args(_parse_args(argv))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    initialize_device(args.device, args.dtype)
    print(f"[verify] device={args.device} dtype={args.dtype} batch={args.batch} budget={args.budget}s conv_mode={args.conv_mode} alpha_split={args.alpha_split}")

    results: list[VerifyResult] = []
    for key in _instances_to_run(args.instance):
        print(
            f"\n--- instance {key} (conv_mode={args.conv_mode}, alpha_split={args.alpha_split}) ---"
        )
        r = verify_instance(
            key,
            conv_mode_arg=args.conv_mode,
            device=args.device,
            subproblem_batch_size=args.batch,
            eta_iters=args.eta_iters,
            max_depth=args.max_depth,
            max_nodes=args.max_nodes,
            time_budget_s=args.budget,
            verbose=args.verbose_bab,
            bound_trace=args.bound_trace,
            alpha_split=args.alpha_split,
        )
        results.append(r)
        print(_format_row(r))

    _emit_summary(results)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or (LOGS_DIR / f"verify_resnet_cifar100_{stamp}.json")
    _emit_json(results, output_path)

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
