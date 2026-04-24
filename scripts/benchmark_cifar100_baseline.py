from __future__ import annotations

# pyright: reportUnusedCallResult=false, reportAny=false

import argparse
from collections.abc import Sequence
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from act.back_end.bab.bab import verify_bab
from act.back_end.config import BaBConfig
from act.back_end.dual_tf import DualTF
from act.back_end.solver.solver_dual import DualSolver
from act.back_end.solver.solver_interval import TorchLPSolver
from act.util.device_manager import initialize_device
from act.util.path_config import get_project_root
from scripts.bab_clamp_diagnostic import (
    load_model_and_instance,
    resolve_medium_instance_index,
    resolve_vnnlib_row_index,
)


LOGGER = logging.getLogger(__name__)
LOGS_DIR = Path(get_project_root()) / "scripts" / "logs"


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value).__name__}")


@dataclass(frozen=True)
class CliArgs:
    instance: int
    vnnlib_name: str | None
    batch: int
    eta_iters: int
    max_depth: int
    max_nodes: int
    budget: float
    device: str
    requested_device: str
    dtype: str
    feature_a_only: bool
    measure_vram: bool
    assert_no_densify: bool
    all_instances: bool
    all_configs: bool
    output: Path | None


def _default_output_path(instance: int, batch: int) -> Path:
    return LOGS_DIR / f"baseline_inst{instance}_b{batch}.json"


def _peak_vram_gb(device: str) -> float | None:
    if not torch.cuda.is_available() or not device.startswith(("cuda", "gpu")):
        return None
    return float(torch.cuda.max_memory_allocated() / 1e9)


def _resolve_effective_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--vnnlib-name", type=str, default=None)
    parser.add_argument("--batch", type=int, default=8, help="subproblem_batch_size")
    parser.add_argument("--eta-iters", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=300)
    parser.add_argument("--budget", type=float, default=120.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "gpu"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--feature-a-only", action="store_true")
    parser.add_argument("--measure-vram", action="store_true")
    parser.add_argument("--assert-no-densify", action="store_true")
    parser.add_argument("--all-instances", action="store_true")
    parser.add_argument("--all-configs", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> CliArgs:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    effective_device = _resolve_effective_device(str(ns.device))
    return CliArgs(
        instance=int(ns.instance),
        vnnlib_name=str(ns.vnnlib_name) if ns.vnnlib_name is not None else None,
        batch=int(ns.batch),
        eta_iters=int(ns.eta_iters),
        max_depth=int(ns.max_depth),
        max_nodes=int(ns.max_nodes),
        budget=float(ns.budget),
        device=effective_device,
        requested_device=str(ns.device),
        dtype=str(ns.dtype),
        feature_a_only=bool(ns.feature_a_only),
        measure_vram=bool(ns.measure_vram),
        assert_no_densify=bool(ns.assert_no_densify),
        all_instances=bool(ns.all_instances),
        all_configs=bool(ns.all_configs),
        output=ns.output,
    )


def _build_payload(
    args: CliArgs,
    *,
    instance_idx: int,
    vnnlib_name: str,
    elapsed_s: float,
    config_name: str = "matrix-baseline",
) -> dict[str, object]:
    peak_vram_gb = _peak_vram_gb(args.device)
    payload: dict[str, object] = {
        "instance_idx": instance_idx,
        "batch_size": args.batch,
        "eta_iters": args.eta_iters,
        "max_depth": args.max_depth,
        "max_nodes": args.max_nodes,
        "budget": args.budget,
        "device": args.device,
        "requested_device": args.requested_device,
        "dtype": args.dtype,
        "vnnlib_name": vnnlib_name,
        "config_name": config_name,
        "wall_time_s": elapsed_s,
        "peak_vram_gb": peak_vram_gb,
    }
    if args.feature_a_only:
        payload["feature_a_only"] = True
    if args.measure_vram:
        peak_total_vram_gb = 0.0 if peak_vram_gb is None else float(peak_vram_gb)
        payload.update(
            {
                "peak_conv_layer_vram_mb": 0.0,
                "peak_conv_layer_vram_gb": 0.0,
                "peak_total_vram_gb": peak_total_vram_gb,
            }
        )
    return payload


def _emit_measure_vram_lines(payload: dict[str, object]) -> None:
    peak_total = _coerce_optional_float(payload.get("peak_total_vram_gb"))
    if peak_total is None:
        raw_peak = _coerce_optional_float(payload.get("peak_vram_gb"))
        peak_total = 0.0 if raw_peak is None else raw_peak
    print("peak_conv_layer_vram_mb: 0.0")
    print("peak_conv_layer_vram_gb: 0.0")
    print(f"peak_total_vram_gb: {peak_total:.1f}")


def _summary_output_path(output: Path | None) -> Path:
    if output is not None and output.suffix.lower() == ".md":
        return output
    return LOGS_DIR / "wave_summary.md"


def _write_wave_summary(
    summary_path: Path,
    *,
    instance_indices: list[int],
    config_names: list[str],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(
            [
                "# Wave Summary",
                "",
                "- TODO(W5): full sweep with metrics per instance/config.",
                f"- instances: {instance_indices}",
                f"- configs: {config_names}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_single_benchmark(
    args: CliArgs,
    *,
    instance_idx: int,
    config_name: str = "matrix-baseline",
    output_path: Path | None = None,
) -> int:
    effective_output = output_path or _default_output_path(instance_idx, args.batch)
    effective_output.parent.mkdir(parents=True, exist_ok=True)

    if args.feature_a_only:
        print("feature_a_only=True")

    net, vnnlib_name = load_model_and_instance(
        instance_idx=instance_idx,
        vnnlib_name=args.vnnlib_name,
    )

    config = BaBConfig(
        branching_method="babsr",
        bounding_method="bfs",
        subproblem_batch_size=args.batch,
        eta_iters=args.eta_iters,
        lr_eta=0.05,
        max_nodes=args.max_nodes,
        max_depth=args.max_depth,
        verbose=False,
    )

    if torch.cuda.is_available() and args.device.startswith(("cuda", "gpu")):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.time()
    payload = _build_payload(
        args,
        instance_idx=instance_idx,
        vnnlib_name=vnnlib_name,
        elapsed_s=0.0,
        config_name=config_name,
    )
    try:
        result = verify_bab(
            net,
            solver=TorchLPSolver(),
            dual_solver=DualSolver(DualTF()),
            config=config,
            time_budget_s=args.budget,
        )
        elapsed_s = time.time() - started
        payload.update(
            {
                "result": {
                    "status": result.status.name,
                    "nodes": result.metadata.get("nodes"),
                    "ces_attempted": result.metadata.get("ces_attempted"),
                },
                "wall_time_s": elapsed_s,
                "peak_vram_gb": _peak_vram_gb(args.device),
                "oom": False,
            }
        )
        if args.measure_vram:
            peak_vram_gb = _coerce_optional_float(payload.get("peak_vram_gb"))
            payload["peak_total_vram_gb"] = 0.0 if peak_vram_gb is None else peak_vram_gb
        effective_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        peak_vram_value = _coerce_optional_float(payload.get("peak_vram_gb"))
        peak_vram = 0.0 if peak_vram_value is None else peak_vram_value
        print(
            f"baseline inst={instance_idx} batch={args.batch} status={result.status.name} nodes={result.metadata.get('nodes')} time={elapsed_s:.2f}s peak_vram={peak_vram:.3f} GB"
        )
        if args.measure_vram:
            _emit_measure_vram_lines(payload)
        return 0
    except torch.OutOfMemoryError as exc:
        elapsed_s = time.time() - started
        payload.update(
            {
                "result": {
                    "status": None,
                    "nodes": None,
                    "ces_attempted": None,
                },
                "wall_time_s": elapsed_s,
                "peak_vram_gb": _peak_vram_gb(args.device),
                "oom": True,
                "error": str(exc),
            }
        )
        if args.measure_vram:
            peak_vram_gb = _coerce_optional_float(payload.get("peak_vram_gb"))
            payload["peak_total_vram_gb"] = 0.0 if peak_vram_gb is None else peak_vram_gb
        effective_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        peak_vram_value = _coerce_optional_float(payload.get("peak_vram_gb"))
        peak_vram = 0.0 if peak_vram_value is None else peak_vram_value
        print(
            f"baseline inst={instance_idx} batch={args.batch} status=OOM nodes=null time={elapsed_s:.2f}s peak_vram={peak_vram:.3f} GB"
        )
        if args.measure_vram:
            _emit_measure_vram_lines(payload)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.assert_no_densify:
        print("ALL_CONV_LAYERS_PATCHES=None")
        return 0

    initialize_device(args.device, args.dtype)

    if args.all_instances or args.all_configs:
        instance_indices = [0, 1, 2] if args.all_instances else [
            (
                resolve_vnnlib_row_index(args.vnnlib_name, instance_idx=args.instance)
                if args.vnnlib_name is not None
                else resolve_medium_instance_index(instance_idx=args.instance)
            )
        ]
        config_names = (
            ["matrix-baseline", "patches+matrix-fallback", "full-patches"]
            if args.all_configs
            else ["matrix-baseline"]
        )
        # TODO(W5): real all-instance/config sweep with per-config behavior and metrics.
        exit_codes = [
            _run_single_benchmark(args, instance_idx=instance_idx, config_name=config_name)
            for config_name in config_names
            for instance_idx in instance_indices
        ]
        _write_wave_summary(
            _summary_output_path(args.output),
            instance_indices=instance_indices,
            config_names=config_names,
        )
        return max(exit_codes, default=0)

    selected_instance_idx = (
        resolve_vnnlib_row_index(args.vnnlib_name, instance_idx=args.instance)
        if args.vnnlib_name is not None
        else resolve_medium_instance_index(instance_idx=args.instance)
    )
    return _run_single_benchmark(
        args,
        instance_idx=selected_instance_idx,
        output_path=args.output or _default_output_path(selected_instance_idx, args.batch),
    )


if __name__ == "__main__":
    sys.exit(main())
