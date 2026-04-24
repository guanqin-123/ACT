from __future__ import annotations

# pyright: reportExplicitAny=false

# pyright: reportUnusedCallResult=false, reportAny=false

import argparse
import copy
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, cast

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
from act.back_end.core import Net
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
KNOWN_3995_VNNLIB = "CIFAR100_resnet_medium_prop_idx_3995_sidx_7978_eps_0.0039.vnnlib"
ABCROWN_BASELINES: dict[str, dict[str, object]] = {
    KNOWN_3995_VNNLIB: {
        "display_name": "α-β-CROWN baseline",
        "status": "CERTIFIED",
        "wall_time_s": 43.533897399902344,
        "nodes": 516,
    }
}


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value).__name__}")


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raise TypeError(f"Expected int-compatible value, got {type(value).__name__}")


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


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    display_name: str
    conv_mode: str
    strict_patches: bool


@dataclass(frozen=True)
class InstanceTarget:
    key: str
    display_name: str
    instance_idx: int
    vnnlib_name: str | None


@dataclass
class LoadedInstance:
    target: InstanceTarget
    resolved_instance_idx: int
    net: Net
    vnnlib_name: str


@dataclass
class BenchmarkResult:
    instance_key: str
    instance_display_name: str
    resolved_instance_idx: int
    vnnlib_name: str
    config_name: str
    config_display_name: str
    output_path: Path
    payload: dict[str, object]
    exit_code: int


@dataclass
class ConvVramTracker:
    peak_conv_layer_vram_mb: float = 0.0
    peak_total_vram_bytes: float = 0.0

    def observe(self, bytes_value: float) -> None:
        self.peak_total_vram_bytes = max(self.peak_total_vram_bytes, float(bytes_value))
        self.peak_conv_layer_vram_mb = max(self.peak_conv_layer_vram_mb, float(bytes_value) / 1e6)


def _default_output_path(instance_key: str, config_name: str, batch: int) -> Path:
    return LOGS_DIR / f"wave5_{instance_key}_{config_name}_b{batch}.json"


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
    loaded: LoadedInstance,
    config: BenchmarkConfig,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "instance_key": loaded.target.key,
        "instance_display_name": loaded.target.display_name,
        "instance_idx": loaded.resolved_instance_idx,
        "batch_size": args.batch,
        "eta_iters": args.eta_iters,
        "max_depth": args.max_depth,
        "max_nodes": args.max_nodes,
        "budget": args.budget,
        "device": args.device,
        "requested_device": args.requested_device,
        "dtype": args.dtype,
        "vnnlib_name": loaded.vnnlib_name,
        "config_name": config.name,
        "config_display_name": config.display_name,
        "conv_mode": config.conv_mode,
        "strict_patches": config.strict_patches,
        "wall_time_s": 0.0,
        "peak_vram_gb": None,
    }
    if args.feature_a_only:
        payload["feature_a_only"] = True
    return payload


def _emit_measure_vram_lines(payload: dict[str, object]) -> None:
    peak_conv_layer_vram_mb = _coerce_optional_float(payload.get("peak_conv_layer_vram_mb")) or 0.0
    peak_conv_layer_vram_gb = _coerce_optional_float(payload.get("peak_conv_layer_vram_gb")) or 0.0
    peak_total = _coerce_optional_float(payload.get("peak_total_vram_gb"))
    if peak_total is None:
        raw_peak = _coerce_optional_float(payload.get("peak_vram_gb"))
        peak_total = 0.0 if raw_peak is None else raw_peak
    print(f"peak_conv_layer_vram_mb: {peak_conv_layer_vram_mb:.1f}")
    print(f"peak_conv_layer_vram_gb: {peak_conv_layer_vram_gb:.1f}")
    print(f"peak_total_vram_gb: {peak_total:.1f}")


def _summary_output_path(output: Path | None) -> Path:
    return output if output is not None else LOGS_DIR / "wave5_summary.md"


def _build_instance_targets(args: CliArgs) -> list[InstanceTarget]:
    if args.all_instances:
        return [
            InstanceTarget(key="instance0", display_name="Instance 0", instance_idx=0, vnnlib_name=None),
            InstanceTarget(key="instance1", display_name="Instance 1", instance_idx=1, vnnlib_name=None),
            InstanceTarget(key="instance2", display_name="Instance 2", instance_idx=2, vnnlib_name=None),
            InstanceTarget(key="instance3995", display_name="Instance 3995", instance_idx=0, vnnlib_name=KNOWN_3995_VNNLIB),
        ]
    if args.vnnlib_name is not None:
        stem = Path(args.vnnlib_name).stem
        return [
            InstanceTarget(
                key=stem.replace("-", "_").replace(".", "_"),
                display_name=f"Instance {stem}",
                instance_idx=0,
                vnnlib_name=args.vnnlib_name,
            )
        ]
    return [
        InstanceTarget(
            key=f"instance{args.instance}",
            display_name=f"Instance {args.instance}",
            instance_idx=args.instance,
            vnnlib_name=None,
        )
    ]


def _build_benchmark_configs(args: CliArgs) -> list[BenchmarkConfig]:
    if args.all_configs:
        return [
            BenchmarkConfig(
                name="baseline_matrix",
                display_name="ACT baseline (matrix)",
                conv_mode="matrix",
                strict_patches=False,
            ),
            BenchmarkConfig(
                name="patches_with_matrix_fallback",
                display_name="ACT patches (mixed)",
                conv_mode="patches",
                strict_patches=False,
            ),
            BenchmarkConfig(
                name="patches_strict",
                display_name="ACT patches (strict)",
                conv_mode="patches",
                strict_patches=True,
            ),
        ]
    return [
        BenchmarkConfig(
            name="baseline_matrix",
            display_name="ACT baseline (matrix)",
            conv_mode="matrix",
            strict_patches=False,
        )
    ]


def _load_instance(target: InstanceTarget) -> LoadedInstance:
    net, vnnlib_name = load_model_and_instance(
        instance_idx=target.instance_idx,
        vnnlib_name=target.vnnlib_name,
    )
    resolved_instance_idx = (
        resolve_vnnlib_row_index(vnnlib_name, instance_idx=target.instance_idx)
        if target.vnnlib_name is not None
        else resolve_medium_instance_index(instance_idx=target.instance_idx)
    )
    return LoadedInstance(
        target=target,
        resolved_instance_idx=resolved_instance_idx,
        net=net,
        vnnlib_name=vnnlib_name,
    )


@contextmanager
def _configured_conv_mode(config: BenchmarkConfig):
    previous_mode = get_conv_mode()
    previous_strict = get_strict_patches()
    set_conv_mode(config.conv_mode)
    set_strict_patches(config.strict_patches)
    try:
        yield
    finally:
        set_conv_mode(previous_mode)
        set_strict_patches(previous_strict)


@contextmanager
def _measure_conv_layer_vram(enabled: bool, tracker: ConvVramTracker):
    if not enabled or not torch.cuda.is_available():
        yield
        return

    import act.back_end.bounds_dispatch as bounds_dispatch

    original_dispatch = bounds_dispatch.dispatch_conv_forward

    def _wrapped_dispatch(*args: Any, **kwargs: Any):
        tracker.observe(torch.cuda.memory_allocated())
        result = original_dispatch(*args, **kwargs)
        torch.cuda.synchronize()
        tracker.observe(torch.cuda.memory_allocated())
        return result

    bounds_dispatch.dispatch_conv_forward = _wrapped_dispatch
    try:
        yield
    finally:
        bounds_dispatch.dispatch_conv_forward = original_dispatch


def _build_bab_config(args: CliArgs) -> BaBConfig:
    return BaBConfig(
        branching_method="babsr",
        bounding_method="bfs",
        subproblem_batch_size=args.batch,
        eta_iters=args.eta_iters,
        lr_eta=0.05,
        max_nodes=args.max_nodes,
        max_depth=args.max_depth,
        verbose=False,
    )


def _update_vram_payload(
    payload: dict[str, object],
    *,
    args: CliArgs,
    tracker: ConvVramTracker,
) -> None:
    peak_vram_gb = _peak_vram_gb(args.device)
    payload["peak_vram_gb"] = peak_vram_gb
    if not args.measure_vram:
        return
    peak_total_vram_gb = (
        peak_vram_gb
        if peak_vram_gb is not None
        else (tracker.peak_total_vram_bytes / 1e9 if tracker.peak_total_vram_bytes > 0 else 0.0)
    )
    peak_conv_layer_vram_mb = tracker.peak_conv_layer_vram_mb
    payload.update(
        {
            "peak_conv_layer_vram_mb": peak_conv_layer_vram_mb,
            "peak_conv_layer_vram_gb": peak_conv_layer_vram_mb / 1000.0,
            "peak_total_vram_gb": peak_total_vram_gb,
        }
    )


def _set_densify_metadata(
    payload: dict[str, object],
    *,
    args: CliArgs,
    config: BenchmarkConfig,
) -> None:
    if config.conv_mode != "patches":
        if args.assert_no_densify:
            print("ALL_CONV_LAYERS_PATCHES=None")
        return
    materializations = get_conv_materialization_count()
    payload["conv_materializations"] = materializations
    payload["all_conv_layers_patches"] = materializations == 0
    if args.assert_no_densify:
        if materializations == 0:
            print("ALL_CONV_LAYERS_PATCHES=True")
        else:
            print(f"ALL_CONV_LAYERS_PATCHES=False ({materializations} materializations)")


def _persist_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_single_benchmark(
    args: CliArgs,
    *,
    loaded: LoadedInstance,
    config: BenchmarkConfig,
    output_path: Path | None = None,
) -> BenchmarkResult:
    effective_output = output_path or _default_output_path(loaded.target.key, config.name, args.batch)
    payload = _build_payload(args, loaded=loaded, config=config)
    tracker = ConvVramTracker()
    if args.feature_a_only:
        print("feature_a_only=True")

    if config.conv_mode == "patches":
        reset_conv_materialization_count()

    if torch.cuda.is_available() and args.device.startswith(("cuda", "gpu")):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tracker.observe(torch.cuda.memory_allocated())

    started = time.time()
    try:
        with _configured_conv_mode(config), _measure_conv_layer_vram(args.measure_vram, tracker):
            result = verify_bab(
                copy.deepcopy(loaded.net),
                solver=TorchLPSolver(),
                dual_solver=DualSolver(DualTF()),
                config=_build_bab_config(args),
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
                "oom": False,
            }
        )
        _update_vram_payload(payload, args=args, tracker=tracker)
        _set_densify_metadata(payload, args=args, config=config)
        _persist_payload(effective_output, payload)
        peak_vram = _coerce_optional_float(payload.get("peak_total_vram_gb"))
        if peak_vram is None:
            peak_vram = _coerce_optional_float(payload.get("peak_vram_gb")) or 0.0
        print(f"baseline inst={loaded.resolved_instance_idx} config={config.name} batch={args.batch} status={result.status.name} nodes={result.metadata.get('nodes')} time={elapsed_s:.2f}s peak_vram={peak_vram:.3f} GB")
        if args.measure_vram:
            _emit_measure_vram_lines(payload)
        return BenchmarkResult(
            instance_key=loaded.target.key,
            instance_display_name=loaded.target.display_name,
            resolved_instance_idx=loaded.resolved_instance_idx,
            vnnlib_name=loaded.vnnlib_name,
            config_name=config.name,
            config_display_name=config.display_name,
            output_path=effective_output,
            payload=payload,
            exit_code=0,
        )
    except torch.OutOfMemoryError as exc:
        status = "OOM"
        exit_code = 2
        error = str(exc)
    except Exception as exc:  # noqa: BLE001
        status = "ERROR"
        exit_code = 1
        error = str(exc)

    elapsed_s = time.time() - started
    payload.update(
        {
            "result": {
                "status": status,
                "nodes": None,
                "ces_attempted": None,
            },
            "wall_time_s": elapsed_s,
            "oom": status == "OOM",
            "error": error,
        }
    )
    _update_vram_payload(payload, args=args, tracker=tracker)
    _set_densify_metadata(payload, args=args, config=config)
    _persist_payload(effective_output, payload)
    peak_vram = _coerce_optional_float(payload.get("peak_total_vram_gb"))
    if peak_vram is None:
        peak_vram = _coerce_optional_float(payload.get("peak_vram_gb")) or 0.0
    print(f"baseline inst={loaded.resolved_instance_idx} config={config.name} batch={args.batch} status={status} nodes=null time={elapsed_s:.2f}s peak_vram={peak_vram:.3f} GB")
    if args.measure_vram:
        _emit_measure_vram_lines(payload)
    return BenchmarkResult(
        instance_key=loaded.target.key,
        instance_display_name=loaded.target.display_name,
        resolved_instance_idx=loaded.resolved_instance_idx,
        vnnlib_name=loaded.vnnlib_name,
        config_name=config.name,
        config_display_name=config.display_name,
        output_path=effective_output,
        payload=payload,
        exit_code=exit_code,
    )


def _format_secs(value: object) -> str:
    number = _coerce_optional_float(value)
    return "—" if number is None else f"{number:.1f}s"


def _format_nodes(value: object) -> str:
    number = _coerce_optional_int(value)
    return "—" if number is None else str(number)


def _format_gb(value: object) -> str:
    number = _coerce_optional_float(value)
    return "—" if number is None else f"{number:.2f} GB"


def _format_vs_abcrown(vnnlib_name: str, wall_time_s: object) -> str:
    baseline = ABCROWN_BASELINES.get(vnnlib_name)
    if baseline is None:
        return "—"
    baseline_wall = _coerce_optional_float(baseline.get("wall_time_s"))
    wall = _coerce_optional_float(wall_time_s)
    if baseline_wall in (None, 0.0) or wall is None:
        return "—"
    return f"{wall / baseline_wall:.2f}×"


def _write_wave_summary(summary_path: Path, results: list[BenchmarkResult]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        grouped[result.instance_key].append(result)

    lines = ["# Wave 5 A/B Benchmark Summary", ""]
    for _instance_key, instance_results in grouped.items():
        instance_results.sort(key=lambda item: item.config_name)
        first = instance_results[0]
        lines.append(f"## {first.instance_display_name} ({first.vnnlib_name})")
        lines.append("| Config | Verdict | Wall-clock | Nodes | Peak VRAM | vs α-β-CROWN |")
        lines.append("|---|---|---|---|---|---|")
        abcrown = ABCROWN_BASELINES.get(first.vnnlib_name)
        if abcrown is not None:
            lines.append(
                "| {display_name} | {status} | {wall} | {nodes} | — | 1.00× |".format(
                    display_name=abcrown["display_name"],
                    status=abcrown["status"],
                    wall=_format_secs(abcrown.get("wall_time_s")),
                    nodes=_format_nodes(abcrown.get("nodes")),
                )
            )
        for result in instance_results:
            payload = result.payload
            outcome = cast(dict[str, object] | None, payload.get("result") if isinstance(payload.get("result"), dict) else None)
            verdict = str(outcome.get("status")) if outcome is not None else "—"
            peak = payload.get("peak_total_vram_gb", payload.get("peak_vram_gb"))
            lines.append(
                "| {config} | {verdict} | {wall} | {nodes} | {peak} | {vs} |".format(
                    config=result.config_display_name,
                    verdict=verdict,
                    wall=_format_secs(payload.get("wall_time_s")),
                    nodes=_format_nodes(outcome.get("nodes") if outcome is not None else None),
                    peak=_format_gb(peak),
                    vs=_format_vs_abcrown(result.vnnlib_name, payload.get("wall_time_s")),
                )
            )
        notes: list[str] = []
        for result in instance_results:
            payload = result.payload
            error = payload.get("error")
            materializations = _coerce_optional_int(payload.get("conv_materializations"))
            if error:
                notes.append(f"- {result.config_display_name}: {error}")
            if materializations is not None:
                notes.append(f"- {result.config_display_name}: conv materializations={materializations}")
        if notes:
            lines.append("")
            lines.append("Notes:")
            lines.extend(notes)
        lines.append("")
    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    initialize_device(args.device, args.dtype)

    instance_targets = _build_instance_targets(args)
    benchmark_configs = _build_benchmark_configs(args)

    loaded_instances = [_load_instance(target) for target in instance_targets]
    results: list[BenchmarkResult] = []
    exit_code = 0
    is_multi_run = args.all_instances or args.all_configs

    for loaded in loaded_instances:
        for config in benchmark_configs:
            result = _run_single_benchmark(
                args,
                loaded=loaded,
                config=config,
                output_path=None if is_multi_run else (args.output or _default_output_path(loaded.target.key, config.name, args.batch)),
            )
            results.append(result)
            exit_code = max(exit_code, result.exit_code)

    if is_multi_run:
        _write_wave_summary(_summary_output_path(args.output), results)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
