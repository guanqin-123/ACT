from __future__ import annotations

# pyright: reportUnusedCallResult=false, reportAny=false

import argparse
import json
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
from scripts.bab_clamp_diagnostic import load_model_and_instance


@dataclass(frozen=True)
class CliArgs:
    instance: int
    batch: int
    eta_iters: int
    max_depth: int
    max_nodes: int
    budget: float
    device: str
    dtype: str
    output: Path | None


def _default_output_path(instance: int, batch: int) -> Path:
    return Path("scripts/logs") / f"baseline_inst{instance}_b{batch}.json"


def _peak_vram_gb(device: str) -> float | None:
    if not torch.cuda.is_available() or not device.startswith(("cuda", "gpu")):
        return None
    return float(torch.cuda.max_memory_allocated() / 1e9)


def _parse_args() -> CliArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8, help="subproblem_batch_size")
    parser.add_argument("--eta-iters", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=300)
    parser.add_argument("--budget", type=float, default=120.0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "gpu"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--output", type=Path, default=None)
    ns = parser.parse_args()
    return CliArgs(
        instance=int(ns.instance),
        batch=int(ns.batch),
        eta_iters=int(ns.eta_iters),
        max_depth=int(ns.max_depth),
        max_nodes=int(ns.max_nodes),
        budget=float(ns.budget),
        device=str(ns.device),
        dtype=str(ns.dtype),
        output=ns.output,
    )


def _build_payload(args: CliArgs, *, vnnlib_name: str, elapsed_s: float) -> dict[str, object]:
    return {
        "instance_idx": args.instance,
        "batch_size": args.batch,
        "eta_iters": args.eta_iters,
        "max_depth": args.max_depth,
        "max_nodes": args.max_nodes,
        "budget": args.budget,
        "device": args.device,
        "dtype": args.dtype,
        "vnnlib_name": vnnlib_name,
        "wall_time_s": elapsed_s,
        "peak_vram_gb": _peak_vram_gb(args.device),
    }


def main() -> int:
    args = _parse_args()

    output_path = args.output or _default_output_path(args.instance, args.batch)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    initialize_device(args.device, args.dtype)
    net, vnnlib_name = load_model_and_instance(args.instance)

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
    payload = _build_payload(args, vnnlib_name=vnnlib_name, elapsed_s=0.0)
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
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        peak_vram = payload["peak_vram_gb"] if payload["peak_vram_gb"] is not None else 0.0
        print(f"baseline inst={args.instance} batch={args.batch} status={result.status.name} nodes={result.metadata.get('nodes')} time={elapsed_s:.2f}s peak_vram={peak_vram:.3f} GB")
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
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        peak_vram = payload["peak_vram_gb"] if payload["peak_vram_gb"] is not None else 0.0
        print(f"baseline inst={args.instance} batch={args.batch} status=OOM nodes=null time={elapsed_s:.2f}s peak_vram={peak_vram:.3f} GB")
        return 2


if __name__ == "__main__":
    sys.exit(main())
