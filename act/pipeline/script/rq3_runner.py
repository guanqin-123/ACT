from __future__ import annotations

import argparse
import traceback

from act.pipeline.script.rq1_config import EXPERIMENT_DEFAULTS
from act.pipeline.script.rq1_runner import run_single_experiment
from act.pipeline.script.rq3_config import (
    RQ3_BENCHMARKS,
    RQ3_DEFAULTS,
    RQ3_METHODS,
    SCALE_FACTORS,
)
from act.util.cli_utils import initialize_from_args


def _scale_str(scale: float) -> str:
    """Format scale factor for directory names (e.g., 0.1 → '0.10')."""
    return f"{scale:.2f}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RQ3 scale sensitivity experiment runner"
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(RQ3_BENCHMARKS.keys()),
        help="Benchmark key (cifar100 or tinyimagenet)",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=RQ3_METHODS,
        help="Method (batch_aniso or batch_iso)",
    )
    parser.add_argument(
        "--scale",
        required=True,
        type=float,
        help="Perturbation scale factor s (e.g., 0.01, 0.05, 0.1, 0.2, 0.3, 0.5)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=RQ3_DEFAULTS["runs"],
        help="Number of independent runs",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=RQ3_DEFAULTS["timeout"],
        help="Per-run timeout in seconds",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "gpu"],
        help="Device to use (cpu/cuda)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Default dtype",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Limit number of instances loaded (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/rq3",
        help="Base output directory for results",
    )
    parser.add_argument(
        "--start-run",
        type=int,
        default=0,
        help="Starting run ID (used by run_rq3.sh for resume)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    initialize_from_args(args)

    scale = args.scale
    method = args.method
    scale_dir = _scale_str(scale)

    original_scale = EXPERIMENT_DEFAULTS["perturb_scale"]

    try:
        EXPERIMENT_DEFAULTS["perturb_scale"] = scale
        scale_output_dir = f"{args.output_dir}/{scale_dir}"

        for run_id in range(args.start_run, args.start_run + args.runs):
            try:
                print(
                    f"🔬 RQ3: benchmark={args.benchmark} method={method} scale={scale} run={run_id}"
                )
                run_single_experiment(
                    benchmark_key=args.benchmark,
                    method_key=method,
                    run_id=run_id,
                    timeout=args.timeout,
                    max_instances=args.max_instances,
                    output_dir=scale_output_dir,
                )
            except Exception as exc:
                print(f"❌ Run {run_id} failed: {exc}")
                traceback.print_exc()
    finally:
        EXPERIMENT_DEFAULTS["perturb_scale"] = original_scale


if __name__ == "__main__":
    main()
