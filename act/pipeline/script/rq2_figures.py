"""
RQ2 Figure: Cumulative violations over wall-clock time.

Reads violation_timestamps from RQ1 result JSONs and generates
a multi-panel figure (one panel per benchmark, 4 curves per panel).

Usage:
    python act/pipeline/script/rq2_figures.py <results_dir> [options]

    # Mean + shaded std (default)
    python act/pipeline/script/rq2_figures.py experiments/rq1

    # Individual run curves
    python act/pipeline/script/rq2_figures.py experiments/rq1 --mode individual

    # Select benchmarks
    python act/pipeline/script/rq2_figures.py experiments/rq1 --benchmarks cifar100 mnist
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = [
    ("batch_aniso", "ACT (ours)", "#2171b5", "-", "o", 2.2),
    ("batch_iso", "Batch-Iso", "#e6550d", "--", "s", 1.8),
    ("seq_aniso", "Seq-Aniso", "#31a354", "-", "^", 1.8),
    ("seq_rand", "Seq-Rand", "#d62728", "-", "D", 1.8),
]

MARKER_EVERY = 50  # place a marker every N grid points

BENCH_DISPLAY = {
    "acasxu": "ACAS Xu",
    "cifar100": "CIFAR-100",
    "tinyimagenet": "TinyImageNet",
    "mnist": "MNIST",
    "trafficsigns": "TrafficSigns",
}

BENCH_ORDER = ["acasxu", "cifar100", "tinyimagenet", "mnist", "trafficsigns"]


def _load_runs(
    results_dir: Path, benchmark: str, method: str, single_run: bool = False
) -> List[dict]:
    method_dir = results_dir / benchmark / method
    if not method_dir.exists():
        return []
    if single_run:
        p = method_dir / "run_0.json"
        if not p.exists():
            return []
        with open(p) as f:
            data = json.load(f)
        return [data] if "violation_timestamps" in data else []
    runs = []
    for p in sorted(method_dir.glob("run_*.json")):
        with open(p) as f:
            data = json.load(f)
        if "violation_timestamps" in data:
            runs.append(data)
        else:
            print(f"  WARN: {p.name} missing violation_timestamps, skipping")
    return runs


def _timestamps_to_cumulative(
    timestamps: List[float], timeout: float, n_points: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    t_grid = np.linspace(0, timeout, n_points)
    ts = np.array(timestamps) if timestamps else np.array([])
    cum = np.searchsorted(ts, t_grid, side="right").astype(float)
    return t_grid, cum


def _plot_mean_shaded(
    ax, results_dir: Path, benchmark: str, timeout: float, n_points: int = 500
):
    for method_key, label, color, ls, mk, lw in METHODS:
        runs = _load_runs(results_dir, benchmark, method_key)
        if not runs:
            continue

        all_cum = []
        for run in runs:
            t_grid, cum = _timestamps_to_cumulative(
                run["violation_timestamps"], timeout, n_points
            )
            all_cum.append(cum)

        all_cum = np.array(all_cum)
        mean = all_cum.mean(axis=0)
        ax.plot(
            t_grid,
            mean,
            label=label,
            color=color,
            ls=ls,
            lw=lw,
            marker=mk,
            markevery=MARKER_EVERY,
            markersize=4,
            markeredgewidth=0.8,
        )

        if all_cum.shape[0] > 1:
            std = all_cum.std(axis=0)
            ax.fill_between(
                t_grid, mean - std, mean + std, alpha=0.15, color=color, linewidth=0
            )


def _plot_single(
    ax, results_dir: Path, benchmark: str, timeout: float, n_points: int = 500
):
    for method_key, label, color, ls, mk, lw in METHODS:
        runs = _load_runs(results_dir, benchmark, method_key, single_run=True)
        if not runs:
            continue
        t_grid, cum = _timestamps_to_cumulative(
            runs[0]["violation_timestamps"], timeout, n_points
        )
        ax.plot(
            t_grid,
            cum,
            label=label,
            color=color,
            ls=ls,
            lw=lw,
            marker=mk,
            markevery=MARKER_EVERY,
            markersize=4,
            markeredgewidth=0.8,
        )


def _plot_individual(
    ax, results_dir: Path, benchmark: str, timeout: float, n_points: int = 500
):
    for method_key, label, color, ls, mk, lw in METHODS:
        runs = _load_runs(results_dir, benchmark, method_key)
        if not runs:
            continue

        for i, run in enumerate(runs):
            t_grid, cum = _timestamps_to_cumulative(
                run["violation_timestamps"], timeout, n_points
            )
            ax.plot(
                t_grid,
                cum,
                color=color,
                ls=ls,
                lw=0.8,
                alpha=0.5,
                label=label if i == 0 else None,
            )


def generate_figure(
    results_dir: Path,
    benchmarks: List[str],
    mode: str = "mean",
    timeout: float = 300.0,
    output: Optional[Path] = None,
):
    n = len(benchmarks)
    if n <= 3:
        ncols, nrows = n, 1
        fig_w = 3.5 * ncols
    else:
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig_w = 3.5 * ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, 2.8 * nrows), squeeze=False)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    for idx, bench in enumerate(benchmarks):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        if mode == "individual":
            _plot_individual(ax, results_dir, bench, timeout)
        elif mode == "single":
            _plot_single(ax, results_dir, bench, timeout)
        else:
            _plot_mean_shaded(ax, results_dir, bench, timeout)

        ax.set_title(BENCH_DISPLAY.get(bench, bench))
        ax.set_xlabel("Time (s)")
        if col == 0:
            ax.set_ylabel("Cumulative Violations")
        ax.set_xlim(0, timeout)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else "0")
        )
        ax.grid(True, alpha=0.3, linewidth=0.5, which="both")

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left", framealpha=0.8)

    for idx in range(len(benchmarks), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout(pad=1.0)

    if output is None:
        output = results_dir / "viol_time"

    output_dir = output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output.stem

    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    # Save individual sub-figures per benchmark
    for idx, bench in enumerate(benchmarks):
        sfig, sax = plt.subplots(1, 1, figsize=(4.0, 3.0))
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 9,
                "axes.labelsize": 10,
                "axes.titlesize": 11,
                "legend.fontsize": 7.5,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
            }
        )
        if mode == "individual":
            _plot_individual(sax, results_dir, bench, timeout)
        elif mode == "single":
            _plot_single(sax, results_dir, bench, timeout)
        else:
            _plot_mean_shaded(sax, results_dir, bench, timeout)

        sax.set_title(BENCH_DISPLAY.get(bench, bench))
        sax.set_xlabel("Time (s)")
        sax.set_ylabel("Cumulative Violations")
        sax.set_xlim(0, timeout)
        sax.set_yscale("symlog", linthresh=1)
        sax.set_ylim(bottom=0)
        sax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else "0")
        )
        sax.grid(True, alpha=0.3, linewidth=0.5, which="both")
        handles, labels = sax.get_legend_handles_labels()
        if handles:
            sax.legend(loc="upper left", framealpha=0.8)

        sfig.tight_layout(pad=0.8)
        sub_pdf = output_dir / f"{stem}_{bench}.pdf"
        sub_png = output_dir / f"{stem}_{bench}.png"
        sfig.savefig(sub_pdf, bbox_inches="tight", dpi=300)
        sfig.savefig(sub_png, bbox_inches="tight", dpi=300)
        plt.close(sfig)
        print(f"Saved: {sub_pdf}")

    return pdf_path, png_path


def main():
    parser = argparse.ArgumentParser(
        description="RQ2: Cumulative violation curves over time"
    )
    parser.add_argument(
        "results_dir", type=str, help="Path to RQ1 results (e.g., experiments/rq1)"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "mean", "individual"],
        default="single",
        help="single: run_0 only (default) | mean: mean+shaded std | individual: all runs",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Benchmarks to plot (default: all found)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="X-axis limit in seconds (default: from data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path stem (default: <results_dir>/viol_time)",
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    if args.benchmarks:
        benchmarks = args.benchmarks
    else:
        benchmarks = [b for b in BENCH_ORDER if (results_dir / b).is_dir()]

    if not benchmarks:
        print("No benchmark results found.")
        sys.exit(1)

    timeout = args.timeout
    if timeout is None:
        for bench in benchmarks:
            for mk, _, _, _, _, _ in METHODS:
                runs = _load_runs(results_dir, bench, mk)
                for r in runs:
                    t = r.get("total_time", 300)
                    if timeout is None or t > timeout:
                        timeout = t
        timeout = timeout or 300.0

    output = Path(args.output) if args.output else None

    print(
        f"RQ2 Figure: {len(benchmarks)} panels, mode={args.mode}, timeout={timeout:.0f}s"
    )
    print(f"Benchmarks: {', '.join(benchmarks)}")

    generate_figure(
        results_dir, benchmarks, mode=args.mode, timeout=timeout, output=output
    )


if __name__ == "__main__":
    main()
