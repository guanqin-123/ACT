"""
RQ2 Figure 2 — Broken-axis: left half zooms into 0–60 s, right half shows full timeline.

The cumulative violation curve is continuous across the break.

Usage:
    python act/pipeline/script/rq2_figure2_broken.py experiments/rq1 \
        --output figures/figure2_broken
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np

# ── constants ───────────────────────────────────────────────────────────────
# (method_key, label, color, linestyle, marker, linewidth)
# Only 3 curves: Batch-Aniso, Batch-Iso, Seq-Fixed
METHODS = [
    ("batch_iso",   "Batch-Iso",  "#e6550d", "--", "s", 1.8),
    ("batch_aniso", "Batch-Ani",  "#d62728", "-",  "o", 2.2),
    ("seq_fix",     "Seq-Fixed",  "#2171b5", "-",  "D", 1.8),
]

BENCH_DISPLAY = {
    "cifar100":     "CIFAR-100",
    "tinyimagenet": "TinyImageNet",
    "trafficsigns": "TrafficSigns",
}

BENCH_ORDER = ["cifar100", "tinyimagenet", "trafficsigns"]


# ── data helpers ────────────────────────────────────────────────────────────
def _load_runs(results_dir: Path, benchmark: str, method: str) -> List[dict]:
    method_dir = results_dir / benchmark / method
    if not method_dir.exists():
        return []
    runs = []
    for p in sorted(method_dir.glob("run_*.json")):
        with open(p) as f:
            data = json.load(f)
        if "violation_timestamps" in data:
            runs.append(data)
    return runs




def _timestamps_to_cumulative(
    timestamps: List[float], timeout: float, n_points: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    t_grid = np.linspace(0, timeout, n_points)
    ts = np.array(sorted(timestamps)) if timestamps else np.array([])
    cum = np.searchsorted(ts, t_grid, side="right").astype(float)
    return t_grid, cum


def _bench_max_time(results_dir: Path, benchmark: str) -> float:
    """Return the maximum timestamp across all methods for a benchmark."""
    max_t = 60.0
    for mk, *_ in METHODS:
        for run in _load_runs(results_dir, benchmark, mk):
            ts = run.get("violation_timestamps", [])
            if ts:
                max_t = max(max_t, max(ts))
            max_t = max(max_t, run.get("total_time", 0))
    return max_t


# ── break markers ───────────────────────────────────────────────────────────
def _draw_break_marks(ax_left, ax_right, d=0.015):
    """Draw diagonal break marks between the two panels."""
    kwargs = dict(color="k", clip_on=False, lw=1.0, zorder=10)
    for ax, x_pos in [(ax_left, 1.0), (ax_right, 0.0)]:
        for y_pos in [0.0, 1.0]:
            ax.plot(
                (x_pos - d, x_pos + d), (y_pos - d * 2, y_pos + d * 2),
                transform=ax.transAxes, **kwargs
            )


# ── log-axis tick helper ─────────────────────────────────────────────────────
def _set_log_ticks(ax, break_time, max_time):
    """Set explicit major ticks on log x-axis, skipping values near break_time."""
    candidates = [
        (300, "5m"), (600, "10m"), (1800, "30m"),
        (3600, "1h"), (7200, "2h"), (10800, "3h"),
    ]
    ticks, tick_labels = [], []
    for val, lbl in candidates:
        if val > break_time * 2.5 and val <= max_time * 1.1:
            ticks.append(val)
            tick_labels.append(lbl)
    # Keep at most 4 ticks to avoid crowding
    if len(ticks) > 4:
        step = len(ticks) // 4 + 1
        ticks = ticks[::step]
        tick_labels = tick_labels[::step]
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


# ── plotting ────────────────────────────────────────────────────────────────
def _plot_on_ax(ax, results_dir, benchmark, timeout, n_points=500, marker_every=50):
    for method_key, label, color, ls, mk, lw in METHODS:
        runs = _load_runs(results_dir, benchmark, method_key)
        if not runs:
            continue
        run = runs[0]
        t_grid, cum = _timestamps_to_cumulative(
            run["violation_timestamps"], timeout, n_points
        )
        ax.plot(
            t_grid, cum,
            label=label, color=color, ls=ls, lw=lw,
            marker=mk, markevery=marker_every, markersize=4, markeredgewidth=0.8,
        )


def generate_broken_figure(
    results_dir: Path,
    benchmarks: List[str],
    break_time: float = 60.0,
    output: Path | None = None,
):
    n = len(benchmarks)
    if n <= 3:
        nrows, ncols_bench = 1, n
    else:
        ncols_bench = 3
        nrows = (n + ncols_bench - 1) // ncols_bench

    left_ratio = 0.50

    fig = plt.figure(figsize=(4.2 * ncols_bench, 2.8 * nrows))

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "mathtext.default": "regular",
    })

    outer = gridspec.GridSpec(nrows, ncols_bench, figure=fig, wspace=0.35, hspace=0.45)

    for idx, bench in enumerate(benchmarks):
        max_time = _bench_max_time(results_dir, bench)
        single_panel = max_time <= break_time * 1.05

        row, col = divmod(idx, ncols_bench)

        if single_panel:
            ax = fig.add_subplot(outer[row, col])
            _plot_on_ax(ax, results_dir, bench, break_time, n_points=500, marker_every=40)
            ax.set_xlim(0, break_time)
            ax.set_xlabel("Time (s)")
            if col == 0:
                ax.set_ylabel("Cumulative Violations")
            ax.set_title(BENCH_DISPLAY.get(bench, bench))
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else "0")
            )
            ax.grid(True, alpha=0.3, linewidth=0.5)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper left", framealpha=0.8)
        else:
            inner = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=outer[row, col],
                width_ratios=[left_ratio, 1 - left_ratio], wspace=0.05
            )
            ax_left = fig.add_subplot(inner[0, 0])
            ax_right = fig.add_subplot(inner[0, 1], sharey=ax_left)

            # plot on both axes
            _plot_on_ax(ax_left, results_dir, bench, break_time, n_points=500, marker_every=40)
            _plot_on_ax(ax_right, results_dir, bench, max_time, n_points=500, marker_every=40)

            # left axis: 0 to break_time
            ax_left.set_xlim(0, break_time)
            ax_left.set_ylim(bottom=0)
            ax_left.spines["right"].set_visible(False)
            ax_left.tick_params(right=False)
            ax_left.set_xlabel("Time (s)")
            if col == 0:
                ax_left.set_ylabel("Cumulative Violations")
            ax_left.grid(True, alpha=0.3, linewidth=0.5)
            ax_left.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else "0")
            )

            # right axis: break_time to max_time (log x-scale)
            ax_right.set_xscale("log")
            ax_right.set_xlim(break_time, max_time)
            ax_right.set_ylim(bottom=0)
            ax_right.spines["left"].set_visible(False)
            ax_right.tick_params(left=False, labelleft=False)
            ax_right.grid(True, alpha=0.3, linewidth=0.5, which="major")
            ax_right.set_xlabel("Time (log)")

            # Explicit ticks: skip values too close to break_time
            _set_log_ticks(ax_right, break_time, max_time)

            # title spanning both
            ax_left.set_title(BENCH_DISPLAY.get(bench, bench), loc="center")

            # legend on left panel only
            handles, labels = ax_left.get_legend_handles_labels()
            if handles:
                ax_left.legend(loc="upper left", framealpha=0.8)
            leg = ax_right.get_legend()
            if leg:
                leg.remove()

            _draw_break_marks(ax_left, ax_right)

    # hide unused panels
    for idx in range(len(benchmarks), nrows * ncols_bench):
        row, col = divmod(idx, ncols_bench)
        ax_empty = fig.add_subplot(outer[row, col])
        ax_empty.set_visible(False)

    # save
    if output is None:
        output = results_dir / "figure2_broken"
    output.parent.mkdir(parents=True, exist_ok=True)

    png_path = output.parent / f"{output.stem}.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {png_path}")

    # ── per-benchmark sub-figures (no title, named by dataset) ──────────
    output_dir = output.parent
    for bench in benchmarks:
        max_time = _bench_max_time(results_dir, bench)
        single_panel = max_time <= break_time * 1.05

        if single_panel:
            sfig, sax = plt.subplots(1, 1, figsize=(4.0, 3.0))
            plt.rcParams.update({
                "font.family": "serif", "font.size": 12,
                "axes.labelsize": 14, "legend.fontsize": 11,
                "xtick.labelsize": 12, "ytick.labelsize": 12,
            })
            _plot_on_ax(sax, results_dir, bench, break_time, n_points=500, marker_every=40)
            sax.set_xlim(0, break_time)
            sax.set_xlabel("Time (s)")
            sax.set_ylabel("Cumulative Violations")
            sax.set_ylim(bottom=0)
            sax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else "0")
            )
            sax.grid(True, alpha=0.3, linewidth=0.5)
            handles, labels = sax.get_legend_handles_labels()
            if handles:
                sax.legend(loc="upper left", framealpha=0.8)
            sfig.tight_layout(pad=0.8)
        else:
            sfig = plt.figure(figsize=(5.0, 3.0))
            plt.rcParams.update({
                "font.family": "serif", "font.size": 12,
                "axes.labelsize": 14, "legend.fontsize": 11,
                "xtick.labelsize": 12, "ytick.labelsize": 12,
            })
            inner = gridspec.GridSpec(1, 2, figure=sfig,
                                     width_ratios=[left_ratio, 1 - left_ratio], wspace=0.05)
            sax_l = sfig.add_subplot(inner[0, 0])
            sax_r = sfig.add_subplot(inner[0, 1], sharey=sax_l)

            _plot_on_ax(sax_l, results_dir, bench, break_time, n_points=500, marker_every=40)
            _plot_on_ax(sax_r, results_dir, bench, max_time, n_points=500, marker_every=40)

            sax_l.set_xlim(0, break_time)
            sax_l.set_ylim(bottom=0)
            sax_l.spines["right"].set_visible(False)
            sax_l.tick_params(right=False)
            sax_l.set_xlabel("Time (s)")
            sax_l.set_ylabel("Cumulative Violations")
            sax_l.grid(True, alpha=0.3, linewidth=0.5)
            sax_l.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else "0")
            )

            sax_r.set_xscale("log")
            sax_r.set_xlim(break_time, max_time)
            sax_r.set_ylim(bottom=0)
            sax_r.spines["left"].set_visible(False)
            sax_r.tick_params(left=False, labelleft=False)
            sax_r.grid(True, alpha=0.3, linewidth=0.5, which="major")
            sax_r.set_xlabel("Time (log)")
            _set_log_ticks(sax_r, break_time, max_time)

            handles, labels = sax_l.get_legend_handles_labels()
            if handles:
                sax_l.legend(loc="upper left", framealpha=0.8)
            leg = sax_r.get_legend()
            if leg:
                leg.remove()

            _draw_break_marks(sax_l, sax_r)

        sub_png = output_dir / f"{bench}.png"
        sfig.savefig(sub_png, bbox_inches="tight", dpi=300)
        plt.close(sfig)
        print(f"Saved: {sub_png}")

    return png_path


def main():
    parser = argparse.ArgumentParser(description="RQ2 Figure 2 with broken x-axis")
    parser.add_argument("results_dir", type=str, help="Path to RQ1 results")
    parser.add_argument("--break-time", type=float, default=60.0,
                        help="Time (s) where x-axis breaks (default: 60)")
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    benchmarks = args.benchmarks or [b for b in BENCH_ORDER if (results_dir / b).is_dir()]
    if not benchmarks:
        print("No benchmark results found.")
        sys.exit(1)

    output = Path(args.output) if args.output else None
    print(f"Generating broken-axis Figure 2: {len(benchmarks)} benchmarks, break={args.break_time}s")
    generate_broken_figure(results_dir, benchmarks, break_time=args.break_time, output=output)


if __name__ == "__main__":
    main()
