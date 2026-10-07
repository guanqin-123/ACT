#!/usr/bin/env python3
"""Draw Figure 7 (label fig:exp-trace) from the collected trace data in rq2/ next to this script.

Writes the three sub-figures, for every traced TinyImageNet query that both configurations (CLIMB enabled / disabled,
FSB, B=256) solve: exp_trace_nodes.pdf (a: processed sub-problems over time since BaB entry), exp_trace_frontier.pdf
(b: pending sub-problems per BaB round) and exp_trace_frontier_gap.pdf (c: per-query disabled-minus-enabled pending
sub-problems per round). Each shows a thin line per query, the mean and the min-max range over the queries still running
as a bold line and shading; (a) and (b) mark each configuration's maximum. rq2/climb/tinyimagenet_2024/traced_runs.csv lists every
traced run (budget, status, processed sub-problems) in drawing order, with the per-wave trace of each solved run under
traces/. Axes are logarithmic (x in powers of two; symmetric y in c). Text is typeset by LaTeX with the paper's fonts
and \\climb macro (shared with make_rq2_scatter.py in the same folder). The PDFs are identical to
evaluation_figures/exp_trace_*.pdf.

Usage, from the data folder:
    python make_rq2_trace.py --out-dir <folder>   # default: the current directory
"""
from __future__ import annotations

import argparse
import bisect
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # importing the sibling scripts must not leave __pycache__ in the data folder
from make_rq1_main_table import read_csv  # noqa: E402
from make_rq2_scatter import STYLE, matplotlib, plt, save  # noqa: E402

ticker = matplotlib.ticker
DATA = Path(__file__).resolve().parent
DATASET = "tinyimagenet_2024"
WIDTH = 395.8225 / 72.27  # paper \linewidth in inches
SOLVED = ("CERTIFIED", "FALSIFIED")
# Drawing code below is copied verbatim from the paper's generator so that the PDFs are byte-identical.
TRACE_COLOURS = {"on_k256": "#D55E00", "off_k256": "#0173B2", None: "#009E73"}  # None: disabled-minus-enabled gap
TRACE_NAMES = {"on_k256": r"\climb enabled", "off_k256": r"\climb disabled"}
TRACE_PANELS = (("exp_trace_nodes", "nodes"), ("exp_trace_frontier", "frontier"), ("exp_trace_frontier_gap", "gap"))
TRACE_SIZE, TRACE_AXES = (0.30, 1.75), [0.30, 0.24, 0.66, 0.70]  # width (share of \linewidth), height (in); axes box
TRACE_GAP_LINEAR = 10.0  # the gap panel's symmetric log y-axis is linear within +-10


def run_pending(rows) -> dict[int, float]:
    """Pending sub-problems after each BaB round (trace column wave >= 1) of one traced run."""
    return {int(float(r["wave"])): float(r["frontier"]) for r in rows if r["event"] == "wave" and float(r["wave"]) >= 1}


def run_processed(rows) -> tuple[list[float], list[float]]:
    """Sample times since BaB entry and processed sub-problems of one traced run."""
    return [float(r["t"]) for r in rows], [float(r["nodes"]) for r in rows]


def value_at(series: tuple[list[float], list[float]], t: float) -> float:
    i = bisect.bisect_right(series[0], t) - 1
    return series[1][i] if i >= 0 else 0.0


def trace_series(runs, panel: str, variant: str | None):
    """Per-query point sequences and the across-query mean, min and max of one Figure 7 panel.
    nodes: processed sub-problems over time, over the queries still running (first processed sub-problem to solve);
    frontier: pending sub-problems per BaB round, over the queries still searching;
    gap: disabled-minus-enabled pending sub-problems per BaB round, over the queries where either run is still searching."""
    if panel == "nodes":
        series = [run_processed(run[variant]) for run in runs]
        queries = [[(t, n) for t, n in zip(*s) if n > 0] for s in series]
        grid = sorted({t for q in queries for t, _ in q})
        values = [[value_at(s, t) for s, q in zip(series, queries) if q and q[0][0] <= t <= q[-1][0]] for t in grid]
    else:
        pending = [{v: run_pending(run[v]) for v in ("on_k256", "off_k256")} for run in runs]
        grid = list(range(1, max(w for p in pending for d in p.values() for w in d) + 1))
        if panel == "frontier":
            queries = [[(w, n) for w, n in sorted(p[variant].items()) if n > 0] for p in pending]
            values = [[p[variant][w] for p in pending if p[variant].get(w, 0) > 0] for w in grid]
        else:
            searching = lambda p, w: p["off_k256"].get(w, 0) > 0 or p["on_k256"].get(w, 0) > 0
            gaps = lambda p, w: p["off_k256"].get(w, 0.0) - p["on_k256"].get(w, 0.0)
            queries = [[(w, gaps(p, w)) for w in grid if searching(p, w)] for p in pending]
            values = [[gaps(p, w) for p in pending if searching(p, w)] for w in grid]
    kept = [(x, v) for x, v in zip(grid, values) if v]
    return (queries, [x for x, _ in kept], [sum(v) / len(v) for _, v in kept],
            [min(v) for _, v in kept], [max(v) for _, v in kept])


def trace_panel(runs, panel: str, width: float):
    """One Figure 7 sub-figure: a thin line per query, the across-query mean as a bold line, and min-max shading."""
    fig = plt.figure(figsize=(TRACE_SIZE[0] * width, TRACE_SIZE[1]))
    ax = fig.add_axes(TRACE_AXES)
    xs, ys, extremes, mean_peaks = [], [], [], []
    for variant, z in ((None, 2),) if panel == "gap" else (("off_k256", 2), ("on_k256", 4)):
        colour = TRACE_COLOURS[variant]
        queries, x, mean, lo, hi = trace_series(runs, panel, variant)
        points = [p for q in queries for p in q]
        ys += [p[1] for p in points]
        for q in queries:  # one line per query: the largest value of one wave need not come from the same query next wave
            ax.step([p[0] for p in q], [p[1] for p in q], where="post", color=colour, lw=0.5, alpha=0.35, zorder=z,
                    label="_query")
        ax.fill_between(x, lo, hi, step="post", color=colour, alpha=0.16, linewidth=0, zorder=z)
        ax.step(x, mean, where="post", color=colour, lw=1.4, zorder=z + 1, label=TRACE_NAMES.get(variant, "_mean"))
        xs += x + [p[0] for p in points]
        if variant:
            i, j = hi.index(max(hi)), mean.index(max(mean))
            extremes.append((variant, colour, hi[i], x[i]))
            mean_peaks.append((variant, colour, mean[j], x[j]))
    low, high = math.floor(math.log2(min(xs))), math.ceil(math.log2(max(xs)))
    high += (high - low) % 2  # ticks at every other power of two
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(ticker.FixedLocator([2.0 ** k for k in range(low, high + 1, 2)]))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlim(2.0 ** low, 2.0 ** high)
    ax.set_xlabel("Time since BaB entry (s)" if panel == "nodes" else "BaB round", fontsize=8)
    ax.set_ylabel({"nodes": "Processed sub-problems", "frontier": "Pending sub-problems", "gap": r"$\Delta$ pending"}[panel],
                  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if panel == "gap":
        ax.axhline(0, color="#777777", lw=0.6, zorder=1)
        down = math.ceil(math.log10(-min(ys))) if min(ys) < -TRACE_GAP_LINEAR else 1
        up = math.ceil(math.log10(max(ys))) if max(ys) > TRACE_GAP_LINEAR else 1
        ax.set_yscale("symlog", linthresh=TRACE_GAP_LINEAR, linscale=0.4)
        ax.yaxis.set_major_locator(ticker.FixedLocator([-(10.0 ** k) for k in range(down, 1, -1)] + [0.0]  # no +-10:
                                                       + [10.0 ** k for k in range(2, up + 1)]))  # too close to 0
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: "0" if v == 0 else ("$-" if v < 0 else "$") + f"10^{{{round(math.log10(abs(v)))}}}$"))
        ax.set_ylim(-(10.0 ** down), 10.0 ** up)
        gap_queries, x_mean, mean, _, _ = trace_series(runs, panel, None)
        gap_points = [p for q in gap_queries for p in q]
        i = mean.index(max(mean))
        extremes = [("max", max(gap_points, key=lambda p: (p[1], -p[0])), (-3, 3), "right", "bottom"),
                    ("max mean", (x_mean[i], mean[i]), (0, -4), "center", "top"),
                    ("min", min(gap_points, key=lambda p: (p[1], p[0])), (4, 0), "left", "center")]
        for word, (x, value), offset, ha, va in extremes:
            ax.plot(x, value, "o", ms=3, mfc="white", mec=TRACE_COLOURS[None], mew=0.8, zorder=6, label="_peak")
            number = f"$-${-value:,.0f}" if value < 0 else f"{value:,.0f}"  # a real minus sign, not a hyphen
            ax.annotate(f"{word} {number}", (x, value), textcoords="offset points", xytext=offset, ha=ha, va=va,
                        color=TRACE_COLOURS[None], fontsize=6.5, zorder=7,
                        bbox={"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})
        return fig
    headroom = 30 if panel == "nodes" else 8  # (a) also holds the legend above its data
    ax.set(yscale="log", ylim=(0.7, headroom * max(value for *_, value, _ in extremes)))
    marks = [(variant, "max" if panel == "nodes" else "peak", colour, value, x) for variant, colour, value, x in extremes]
    if panel == "frontier":  # also mark where each configuration's mean pending count peaks
        marks += [(variant, "max mean", colour, value, x) for variant, colour, value, x in mean_peaks]
    # (b) is crowded around its points: these labels sit in empty space (left side, upper right) with a thin leader line
    # value: text position (axes fraction), alignment, where the leader leaves the text, number on its own line
    leaders = {("frontier", "on_k256", "peak"): ((0.03, 0.60), "left", (1, 0.5), False),
               ("frontier", "on_k256", "max mean"): ((0.99, 0.08), "right", (0, 0.5), True),
               ("frontier", "off_k256", "max mean"): ((0.98, 0.95), "right", (0.5, 0.5), False)}
    for variant, word, colour, value, x in marks:
        disabled = variant == "off_k256"
        ax.plot(x, value, "o", ms=3, mfc="white", mec=colour, mew=0.8, zorder=6, label="_peak")
        style = {"color": colour, "fontsize": 6.5, "zorder": 7,
                 "bbox": {"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}}
        if (panel, variant, word) in leaders:
            at, ha, leave, two_lines = leaders[(panel, variant, word)]
            ax.annotate(f"{word}{chr(10) if two_lines else ' '}{value:,.0f}", (x, value), xytext=at, textcoords="axes fraction",
                        ha=ha, va="center", multialignment=ha,
                        arrowprops={"arrowstyle": "-", "color": colour, "lw": 0.5, "shrinkA": 0, "shrinkB": 2.5,
                                    "relpos": leave}, **style)
            continue
        above = disabled and panel == "frontier"  # in (a) the legend is above, so the disabled label goes left
        ax.annotate(f"{word} {value:,.0f}", (x, value), textcoords="offset points",
                    xytext=(-3, 3) if above else (-5, 0) if disabled else (4, 0), ha="right" if disabled else "left",
                    va="bottom" if above else "center", **style)
    if panel == "nodes":  # the legend sits in this panel's empty upper left (no query is past its root yet)
        lines = {line.get_label(): line for line in ax.lines}
        names = (r"\climb enabled", r"\climb disabled")
        ax.legend([lines[n] for n in names], names, loc="upper left", fontsize=6.5, handlelength=1.0, handletextpad=0.4,
                  borderpad=0.2, labelspacing=0.2, borderaxespad=0.2, facecolor="white", edgecolor="none", framealpha=1)
    return fig


def solved_by_both(data: Path) -> list[dict[str, list[dict[str, str]]]]:
    folder = data / "rq2" / "climb" / DATASET
    pairs: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for run in read_csv(folder / "traced_runs.csv"):
        pairs.setdefault((run["set"], run["query"]), {})[run["variant"]] = run
    return [{variant: read_csv(folder / runs[variant]["trace"]) for variant in ("on_k256", "off_k256")}
            for runs in pairs.values() if all(runs[variant]["status"] in SOLVED for variant in ("on_k256", "off_k256"))]


def draw(data: Path, out_dir: Path) -> list[Path]:
    solved = solved_by_both(data)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with matplotlib.rc_context(STYLE):
        for name, panel in TRACE_PANELS:
            paths.append(out_dir / f"{name}.pdf")
            save(trace_panel(solved, panel, WIDTH), paths[-1])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA,
                        help="data folder that contains rq2/ (default: the folder of this script)")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="where to write the PDFs (default: .)")
    args = parser.parse_args()
    for path in draw(args.data, args.out_dir):
        print(path)


if __name__ == "__main__":
    main()
