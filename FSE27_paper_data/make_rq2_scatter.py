#!/usr/bin/env python3
"""Draw Figure 6 (label fig:exp-scatter) from the collected data in rq2/ next to this script.

Writes the two sub-figures, each with its own legend: exp_rq2_scatter_nodes.pdf (processed sub-problems) and
exp_rq2_scatter_time.pdf (verification time). Each point is a CIFAR-100 or
TinyImageNet query solved with CLIMB both disabled (x axis) and enabled (y axis); SAT-ReLU is not reported. All
text is typeset by LaTeX with the paper's fonts and its \\climb macro (\\textsc{Climb}), so LaTeX with the
libertine, newtx and zi4 packages must be installed. The query selection and the comparison come from
make_rq2_table.py in the same folder. The PDFs are identical to evaluation_figures/exp_rq2_scatter_*.pdf.

Usage, from the data folder:
    python make_rq2_scatter.py --out-dir <folder>   # default: the current directory
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogLocator  # noqa: E402

sys.dont_write_bytecode = True  # importing the sibling script must not leave __pycache__ in the data folder
from make_rq2_table import DATASETS, Run, load  # noqa: E402

DATA = Path(__file__).resolve().parent
WIDTH = 394.344 / 72.27  # paper \linewidth in inches
MARKERS = {"cifar100_2024": ("o", "#0173b2", 20), "tinyimagenet_2024": ("^", "#de8f05", 22)}
PREAMBLE = (r"\usepackage[T1]{fontenc}"
            r"\usepackage[tt=false,type1=true]{libertine}"
            r"\usepackage[varqu]{zi4}"
            r"\usepackage[libertine]{newtxmath}"
            r"\usepackage{xspace}"
            r"\newcommand{\climb}{\textsc{Climb}\xspace}")
STYLE = {"text.usetex": True, "text.latex.preamble": PREAMBLE,
         "font.family": "serif", "font.serif": ["Computer Modern Roman"],
         "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 9, "legend.fontsize": 8,
         "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
         "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e0e0e0", "grid.linewidth": 0.5,
         "axes.edgecolor": "#cccccc", "axes.linewidth": 0.8, "savefig.bbox": None}


def save(fig: plt.Figure, path: Path) -> None:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    path.write_bytes(buffer.getvalue())


def panel(pairs: dict[str, list[tuple[Run, Run]]], field: str, path: Path) -> None:
    fig = plt.figure(figsize=(0.48 * WIDTH, 2.25))
    ax = fig.add_axes((0.22, 0.22, 0.74, 0.74))
    values: list[float] = []
    for dataset, label in DATASETS:
        marker, colour, size = MARKERS[dataset]
        points = [(off[field], on[field]) for on, off in pairs[dataset]
                  if on[field] and off[field] and on[field] > 0 and off[field] > 0]
        if points:
            ax.scatter([p[0] for p in points], [p[1] for p in points], marker=marker, s=size, lw=0.4,
                       edgecolors="white", facecolors=colour, label=label, alpha=0.7)
        values += [v for p in points for v in p]
    low, high = 1.0, 10.0
    if values:
        low, high = min(values), max(values)
        pad = max((high / low) ** 0.05, 1.1)
        low, high = low / pad, high * pad
    ax.plot([low, high], [low, high], color="gray", lw=0.8, ls="--", zorder=1)
    ax.fill_between([low, high], [low, high], low, color="#b0c4de", alpha=0.12, zorder=0)
    ax.text(0.96, 0.04, r"\climb better", transform=ax.transAxes, color="#444444", fontsize=8,
            ha="right", va="bottom")
    ax.set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high))
    ax.set_xlabel(r"\climb disabled", labelpad=2)
    ax.set_ylabel(r"\climb enabled", labelpad=2)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10.0, numticks=5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", facecolor="white", edgecolor="none", framealpha=1)
    save(fig, path)


def draw(data: Path, out_dir: Path) -> list[Path]:
    pairs = {dataset: load(data, dataset)[1] for dataset, _ in DATASETS}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"exp_rq2_scatter_{name}.pdf" for name in ("nodes", "time")]
    with matplotlib.rc_context(STYLE):
        for field, path in zip(("nodes", "time"), paths):
            panel(pairs, field, path)
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
