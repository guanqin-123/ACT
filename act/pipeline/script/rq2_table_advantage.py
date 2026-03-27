"""
RQ2 Table: Batch advantage over sequential baseline at fixed time budgets.

For each benchmark and time cutoff, shows how many more violations the batch
approach (anisotropic and isotropic) found compared to the sequential baseline.

Sequential baseline: seq_fix for all benchmarks.

Usage:
    python act/pipeline/script/rq2_table_advantage.py experiments/rq1
    python act/pipeline/script/rq2_table_advantage.py experiments/rq1 --output figures/table_advantage
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Time cutoffs in seconds
CUTOFFS_SEC = [60, 300, 1800, 3600]
CUTOFF_LABELS = ["1 min", "5 min", "30 min", "60 min"]

# Benchmarks in display order
BENCHMARKS = ["trafficsigns", "cifar100", "tinyimagenet"]
BENCH_DISPLAY = {
    "cifar100": r"\textsc{Cifar100}",
    "tinyimagenet": r"\textsc{TinyImageNet}",
    "trafficsigns": r"\textsc{TrafficSigns}",
}

# Sequential baseline per benchmark
SEQ_BASELINE = {
    "cifar100": "seq_fix",
    "tinyimagenet": "seq_fix",
    "trafficsigns": "seq_fix",
}


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


def _mean_violations_at_cutoffs(runs: List[dict], cutoffs: List[float]) -> np.ndarray:
    """Return mean cumulative violations at each cutoff across runs."""
    if not runs:
        return np.full(len(cutoffs), np.nan)
    all_counts = []
    for run in runs:
        ts = np.array(run.get("violation_timestamps", []))
        counts = [int(np.searchsorted(ts, t, side="right")) for t in cutoffs]
        all_counts.append(counts)
    return np.mean(all_counts, axis=0)


def _max_total_time(runs: List[dict]) -> float:
    if not runs:
        return 0.0
    return max(r.get("total_time", 0) for r in runs)


def generate_table(
    results_dir: Path,
    benchmarks: List[str],
    output: Optional[Path] = None,
) -> str:
    """Generate the advantage table and return LaTeX string."""

    # Collect data: {benchmark: {method: np.array of mean violations at cutoffs}}
    data: Dict[str, Dict[str, np.ndarray]] = {}
    max_times: Dict[str, Dict[str, float]] = {}

    methods_needed = {"batch_aniso", "batch_iso"}
    for bm in benchmarks:
        methods_needed.add(SEQ_BASELINE[bm])

    for bm in benchmarks:
        data[bm] = {}
        max_times[bm] = {}
        for method in ["batch_aniso", "batch_iso", SEQ_BASELINE[bm]]:
            runs = _load_runs(results_dir, bm, method)
            data[bm][method] = _mean_violations_at_cutoffs(runs, CUTOFFS_SEC)
            max_times[bm][method] = _max_total_time(runs)

    # Build table rows
    n_cutoffs = len(CUTOFFS_SEC)
    n_bench = len(benchmarks)

    # Text table
    lines = []
    header = f"{'Time':<10}"
    for bm in benchmarks:
        header += f"  {bm:>30}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, (cutoff, label) in enumerate(zip(CUTOFFS_SEC, CUTOFF_LABELS)):
        row = f"{label:<10}"
        for bm in benchmarks:
            seq_key = SEQ_BASELINE[bm]
            seq_val = data[bm][seq_key][i]
            ani_val = data[bm]["batch_aniso"][i]
            iso_val = data[bm]["batch_iso"][i]
            diff_ani = ani_val - seq_val
            diff_iso = iso_val - seq_val
            cell = f"+{diff_iso:,.0f} / +{diff_ani:,.0f}"
            row += f"  {cell:>30}"
        lines.append(row)

    text_table = "\n".join(lines)
    print(text_table)
    print()

    # LaTeX table
    ncols = n_bench
    col_spec = "l" + "r" * ncols
    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(
        r"\caption{Additional violations found by the batch approach over the sequential baseline at equal wall-clock budgets. Each cell reports $\Delta_{\mathrm{Iso}}$ / $\Delta_{\mathrm{Ani}}$, where $\Delta = \text{Batch} - \text{Seq}$. Sequential baseline: \texttt{seq\_fix} for all benchmarks.}"
    )
    latex_lines.append(r"\label{tab:batch-advantage}")
    latex_lines.append(r"\small")
    latex_lines.append(r"\begin{tabular}{" + col_spec + "}")
    latex_lines.append(r"\toprule")

    # Header row
    header_cells = [r"\textbf{Time}"]
    for bm in benchmarks:
        header_cells.append(BENCH_DISPLAY[bm])
    latex_lines.append(" & ".join(header_cells) + r" \\")
    latex_lines.append(r"\midrule")

    # Data rows
    for i, (cutoff, label) in enumerate(zip(CUTOFFS_SEC, CUTOFF_LABELS)):
        cells = [label]
        for bm in benchmarks:
            seq_key = SEQ_BASELINE[bm]
            seq_val = data[bm][seq_key][i]
            ani_val = data[bm]["batch_aniso"][i]
            iso_val = data[bm]["batch_iso"][i]
            diff_ani = ani_val - seq_val
            diff_iso = iso_val - seq_val

            # Format with sign and thousands separator
            def _fmt(v: float) -> str:
                sign = "+" if v >= 0 else ""
                return f"{sign}{v:,.0f}"

            cell = f"{_fmt(diff_iso)}/{_fmt(diff_ani)}"
            cells.append(cell)
        latex_lines.append(" & ".join(cells) + r" \\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    latex_str = "\n".join(latex_lines)
    print(latex_str)

    # Save
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        tex_path = output.with_suffix(".tex")
        txt_path = output.with_suffix(".txt")
        tex_path.write_text(latex_str + "\n")
        txt_path.write_text(text_table + "\n")
        print(f"\nSaved: {tex_path}")
        print(f"Saved: {txt_path}")

    return latex_str


def main():
    parser = argparse.ArgumentParser(
        description="RQ2: Batch advantage table at fixed time budgets"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to RQ1 results (e.g., experiments/rq1)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Benchmarks to include (default: cifar100 tinyimagenet mnist trafficsigns)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path stem (e.g., figures/table_advantage)",
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    benchmarks = args.benchmarks or BENCHMARKS

    output = Path(args.output) if args.output else None
    generate_table(results_dir, benchmarks, output)


if __name__ == "__main__":
    main()
