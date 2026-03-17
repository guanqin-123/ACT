"""
RQ3 Results Aggregator: Scale Factor Sensitivity.

Scans RQ3 experiment results for both batch_aniso and batch_iso,
produces tables showing how each method responds to varying scale factor s.

Usage:
    python act/pipeline/script/rq3_aggregator.py experiments/rq3

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCALE_FACTORS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
METHODS = ["batch_aniso", "batch_iso"]
BENCHMARK_ORDER = ["cifar100", "tinyimagenet"]

BENCHMARK_DISPLAY = {
    "cifar100": r"\textsc{Cifar100}",
    "tinyimagenet": r"\textsc{TinyImageNet}",
}

BENCHMARK_TEXT = {
    "cifar100": "Cifar100",
    "tinyimagenet": "TinyImageNet",
}

METHOD_DISPLAY = {
    "batch_aniso": "Batch-Aniso",
    "batch_iso": "Batch-Iso",
}

METHOD_LATEX = {
    "batch_aniso": r"\textsc{Batch\text{-}Aniso}",
    "batch_iso": r"\textsc{Batch\text{-}Iso}",
}


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return math.sqrt(sum((x - avg) ** 2 for x in values) / len(values))


def _scale_str(scale: float) -> str:
    return f"{scale:.2f}"


# Stats key: (benchmark, method, scale)
Stats = Dict[str, Dict[str, Dict[float, Dict[str, Any]]]]


class ResultsAggregator:
    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)
        self._stats: Stats = {}

    def scan_and_compute(self) -> Stats:
        stats: Stats = {}
        for benchmark in BENCHMARK_ORDER:
            stats[benchmark] = {}
            for method in METHODS:
                stats[benchmark][method] = {}
                for scale in SCALE_FACTORS:
                    method_dir = (
                        self.results_dir / _scale_str(scale) / benchmark / method
                    )
                    violations: List[float] = []
                    if method_dir.is_dir():
                        for rf in sorted(method_dir.glob("run_*.json")):
                            try:
                                with open(rf, "r", encoding="utf-8") as fh:
                                    data = json.load(fh)
                                if data.get("violations") is not None:
                                    violations.append(float(data["violations"]))
                            except (json.JSONDecodeError, OSError) as exc:
                                print(
                                    f"WARNING: skipping {rf} — {exc}", file=sys.stderr
                                )

                    if violations:
                        stats[benchmark][method][scale] = {
                            "mean": _mean(violations),
                            "std": _std(violations),
                            "runs": len(violations),
                        }
                    else:
                        stats[benchmark][method][scale] = {
                            "mean": None,
                            "std": None,
                            "runs": 0,
                        }

        self._stats = stats
        return stats

    def _fmt(self, st: Dict[str, Any]) -> str:
        if st["mean"] is None:
            return "---"
        m = int(round(st["mean"]))
        if st["runs"] > 1 and st["std"] and st["std"] > 0:
            s = int(round(st["std"]))
            return f"{m}±{s}"
        return str(m)

    def _fmt_latex(self, st: Dict[str, Any]) -> str:
        if st["mean"] is None:
            return "---"
        m = int(round(st["mean"]))
        if st["runs"] > 1 and st["std"] and st["std"] > 0:
            s = int(round(st["std"]))
            return f"${m} \\pm {s}$"
        return str(m)

    def to_text(self) -> None:
        if not self._stats:
            self.scan_and_compute()

        w_bm, w_method, w_val = 14, 14, 10
        scale_hdr = "  ".join(f"{'s=' + str(s):>{w_val}}" for s in SCALE_FACTORS)
        sep = "-" * (w_bm + w_method + 4 + len(scale_hdr))

        print(sep)
        print(f"{'Benchmark':<{w_bm}}  {'Method':<{w_method}}  {scale_hdr}")
        print(sep)

        for benchmark in BENCHMARK_ORDER:
            bm_label = BENCHMARK_TEXT.get(benchmark, benchmark)
            for i, method in enumerate(METHODS):
                label = bm_label if i == 0 else ""
                m_label = METHOD_DISPLAY.get(method, method)
                cells = []
                for scale in SCALE_FACTORS:
                    st = self._stats[benchmark][method][scale]
                    cells.append(self._fmt(st).rjust(w_val))
                print(f"{label:<{w_bm}}  {m_label:<{w_method}}  {'  '.join(cells)}")
            print(sep)

    def to_csv(self, output_path: Path) -> None:
        if not self._stats:
            self.scan_and_compute()

        csv_file = Path(output_path) / "rq3_results.csv"
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        columns = ["benchmark", "method"] + [
            f"s_{_scale_str(s)}" for s in SCALE_FACTORS
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for benchmark in BENCHMARK_ORDER:
                for method in METHODS:
                    row: Dict[str, Any] = {"benchmark": benchmark, "method": method}
                    for scale in SCALE_FACTORS:
                        st = self._stats[benchmark][method][scale]
                        row[f"s_{_scale_str(scale)}"] = (
                            f"{st['mean']:.1f}" if st["mean"] is not None else ""
                        )
                    writer.writerow(row)
        print(f"CSV written to {csv_file}")

    def to_latex(self, output_path: Path) -> None:
        if not self._stats:
            self.scan_and_compute()

        tex_file = Path(output_path) / "table3_rq3_scale.tex"
        tex_file.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = []
        lines.append(r"\begin{tabular}{l l r r r r r r}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Benchmark} & \textbf{Method} "
            r"& \multicolumn{6}{c}{\textbf{Violations at scale factor $s$}} \\"
        )
        lines.append(r"\cmidrule(lr){3-8}")
        scale_cols = " & ".join(f"${s}$" for s in SCALE_FACTORS)
        lines.append(f" & & {scale_cols} \\\\")
        lines.append(r"\midrule")

        for bi, benchmark in enumerate(BENCHMARK_ORDER):
            bm_display = BENCHMARK_DISPLAY.get(benchmark, benchmark)
            for mi, method in enumerate(METHODS):
                bm_col = bm_display if mi == 0 else ""
                m_col = METHOD_LATEX.get(method, method)
                cells = []
                for scale in SCALE_FACTORS:
                    st = self._stats[benchmark][method][scale]
                    cells.append(self._fmt_latex(st))
                line = f"{bm_col:<25s} & {m_col:<30s} & " + " & ".join(cells) + r" \\"
                lines.append(line)
            if bi < len(BENCHMARK_ORDER) - 1:
                lines.append(r"\addlinespace")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        with open(tex_file, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"LaTeX table written to {tex_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate RQ3 scale sensitivity results (aniso + iso).",
    )
    parser.add_argument("results_dir", type=str)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else results_dir

    aggregator = ResultsAggregator(results_dir)
    aggregator.scan_and_compute()

    total = sum(
        st["runs"]
        for bm in aggregator._stats.values()
        for m in bm.values()
        for st in m.values()
    )
    if total == 0:
        print("WARNING: no results found", file=sys.stderr)
    else:
        print(f"Found {total} run(s).\n")

    aggregator.to_text()
    print()
    aggregator.to_csv(output_dir)
    aggregator.to_latex(output_dir)


if __name__ == "__main__":
    main()
