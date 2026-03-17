"""
RQ2 Unique Violation Analysis.

Compares two fuzzing methods to determine which specification instances
are solved by each, producing Table tab:unique-viol from the paper.

A spec instance is "solved" if at least one counterexample was found
for it across ANY of the 5 runs.

Usage:
    python act/pipeline/script/rq2_unique_violations.py <results_dir>
    python act/pipeline/script/rq2_unique_violations.py experiments/rq1 --method-a batch_aniso --method-b batch_iso
    python act/pipeline/script/rq2_unique_violations.py experiments/rq1 --output reports/

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants — ordering and display names matching paper Table tab:unique-viol
# ---------------------------------------------------------------------------
BENCHMARK_ORDER: List[str] = [
    "acasxu",
    "cifar100",
    "tinyimagenet",
    "mnist",
    "trafficsigns",
]

BENCHMARK_DISPLAY: Dict[str, str] = {
    "acasxu": r"\textsc{AcasXu}",
    "cifar100": r"\textsc{Cifar100}",
    "tinyimagenet": r"\textsc{TinyImageNet}",
    "mnist": r"\textsc{Mnist}",
    "trafficsigns": r"\textsc{TrafficSigns}",
}

# Expected instance counts (from rq1_config.py)
EXPECTED_INSTANCES: Dict[str, int] = {
    "acasxu": 185,
    "cifar100": 199,
    "tinyimagenet": 199,
    "mnist": 30,
    "trafficsigns": 44,
}

CSV_COLUMNS: List[str] = [
    "benchmark",
    "specs",
    "method_a_name",
    "method_a_solved",
    "method_b_name",
    "method_b_solved",
    "only_a",
    "only_b",
    "both",
]


# ---------------------------------------------------------------------------
# UniqueViolationAnalyzer
# ---------------------------------------------------------------------------
class UniqueViolationAnalyzer:
    """Analyze per-instance violation data to find unique solves per method."""

    def __init__(self, results_dir: Path, method_a: str, method_b: str) -> None:
        self.results_dir = Path(results_dir)
        self.method_a = method_a
        self.method_b = method_b
        self._stats: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # _load_solved_indices
    # ------------------------------------------------------------------
    def _load_solved_indices(self, benchmark: str, method: str) -> Set[int]:
        """Load union of violated_seed_indices across all runs for a (benchmark, method).

        Returns:
            Set of spec instance indices that were solved in at least one run.
        """
        method_dir = self.results_dir / benchmark / method
        if not method_dir.is_dir():
            return set()

        solved: Set[int] = set()
        for rf in sorted(method_dir.glob("run_*.json")):
            try:
                with open(rf, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"WARNING: skipping {rf} — {exc}", file=sys.stderr)
                continue

            # Collect from top-level all_violated_indices (if present)
            if "all_violated_indices" in data:
                solved.update(data["all_violated_indices"])

            # Also collect from per_group_results
            for group in data.get("per_group_results", []):
                for idx in group.get("violated_seed_indices", []):
                    solved.add(idx)

        return solved

    # ------------------------------------------------------------------
    # compute_stats
    # ------------------------------------------------------------------
    def compute_stats(self) -> List[Dict[str, Any]]:
        """Compute unique violation stats for each benchmark.

        Returns:
            List of dicts, one per benchmark, with overlap analysis.
        """
        stats: List[Dict[str, Any]] = []

        for benchmark in BENCHMARK_ORDER:
            bm_dir = self.results_dir / benchmark
            if not bm_dir.is_dir():
                stats.append(
                    {
                        "benchmark": benchmark,
                        "specs": EXPECTED_INSTANCES.get(benchmark, 0),
                        "a_solved": 0,
                        "b_solved": 0,
                        "only_a": 0,
                        "only_b": 0,
                        "both": 0,
                        "has_data": False,
                    }
                )
                continue

            solved_a = self._load_solved_indices(benchmark, self.method_a)
            solved_b = self._load_solved_indices(benchmark, self.method_b)

            only_a = solved_a - solved_b
            only_b = solved_b - solved_a
            both = solved_a & solved_b

            stats.append(
                {
                    "benchmark": benchmark,
                    "specs": EXPECTED_INSTANCES.get(benchmark, 0),
                    "a_solved": len(solved_a),
                    "b_solved": len(solved_b),
                    "only_a": len(only_a),
                    "only_b": len(only_b),
                    "both": len(both),
                    "has_data": bool(solved_a or solved_b),
                }
            )

        self._stats = stats
        return stats

    # ------------------------------------------------------------------
    # to_text
    # ------------------------------------------------------------------
    def to_text(self) -> None:
        """Print text table to stdout."""
        if not self._stats:
            self.compute_stats()

        a_name = self.method_a.replace("_", "-").title()
        b_name = self.method_b.replace("_", "-").title()

        w_bm = 15
        w_sp = 6
        w_val = 12

        sep = "-" * (w_bm + w_sp + w_val * 5 + 6)

        header = (
            f"{'Benchmark':<{w_bm}} "
            f"{'Specs':>{w_sp}} "
            f"{a_name + ' solved':>{w_val}} "
            f"{b_name + ' solved':>{w_val}} "
            f"{'Only ' + a_name:>{w_val}} "
            f"{'Only ' + b_name:>{w_val}} "
            f"{'Both':>{w_val}}"
        )

        print(sep)
        print(header)
        print(sep)

        for row in self._stats:
            bm = row["benchmark"]
            if not row["has_data"]:
                print(
                    f"{bm:<{w_bm}} "
                    f"{row['specs']:>{w_sp}} "
                    f"{'---':>{w_val}} "
                    f"{'---':>{w_val}} "
                    f"{'---':>{w_val}} "
                    f"{'---':>{w_val}} "
                    f"{'---':>{w_val}}"
                )
            else:
                print(
                    f"{bm:<{w_bm}} "
                    f"{row['specs']:>{w_sp}} "
                    f"{row['a_solved']:>{w_val}} "
                    f"{row['b_solved']:>{w_val}} "
                    f"{row['only_a']:>{w_val}} "
                    f"{row['only_b']:>{w_val}} "
                    f"{row['both']:>{w_val}}"
                )

        print(sep)

    # ------------------------------------------------------------------
    # to_csv
    # ------------------------------------------------------------------
    def to_csv(self, output_path: Path) -> None:
        """Write unique_violations.csv."""
        if not self._stats:
            self.compute_stats()

        csv_file = Path(output_path) / "unique_violations.csv"
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_file, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in self._stats:
                writer.writerow(
                    {
                        "benchmark": row["benchmark"],
                        "specs": row["specs"],
                        "method_a_name": self.method_a,
                        "method_a_solved": row["a_solved"],
                        "method_b_name": self.method_b,
                        "method_b_solved": row["b_solved"],
                        "only_a": row["only_a"],
                        "only_b": row["only_b"],
                        "both": row["both"],
                    }
                )

        print(f"CSV written to {csv_file}")

    # ------------------------------------------------------------------
    # to_latex
    # ------------------------------------------------------------------
    def to_latex(self, output_path: Path) -> None:
        """Write table2_unique_viol.tex matching paper Table tab:unique-viol."""
        if not self._stats:
            self.compute_stats()

        a_label = (
            self.method_a.replace("_", r"\text{-}")
            .replace("batch", r"Batch")
            .replace("seq", r"Seq")
        )
        b_label = (
            self.method_b.replace("_", r"\text{-}")
            .replace("batch", r"Batch")
            .replace("seq", r"Seq")
        )

        tex_file = Path(output_path) / "table2_unique_viol.tex"
        tex_file.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = []
        lines.append(r"\begin{tabular}{l r r r r r}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Benchmark} & \textbf{Specs} "
            rf"& \textbf{{{a_label} solved}} & \textbf{{{b_label} solved}} "
            rf"& \textbf{{Only {a_label}}} & \textbf{{Only {b_label}}} \\"
        )
        lines.append(r"\midrule")

        for i, row in enumerate(self._stats):
            bm_display = BENCHMARK_DISPLAY.get(row["benchmark"], row["benchmark"])
            if not row["has_data"]:
                lines.append(
                    f"{bm_display:<25s} & {row['specs']} & --- & --- & --- & --- \\\\"
                )
            else:
                lines.append(
                    f"{bm_display:<25s} & {row['specs']} "
                    f"& {row['a_solved']} & {row['b_solved']} "
                    f"& {row['only_a']} & {row['only_b']} \\\\"
                )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        with open(tex_file, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        print(f"LaTeX table written to {tex_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RQ2: Unique violation analysis — compare two fuzzing methods.",
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to RQ1 results directory (e.g., experiments/rq1)",
    )
    parser.add_argument(
        "--method-a",
        type=str,
        default="batch_aniso",
        help="First method to compare (default: batch_aniso)",
    )
    parser.add_argument(
        "--method-b",
        type=str,
        default="batch_iso",
        help="Second method to compare (default: batch_iso)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output directory for CSV/LaTeX (default: same as results_dir)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(
            f"ERROR: results directory does not exist: {results_dir}", file=sys.stderr
        )
        sys.exit(1)
    if not results_dir.is_dir():
        print(f"ERROR: not a directory: {results_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else results_dir

    analyzer = UniqueViolationAnalyzer(results_dir, args.method_a, args.method_b)
    analyzer.compute_stats()

    # Check if any data found
    has_any = any(row["has_data"] for row in analyzer._stats)
    if not has_any:
        print(
            f"WARNING: no violation data found for methods '{args.method_a}' "
            f"and '{args.method_b}' under {results_dir}",
            file=sys.stderr,
        )

    analyzer.to_text()
    print()
    analyzer.to_csv(output_dir)
    analyzer.to_latex(output_dir)


if __name__ == "__main__":
    main()
