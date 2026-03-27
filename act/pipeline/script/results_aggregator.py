import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from act.util.path_config import get_project_root

# ---------------------------------------------------------------------------
# Constants — ordering and display names matching paper Table 1
# ---------------------------------------------------------------------------
BENCHMARK_ORDER: List[str] = [
    "trafficsigns",
    "cifar100",
    "tinyimagenet",
]

BENCHMARK_DISPLAY: Dict[str, str] = {
    "cifar100": r"\textsc{Cifar100}",
    "tinyimagenet": r"\textsc{TinyImageNet}",
    "trafficsigns": r"\textsc{TrafficSigns}",
}

METHOD_ORDER: List[str] = [
    "seq_fix",
    "batch_iso",
    "batch_aniso",
]

# paper_name values from rq1_config.py METHODS dict
METHOD_PAPER_NAME: Dict[str, str] = {
    "batch_aniso": r"\textbf{\sysname}",
    "batch_iso": r"\textsc{Batch\text{-}Iso}",
    "seq_fix": r"\textsc{Seq\text{-}Fixed}",
}

METHOD_TEXT_NAME: Dict[str, str] = {
    "batch_aniso": "Batch-Ani",
    "batch_iso": "Batch-Iso",
    "seq_fix": "Seq-Fixed",
}


# ---------------------------------------------------------------------------
# Pure-Python statistics helpers (no numpy dependency)
# ---------------------------------------------------------------------------
def _mean(values: List[float]) -> float:
    """Compute arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: List[float]) -> float:
    """Compute population standard deviation."""
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    variance = sum((x - avg) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# ResultsAggregator
# ---------------------------------------------------------------------------
class ResultsAggregator:
    """Scan experiment JSON results, compute stats, and emit tables."""

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)
        self._raw: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self._stats: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # scan_results
    # ------------------------------------------------------------------
    def scan_results(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Scan ``{results_dir}/{benchmark}/{method}/run_*.json``.

        Returns:
            Nested dict ``{benchmark: {method: [run_dict, ...]}}``.
        """
        results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        for benchmark in BENCHMARK_ORDER:
            bm_dir = self.results_dir / benchmark
            if not bm_dir.is_dir():
                continue
            results.setdefault(benchmark, {})

            for method in METHOD_ORDER:
                method_dir = bm_dir / method
                if not method_dir.is_dir():
                    continue

                run_files = sorted(method_dir.glob("run_*.json"))
                runs: List[Dict[str, Any]] = []
                for rf in run_files:
                    try:
                        with open(rf, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        runs.append(data)
                    except (json.JSONDecodeError, OSError) as exc:
                        print(
                            f"WARNING: skipping {rf} — {exc}",
                            file=sys.stderr,
                        )
                if runs:
                    results[benchmark][method] = runs

        self._raw = results
        return results

    # ------------------------------------------------------------------
    # compute_stats
    # ------------------------------------------------------------------
    def compute_stats(self) -> List[Dict[str, Any]]:
        """Compute summary statistics for every (benchmark, method) pair.

        Must call :meth:`scan_results` first.

        Returns:
            List of dicts (one per benchmark×method), sorted by paper order.
        """
        if not self._raw:
            self.scan_results()

        stats: List[Dict[str, Any]] = []

        # Pre-compute throughput lookup for speedup calculation
        throughput_lookup: Dict[str, Dict[str, float]] = {}

        for benchmark in BENCHMARK_ORDER:
            bm_data = self._raw.get(benchmark, {})
            throughput_lookup[benchmark] = {}

            for method in METHOD_ORDER:
                runs = bm_data.get(method, [])
                if not runs:
                    continue
                tp_vals = [
                    r["throughput"] for r in runs if r.get("throughput") is not None
                ]
                if tp_vals:
                    throughput_lookup[benchmark][method] = _mean(tp_vals)

        # Build stats rows
        for benchmark in BENCHMARK_ORDER:
            bm_data = self._raw.get(benchmark, {})

            for method in METHOD_ORDER:
                runs = bm_data.get(method, [])
                num_runs = len(runs)

                if num_runs == 0:
                    stats.append(
                        {
                            "benchmark": benchmark,
                            "method": method,
                            "violations_mean": None,
                            "violations_std": None,
                            "coverage_mean": None,
                            "coverage_std": None,
                            "ttfv_mean": None,
                            "ttfv_std": None,
                            "throughput_mean": None,
                            "throughput_std": None,
                            "num_runs": 0,
                            "speedup": None,
                        }
                    )
                    continue

                violations = [
                    r["violations"] for r in runs if r.get("violations") is not None
                ]
                coverages = [
                    r["coverage"] for r in runs if r.get("coverage") is not None
                ]
                ttfvs = [r["ttfv"] for r in runs if r.get("ttfv") is not None]
                throughputs = [
                    r["throughput"] for r in runs if r.get("throughput") is not None
                ]

                # Speedup: batch_aniso throughput / seq_fix throughput
                speedup: Optional[float] = None
                if method == "batch_aniso":
                    ba_tp = throughput_lookup.get(benchmark, {}).get("batch_aniso")
                    sr_tp = throughput_lookup.get(benchmark, {}).get("seq_fix")
                    if ba_tp is not None and sr_tp is not None and sr_tp > 0:
                        speedup = ba_tp / sr_tp

                stats.append(
                    {
                        "benchmark": benchmark,
                        "method": method,
                        "violations_mean": _mean(violations) if violations else None,
                        "violations_std": _std(violations) if violations else None,
                        "coverage_mean": _mean(coverages) if coverages else None,
                        "coverage_std": _std(coverages) if coverages else None,
                        "ttfv_mean": _mean(ttfvs) if ttfvs else None,
                        "ttfv_std": _std(ttfvs) if ttfvs else None,
                        "throughput_mean": _mean(throughputs) if throughputs else None,
                        "throughput_std": _std(throughputs) if throughputs else None,
                        "num_runs": num_runs,
                        "speedup": speedup,
                    }
                )

        self._stats = stats
        return stats

    # ------------------------------------------------------------------
    # LaTeX formatting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_violations(
        mean: Optional[float], std: Optional[float], num_runs: int
    ) -> str:
        if mean is None:
            return "---"
        m = int(round(mean))
        if num_runs <= 1 or std is None or std == 0.0:
            return str(m)
        s = int(round(std))
        return f"${m} \\pm {s}$"

    @staticmethod
    def _fmt_coverage(
        mean: Optional[float], std: Optional[float], num_runs: int
    ) -> str:
        if mean is None:
            return "---"
        pct = mean * 100.0
        if num_runs <= 1 or std is None or std == 0.0:
            return f"{pct:.1f}"
        spct = std * 100.0
        return f"${pct:.1f} \\pm {spct:.1f}$"

    @staticmethod
    def _fmt_ttfv(mean: Optional[float], std: Optional[float], num_runs: int) -> str:
        if mean is None:
            return "---"
        if num_runs <= 1 or std is None or std == 0.0:
            return f"{mean:.2f}"
        return f"${mean:.2f} \\pm {std:.2f}$"

    @staticmethod
    def _fmt_throughput(
        mean: Optional[float], std: Optional[float], num_runs: int
    ) -> str:
        if mean is None:
            return "---"
        if num_runs <= 1 or std is None or std == 0.0:
            return f"{mean:.1f}"
        return f"${mean:.1f} \\pm {std:.1f}$"

    # ------------------------------------------------------------------
    # to_latex
    # ------------------------------------------------------------------
    def to_latex(self, output_path: Path) -> None:
        """Write ``table1_latex.tex`` matching the paper Table 1 format."""
        if not self._stats:
            self.compute_stats()

        tex_file = Path(output_path) / "table1_latex.tex"
        tex_file.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = []
        lines.append(r"\begin{tabular}{l l r r r}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Benchmark} & \textbf{Method} "
            r"& \textbf{Violations} & \textbf{TTFV\,(s)} & \textbf{Thpt.\,(it/s)} \\"
        )
        lines.append(r"\midrule")

        # Build a fast lookup: (benchmark, method) -> row dict
        lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in self._stats:
            lookup[(row["benchmark"], row["method"])] = row

        for bm_idx, benchmark in enumerate(BENCHMARK_ORDER):
            bm_display = BENCHMARK_DISPLAY.get(benchmark, benchmark)
            lines.append(bm_display)

            for method in METHOD_ORDER:
                row = lookup.get((benchmark, method))
                if row is None or row["num_runs"] == 0:
                    paper_name = METHOD_PAPER_NAME.get(method, method)
                    lines.append(f"  & {paper_name}  & --- & --- & --- \\\\")
                    continue

                nr = row["num_runs"]
                viol = self._fmt_violations(
                    row["violations_mean"], row["violations_std"], nr
                )
                ttfv = self._fmt_ttfv(row["ttfv_mean"], row["ttfv_std"], nr)
                thpt = self._fmt_throughput(
                    row["throughput_mean"], row["throughput_std"], nr
                )
                paper_name = METHOD_PAPER_NAME.get(method, method)

                lines.append(f"  & {paper_name:<35s} & {viol} & {ttfv} & {thpt} \\\\")

            # \midrule between benchmarks, \bottomrule after the last
            if bm_idx < len(BENCHMARK_ORDER) - 1:
                lines.append(r"\midrule")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        with open(tex_file, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        print(f"LaTeX table written to {tex_file}")

    # ------------------------------------------------------------------
    # to_text
    # ------------------------------------------------------------------
    def to_text(self) -> None:
        """Print a formatted text table to stdout for quick inspection."""
        if not self._stats:
            self.compute_stats()

        # Column widths
        w_bm = 15
        w_method = 14
        w_viol = 16
        w_cov = 16
        w_ttfv = 14
        w_thpt = 16

        sep = "-" * (w_bm + w_method + w_viol + w_cov + w_ttfv + w_thpt + 5)

        header = (
            f"{'Benchmark':<{w_bm}} "
            f"{'Method':<{w_method}} "
            f"{'Violations':>{w_viol}} "
            f"{'Coverage(%)':>{w_cov}} "
            f"{'TTFV(s)':>{w_ttfv}} "
            f"{'Thpt(it/s)':>{w_thpt}}"
        )

        header = (
            f"{'Benchmark':<{w_bm}} "
            f"{'Method':<{w_method}} "
            f"{'Violations':>{w_viol}} "
            f"{'Coverage(%)':>{w_cov}} "
            f"{'TTFV(s)':>{w_ttfv}} "
            f"{'Thpt(it/s)':>{w_thpt}}"
        )

        print(sep)
        print(header)
        print(sep)

        lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in self._stats:
            lookup[(row["benchmark"], row["method"])] = row

        for bm_idx, benchmark in enumerate(BENCHMARK_ORDER):
            for m_idx, method in enumerate(METHOD_ORDER):
                row = lookup.get((benchmark, method))
                bm_label = benchmark if m_idx == 0 else ""
                method_label = METHOD_TEXT_NAME.get(method, method)

                if row is None or row["num_runs"] == 0:
                    print(
                        f"{bm_label:<{w_bm}} "
                        f"{method_label:<{w_method}} "
                        f"{'---':>{w_viol}} "
                        f"{'---':>{w_cov}} "
                        f"{'---':>{w_ttfv}} "
                        f"{'---':>{w_thpt}}"
                    )
                    continue

                nr = row["num_runs"]

                # Violations
                if row["violations_mean"] is not None:
                    vm = int(round(row["violations_mean"]))
                    if nr > 1 and row["violations_std"] and row["violations_std"] > 0:
                        vs = int(round(row["violations_std"]))
                        viol_str = f"{vm}±{vs}"
                    else:
                        viol_str = str(vm)
                else:
                    viol_str = "---"

                # Coverage
                if row["coverage_mean"] is not None:
                    cm = row["coverage_mean"] * 100.0
                    if nr > 1 and row["coverage_std"] and row["coverage_std"] > 0:
                        cs = row["coverage_std"] * 100.0
                        cov_str = f"{cm:.2f}±{cs:.2f}"
                    else:
                        cov_str = f"{cm:.2f}"
                else:
                    cov_str = "---"

                # TTFV
                if row["ttfv_mean"] is not None:
                    if nr > 1 and row["ttfv_std"] and row["ttfv_std"] > 0:
                        ttfv_str = f"{row['ttfv_mean']:.1f}±{row['ttfv_std']:.1f}"
                    else:
                        ttfv_str = f"{row['ttfv_mean']:.1f}"
                else:
                    ttfv_str = "---"

                # Throughput
                if row["throughput_mean"] is not None:
                    if nr > 1 and row["throughput_std"] and row["throughput_std"] > 0:
                        thpt_str = (
                            f"{row['throughput_mean']:.1f}±{row['throughput_std']:.1f}"
                        )
                    else:
                        thpt_str = f"{row['throughput_mean']:.1f}"
                else:
                    thpt_str = "---"

                # Speedup
                if row["speedup"] is not None:
                    spd_str = f"{row['speedup']:.1f}x"
                else:
                    spd_str = "---"

                print(
                    f"{bm_label:<{w_bm}} "
                    f"{method_label:<{w_method}} "
                    f"{viol_str:>{w_viol}} "
                    f"{cov_str:>{w_cov}} "
                    f"{ttfv_str:>{w_ttfv}} "
                    f"{thpt_str:>{w_thpt}}"
                )

            if bm_idx < len(BENCHMARK_ORDER) - 1:
                print()

        print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point: parse args, aggregate, and output."""
    parser = argparse.ArgumentParser(
        description="Aggregate RQ1 experiment results into CSV, LaTeX, and text tables.",
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to results directory (e.g. experiments/rq1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output directory for CSV/LaTeX files (default: same as results_dir)",
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

    output_dir = (
        Path(args.output) if args.output else Path(get_project_root()) / "figures"
    )

    aggregator = ResultsAggregator(results_dir)
    raw = aggregator.scan_results()

    # Check if any results were found
    total_runs = sum(len(runs) for bm_data in raw.values() for runs in bm_data.values())
    if total_runs == 0:
        print(
            f"ERROR: no run_*.json files found under {results_dir}/{{benchmark}}/{{method}}/",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {total_runs} run(s) across {len(raw)} benchmark(s).\n")

    aggregator.compute_stats()
    aggregator.to_text()
    print()
    aggregator.to_latex(output_dir)


if __name__ == "__main__":
    main()
