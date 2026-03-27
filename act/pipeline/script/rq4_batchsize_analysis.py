from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math

from scipy.stats import mannwhitneyu

from act.util.path_config import get_project_root

# ---------------------------------------------------------------------------
# Per-benchmark configuration
# ---------------------------------------------------------------------------
BENCHMARK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "tinyimagenet": {
        "b_values": [1, 10, 50, 100, 199],
        "label": "TinyImageNet",
    },
    "cifar100": {
        "b_values": [1, 10, 50, 99],
        "label": "CIFAR-100",
    },
}


def get_configs_for_benchmark(
    benchmark: str,
) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Generate (method, B) configs and labels for a benchmark.

    Args:
        benchmark: Key into ``BENCHMARK_CONFIGS``.

    Returns:
        (configs, labels) where configs is a list of (method, B) tuples
        and labels is a list of human-readable column/row labels.
    """
    b_values = BENCHMARK_CONFIGS[benchmark]["b_values"]
    configs: List[Tuple[str, int]] = []
    labels: List[str] = []
    for method in ["batch_iso", "batch_aniso"]:
        prefix = "Iso" if "iso" in method and "aniso" not in method else "Ani"
        for b in b_values:
            configs.append((method, b))
            labels.append(f"{prefix}-{b}")
    return configs, labels


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def cohens_d(group_a: List[float], group_b: List[float]) -> float:
    """Cohen's d effect size (pooled standard deviation).

    d = (mean_A - mean_B) / pooled_std.
    Positive d indicates group_a has larger values; negative indicates group_b.
    Returns 0.0 if pooled std is zero (identical groups).
    Conventional thresholds: small |d|≥0.2, medium |d|≥0.5, large |d|≥0.8.
    """
    n_a, n_b = len(group_a), len(group_b)
    mean_a = sum(group_a) / n_a
    mean_b = sum(group_b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in group_a) / (n_a - 1) if n_a > 1 else 0.0
    var_b = sum((x - mean_b) ** 2 for x in group_b) / (n_b - 1) if n_b > 1 else 0.0
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0.0:
        return 0.0
    return (mean_a - mean_b) / pooled_std


def benjamini_hochberg(
    p_values: List[Tuple[int, int, float]],
) -> Dict[Tuple[int, int], float]:
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Less conservative than Holm-Bonferroni; controls the false discovery rate
    rather than the family-wise error rate.  Preferred when the number of
    comparisons is large relative to the minimum achievable p-value (e.g.,
    Mann-Whitney with small n and many pairs).

    Args:
        p_values: List of (i, j, raw_p) triples.

    Returns:
        Dict mapping (i, j) → BH-adjusted p-value.
    """
    n = len(p_values)
    if n == 0:
        return {}

    # Sort by p descending for the step-up procedure
    sorted_pairs = sorted(p_values, key=lambda x: x[2], reverse=True)
    adjusted: List[Tuple[int, int, float]] = []
    running_min = 1.0
    for rank, (i, j, p) in enumerate(sorted_pairs):
        # BH adjusted p = p * n / (n - rank)
        p_adj = min(p * n / (n - rank), 1.0)
        running_min = min(running_min, p_adj)
        adjusted.append((i, j, running_min))

    result: Dict[Tuple[int, int], float] = {}
    for i, j, p_adj in adjusted:
        result[(i, j)] = p_adj
    return result


def get_symbol(d: float, p_adj: float) -> str:
    """Assign comparison symbol for (row, col) pair using Cohen's d.

    Args:
        d: Cohen's d for row vs col (positive → row has larger mean).
        p_adj: BH-adjusted p-value.

    Returns:
        Symbol indicating row vs col:
        - ``≡``: no significant difference or negligible effect (|d| < 6.0)
        - ``✓`` / ``✓✓`` / ``✓✓✓``: row significantly better (small/medium/large)
        - ``✗`` / ``✗✗`` / ``✗✗✗``: row significantly worse (small/medium/large)
    """
    if p_adj >= 0.05:
        return "≡"
    abs_d = abs(d)
    if abs_d < 6.0:
        return "≡"  # negligible effect size despite significant p
    if d > 0:  # row better (higher violations/spec for row)
        if abs_d >= 13.0:
            return "✓✓✓"
        if abs_d >= 9.0:
            return "✓✓"
        return "✓"
    else:  # col better
        if abs_d >= 13.0:
            return "✗✗✗"
        if abs_d >= 9.0:
            return "✗✗"
        return "✗"


def flip_symbol(sym: str) -> str:
    """Mirror a symbol: ✓ ↔ ✗, preserving count."""
    return sym.replace("✓", "X").replace("✗", "✓").replace("X", "✗")


# ---------------------------------------------------------------------------
# BatchSizeAnalyzer
# ---------------------------------------------------------------------------
class BatchSizeAnalyzer:
    """Analyze batch-size experiment results with pairwise statistical tests."""

    def __init__(self, results_dir: str, benchmark: str = "tinyimagenet") -> None:
        self.results_dir = Path(results_dir)
        self.benchmark = benchmark
        self.configs, self.labels = get_configs_for_benchmark(benchmark)
        self.benchmark_label = BENCHMARK_CONFIGS[benchmark]["label"]
        # (method, B) → list of violations_per_spec floats
        self._data: Dict[Tuple[str, int], List[float]] = {}
        # N×N symbol matrix
        self._matrix: List[List[str]] = []
        # N×N Cohen's d matrix
        self._d_matrix: List[List[Optional[float]]] = []

    # ------------------------------------------------------------------
    # scan_results
    # ------------------------------------------------------------------
    def scan_results(self) -> Dict[Tuple[str, int], List[float]]:
        """Scan ``{results_dir}/B_*/{benchmark}/*/run_*.json``.

        Groups results by (method, B) with violations_per_spec metric.

        Returns:
            Dict mapping (method, B) → list of violations_per_spec values.
        """
        data: Dict[Tuple[str, int], List[float]] = {}
        pattern = f"B_*/{self.benchmark}/*/run_*.json"

        for run_file in sorted(self.results_dir.glob(pattern)):
            parts = run_file.parts
            # Extract B from path: e.g. B_001 → 1
            b_dir = parts[-4]
            try:
                batch_size = int(b_dir.replace("B_", ""))
            except ValueError:
                print(
                    f"WARNING: cannot parse batch size from {b_dir}, skipping {run_file}",
                    file=sys.stderr,
                )
                continue

            # Extract method from path
            method = parts[-2]

            try:
                with open(run_file, "r", encoding="utf-8") as fh:
                    result = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"WARNING: skipping {run_file} — {exc}", file=sys.stderr)
                continue

            violations = result.get("violations", 0)
            instances_processed = result.get("instances_processed", 0)
            violations_per_spec = violations / max(instances_processed, 1)

            key = (method, batch_size)
            data.setdefault(key, []).append(violations_per_spec)

        self._data = data
        return data

    # ------------------------------------------------------------------
    # compute_matrix
    # ------------------------------------------------------------------
    def compute_matrix(self) -> List[List[str]]:
        """Compute N×N pairwise comparison matrix.

        Uses Mann-Whitney U (exact, two-sided) with Benjamini-Hochberg FDR
        correction and Cohen's d effect size.

        Returns:
            N×N list of symbol strings.
        """
        if not self._data:
            self.scan_results()

        n = len(self.configs)
        # Gather data vectors in config order
        vectors: List[List[float]] = []
        for method, b in self.configs:
            vectors.append(self._data.get((method, b), []))

        # Phase 1: Compute raw p-values and Cohen's d for all unique pairs
        raw_pvalues: List[Tuple[int, int, float]] = []
        d_cache: Dict[Tuple[int, int], float] = {}

        for i in range(n):
            for j in range(i + 1, n):
                if not vectors[i] or not vectors[j]:
                    continue
                d = cohens_d(vectors[i], vectors[j])
                d_cache[(i, j)] = d
                try:
                    _, p = mannwhitneyu(
                        vectors[i],
                        vectors[j],
                        alternative="two-sided",
                        method="exact",
                    )
                except ValueError:
                    # Fallback for edge cases (e.g., all identical values)
                    p = 1.0
                raw_pvalues.append((i, j, p))

        # Phase 2: Benjamini-Hochberg FDR correction
        adjusted = benjamini_hochberg(raw_pvalues)

        # Phase 3: Build full symmetric matrix
        matrix: List[List[str]] = [["" for _ in range(n)] for _ in range(n)]
        d_matrix: List[List[Optional[float]]] = [
            [None for _ in range(n)] for _ in range(n)
        ]

        for i in range(n):
            matrix[i][i] = "—"
            d_matrix[i][i] = None

        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in d_cache:
                    # Missing data — mark as cannot compare
                    matrix[i][j] = "?"
                    matrix[j][i] = "?"
                    continue

                d = d_cache[(i, j)]
                p_adj = adjusted.get((i, j), 1.0)
                sym = get_symbol(d, p_adj)

                # (i, j) in upper triangle: row=i vs col=j
                matrix[i][j] = sym
                matrix[j][i] = flip_symbol(sym)
                d_matrix[i][j] = d
                d_matrix[j][i] = -d

        self._matrix = matrix
        self._d_matrix = d_matrix
        return matrix

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(self, format: str = "text") -> None:
        """Execute full pipeline: scan → compute → output."""
        self.scan_results()

        # Check if any data found
        if not self._data:
            print(
                f"ERROR: no run_*.json files found under "
                f"{self.results_dir}/B_*/{self.benchmark}/*/",
                file=sys.stderr,
            )
            sys.exit(1)

        # Report data summary
        total_runs = sum(len(v) for v in self._data.values())
        configs_found = len(self._data)
        print(
            f"[{self.benchmark_label}] "
            f"Found {total_runs} run(s) across {configs_found} configuration(s).\n"
        )

        self.compute_matrix()

        if format == "text":
            self.to_text()
        elif format == "latex":
            self.to_latex(self.results_dir)
        elif format == "csv":
            self.to_csv(self.results_dir)
        else:
            print(f"ERROR: unknown format '{format}'", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------
    # to_text
    # ------------------------------------------------------------------
    def to_text(self) -> None:
        """Print text table to stdout."""
        if not self._matrix:
            self.compute_matrix()

        n = len(self.configs)
        col_w = 8
        label_w = 8

        # Benchmark header
        print(f"\n=== {self.benchmark_label} ===\n")

        # Header
        header = f"{'':>{label_w}}"
        for label in self.labels:
            header += f" {label:>{col_w}}"
        sep = "-" * len(header)

        print(sep)
        print(header)
        print(sep)

        for i in range(n):
            row_str = f"{self.labels[i]:>{label_w}}"
            for j in range(n):
                row_str += f" {self._matrix[i][j]:>{col_w}}"
            print(row_str)

        print(sep)

    # ------------------------------------------------------------------
    # to_csv
    # ------------------------------------------------------------------
    def to_csv(self, output_path: Path) -> None:
        """Write pairwise comparison CSV."""
        if not self._matrix:
            self.compute_matrix()

        csv_file = Path(output_path) / f"rq_batchsize_stats_{self.benchmark}.csv"
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        n = len(self.configs)
        with open(csv_file, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([""] + self.labels)
            for i in range(n):
                writer.writerow([self.labels[i]] + self._matrix[i])

        print(f"CSV written to {csv_file}")

    # ------------------------------------------------------------------
    # to_latex
    # ------------------------------------------------------------------
    def to_latex(self, output_path: Path) -> None:
        """Write rq_batchsize_table_{benchmark}.tex with booktabs N×N comparison matrix."""
        if not self._matrix:
            self.compute_matrix()

        tex_file = Path(output_path) / f"rq_batchsize_table_{self.benchmark}.tex"
        tex_file.parent.mkdir(parents=True, exist_ok=True)

        n = len(self.configs)
        col_spec = "l " + "c" * n
        lines: List[str] = []

        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(
            r"\caption{Pairwise comparison of batch-size configurations on "
            + self.benchmark_label
            + r" (violations/spec)."
        )
        lines.append(
            r"  \ding{61}: no significant difference; "
            r"\ding{51}/\ding{51}\ding{51}/\ding{51}\ding{51}\ding{51}: "
            r"row significantly better (small/medium/large effect);"
        )
        lines.append(
            r"  \ding{55}/\ding{55}\ding{55}/\ding{55}\ding{55}\ding{55}: "
            r"row significantly worse. Wilcoxon rank-sum test with "
            r"Benjamini-Hochberg FDR correction ($\alpha=0.05$);"
        )
        lines.append(
            r"  effect size: Cohen's $d$ (small $|d|\geq6$, medium $|d|\geq9$, large $|d|\geq13$).}"
        )
        lines.append(rf"\label{{tab:batchsize-{self.benchmark}}}")
        lines.append(r"\small")
        lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")

        # Header row
        header = " "
        for label in self.labels:
            header += f" & {label}"
        header += r" \\"
        lines.append(header)
        lines.append(r"\midrule")

        # Symbol → LaTeX mapping
        sym_map = {
            "—": "---",
            "≡": r"$\equiv$",
            "✓": r"\ding{51}",
            "✓✓": r"\ding{51}\ding{51}",
            "✓✓✓": r"\ding{51}\ding{51}\ding{51}",
            "✗": r"\ding{55}",
            "✗✗": r"\ding{55}\ding{55}",
            "✗✗✗": r"\ding{55}\ding{55}\ding{55}",
            "?": "?",
        }

        for i in range(n):
            row = self.labels[i]
            for j in range(n):
                sym = self._matrix[i][j]
                tex_sym = sym_map.get(sym, sym)
                row += f" & {tex_sym}"
            row += r" \\"
            lines.append(row)

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        with open(tex_file, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        print(f"LaTeX table written to {tex_file}")


# ---------------------------------------------------------------------------
# Merged LaTeX table (CIFAR-100 + TinyImageNet side by side)
# ---------------------------------------------------------------------------
SYM_MAP = {
    "—": "---",
    "≡": r"$\equiv$",
    "✓": r"\ding{51}",
    "✓✓": r"\ding{51}\ding{51}",
    "✓✓✓": r"\ding{51}\ding{51}\ding{51}",
    "✗": r"\ding{55}",
    "✗✗": r"\ding{55}\ding{55}",
    "✗✗✗": r"\ding{55}\ding{55}\ding{55}",
    "?": "?",
}


def generate_merged_latex(results_dir: str, output_path: Optional[str] = None) -> None:
    """Generate a merged side-by-side LaTeX table for CIFAR-100 and TinyImageNet."""
    # Build both analyzers
    left_bm, right_bm = "cifar100", "tinyimagenet"
    left = BatchSizeAnalyzer(results_dir, benchmark=left_bm)
    right = BatchSizeAnalyzer(results_dir, benchmark=right_bm)
    left.scan_results()
    right.scan_results()

    if not left._data:
        print(f"ERROR: no data found for {left_bm}", file=sys.stderr)
        sys.exit(1)
    if not right._data:
        print(f"ERROR: no data found for {right_bm}", file=sys.stderr)
        sys.exit(1)

    left.compute_matrix()
    right.compute_matrix()

    left.to_text()
    right.to_text()

    n_left = len(left.configs)
    n_right = len(right.configs)
    n_rows = max(n_left, n_right)

    # Column spec: left_label + left_data + gap + right_label + right_data
    col_spec = "l " + "c" * n_left + " c l " + "c" * n_right

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Pairwise comparison of batch-size configurations (violations/spec)."
    )
    lines.append(
        r"  \ding{61}: no significant difference; "
        r"\ding{51}/\ding{51}\ding{51}/\ding{51}\ding{51}\ding{51}: "
        r"row significantly better (small/medium/large effect);"
    )
    lines.append(
        r"  \ding{55}/\ding{55}\ding{55}/\ding{55}\ding{55}\ding{55}: "
        r"row significantly worse. Wilcoxon rank-sum test with "
        r"Benjamini-Hochberg FDR correction ($\alpha=0.05$);"
    )
    lines.append(
        r"  effect size: Cohen's $d$ (small $|d|\geq6$, medium $|d|\geq9$, large $|d|\geq13$).}"
    )
    lines.append(r"\label{tab:batchsize-merged}")
    lines.append(r"\small")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Multicolumn header
    left_col_start = 2
    left_col_end = 1 + n_left
    right_col_start = left_col_end + 3  # +1 gap col, +1 label col, +1 for start
    right_col_end = right_col_start + n_right - 1
    lines.append(
        f"  & \\multicolumn{{{n_left}}}{{c}}{{{left.benchmark_label}}} "
        f"& & & \\multicolumn{{{n_right}}}{{c}}{{{right.benchmark_label}}} \\\\"
    )
    lines.append(
        f"\\cmidrule(lr){{{left_col_start}-{left_col_end}}} "
        f"\\cmidrule(lr){{{right_col_start}-{right_col_end}}}"
    )

    # Sub-header row (column labels)
    header = " "
    for label in left.labels:
        header += f" & {label}"
    header += " & & "
    for label in right.labels:
        header += f" & {label}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Data rows
    for row_idx in range(n_rows):
        row = ""
        # Left side
        if row_idx < n_left:
            row += left.labels[row_idx]
            for j in range(n_left):
                sym = left._matrix[row_idx][j]
                row += f" & {SYM_MAP.get(sym, sym)}"
        else:
            row += " " + " &" * n_left

        # Gap + right label
        if row_idx < n_right:
            row += f" & & {right.labels[row_idx]}"
            for j in range(n_right):
                sym = right._matrix[row_idx][j]
                row += f" & {SYM_MAP.get(sym, sym)}"
        else:
            row += " & &" + " &" * n_right

        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out_dir = Path(output_path) if output_path else Path(get_project_root()) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_file = out_dir / "rq_batchsize_table_merged.tex"
    with open(tex_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Merged LaTeX table written to {tex_file}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def run_self_test() -> None:
    """Built-in validation against known statistical results."""
    errors: List[str] = []

    # ------------------------------------------------------------------
    # Test 1: identical groups → A12=0.5, p≥0.05, symbol=≡
    # ------------------------------------------------------------------
    g1 = [10.0, 10.0, 10.0, 10.0, 10.0]
    g2 = [10.0, 10.0, 10.0, 10.0, 10.0]
    d = cohens_d(g1, g2)
    if d != 0.0:
        errors.append(f"Test1 Cohen's d failed: expected 0.0, got {d}")
    try:
        _, p = mannwhitneyu(g1, g2, alternative="two-sided", method="exact")
    except ValueError:
        p = 1.0
    if p < 0.05:
        errors.append(f"Test1 p failed: expected ≥0.05, got {p}")
    sym = get_symbol(d, p)
    if sym != "≡":
        errors.append(f"Test1 symbol failed: expected ≡, got {sym}")

    # ------------------------------------------------------------------
    # Test 2: well-separated groups → large effect (|d|≥13), p<0.05
    # mean_g3=3, mean_g4=52, pooled_std≈1.58 → d≈-31
    # ------------------------------------------------------------------
    g3 = [1.0, 2.0, 3.0, 4.0, 5.0]
    g4 = [50.0, 51.0, 52.0, 53.0, 54.0]
    d = cohens_d(g3, g4)
    if not abs(d) >= 0.8:
        errors.append(f"Test2 Cohen's d failed: expected |d|≥0.8, got {d}")
    _, p = mannwhitneyu(g3, g4, alternative="two-sided", method="exact")
    if p >= 0.05:
        errors.append(f"Test2 p failed: expected <0.05, got {p}")
    sym = get_symbol(d, p)
    if sym not in ("✓✓✓", "✗✗✗"):
        errors.append(f"Test2 symbol failed: expected ✓✓✓ or ✗✗✗, got {sym}")

    # ------------------------------------------------------------------
    # Test 3: Benjamini-Hochberg FDR correction
    # BH step-up (descending order): p * n / (n - rank)
    #   n=3, sorted desc: (1,2,0.06), (0,2,0.04), (0,1,0.01)
    #   rank0: 0.06*3/3=0.06, running_min=0.06
    #   rank1: 0.04*3/2=0.06, running_min=0.06
    #   rank2: 0.01*3/1=0.03, running_min=0.03
    # ------------------------------------------------------------------
    raw = [(0, 1, 0.01), (0, 2, 0.04), (1, 2, 0.06)]
    adj = benjamini_hochberg(raw)
    # (0,1) p=0.01 → adj=0.03 → significant
    if adj[(0, 1)] >= 0.05:
        errors.append(f"Test3 first pair should be significant: adj_p={adj[(0, 1)]}")
    # (0,2) p=0.04 → adj=0.06 → not significant
    if adj[(0, 2)] < 0.05:
        errors.append(
            f"Test3 second pair should not be significant: adj_p={adj[(0, 2)]}"
        )
    # (1,2) p=0.06 → adj=0.06 → not significant
    if adj[(1, 2)] < 0.05:
        errors.append(
            f"Test3 third pair should not be significant: adj_p={adj[(1, 2)]}"
        )

    # ------------------------------------------------------------------
    # Test 4: flip_symbol symmetry
    # ------------------------------------------------------------------
    if flip_symbol("✓✓✓") != "✗✗✗":
        errors.append(f"Test4a flip failed: {flip_symbol('✓✓✓')}")
    if flip_symbol("✗") != "✓":
        errors.append(f"Test4b flip failed: {flip_symbol('✗')}")
    if flip_symbol("≡") != "≡":
        errors.append(f"Test4c flip failed: {flip_symbol('≡')}")
    if flip_symbol("—") != "—":
        errors.append(f"Test4d flip failed: {flip_symbol('—')}")

    # ------------------------------------------------------------------
    # Test 5: get_symbol boundary cases with Cohen's d
    # ------------------------------------------------------------------
    # p >= 0.05 always yields ≡
    if get_symbol(5.0, 0.10) != "≡":
        errors.append("Test5a: p≥0.05 should always yield ≡")
    if get_symbol(-5.0, 0.06) != "≡":
        errors.append("Test5b: p≥0.05 should always yield ≡")

    # Significant but negligible effect (|d| < 6) → ≡
    if get_symbol(0.0, 0.01) != "≡":
        errors.append("Test5c: d=0.0 + sig should yield ≡")
    if get_symbol(5.9, 0.01) != "≡":
        errors.append("Test5d: d=5.9 + sig should yield ≡ (negligible)")

    # Small effect (6 ≤ |d| < 9)
    if get_symbol(7.0, 0.01) != "✓":
        errors.append(
            f"Test5e: d=7.0 + sig should yield ✓, got {get_symbol(7.0, 0.01)}"
        )
    if get_symbol(-7.0, 0.01) != "✗":
        errors.append(
            f"Test5f: d=-7.0 + sig should yield ✗, got {get_symbol(-7.0, 0.01)}"
        )

    # Medium effect (9 ≤ |d| < 13)
    if get_symbol(10.0, 0.01) != "✓✓":
        errors.append(
            f"Test5g: d=10.0 + sig should yield ✓✓, got {get_symbol(10.0, 0.01)}"
        )
    if get_symbol(-10.0, 0.01) != "✗✗":
        errors.append(
            f"Test5h: d=-10.0 + sig should yield ✗✗, got {get_symbol(-10.0, 0.01)}"
        )

    # Large effect (|d| ≥ 13)
    if get_symbol(15.0, 0.01) != "✓✓✓":
        errors.append(
            f"Test5i: d=15.0 + sig should yield ✓✓✓, got {get_symbol(15.0, 0.01)}"
        )
    if get_symbol(-15.0, 0.01) != "✗✗✗":
        errors.append(
            f"Test5j: d=-15.0 + sig should yield ✗✗✗, got {get_symbol(-15.0, 0.01)}"
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        print(f"\n❌ {len(errors)} self-test(s) failed", file=sys.stderr)
        sys.exit(1)

    print("✅ All self-tests passed")
    sys.exit(0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Batch-size experiment analysis: pairwise Vargha-Delaney A12 + "
            "Holm-Bonferroni corrected Wilcoxon tests."
        ),
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="experiments/rq_batchsize",
        help="Path to batch-size results directory (default: experiments/rq_batchsize)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "latex", "csv"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--benchmark",
        choices=["tinyimagenet", "cifar100", "all"],
        default="all",
        help="Benchmark to analyze (default: all)",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Generate a merged side-by-side LaTeX table (CIFAR-100 + TinyImageNet)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in self-tests and exit",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(
            f"ERROR: results directory does not exist: {results_dir}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not results_dir.is_dir():
        print(f"ERROR: not a directory: {results_dir}", file=sys.stderr)
        sys.exit(1)

    if args.merged:
        generate_merged_latex(args.results_dir)
        return

    # Determine which benchmarks to run
    if args.benchmark == "all":
        benchmarks = list(BENCHMARK_CONFIGS.keys())
    else:
        benchmarks = [args.benchmark]

    for benchmark in benchmarks:
        analyzer = BatchSizeAnalyzer(args.results_dir, benchmark=benchmark)
        analyzer.run(format=args.format)


if __name__ == "__main__":
    main()
