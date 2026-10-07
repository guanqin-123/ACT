#!/usr/bin/env python3
"""Build the RQ3 table (label tab:exp-rq3) from the collected data in rq3/ next to this script.

Full CLIMB (on_k256) versus component ablations on the CIFAR-100 and TinyImageNet queries in
rq3/climb/<dataset>/queries.csv, using the configuration CSVs in the same folder; SAT-ReLU is not
reported. Search and time are geometric-mean ratios to full CLIMB on common completions. The solved
definition and comparison helpers are shared with the RQ1/RQ2 scripts in this folder. The output is
byte-identical to evaluation_tables/exp_rq3.tex.

Usage, from the data folder:
    python make_rq3_table.py                   # print the table
    python make_rq3_table.py --out table.tex   # write it to a file
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # importing sibling scripts must not leave __pycache__ in the data folder
from make_rq1_main_table import fmt, latex_table, series  # noqa: E402
from make_rq2_table import DATASETS, compare, load_run, queries  # noqa: E402

DATA = Path(__file__).resolve().parent
VARIANTS = [("coarsening_off_k256", "w/o coarsening"), ("propagation_off_k256", "w/o propagation"),
            ("merge_off_k256", "w/o merging"), ("terminal_off_k256", "w/o terminal-LP cores")]
CAPTION = (r"Component ablations (FSB, $B=256$; query counts in parentheses). Search and time "
           r"are geometric-mean ratios to full \climb on common completions; values above 1 favour full \climb. ")


def build_table(data: Path) -> str:
    columns: dict[str, dict[str, tuple[int, float | None, float | None]]] = {}
    counts: dict[str, int] = {}
    for dataset, _ in DATASETS:
        folder = data / "rq3" / "climb" / dataset
        keys = queries(folder)
        full = load_run(folder / "on_k256.csv")
        counts[dataset] = len(keys)
        col: dict[str, tuple[int, float | None, float | None]] = {
            "full": (sum(bool(full.get(k, {}).get("solved")) for k in keys), None, None)}
        columns[dataset] = col
        for variant, _ in VARIANTS:
            summary, _ = compare(full, load_run(folder / f"{variant}.csv"), keys)
            col[variant] = (summary["off_solved"], summary["search"], summary["time"])
    head = [" & " + " & ".join(r"\multicolumn{3}{c}{" + f"{label} ({counts[dataset]})" + "}"
                               for dataset, label in DATASETS) + r" \\",
            "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(DATASETS))),
            "Configuration & " + " & ".join(["Solved & Search & Time"] * len(DATASETS)) + r" \\"]
    body: list[str] = []
    for variant, label in [("full", r"Full \climb"), *VARIANTS]:
        cells: list[str] = []
        for dataset, _ in DATASETS:
            solved, search, time = columns[dataset][variant]
            cells += [str(solved), fmt(search, "time"), fmt(time, "time")]
        body.append(label + " & " + " & ".join(cells) + r" \\")
    loss = {label: sum(columns[d]["full"][0] - columns[d][v][0] for d, _ in DATASETS)
            for v, label in VARIANTS}
    worst = max(loss.values(), default=0)
    takeaway = (f"Removing {series([lab.removeprefix('w/o ') for lab, n in loss.items() if n == worst], 'or')} "
                f"loses the most solved queries ({worst} across all datasets)." if worst > 0 else
                "No ablation reduces the number of solved queries.")
    return latex_table(CAPTION + takeaway, "tab:exp-rq3", body,
                       spec="l" + "rrr" * len(DATASETS), head="\n".join(head))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA,
                        help="data folder that contains rq3/ (default: the folder of this script)")
    parser.add_argument("--out", type=Path, help="write the table to this file instead of printing it")
    args = parser.parse_args()
    table = build_table(args.data)
    if args.out:
        args.out.write_text(table, encoding="utf-8")
    else:
        print(table, end="")


if __name__ == "__main__":
    main()
