#!/usr/bin/env python3
"""Build the RQ4 batch table (label tab:exp-rq4-batch) from the collected data in rq4/ next to this script.

CLIMB enabled versus disabled at batch limits 64, 256 and 1024 on the CIFAR-100 and TinyImageNet
queries in rq4/climb/<dataset>/queries.csv, using on_k<batch>.csv and off_k<batch>.csv. SAT-ReLU and
budget/re-check sensitivity are not reported. Search and time are off/on geometric-mean ratios on
common completions. The solved definition and comparison helpers are shared with the RQ1/RQ2 scripts
in this folder. The output is byte-identical to evaluation_tables/exp_rq4_batch.tex.

Usage, from the data folder:
    python make_rq4_table.py                   # print the table
    python make_rq4_table.py --out table.tex   # write it to a file
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True  # importing sibling scripts must not leave __pycache__ in the data folder
from make_rq1_main_table import fmt, latex_table  # noqa: E402
from make_rq2_table import DATASETS, compare, load_run, queries  # noqa: E402

DATA = Path(__file__).resolve().parent
BATCHES = (64, 256, 1024)
CAPTION = (r"Batch sensitivity (FSB, $T=100$\,s; query counts in parentheses). "
           r"Solved gives the queries solved with \climb enabled/disabled. Search and time are off/on geometric-mean "
           r"ratios on common completions; values above 1 favour \climb. ")


def build_table(data: Path) -> str:
    columns: dict[str, dict[int, dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    for dataset, _ in DATASETS:
        folder = data / "rq4" / "climb" / dataset
        keys = queries(folder)
        counts[dataset] = len(keys)
        columns[dataset] = {}
        for batch in BATCHES:
            summary, _ = compare(load_run(folder / f"on_k{batch}.csv"),
                                 load_run(folder / f"off_k{batch}.csv"), keys)
            columns[dataset][batch] = summary
    head = [" & " + " & ".join(r"\multicolumn{3}{c}{" + f"{label} ({counts[dataset]})" + "}"
                               for dataset, label in DATASETS) + r" \\",
            "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(DATASETS))),
            "$B$ & " + " & ".join(["Solved & Search & Time"] * len(DATASETS)) + r" \\"]
    body: list[str] = []
    for batch in BATCHES:
        cells: list[str] = []
        for dataset, _ in DATASETS:
            summary = columns[dataset][batch]
            cells += [f"{summary['on_solved']}/{summary['off_solved']}",
                      fmt(summary["search"], "time"), fmt(summary["time"], "time")]
        body.append(f"{batch} & " + " & ".join(cells) + r" \\")
    wins = [columns[d][b]["on_solved"] > columns[d][b]["off_solved"] for d, _ in DATASETS for b in BATCHES]
    takeaway = (r"Enabling \climb solves more queries at every tested batch limit on every dataset."
                if wins and all(wins) else rf"Enabling \climb solves more queries in {sum(wins)} of {len(wins)} settings.")
    return latex_table(CAPTION + takeaway, "tab:exp-rq4-batch", body,
                       spec="r" + "rrr" * len(DATASETS), head="\n".join(head))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA,
                        help="data folder that contains rq4/ (default: the folder of this script)")
    parser.add_argument("--out", type=Path, help="write the table to this file instead of printing it")
    args = parser.parse_args()
    table = build_table(args.data)
    if args.out:
        args.out.write_text(table, encoding="utf-8")
    else:
        print(table, end="")


if __name__ == "__main__":
    main()
