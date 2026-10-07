#!/usr/bin/env python3
"""Build the Transformer table (label tab:exp-rq1-text) from the collected data in rq1/ next to this script.

Counts and mean/median times come from our own runs (rq1/<verifier>/<slice>/results.csv); times are each
verifier's own verification times over its solved queries. NeuralSAT is N/A on the l1 slices (an l1 ball cannot
be passed through the box-only VNN-LIB interface), and nnenum is not in this table (it cannot load the
Transformer models). The solved definition and formatting are shared with make_rq1_main_table.py in the same
folder. The output is byte-identical to evaluation_tables/exp_rq1_text.tex.

Usage, from the data folder:
    python make_rq1_text_table.py                   # print the table
    python make_rq1_text_table.py --out table.tex   # write it to a file
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True  # importing the sibling script must not leave __pycache__ in the data folder
from make_rq1_main_table import (EXECUTION_FAILURES, FIELDS, fmt, is_solved, latex_table,  # noqa: E402
                                 number, read_csv, takeaway)

DATA = Path(__file__).resolve().parent
SLICES = [("sst_p1", r"SST $\ell_1$"), ("sst_pinf", r"SST $\ell_\infty$"), ("yelp_p1", r"Yelp $\ell_1$"),
          ("yelp_pinf", r"Yelp $\ell_\infty$")]
PLANNED = 60
VERIFIERS = [("climb", r"\climb"), ("abcrown", r"$\alpha,\beta$-CROWN"), ("neuralsat", "NeuralSAT")]
NOT_APPLICABLE = {("neuralsat", "sst_p1"), ("neuralsat", "yelp_p1")}
CAPTION = (r"Coverage and solved-query mean/median time for encoder-only Transformer classifiers "
           r"($T=30$\,s, attacks disabled). ")
EMPTY: dict[str, Any] = {"n": 0, "solved": None, "cert": None, "fals": None, "mean": None, "med": None}


def summarize(verifier: str, name: str, folder: Path) -> dict[str, Any]:
    if (verifier, name) in NOT_APPLICABLE:
        return {"state": "na", **EMPTY}
    path = folder / "results.csv"
    if not path.exists():
        return {"state": "pending", **EMPTY}
    queries: dict[int, dict[str, str]] = {}
    for r in read_csv(path):
        row = number(r.get("row"))
        if row is not None:
            queries[int(row)] = r
    solved = [r for r in queries.values() if is_solved(r)]
    times = [float(r["verification_s"]) for r in solved]
    failures = verifier == "climb" and any(r.get("status") in EXECUTION_FAILURES or
                                           r.get("runner_state") in ("error", "watchdog") for r in queries.values())
    if queries and all(r.get("error_class") == "unsupported_op" for r in queries.values()):
        state = "na"
    elif len(queries) != PLANNED or failures:
        state = "partial"
    else:
        state = "complete"
    return {"state": state, "n": len(queries), "solved": len(solved),
            "cert": sum(r["status"] == "CERTIFIED" for r in solved),
            "fals": sum(r["status"] == "FALSIFIED" for r in solved),
            "mean": statistics.mean(times) if times else None, "med": statistics.median(times) if times else None}


def slice_block(label: str, summaries: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, int]]:
    """Rows of one slice; bold marks the best shown value among verifiers that ran every query."""
    full = {v: s for v, s in summaries.items() if s["state"] in ("complete", "partial") and s["n"] == PLANNED}
    best: dict[str, float] = {}
    for field, kind in FIELDS:
        values = [float(fmt(s[field], kind).replace(",", "")) for s in full.values() if s[field] is not None]
        if values and (kind == "time" or max(values) > 0):
            best[field] = max(values) if kind == "count" else min(values)
    lines: list[str] = []
    size = len(VERIFIERS)
    for i, (verifier, name) in enumerate(VERIFIERS):
        s = summaries[verifier]
        lead = ([rf"\multirow{{{size}}}{{*}}{{{label}}}", rf"\multirow{{{size}}}{{*}}{{{fmt(PLANNED, 'count')}}}"]
                if i == 0 else ["", ""])
        if s["state"] in ("na", "pending"):
            cells = ["N/A" if s["state"] == "na" else r"--$^\dagger$"] + ["--"] * (len(FIELDS) - 1)
        else:
            cells = []
            for field, kind in FIELDS:
                value = fmt(s[field], kind)
                if verifier in full and value != "--" and float(value.replace(",", "")) == best.get(field):
                    value = r"\textbf{" + value + "}"
                cells.append(value)
            if s["state"] == "partial":
                cells[0] += r"$^\dagger$"
        lines.append(" & ".join([*lead, name, *cells]) + r" \\")
    return lines, {v: s["solved"] for v, s in full.items()}


def build_table(data: Path) -> str:
    body: list[str] = []
    blocks: list[tuple[str, dict[str, int]]] = []
    for name, label in SLICES:
        summaries = {v: summarize(v, name, data / "rq1" / v / name) for v, _ in VERIFIERS}
        lines, solved = slice_block(label, summaries)
        body += ([r"\midrule"] if body else []) + lines
        blocks.append((label, solved))
    return latex_table(CAPTION + takeaway(blocks, VERIFIERS), "tab:exp-rq1-text", body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA,
                        help="data folder that contains rq1/ (default: the folder of this script)")
    parser.add_argument("--out", type=Path, help="write the table to this file instead of printing it")
    args = parser.parse_args()
    table = build_table(args.data)
    if args.out:
        args.out.write_text(table, encoding="utf-8")
    else:
        print(table, end="")


if __name__ == "__main__":
    main()
