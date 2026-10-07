#!/usr/bin/env python3
"""Build the RQ2 table (label tab:exp-rq2) from the collected data in rq2/ next to this script.

CLIMB enabled (on_k256) versus disabled (off_k256) on the RQ2 queries of CIFAR-100 and TinyImageNet
(rq2/climb/<dataset>/queries.csv); SAT-ReLU is not reported. Processed sub-problem totals, their reduction and
the off/on geometric-mean search and time ratios use common completions (queries solved by both runs). The
solved definition is shared with make_rq1_main_table.py in the same folder. The output is byte-identical to
evaluation_tables/exp_rq2.tex.

Usage, from the data folder:
    python make_rq2_table.py                   # print the table
    python make_rq2_table.py --out table.tex   # write it to a file
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True  # importing the sibling script must not leave __pycache__ in the data folder
from make_rq1_main_table import fmt, is_solved, latex_table, number, read_csv  # noqa: E402

DATA = Path(__file__).resolve().parent
DATASETS = [("cifar100_2024", "CIFAR-100"), ("tinyimagenet_2024", "TinyImageNet")]
ON, OFF = "on_k256", "off_k256"
VERDICTS = ("CERTIFIED", "FALSIFIED", "UNKNOWN", "TIMEOUT")
HEAD = r"Dataset & $n$ & Solved & Only & Processed (off$\to$on) & Red. & Search & Time \\"
CAPTION = (r"Enabled/disabled solved and exclusive counts ($B=256$, $T=100$\,s; FSB unless noted). "
           r"Processed totals, their reduction, and off/on geometric-mean search and time ratios use common "
           r"completions. ")

Run = dict[str, Any]
Query = tuple[str, str]


def queries(folder: Path) -> list[Query]:
    return [(r["onnx"], r["vnnlib"]) for r in read_csv(folder / "queries.csv")]


def load_run(path: Path) -> dict[Query, Run]:
    runs: dict[Query, Run] = {}
    for r in read_csv(path):
        exit_code = number(r.get("exit_code"))
        runs[(r["onnx"], r["vnnlib"])] = {"status": r["status"], "solved": is_solved(r),
                                         "exit": None if exit_code is None else int(exit_code),
                                         "nodes": number(r.get("nodes")), "time": number(r.get("verification_s"))}
    return runs


def gmean(values: list[float]) -> float | None:
    return math.exp(statistics.mean(math.log(x) for x in values)) if values else None


def compare(on: dict[Query, Run], off: dict[Query, Run], keys: list[Query]) -> tuple[dict[str, Any], list[tuple[Run, Run]]]:
    """CLIMB on vs off over the planned queries; off/on ratios and totals use queries solved by both runs."""
    outcome: Counter[str] = Counter()
    node_ratios: list[float] = []
    time_ratios: list[float] = []
    nodes_on = nodes_off = 0.0
    both: list[tuple[Run, Run]] = []
    for key in keys:
        x, y = on.get(key), off.get(key)
        if x is None or y is None:
            outcome["missing"] += 1
            continue
        if {x["status"], y["status"]} >= {"CERTIFIED", "FALSIFIED"}:
            kind = "conflict"
        elif any(z["exit"] not in (0, None) or z["status"] not in VERDICTS for z in (x, y)):
            kind = "error"
        else:
            kind = ("both" if x["solved"] and y["solved"] else "on_only" if x["solved"] else
                    "off_only" if y["solved"] else "neither")
        outcome[kind] += 1
        if kind != "both":
            continue
        for ratios, field in ((node_ratios, "nodes"), (time_ratios, "time")):
            if x[field] and y[field] and x[field] > 0 and y[field] > 0:
                ratios.append(y[field] / x[field])
        if x["nodes"] and y["nodes"]:
            nodes_on += x["nodes"]
            nodes_off += y["nodes"]
        both.append((x, y))
    summary = {"n": len(keys), "on_solved": sum(bool(on.get(k, {}).get("solved")) for k in keys),
               "off_solved": sum(bool(off.get(k, {}).get("solved")) for k in keys),
               "on_only": outcome["on_only"], "off_only": outcome["off_only"], "nodes_on": nodes_on,
               "nodes_off": nodes_off, "search": gmean(node_ratios), "time": gmean(time_ratios)}
    return summary, both


def load(data: Path, dataset: str) -> tuple[dict[str, Any], list[tuple[Run, Run]]]:
    folder = data / "rq2" / "climb" / dataset
    return compare(load_run(folder / f"{ON}.csv"), load_run(folder / f"{OFF}.csv"), queries(folder))


def percent(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.0f}"


def build_table(data: Path) -> str:
    body: list[str] = []
    reductions: list[float | None] = []
    for dataset, label in DATASETS:
        s, _ = load(data, dataset)
        reduction = 1 - s["nodes_on"] / s["nodes_off"] if s["nodes_off"] else None
        reductions.append(reduction)
        body.append(" & ".join([label, str(s["n"]), f"{s['on_solved']}/{s['off_solved']}",
                                f"{s['on_only']}/{s['off_only']}",
                                f"{fmt(s['nodes_off'], 'count')}$\\to${fmt(s['nodes_on'], 'count')}",
                                percent(reduction) + r"\%", fmt(s["search"], "time") + r"$\times$",
                                fmt(s["time"], "time") + r"$\times$"]) + r" \\")
    takeaway = ""
    known = [r for r in reductions if r is not None]
    if reductions and len(known) == len(reductions) and all(r > 0 for r in known):
        low, high = percent(min(known)), percent(max(known))
        takeaway = (rf"\climb reduces processed sub-problems by {low if low == high else f'{low}--{high}'}\% "
                    r"on these workloads.")
    return latex_table(CAPTION + takeaway, "tab:exp-rq2", body, spec="lrrrrrrr", head=HEAD)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA,
                        help="data folder that contains rq2/ (default: the folder of this script)")
    parser.add_argument("--out", type=Path, help="write the table to this file instead of printing it")
    args = parser.parse_args()
    table = build_table(args.data)
    if args.out:
        args.out.write_text(table, encoding="utf-8")
    else:
        print(table, end="")


if __name__ == "__main__":
    main()
