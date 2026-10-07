#!/usr/bin/env python3
"""Build the RQ1 main table (label tab:exp-rq1) from the collected data in rq1/ next to this script.

Counts come from our own runs (rq1/<verifier>/<dataset>/results.csv). Mean/median times are over each
verifier's solved queries: CLIMB uses its own verification time; each baseline uses time_used_s from
rq1/<verifier>/<dataset>/solved_query_times.csv (VNN-COMP raw run time when VNN-COMP credits the query,
our local time otherwise). The output is byte-identical to evaluation_tables/exp_rq1_main.tex.

Usage, from the data folder:
    python make_rq1_main_table.py                   # print the table
    python make_rq1_main_table.py --out table.tex   # write it to a file
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parent
DATASETS = [("tinyimagenet_2024", "TinyImageNet", 200), ("cifar100_2024", "CIFAR-100", 200),
            ("cora_2024", "CORA", 180), ("safenlp_2024", "SafeNLP", 1080)]
VERIFIERS = [("climb", r"\climb"), ("abcrown", r"$\alpha,\beta$-CROWN"), ("neuralsat", "NeuralSAT"),
             ("nnenum", "nnenum")]
FIELDS = [("solved", "count"), ("cert", "count"), ("fals", "count"), ("mean", "time"), ("med", "time")]
HEAD = r"Dataset & \#Instances & Verifier & Solved & Certified & Falsified & Mean (sec) & Median (sec) \\"
CAPTION = (r"Coverage and mean/median time over each verifier's solved VNN-COMP queries, with attacks disabled. "
           r"Baseline times use official VNN-COMP 2026 raw runtimes for correctly solved queries and local times "
           r"otherwise. Bold: best per dataset; $^\dagger$: partial; N/A: unsupported. ")
EXECUTION_FAILURES = ("ERROR", "WATCHDOG", "RUNNER_ERROR")


def number(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        x = float(text)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def fmt(value: float | None, kind: str) -> str:
    if value is None:
        return "--"
    return f"{int(round(value)):,}" if kind == "count" else f"{value:.2f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return [r for r in csv.DictReader(fh) if None not in r and None not in r.values()]


def is_solved(r: dict[str, str]) -> bool:
    """VNN-COMP convention: certified, or falsified with a witness valid at 1e-6, within the time limit."""
    t, limit = number(r.get("verification_s")), number(r.get("timeout_s"))
    within = r.get("status") in ("CERTIFIED", "FALSIFIED") and t is not None and limit is not None and 0 <= t <= limit
    return within and (r.get("status") != "FALSIFIED" or r.get("witness_valid_1e6") == "True")


def summarize(verifier: str, folder: Path, planned: int) -> dict[str, Any]:
    path = folder / "results.csv"
    if not path.exists():
        return {"state": "pending", "n": 0, "solved": None, "cert": None, "fals": None, "mean": None, "med": None}
    queries: dict[int, dict[str, str]] = {}
    for r in read_csv(path):
        row = number(r.get("row"))
        if row is not None:
            queries[int(row)] = r  # reruns append; the last record of a query counts
    solved = {k: r for k, r in queries.items() if is_solved(r)}
    if verifier == "climb":
        times = [float(r["verification_s"]) for r in solved.values()]
    else:
        timing = folder / "solved_query_times.csv"
        rows = read_csv(timing) if timing.exists() else []
        if sorted(int(r["row"]) for r in rows) != sorted(solved):
            raise ValueError(f"{timing}: rows differ from the solved queries in {path}")
        times = [float(r["time_used_s"]) for r in rows]
    failures = verifier == "climb" and any(r.get("status") in EXECUTION_FAILURES or
                                           r.get("runner_state") in ("error", "watchdog") for r in queries.values())
    if queries and all(r.get("error_class") == "unsupported_op" for r in queries.values()):
        state = "na"
    elif len(queries) != planned or failures:
        state = "partial"
    else:
        state = "complete"
    return {"state": state, "n": len(queries), "solved": len(solved),
            "cert": sum(r["status"] == "CERTIFIED" for r in solved.values()),
            "fals": sum(r["status"] == "FALSIFIED" for r in solved.values()),
            "mean": statistics.mean(times) if times else None, "med": statistics.median(times) if times else None}


def dataset_block(label: str, planned: int,
                  summaries: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, int]]:
    """Rows of one dataset; bold marks the best shown value (counts among verifiers that ran every query)."""
    full = {v: s for v, s in summaries.items() if s["state"] in ("complete", "partial") and s["n"] == planned}
    best: dict[str, float] = {}
    for field, kind in FIELDS:
        pool = full.values() if kind == "count" else [s for v, s in summaries.items() if v in full or v != "climb"]
        values = [float(fmt(s[field], kind).replace(",", "")) for s in pool if s[field] is not None]
        if values and (kind == "time" or max(values) > 0):
            best[field] = max(values) if kind == "count" else min(values)
    lines: list[str] = []
    size = len(VERIFIERS)
    for i, (verifier, name) in enumerate(VERIFIERS):
        s = summaries[verifier]
        lead = ([rf"\multirow{{{size}}}{{*}}{{{label}}}", rf"\multirow{{{size}}}{{*}}{{{fmt(planned, 'count')}}}"]
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
        if verifier != "climb":
            for index, field in ((-2, "mean"), (-1, "med")):
                value = fmt(s[field], "time")
                if value != "--" and float(value) == best.get(field):
                    value = r"\textbf{" + value + "}"
                cells[index] = value
        lines.append(" & ".join([*lead, name, *cells]) + r" \\")
    return lines, {v: s["solved"] for v, s in full.items()}


def series(items: Iterable[str], conj: str = "and") -> str:
    words = list(items)
    return f" {conj} ".join(words) if len(words) < 3 else ", ".join(words[:-1]) + f", {conj} " + words[-1]


def takeaway(blocks: list[tuple[str, dict[str, int]]], verifiers: list[tuple[str, str]] = VERIFIERS) -> str:
    names = dict(verifiers)
    lead: list[str] = []
    trail: dict[str, list[str]] = {}
    for label, solved in blocks:
        if "climb" not in solved or len(solved) < 2:
            continue
        leaders = [v for v, _ in verifiers if solved.get(v) == max(solved.values())]
        if "climb" in leaders:
            lead.append(label + (" (tied)" if len(leaders) > 1 else ""))
        else:
            trail.setdefault(leaders[0], []).append(label)
    behind = " and than ".join(f"{names[v]} on {series(labels)}" for v, labels in trail.items())
    if lead:
        return rf"\climb solves the most queries on {series(lead)}" + (f", but fewer than {behind}." if trail else ".")
    return rf"\climb solves fewer queries than {behind}." if trail else ""


def latex_table(caption: str, label: str, body: list[str], spec: str = "lclrrrrr", head: str = HEAD) -> str:
    return "\n".join([r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
                      r"\caption{" + caption.strip() + "}", r"\label{" + label + "}",
                      r"\setbox0=\hbox{\begin{tabular}{" + spec + "}", r"\toprule", head, r"\midrule", *body,
                      r"\bottomrule", r"\end{tabular}}",
                      r"\leavevmode\ifdim\wd0>\linewidth\resizebox{\linewidth}{!}{\box0}\else\box0\fi",
                      r"\end{table}", ""])


def build_table(data: Path) -> str:
    body: list[str] = []
    blocks: list[tuple[str, dict[str, int]]] = []
    for dataset, label, planned in DATASETS:
        summaries = {v: summarize(v, data / "rq1" / v / dataset, planned) for v, _ in VERIFIERS}
        lines, solved = dataset_block(label, planned, summaries)
        body += ([r"\midrule"] if body else []) + lines
        blocks.append((label, solved))
    return latex_table(CAPTION + takeaway(blocks), "tab:exp-rq1", body)


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
