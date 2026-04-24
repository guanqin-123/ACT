from __future__ import annotations

# pyright: reportUnusedCallResult=false, reportUnknownMemberType=false, reportAny=false, reportImplicitStringConcatenation=false

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import torch

from act.back_end.bab.bab import verify_bab
from act.back_end.config import BaBConfig
from act.back_end.dual_tf import DualTF
from act.back_end.solver.solver_dual import DualSolver
from act.back_end.solver.solver_interval import TorchLPSolver
from act.util.device_manager import initialize_device
from scripts.bab_clamp_diagnostic import BENCHMARK_DIR, INSTANCES_CSV, load_model_and_instance


NODE_RE = re.compile(
    r"\[BaB\]\s+iter=\s*(?P<iter>\d+)\s+node=\s*(?P<node>\d+)\s+depth=\s*(?P<depth>\d+)\s+lb=(?P<lb>[+-]?\d+\.\d+)\s+"
    r"raw=(?P<raw>[+-]?\d+\.\d+)(?:\s+CLAMP)?\s+\[(?P<mark>CERT|open)\]"
)


PerNodeRecord = list[int | float | bool]


@dataclass(frozen=True)
class AnchorConfig:
    max_depth: int
    max_nodes: int
    subproblem_batch_size: int
    eta_iters: int
    branching_method: str
    bounding_method: str
    budget: float
    device: str
    dtype: str
    seed: int


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self._streams: tuple[TextIO, ...] = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _fixed_config() -> AnchorConfig:
    return AnchorConfig(
        max_depth=5,
        max_nodes=40,
        subproblem_batch_size=4,
        eta_iters=5,
        branching_method="babsr",
        bounding_method="bfs",
        budget=60.0,
        device="cuda",
        dtype="float32",
        seed=42,
    )


def _config_hash(config: AnchorConfig) -> str:
    payload = json.dumps(config.__dict__, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _available_medium_instance_count() -> int:
    with open(INSTANCES_CSV, encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return sum(
        1
        for onnx_rel, vnnlib_rel, _timeout_s in rows
        if Path(onnx_rel).name == "CIFAR100_resnet_medium.onnx"
        and (BENCHMARK_DIR / vnnlib_rel).exists()
    )


def _parse_per_node(log_text: str) -> tuple[list[PerNodeRecord], list[str]]:
    per_node: list[PerNodeRecord] = []
    lb_tokens: list[str] = []
    for match in NODE_RE.finditer(log_text):
        lb_token = match.group("lb")
        lb_tokens.append(lb_token)
        per_node.append(
            [
                int(match.group("iter")),
                int(match.group("node")),
                int(match.group("depth")),
                float(lb_token),
                float(match.group("raw")),
                match.group("mark") == "CERT",
            ]
        )
    return per_node, lb_tokens


def _lb_trace_sha256(lb_tokens: list[str]) -> str:
    return hashlib.sha256("\t".join(lb_tokens).encode("utf-8")).hexdigest()


def _run_instance(instance_idx: int, config: AnchorConfig) -> dict[str, object]:
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    net, _ = load_model_and_instance(instance_idx)
    bab_config = BaBConfig(
        max_depth=config.max_depth,
        max_nodes=config.max_nodes,
        subproblem_batch_size=config.subproblem_batch_size,
        eta_iters=config.eta_iters,
        branching_method=config.branching_method,
        bounding_method=config.bounding_method,
        verbose=True,
    )

    capture = io.StringIO()
    with redirect_stdout(Tee(sys.stdout, capture)):
        result = verify_bab(
            net,
            solver=TorchLPSolver(),
            dual_solver=DualSolver(DualTF()),
            config=bab_config,
            time_budget_s=config.budget,
        )

    per_node, lb_tokens = _parse_per_node(capture.getvalue())
    return {
        "status": result.status.name,
        "nodes": result.metadata.get("nodes"),
        "per_node": per_node,
        "lb_trace_sha256": _lb_trace_sha256(lb_tokens),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/logs/parity_anchor.json"),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = _fixed_config()
    initialize_device(config.device, config.dtype)

    instances: dict[str, dict[str, object]] = {}
    max_instances = _available_medium_instance_count()
    candidate_idx = 0
    failures: list[dict[str, int | str]] = []

    while len(instances) < 3 and candidate_idx < max_instances:
        try:
            instances[str(candidate_idx)] = _run_instance(candidate_idx, config)
        except Exception as exc:  # pragma: no cover - diagnostic script
            failures.append({"instance_idx": candidate_idx, "error": str(exc)})
        candidate_idx += 1

    payload: dict[str, object] = {
        "config_hash": _config_hash(config),
        "instances": instances,
    }
    if failures:
        payload["load_failures"] = failures

    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote parity anchor: {args.output} instances={list(instances.keys())}")
    return 0 if len(instances) == 3 else 1


if __name__ == "__main__":
    sys.exit(main())
