"""Run one CIFAR-100 ResNet-medium instance through verify_bab with the new
raw_lb / CLAMP diagnostic enabled.

Mirrors the PHASE2 setup from ipynb/bab_neuron_split_demo.ipynb but with
verbose=True so we can read clamp_hits per iteration and confirm whether the
deep-depth "bound barely moves" pattern is the parent-margin clamp masking
Adam regression.
"""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false, reportAttributeAccessIssue=false, reportAny=false, reportCallIssue=false, reportArgumentType=false, reportUnusedVariable=false

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import torch

from act.back_end.bab.bab import verify_bab
from act.back_end.bab.trace import BoundTrace
from act.back_end.config import BaBConfig
from act.back_end.dual_tf import DualTF
from act.back_end.solver.solver_dual import DualSolver
from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.verifiable_model import (
    InputLayer,
    InputSpecLayer,
    OutputSpecLayer,
    VerifiableModel,
)
from act.front_end.vnnlib_loader.onnx_converter import (
    convert_onnx_to_pytorch,
    get_onnx_input_shape,
)
from act.front_end.vnnlib_loader.vnnlib_parser import (
    extract_label_from_vnnlib,
    parse_vnnlib_queries,
    parse_vnnlib_to_tensors,
)
from act.pipeline.verification.torch2act import TorchToACT
from act.util.device_manager import initialize_device


BENCHMARK_DIR = Path(
    "/home/guanqinzhang/data/guanqin/newACT/ACT/data/vnnlib/cifar100_2024"
)
INSTANCES_CSV = BENCHMARK_DIR / "instances.csv"
ONNX_DIR = BENCHMARK_DIR / "onnx"

LOGGER = logging.getLogger(__name__)


def _load_medium_rows() -> list[tuple[int, str, str, float]]:
    with open(INSTANCES_CSV, encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    return [
        (i, onnx_rel, vnnlib_rel, float(timeout_s))
        for i, (onnx_rel, vnnlib_rel, timeout_s) in enumerate(rows)
        if Path(onnx_rel).name == "CIFAR100_resnet_medium.onnx"
        and (BENCHMARK_DIR / vnnlib_rel).exists()
    ]


def resolve_medium_instance_index(
    instance_idx: int | None = 0,
) -> int:
    medium_rows = _load_medium_rows()
    if not medium_rows:
        raise RuntimeError("No CIFAR100_resnet_medium instances found under BENCHMARK_DIR.")

    resolved_idx = 0 if instance_idx is None else int(instance_idx)
    if resolved_idx >= len(medium_rows):
        raise IndexError(
            f"instance_idx={resolved_idx} out of range (have {len(medium_rows)} instances)"
        )
    return resolved_idx


def resolve_vnnlib_row_index(
    vnnlib_name: str,
    *,
    instance_idx: int | None = None,
) -> int:
    medium_rows = _load_medium_rows()
    if not medium_rows:
        raise RuntimeError("No CIFAR100_resnet_medium instances found under BENCHMARK_DIR.")

    if instance_idx not in (None, 0):
        LOGGER.warning(
            "Both instance_idx=%s and vnnlib_name=%s were provided; vnnlib_name wins.",
            instance_idx,
            vnnlib_name,
        )

    for csv_idx, _onnx_rel, vnnlib_rel, _timeout_s in medium_rows:
        if Path(vnnlib_rel).name == vnnlib_name:
            return csv_idx
    raise ValueError(
        f"Unknown CIFAR100_resnet_medium vnnlib_name={vnnlib_name!r}; "
        f"expected one of the entries from {INSTANCES_CSV}."
    )


def load_model_and_instance(
    instance_idx: int = 0,
    *,
    vnnlib_name: str | None = None,
):
    """Load the ResNet-medium ONNX and one CIFAR-100 VNNLIB instance.

    Returns (net, vnnlib_name). `net` is an ACT Net ready for verify_bab.
    """
    medium_rows = _load_medium_rows()
    if vnnlib_name is not None:
        resolved_row_idx = resolve_vnnlib_row_index(
            vnnlib_name,
            instance_idx=instance_idx,
        )
        matching_rows = [row for row in medium_rows if row[0] == resolved_row_idx]
        if not matching_rows:
            raise ValueError(
                f"Resolved vnnlib_name={vnnlib_name!r} to CSV row {resolved_row_idx}, "
                "but no matching CIFAR100_resnet_medium entry was found."
            )
        idx, _onnx_rel, vnnlib_rel, _timeout_s = matching_rows[0]
    else:
        resolved_idx = resolve_medium_instance_index(instance_idx=instance_idx)
        idx, _onnx_rel, vnnlib_rel, _timeout_s = medium_rows[resolved_idx]
    vnnlib_path = BENCHMARK_DIR / vnnlib_rel
    onnx_path = ONNX_DIR / "CIFAR100_resnet_medium.onnx"

    t0 = time.time()
    pytorch_model = convert_onnx_to_pytorch(onnx_path, simplify=True)
    pytorch_model.eval()
    model_load_s = time.time() - t0
    input_shape = get_onnx_input_shape(onnx_path)

    label = extract_label_from_vnnlib(vnnlib_path)
    if label is None:
        raise RuntimeError(f"Missing label comment in {vnnlib_path.name}")
    input_tensor, _unused_bounds = parse_vnnlib_to_tensors(vnnlib_path, input_shape)
    labeled = LabeledInputTensor(
        tensor=input_tensor, label=torch.tensor([label], dtype=torch.int64)
    )
    in_spec, out_spec = parse_vnnlib_queries(vnnlib_path, labeled_tensor=labeled)[0]

    wrapped = VerifiableModel(
        input_layer=InputLayer(
            labeled, labeled.tensor.shape, labeled.tensor.dtype,
            layout="CHW", dataset_name="cifar100_2024",
        ),
        input_spec=InputSpecLayer(spec=in_spec),
        model=pytorch_model,
        output_spec=OutputSpecLayer(spec=out_spec),
    )
    net = TorchToACT(wrapped).run()

    print(f"[setup] model loaded in {model_load_s:.2f}s, instance_idx={idx}")
    print(f"[setup] vnnlib: {vnnlib_path.name}")
    print(f"[setup] input_shape={input_shape} label={label}")
    return net, vnnlib_path.name


def run_diagnostic(
    net,
    *,
    device: str = "cpu",
    dtype: str = "float32",
    subproblem_batch_size: int = 8,
    eta_iters: int = 10,
    max_depth: int = 20,
    max_nodes: int = 300,
    time_budget_s: float = 120.0,
    record_trace: bool = False,
):
    config = BaBConfig(
        branching_method="babsr",
        bounding_method="bfs",
        subproblem_batch_size=subproblem_batch_size,
        eta_iters=eta_iters,
        lr_eta=0.05,
        max_nodes=max_nodes,
        max_depth=max_depth,
        record_bound_trace=record_trace,
        verbose=True,
    )
    trace = BoundTrace() if record_trace else None

    print(
        f"[config] device={device} dtype={dtype} "
        f"batch={subproblem_batch_size} eta_iters={eta_iters} "
        f"max_depth={max_depth} max_nodes={max_nodes} budget={time_budget_s}s"
    )

    t0 = time.time()
    result = verify_bab(
        net,
        solver=None,
        dual_solver=DualSolver(DualTF()),
        config=config,
        time_budget_s=time_budget_s,
        trace=trace,
    )
    elapsed = time.time() - t0
    print(
        f"\n[result] status={result.status.name} "
        f"nodes={result.metadata.get('nodes')} time={elapsed:.2f}s"
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--batch", type=int, default=8, help="subproblem_batch_size")
    parser.add_argument("--eta-iters", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=300)
    parser.add_argument("--budget", type=float, default=120.0)
    args = parser.parse_args()

    # device_manager must be initialised BEFORE loading the ONNX model;
    # the ONNX converter respects the default device/dtype set here.
    initialize_device(args.device, args.dtype)
    net, vnnlib_name = load_model_and_instance(args.instance)
    run_diagnostic(
        net,
        device=args.device,
        dtype=args.dtype,
        subproblem_batch_size=args.batch,
        eta_iters=args.eta_iters,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        time_budget_s=args.budget,
    )
