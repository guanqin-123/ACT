#!/usr/bin/env python3
"""ACT VNN-COMP 2026 single-instance runner.

Wraps the ACT pipeline (arbitrary onnx+vnnlib load, ACTFuzzer mutation pre-attack for
FALSIFICATION, dual_alpha_eta BaB for CERTIFICATION) and emits the VNN-COMP result contract:

    line 1 : unsat | sat | timeout | unknown
    if sat : lines 2+ = the counterexample as a VNNLIB-1.0 flat assignment
             ``((X_0 v)...(Y_0 v)...)``. This flat form is the only layout the
             VNN-COMP scoring re-checker parses, so it is emitted for every
             instance regardless of the input spec's VNNLIB version.

Both the verifier's CE (result.counterexample) and the fuzzer's CE (.input) are
in model input space; the output Y is (re)computed by running the raw
ONNX-converted model on the CE input, so it always matches the network the
harness re-executes. Invoked by run_instance.sh as:

    python act_run_instance.py <onnx> <vnnlib> <results_file> <timeout_s> [opts]

We use llm_probe for LLM-guided BaB.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

import torch

from act.util.device_manager import initialize_device


def _write_result(out_path: str, token: str, *, x=None, y=None) -> None:
    """Emit the result token, plus a ``sat`` witness as a flat ``((X_i v)...(Y_j v)...)``.
    VNN-COMP scoring parses only this VNNLIB-1.0 flat form (row-major ``X_i``), so it
    is used for every spec version; the tensor layout would fail the CE re-check."""
    with open(out_path, "w") as f:
        if token != "sat" or x is None or y is None:
            f.write(token + "\n")
            return
        xf = x.detach().cpu().flatten().tolist()
        yf = y.detach().cpu().flatten().tolist()
        f.write("sat\n(")
        for i, v in enumerate(xf):
            f.write(f"(X_{i} {v:.16g})\n")
        for j, v in enumerate(yf):
            f.write(f"(Y_{j} {v:.16g})\n")
        f.write(")\n")


def fuzz_precheck(wrapped_model, seeds, budget, scale=0.5):
    """Anisotropic-PGD pre-attack (the FALSIFY path); returns (counterexample_or_None, elapsed_s)."""
    from act.pipeline.fuzzing import ACTFuzzer, FuzzingConfig
    from act.util.device_manager import get_default_device

    device = get_default_device()
    wrapped_model = wrapped_model.to(device)
    seeds = [
        type(s)(tensor=s.tensor.to(device), label=(s.label.to(device) if s.label is not None else None))
        for s in seeds
    ]
    cfg = FuzzingConfig.from_yaml(
        timeout_seconds=float(budget),
        max_iterations=10_000_000,
        mutation_weights={"gradient": 0.0, "pgd": 1.0, "activation": 0.0,
                          "boundary": 0.0, "random": 0.0},
        perturb_mode="adaptive_perdim",
        perturb_scale=scale,
        save_counterexamples=False,
        stop_on_first_violation=True,
        verbose=0,
    )
    report = ACTFuzzer(wrapped_model=wrapped_model, initial_seeds=seeds, config=cfg).fuzz()
    ce = report.counterexamples[0] if report.counterexamples else None
    return ce, report.total_time


def build_fast_config(config_label, *, llm_backend="mock", llm_decisions="split,frontier,refine",
                      llm_timeout=30.0, llm_model="", llm_cadence=1, llm_neuron_topk=0,
                      llm_log=False, multi_split_levels=4, max_depth=1_000_000,
                      max_nodes=1_000_000_000, solver_tier="dual_alpha_eta", dual_n_iters=100):
    """BaBConfig for real VNNLIB instances: ``fsb``/``babsr`` keep single-neuron splits,
    ``gain``/``gain+llm`` use joint-split depth, and only ``gain+llm`` enables the LLM probe."""
    from act.back_end.config import BaBConfig

    branching_method = config_label if config_label in ("fsb", "babsr") else "gain"
    common: dict[str, Any] = dict(
        solver_tier=solver_tier,
        branching_method=branching_method,
        bounding_method="topk",
        bounding_order="depth_lb",
        frontier_cap=25000,
        max_depth=max_depth,
        max_nodes=max_nodes,
        dual_n_iters=dual_n_iters,
        lr_alpha=0.25,
        lr_beta=0.1,
        lr_decay=0.98,
        incremental_start_enabled=True,
        per_class_alpha=True,
        reuse_root_bounds=True,
        intermediate_refine="all",
        presplit_levels=0,
        eta_only_children=False,
        multi_split_levels=1 if branching_method != "gain" else max(1, int(multi_split_levels)),
    )
    if config_label != "gain+llm":
        return BaBConfig(**common)
    cfg = BaBConfig(
        llm_probe_enabled=True,
        llm_probe_backend=llm_backend,
        llm_probe_decisions=llm_decisions,
        llm_probe_timeout=llm_timeout,
        llm_probe_cadence=llm_cadence,
        llm_probe_neuron_topk=llm_neuron_topk,
        llm_probe_log=llm_log,
        **common,
    )
    if llm_model:
        cfg.llm_probe_model = llm_model
    return cfg


def _sr_from_paths(onnx_path, vnnlib_path, category="custom"):
    """Build one spec_result from an arbitrary (onnx, vnnlib) pair (ONNX->torch, input-shape
    probe, VNNLIB parse+validate). Raises SystemExit on missing or invalid inputs."""
    from act.front_end.spec_creator_base import LabeledInputTensor
    from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator
    from act.front_end.vnnlib_loader.data_model_loader import _parse_vnnlib_with_shape_probe
    from act.front_end.vnnlib_loader.onnx_converter import convert_onnx_to_pytorch, get_onnx_input_shape
    from act.front_end.vnnlib_loader.vnnlib_parser import extract_label_from_vnnlib

    onnx_p, vnnlib_p = Path(onnx_path), Path(vnnlib_path)
    if not onnx_p.exists():
        raise SystemExit(f"ONNX not found: {onnx_p}")
    if not vnnlib_p.exists():
        raise SystemExit(f"VNNLIB not found: {vnnlib_p}")
    model = convert_onnx_to_pytorch(onnx_p, simplify=True)
    model.eval()
    try:
        input_shape = get_onnx_input_shape(onnx_p)
    except Exception:
        input_shape = None
    input_tensor, _meta = _parse_vnnlib_with_shape_probe(vnnlib_p, model, input_shape)
    lbl = extract_label_from_vnnlib(vnnlib_p)
    label = torch.tensor([lbl], dtype=torch.int64) if lbl is not None else None
    instance_data = {
        "model": model,
        "labeled_tensor": LabeledInputTensor(tensor=input_tensor, label=label),
        "vnnlib_path": str(vnnlib_p),
    }
    sr = VNNLibSpecCreator()._create_specs_for_single_instance(
        category, f"{onnx_p.stem}__{vnnlib_p.stem}", instance_data, validate_shapes=True)
    if sr is None:
        raise SystemExit(f"Spec creation failed (unsupported/invalid spec) for {vnnlib_p.name}")
    return sr


def main() -> None:
    ap = argparse.ArgumentParser(description="ACT VNN-COMP 2026 single-instance runner")
    ap.add_argument("onnx")
    ap.add_argument("vnnlib")
    ap.add_argument("output")
    ap.add_argument("timeout", type=float)
    ap.add_argument("--config", default="gain", choices=["fsb", "babsr", "gain", "gain+llm"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--attack-seconds", type=float, default=10.0)
    ap.add_argument("--attack-scale", type=float, default=0.5)
    ap.add_argument("--max-batch-size", default="auto",
                    help="int or 'auto' (net/GPU-aware, avoids OOM)")
    ap.add_argument("--margin", type=float, default=5.0,
                    help="seconds reserved before the harness kill (timeout+60)")
    ap.add_argument("--llm-backend", default="openrouter")
    ap.add_argument("--llm-model", default="google/gemini-2.5-flash-lite")
    ap.add_argument("--llm-timeout", type=float, default=30.0,
                    help="per-call LLM wall-clock cap; a slower reply falls back to baseline")
    ap.add_argument("--solver-tier", default="auto",
                    choices=["auto", "lp", "dual", "dual_alpha", "dual_alpha_eta"],
                    help="'auto' = cheap one-shot 'dual' bound, then escalate to 'dual_alpha_eta'")
    args = ap.parse_args()

    t0 = time.time()
    initialize_device(args.device, args.dtype)

    from act.back_end.bab.bab import clear_violation_check_module_cache, verify_bab_batched
    from act.back_end.solver.solver_torchlp import TorchLPSolver
    from act.front_end.model_synthesis import synthesize_models_from_specs
    from act.pipeline.verification.torch2act import TorchToACT
    from act.util.stats import VerifyStatus

    def remaining() -> float:
        return args.timeout - (time.time() - t0) - args.margin

    try:
        sr = _sr_from_paths(args.onnx, args.vnnlib)
    except SystemExit as exc:
        print(f"[load failed] {exc}", file=sys.stderr)
        _write_result(args.output, "unknown")
        return

    raw_model = sr[2]
    param = next(raw_model.parameters(), None)

    def raw_forward(x):
        if param is not None:
            x = x.to(device=param.device, dtype=param.dtype)
        with torch.no_grad():
            return raw_model(x)

    wrapped = next(iter(synthesize_models_from_specs([sr]).values()))

    if args.attack_seconds > 0 and remaining() > 1.0:
        try:
            ce, _ = fuzz_precheck(wrapped, sr[3], min(args.attack_seconds, remaining()), args.attack_scale)
        except Exception as exc:
            ce = None
            print(f"[attack skipped] {exc}", file=sys.stderr)
        if ce is not None:
            x = ce.input if hasattr(ce, "input") else ce
            _write_result(args.output, "sat", x=x, y=raw_forward(x))
            return

    if remaining() <= 1.0:
        _write_result(args.output, "timeout")
        return
    try:
        net = TorchToACT(wrapped).run()

        def _verify(tier, budget):
            cfg = build_fast_config(args.config, llm_backend=args.llm_backend, llm_model=args.llm_model,
                                    llm_timeout=args.llm_timeout, solver_tier=tier)
            clear_violation_check_module_cache()
            return verify_bab_batched(net, solver_factory=TorchLPSolver, config=cfg,
                                      max_batch_size=args.max_batch_size, time_budget_s=max(1.0, budget))

        if args.solver_tier == "auto":
            # The one-shot 'dual' bound certifies tight nets (e.g. ViT attention) at the root in
            # ~0.2s; escalate to the slower iterative alpha+eta tier + BaB only if still UNKNOWN.
            result = _verify("dual", min(remaining(), 15.0))
            if result.status not in (VerifyStatus.CERTIFIED, VerifyStatus.FALSIFIED) and remaining() > 1.0:
                result = _verify("dual_alpha_eta", remaining())
        else:
            result = _verify(args.solver_tier, remaining())
    except Exception as exc:
        print(f"[verify error] {exc}", file=sys.stderr)
        _write_result(args.output, "unknown")
        return

    if result.status == VerifyStatus.CERTIFIED:
        _write_result(args.output, "unsat")
    elif result.status == VerifyStatus.FALSIFIED and result.counterexample is not None:
        x = result.counterexample
        _write_result(args.output, "sat", x=x, y=raw_forward(x))
    elif result.status == VerifyStatus.TIMEOUT:
        _write_result(args.output, "timeout")
    else:
        _write_result(args.output, "unknown")


if __name__ == "__main__":
    main()
