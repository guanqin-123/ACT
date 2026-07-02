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
import copy
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

import torch
import torch.fx as fx  # noqa: E402
import torch.nn as nn  # noqa: E402

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
        f.write("sat\n(\n")
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


def build_fast_config(config_label, *, llm_backend="mock",
                      llm_decisions="split,frontier,refine,input_split",
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


# -----------------------------------------------------------------------------
# Merge split-ReLUs: invert the ReluSplitter benchmark transformation.
#
# ReluSplitter rewrites a linear pre-activation  z = W_orig x + b_orig  into a
# DENSE -> ReLU -> DENSE "sandwich" that computes the SAME affine map by
# exploiting  a = ReLU(a) - ReLU(-a): each base row (w,b) is emitted as an
# anti-parallel pair (+(w,b), -(w,b)) in the first DENSE, and the second DENSE
# recombines the pair with opposite-sign weights so its output is again the
# linear z. The spurious (always-unstable) ReLUs only loosen the dual
# relaxation, leaving pct>=0.4 instances "unknown". This pass detects any
# DENSE -> ReLU -> DENSE sandwich that is PROVABLY a global affine map and
# collapses it back to a single DENSE -- exactly semantics-preserving. It is a
# strict no-op unless global affinity is certified, so genuine ReLU layers
# (e.g. ACAS Xu) are left untouched.
#
# Exact soundness certificate (over R^d): group the first DENSE's rows by their
# AUGMENTED direction (w_i | b_i); rows n in a group satisfy a_n = t_n * a_rep.
# With ReLU(t*a)=t*ReLU(a) (t>0) and ReLU(t*a)=|t|*(ReLU(a)-a) (t<0), output k
# receives  R_g[k]*ReLU(a_rep) + L_g[k]*a_rep, where R_g[k]=sum_n W2[k,n]|t_n|.
# The sandwich is affine  <=>  R_g[k]==0 for every group g and output k. The
# collapsed map is  M[k]=sum_g L_g[k]*w_rep_g,  m[k]=b2[k]+sum_g L_g[k]*b_rep_g
# with  L_g[k] = -sum_{n: t_n<0} W2[k,n]|t_n|  (closed form, exact).
# -----------------------------------------------------------------------------

# grouping tolerance on |cos| between augmented rows (exact copies give |cos|==1)
_MERGE_RTOL = 1e-6
# affinity certificate threshold: R is exactly 0 for a true split, O(0.1..1) for
# a genuine ReLU layer, so this cleanly separates them while tolerating ULP.
_MERGE_AFFINE_TOL = 1e-8


def _iter_dense_relu_dense(gm: fx.GraphModule):
    """Yield (l1_node, relu_node, l2_node) for sole-consumer Linear->ReLU->Linear chains.

    Only fires when the ReLU output feeds nothing but the next Linear (and the
    first Linear feeds nothing but the ReLU), so merging cannot affect any other
    consumer of the intermediate activations.
    """
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.op != "call_module" or not isinstance(modules.get(node.target), nn.Linear):
            continue
        if len(node.users) != 1:
            continue
        relu = next(iter(node.users))
        if relu.op != "call_module" or not isinstance(modules.get(relu.target), nn.ReLU):
            continue
        if len(relu.users) != 1 or len(relu.args) != 1 or relu.args[0] is not node:
            continue
        l2 = next(iter(relu.users))
        if l2.op != "call_module" or not isinstance(modules.get(l2.target), nn.Linear):
            continue
        if not l2.args or l2.args[0] is not relu:
            continue
        yield node, relu, l2


def _certify_affine_collapse(l1: nn.Linear, l2: nn.Linear):
    """Return (M, m, n_merged) in float64 if l1->ReLU->l2 is a global affine map, else None.

    n_merged = number of rows removed = sum over augmented-direction groups of
    (group_size - 1). Computation is in float64; the caller casts to model dtype.
    """
    dev = l1.weight.device
    W1 = l1.weight.detach().double()
    out1, in1 = W1.shape
    b1 = l1.bias.detach().double() if l1.bias is not None else torch.zeros(out1, dtype=torch.float64, device=dev)
    W2 = l2.weight.detach().double()
    out2 = W2.shape[0]
    b2 = l2.bias.detach().double() if l2.bias is not None else torch.zeros(out2, dtype=torch.float64, device=dev)

    aug = torch.cat([W1, b1.unsqueeze(1)], dim=1)
    norms = aug.norm(dim=1)
    unit = aug / norms.clamp_min(1e-30).unsqueeze(1)

    visited = [False] * out1
    groups: list[list[tuple[int, float]]] = []
    for i in range(out1):
        if visited[i]:
            continue
        visited[i] = True
        grp = [(i, 1.0)]
        if norms[i] > 1e-30:
            ui = unit[i]
            for j in range(i + 1, out1):
                if visited[j] or norms[j] <= 1e-30:
                    continue
                if abs(abs(float(ui @ unit[j])) - 1.0) <= _MERGE_RTOL:
                    t = float(aug[j] @ aug[i] / (aug[i] @ aug[i]))
                    grp.append((j, t))
                    visited[j] = True
        groups.append(grp)

    thr = _MERGE_AFFINE_TOL * max(1.0, float(W2.abs().max()) if W2.numel() else 1.0)
    M = torch.zeros(out2, in1, dtype=torch.float64, device=dev)
    m = b2.clone()
    n_merged = 0
    for grp in groups:
        idx = [n for n, _ in grp]
        ts = torch.tensor([t for _, t in grp], dtype=torch.float64, device=dev)
        cols = W2[:, idx]
        relu_coeff = (cols * ts.abs().unsqueeze(0)).sum(dim=1)
        if float(relu_coeff.abs().max()) > thr:
            return None
        neg = ts < 0
        if bool(neg.any()):
            lin_coeff = -(cols[:, neg] * ts[neg].abs().unsqueeze(0)).sum(dim=1)
            rep = grp[0][0]
            M += torch.outer(lin_coeff, W1[rep])
            m += lin_coeff * b1[rep]
        n_merged += len(grp) - 1

    return None if n_merged == 0 else (M, m, n_merged)


def _splice_affine(gm: fx.GraphModule, l1_node, relu_node, l2_node, M, m) -> None:
    """Replace the sandwich with a single Linear: reuse l1's node (rewrite its
    weights), rewire l2's consumers to l1, and erase the dead ReLU + l2 nodes."""
    l1 = dict(gm.named_modules())[l1_node.target]
    dtype, dev = l1.weight.dtype, l1.weight.device
    l1.weight = nn.Parameter(M.to(dtype=dtype, device=dev), requires_grad=False)
    l1.bias = nn.Parameter(m.to(dtype=dtype, device=dev), requires_grad=False)
    l1.out_features, l1.in_features = int(M.shape[0]), int(M.shape[1])
    for consumer in list(l2_node.users):
        consumer.replace_input_with(l2_node, l1_node)
    gm.graph.erase_node(l2_node)
    gm.graph.erase_node(relu_node)
    gm.graph.lint()
    gm.recompile()


def merge_split_relus(model: nn.Module):
    """Collapse every provably-affine DENSE->ReLU->DENSE sandwich into one DENSE.

    Returns (merged_model, n_merged). The input ``model`` is never mutated (work
    happens on a deepcopy); when nothing is merged the ORIGINAL object is
    returned so callers can keep using it unchanged.
    """
    if not isinstance(model, fx.GraphModule):
        return model, 0
    gm = copy.deepcopy(model)
    total = 0
    while True:
        for l1_node, relu_node, l2_node in _iter_dense_relu_dense(gm):
            mods = dict(gm.named_modules())
            certified = _certify_affine_collapse(mods[l1_node.target], mods[l2_node.target])
            if certified is None:
                continue
            M, m, n = certified
            _splice_affine(gm, l1_node, relu_node, l2_node, M, m)
            total += n
            break
        else:
            break
    return (gm, total) if total else (model, 0)


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
    ap.add_argument("--fuzzing-seconds", type=float, default=10.0)
    ap.add_argument("--fuzzing-scale", type=float, default=0.5)
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
    ap.add_argument("--input-split-dims", type=int, default=10,
                    help="input dimension threshold at or below which BaB switches to "
                         "input-domain splitting with full per-node bound recomputation "
                         "(the ACAS Xu regime); 0 disables the profile")
    args = ap.parse_args()

    cuda_ok = torch.cuda.is_available()
    print(f"[env] torch={torch.__version__} cuda_build={torch.version.cuda} "
          f"cuda_available={cuda_ok} "
          f"device_count={torch.cuda.device_count()} resolved_device={args.device}",
          file=sys.stderr, flush=True)
    if not cuda_ok:
        print("[env] WARNING: running on CPU - expect timeouts. torch cannot see a GPU; "
              "check the NVIDIA driver (nvidia-smi must work).",
              file=sys.stderr, flush=True)

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

    input_dim = int(sr[3][0].tensor.numel()) if sr[3] else 0
    low_dim = 0 < input_dim <= args.input_split_dims
    if low_dim:
        print(f"[profile] input_dim={input_dim} <= {args.input_split_dims}: "
              f"input-split BaB with per-node bound recompute", file=sys.stderr, flush=True)

    def raw_forward(x):
        if param is not None:
            x = x.to(device=param.device, dtype=param.dtype)
        with torch.no_grad():
            return raw_model(x)

    verify_model, n_merged = merge_split_relus(raw_model)
    if n_merged:
        print(f"[merge] fused {n_merged} split-ReLU neurons", file=sys.stderr, flush=True)
        sr = tuple(verify_model if i == 2 else v for i, v in enumerate(sr))

    wrapped = next(iter(synthesize_models_from_specs([sr]).values()))

    if args.fuzzing_seconds > 0 and remaining() > 1.0:
        try:
            ce, _ = fuzz_precheck(wrapped, sr[3], min(args.fuzzing_seconds, remaining()), args.fuzzing_scale)
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
            if low_dim:
                # Low-dim regime (ACAS Xu-style): bisect the input domain and recompute
                # every child's intermediate bounds on its own sub-box. Neuron splits and
                # frozen root bounds - the large-net defaults - certify nothing here: the
                # branching gain of an input split lives entirely in the recomputed
                # intermediate relaxations. Uncapped frontier, since any eviction makes
                # certification permanently impossible for the run.
                cfg.branching_method = "width"
                cfg.multi_split_levels = 1
                cfg.reuse_root_bounds = False
                cfg.intermediate_refine = "none"
                cfg.frontier_cap = 0
            clear_violation_check_module_cache()
            return verify_bab_batched(net, solver_factory=TorchLPSolver, config=cfg,
                                      max_batch_size=args.max_batch_size, time_budget_s=max(1.0, budget))

        if args.solver_tier == "auto":
            if low_dim:
                # With input splits + per-node recompute, the cheap one-shot 'dual'
                # bound is the workhorse (ACAS Xu prop_1 certifies in ~0.1s / 500
                # nodes vs 17s with alpha+eta); escalate only if it can't close.
                result = _verify("dual", remaining())
                if result.status not in (VerifyStatus.CERTIFIED, VerifyStatus.FALSIFIED) and remaining() > 1.0:
                    result = _verify("dual_alpha_eta", remaining())
            else:
                # The one-shot 'dual' bound certifies tight nets (e.g. ViT attention) at the
                # root in ~0.2s; escalate to the iterative alpha+eta tier + BaB only if
                # still UNKNOWN.
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
