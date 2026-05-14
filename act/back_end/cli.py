#!/usr/bin/env python3
"""
ACT Back-End Command-Line Interface.

Provides CLI tools for core verification operations:
- Network verification (single-shot and branch-and-bound)
- Network factory (generate example networks from YAML)
- Network serialization (save/load ACT Net structures)
- Analysis and constraint inspection

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

import argparse
import datetime
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from act.util.cli_utils import add_device_args, initialize_from_args


def _make_solver(solver_name: str):
    """Instantiate a solver backend by name."""
    from act.back_end.solver.solver_interval import TorchLPSolver

    if solver_name == "gurobi":
        from act.back_end.solver.solver_gurobi import GurobiSolver

        return GurobiSolver()
    if solver_name == "torch":
        return TorchLPSolver()
    # "auto": try Gurobi, fall back to TorchLP
    try:
        from act.back_end.solver.solver_gurobi import GurobiSolver

        return GurobiSolver()
    except Exception:
        return TorchLPSolver()


def run_verification(args, backend_cfg):
    """Run verification on a network using *backend_cfg*."""
    from act.back_end.serialization.serialization import load_net_from_file
    from act.back_end.verifier import verify_once
    from act.back_end.bab import verify_bab_batched
    from act.util.stats import VerifyStatus

    net = load_net_from_file(args.network)
    print(f"Loaded {len(net.layers)}-layer net; solver={backend_cfg.solver}")

    if backend_cfg.bab_enabled:
        results = verify_bab_batched(
            net,
            solver_factory=lambda: _make_solver(backend_cfg.solver),
            config=backend_cfg.bab,
            time_budget_s=backend_cfg.timeout,
        )
        all_certified = True
        multi = len(results) > 1
        for i, result in enumerate(results):
            prefix = f"Sample {i}: " if multi else ""
            print(f"{prefix}{result.status}")
            if backend_cfg.verbose and result.metadata:
                for k, v in result.metadata.items():
                    print(f"  {k}: {v}")
            if result.status != VerifyStatus.CERTIFIED:
                all_certified = False
        return 0 if all_certified else 1

    for i, lane in enumerate(verify_once(net=net)):
        print(f"Lane {i}: {lane.status}")
        if backend_cfg.verbose and lane.metadata:
            for k, v in lane.metadata.items():
                print(f"  {k}: {v}")
    return 0


def run_verify_all(args, backend_cfg):
    """Verify every .json net under a directory in one Python process."""
    import glob
    import os
    from act.back_end.serialization.serialization import load_net_from_file
    from act.back_end.verifier import verify_once
    from act.back_end.bab import verify_bab_batched
    from act.util.stats import VerifyStatus

    nets_dir = args.verify_all
    if not os.path.isdir(nets_dir):
        print(f"❌ --verify-all: directory not found: {nets_dir}")
        return 1

    paths = sorted(
        p for p in glob.glob(os.path.join(nets_dir, "*.json"))
        if "_meta" not in os.path.basename(p) and "manifest" not in os.path.basename(p)
    )
    if not paths:
        print(f"❌ --verify-all: no .json nets under {nets_dir}")
        return 1

    print(f"\n{'=' * 80}")
    print(f"ACT VERIFY-ALL  ({len(paths)} nets, solver={backend_cfg.solver}, "
          f"bab={'on' if backend_cfg.bab_enabled else 'off'})")
    print(f"{'=' * 80}\n")

    total = len(paths)
    failures: list = []
    errors: list = []

    for idx, net_path in enumerate(paths, 1):
        name = os.path.basename(net_path)
        try:
            net = load_net_from_file(net_path)
            if backend_cfg.bab_enabled:
                results = verify_bab_batched(
                    net,
                    solver_factory=lambda: _make_solver(backend_cfg.solver),
                    config=backend_cfg.bab,
                    time_budget_s=backend_cfg.timeout,
                )
            else:
                results = list(verify_once(net=net))
            statuses = [r.status for r in results]
            tag = "  ".join(s.name for s in statuses)
            print(f"[{idx:>3}/{total}] {name}: {tag}")
            for r in results:
                if r.status not in (VerifyStatus.CERTIFIED, VerifyStatus.UNKNOWN):
                    failures.append((name, r.status))
        except Exception as e:  # noqa: BLE001 — surface per-net error, keep iterating
            print(f"[{idx:>3}/{total}] {name}: ❌ ERROR — {e}")
            errors.append((name, str(e)))

    print(f"\n{'=' * 80}")
    print(f"verify-all summary:  total={total}  failures={len(failures)}  errors={len(errors)}")
    if failures:
        print("  Failures (non-CERTIFIED/UNKNOWN):")
        for n, s in failures[:20]:
            print(f"    - {n}: {s.name}")
    if errors:
        print("  Errors:")
        for n, e in errors[:20]:
            print(f"    - {n}: {e[:120]}")
    print(f"{'=' * 80}\n")

    # Exit 0 if no hard errors (UNKNOWN is acceptable). FALSIFIED counts as failure.
    return 1 if (failures or errors) else 0


def run_network_factory(args, backend_cfg):
    """Generate example networks using TF-aware NetFactory."""
    print(f"\n{'=' * 80}")
    print(f"ACT NETWORK FACTORY")
    print(f"{'=' * 80}\n")

    from act.back_end.net_factory import NetFactory

    gen = backend_cfg.generation

    if gen.tf_targets:
        print(f"TF targets: {gen.tf_targets} (mode: {gen.registry_mode})")
    print(f"Config: {gen.gen_config_path}")
    print(f"Output: {gen.output_dir}")
    print(f"Instances: {gen.num_instances}, Seed: {gen.base_seed}")
    print()

    try:
        factory = NetFactory(
            gen_config_path=gen.gen_config_path,
            output_dir=gen.output_dir,
            base_seed=gen.base_seed,
            num_instances=gen.num_instances,
            name_prefix=gen.name_prefix,
            tf_targets=gen.tf_targets,
            registry_mode=gen.registry_mode,
            write_manifest=gen.write_manifest,
        )
        factory.generate()
        print(f"\n{'=' * 80}")
        print(f"✓ Network generation complete")
        print(f"{'=' * 80}\n")

        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if backend_cfg.verbose:
            import traceback

            traceback.print_exc()
        return 1


def run_network_info(args):
    """Display information about a network."""
    print(f"\n{'=' * 80}")
    print(f"NETWORK INFORMATION")
    print(f"{'=' * 80}\n")

    from act.back_end.serialization.serialization import load_net_from_file
    from act.back_end.layer_schema import LayerKind

    print(f"Loading network from: {args.network}\n")
    net = load_net_from_file(args.network)

    # Basic info
    print(f"Network: {Path(args.network).stem}")
    print(f"Total layers: {len(net.layers)}")
    print(f"Predecessors: {sum(len(p) for p in net.preds.values())} edges")
    print(f"Successors: {sum(len(s) for s in net.succs.values())} edges")

    # Layer breakdown by kind
    layer_kinds = {}
    for layer in net.layers:
        kind = layer.kind
        layer_kinds[kind] = layer_kinds.get(kind, 0) + 1

    print(f"\nLayer breakdown:")
    for kind, count in sorted(layer_kinds.items()):
        print(f"  {kind:20s}: {count}")

    # Detailed layer info if verbose
    if args.verbose:
        print(f"\n{'=' * 80}")
        print(f"DETAILED LAYER INFORMATION")
        print(f"{'=' * 80}\n")

        for layer in net.layers:
            print(f"Layer {layer.id}: {layer.kind}")
            print(f"  In vars: {layer.in_vars}")
            print(f"  Out vars: {layer.out_vars}")
            if layer.params:
                print(f"  Params: {layer.params}")

            # Show predecessors
            preds = net.preds.get(layer.id, [])
            if preds:
                print(f"  Predecessors: {preds}")

            # Show successors
            succs = net.succs.get(layer.id, [])
            if succs:
                print(f"  Successors: {succs}")
            print()

    print(f"{'=' * 80}\n")
    return 0


def run_serialization_test(args):
    """Test network serialization (save/load round-trip)."""
    print(f"\n{'=' * 80}")
    print(f"SERIALIZATION TEST")
    print(f"{'=' * 80}\n")

    from act.back_end.serialization.test_serialization import main as test_main

    print("Running serialization tests...\n")
    result = test_main()

    print(f"\n{'=' * 80}")
    if result == 0:
        print("✓ All serialization tests passed")
    else:
        print("❌ Some serialization tests failed")
    print(f"{'=' * 80}\n")

    return result


def list_examples(args):
    """List available example networks."""
    print(f"\n{'=' * 80}")
    print(f"AVAILABLE EXAMPLE NETWORKS")
    print(f"{'=' * 80}\n")

    from act.pipeline.verification.model_factory import ModelFactory

    factory = ModelFactory()
    names = factory.list_networks()
    print(f"Total networks: {len(names)}\n")

    # Group by category (inferred from filename)
    categories: dict = {}
    for name in names:
        info = factory.get_network_info(name)
        nl = name.lower()
        if "mnist" in nl:
            cat = "MNIST Classification"
        elif "cifar" in nl:
            cat = "CIFAR Classification"
        elif "control" in nl:
            cat = "Control Systems"
        elif "reachability" in nl:
            cat = "Reachability Analysis"
        else:
            cat = "Generated"
        categories.setdefault(cat, []).append((name, info))

    for cat, nets in sorted(categories.items()):
        print(f"{cat} ({len(nets)} networks):")
        print("-" * 70)
        for name, info in sorted(nets):
            shape = info.get("input_shape", "?")
            layers = info.get("num_layers", "?")
            print(f"  {name:40s}  shape={shape}  layers={layers}")
        print()

    print(f"{'=' * 80}")
    print("To generate networks: python -m act.back_end --generate")
    print(f"{'=' * 80}\n")

    return 0


def _bench_default_path(kind: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("act", "pipeline", "log", f"bench_{kind}_{ts}.json")


def _write_bench_result(out_path: str, result: object) -> None:
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"Wrote {out_path}")


def _run_bench_cnn(out_path: str) -> int:
    import torch
    from act.back_end.serialization.serialization import load_net_from_file
    from act.back_end.analyze import analyze
    from act.back_end.core import Fact, ConSet
    from act.back_end.verifier import find_entry_layer_id, gather_input_spec_layers, seed_from_input_specs

    nets = sorted(
        p for p in glob.glob("act/back_end/examples/nets/cnn2d_plain_*.json")
        if "_meta" not in p
    )
    if not nets:
        print("No CNN example nets found at act/back_end/examples/nets/cnn2d_plain_*.json")
        return 1

    results: Dict[str, Any] = {}
    for path in nets:
        net = load_net_from_file(path)
        entry = find_entry_layer_id(net)
        seed = seed_from_input_specs(gather_input_spec_layers(net))
        fact = Fact(bounds=seed, cons=ConSet())
        for _ in range(2):
            analyze(net, entry, fact)
        times: List[float] = []
        for _ in range(5):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            analyze(net, entry, fact)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        results[path] = {
            "mean": statistics.mean(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0,
            "all": times,
        }
        print(f"  {path}: mean={results[path]['mean']:.4f}s")

    _write_bench_result(out_path, results)
    return 0


def _run_bench_hybridz(out_path: str) -> int:
    import torch
    from act.back_end.core import Net, Layer, Bounds, Fact, ConSet
    from act.back_end.layer_schema import LayerKind
    from act.back_end.analyze import analyze
    from act.back_end.transfer_functions import set_transfer_function_mode
    from act.front_end.specs import OutputSpec

    def _build_net(B: int = 1, n_in: int = 8, n_hid: int = 16, n_out: int = 8) -> Net:
        layers: List[Any] = []
        next_id = 0
        next_var = 0

        def alloc_vars(n: int) -> List[int]:
            nonlocal next_var
            vs = list(range(next_var, next_var + n))
            next_var += n
            return vs

        in_v = alloc_vars(n_in)
        layers.append(Layer(id=next_id, kind=LayerKind.INPUT.value,
            params={"shape": (B, n_in), "dtype": "torch.float32"},
            in_vars=[], out_vars=in_v))
        next_id += 1
        layers.append(Layer(id=next_id, kind=LayerKind.INPUT_SPEC.value,
            params={"kind": "BOX",
                    "lb": torch.full((B, n_in), -1.0),
                    "ub": torch.full((B, n_in),  1.0)},
            in_vars=in_v, out_vars=in_v))
        next_id += 1
        h1_v = alloc_vars(n_hid)
        W1 = torch.randn(n_hid, n_in)
        b1 = torch.zeros(n_hid)
        layers.append(Layer(id=next_id, kind=LayerKind.DENSE.value,
            params={"weight": W1, "in_features": n_in, "out_features": n_hid,
                    "weight_pos": W1.clamp(min=0), "weight_neg": W1.clamp(max=0),
                    "bias": b1, "input_shape": (n_in,)},
            in_vars=in_v, out_vars=h1_v))
        next_id += 1
        layers.append(Layer(id=next_id, kind=LayerKind.RELU.value,
            params={"input_shape": (n_hid,)},
            in_vars=h1_v, out_vars=h1_v))
        next_id += 1
        out_v = alloc_vars(n_out)
        W2 = torch.randn(n_out, n_hid)
        b2 = torch.zeros(n_out)
        layers.append(Layer(id=next_id, kind=LayerKind.DENSE.value,
            params={"weight": W2, "in_features": n_hid, "out_features": n_out,
                    "weight_pos": W2.clamp(min=0), "weight_neg": W2.clamp(max=0),
                    "bias": b2, "input_shape": (n_hid,)},
            in_vars=h1_v, out_vars=out_v))
        next_id += 1
        assert_params = OutputSpec(
            kind="LINEAR_LE",
            c=torch.zeros(n_out),
            d=torch.tensor(1.0),
        ).encode_linear(B=B, n_out=n_out, device=torch.device("cpu"), dtype=torch.float32)
        layers.append(Layer(id=next_id, kind=LayerKind.ASSERT.value,
            params=assert_params, in_vars=out_v, out_vars=out_v))
        preds = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]}
        succs = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []}
        return Net(layers=layers, preds=preds, succs=succs)

    torch.manual_seed(42)
    net = _build_net()
    set_transfer_function_mode("hybridz")
    entry_id = next(l.id for l in net.layers if l.kind == LayerKind.INPUT.value)
    spec_layer = next(l for l in net.layers if l.kind == LayerKind.INPUT_SPEC.value)
    import torch as _torch
    lb_t = cast(_torch.Tensor, spec_layer.params["lb"])
    ub_t = cast(_torch.Tensor, spec_layer.params["ub"])
    seed = Bounds(lb_t.clone(), ub_t.clone())
    fact = Fact(bounds=seed, cons=ConSet())
    for _ in range(2):
        analyze(net, entry_id, fact)
    times: List[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        analyze(net, entry_id, fact)
        times.append(time.perf_counter() - t0)
    result = {
        "mean": statistics.mean(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
    }
    print(f"  hybridz synthetic 4-layer MLP: mean={result['mean']:.4f}s")
    _write_bench_result(out_path, result)
    return 0


def run_bench(args) -> int:
    """Run timing benchmarks for CNN and/or HybridZ analyze() code paths."""
    kind = args.bench
    bench_out = getattr(args, "bench_out", None)

    print(f"\n{'=' * 80}")
    print(f"ACT BENCH: {kind.upper()}")
    print(f"{'=' * 80}\n")

    if kind in ("cnn", "all"):
        out_path = bench_out if (bench_out and kind == "cnn") else _bench_default_path("cnn")
        print(f"--- CNN benchmark ---")
        rc = _run_bench_cnn(out_path)
        if rc != 0:
            return rc

    if kind in ("hybridz", "all"):
        out_path = bench_out if (bench_out and kind == "hybridz") else _bench_default_path("hybridz")
        print(f"\n--- HybridZ benchmark ---")
        rc = _run_bench_hybridz(out_path)
        if rc != 0:
            return rc

    print(f"\n{'=' * 80}")
    print(f"Bench complete")
    print(f"{'=' * 80}\n")
    return 0


def run_diff_nets(args) -> int:
    """Load two ACT Net JSON files and print a unified-diff-style layer comparison."""
    from act.back_end.serialization.serialization import load_net_from_file

    path_a, path_b = args.diff_nets

    try:
        net_a = load_net_from_file(path_a)
    except Exception as e:
        print(f"Error loading {path_a}: {e}")
        return 1

    try:
        net_b = load_net_from_file(path_b)
    except Exception as e:
        print(f"Error loading {path_b}: {e}")
        return 1

    print(f"\n{'=' * 80}")
    print(f"NET DIFF")
    print(f"  A: {path_a}")
    print(f"  B: {path_b}")
    print(f"{'=' * 80}\n")

    la, lb = len(net_a.layers), len(net_b.layers)
    marker = "  " if la == lb else "!"
    print(f"{marker} Layer count: A={la}  B={lb}")

    n_common = min(la, lb)
    for i in range(n_common):
        lyr_a = net_a.layers[i]
        lyr_b = net_b.layers[i]
        diffs: List[str] = []
        if lyr_a.kind != lyr_b.kind:
            diffs.append(f"kind: {lyr_a.kind!r} -> {lyr_b.kind!r}")
        if len(lyr_a.in_vars) != len(lyr_b.in_vars):
            diffs.append(f"in_vars: {len(lyr_a.in_vars)} -> {len(lyr_b.in_vars)}")
        if len(lyr_a.out_vars) != len(lyr_b.out_vars):
            diffs.append(f"out_vars: {len(lyr_a.out_vars)} -> {len(lyr_b.out_vars)}")
        keys_a = set(lyr_a.params.keys())
        keys_b = set(lyr_b.params.keys())
        if keys_a != keys_b:
            only_a = sorted(keys_a - keys_b)
            only_b = sorted(keys_b - keys_a)
            if only_a:
                diffs.append(f"params only in A: {only_a}")
            if only_b:
                diffs.append(f"params only in B: {only_b}")
        if diffs:
            print(f"! Layer {i:2d} ({lyr_a.kind:20s}): " + "; ".join(diffs))
        else:
            print(f"  Layer {i:2d} ({lyr_a.kind:20s}): identical")

    if la != lb:
        extra_net = net_a if la > lb else net_b
        extra_side = "A" if la > lb else "B"
        for i in range(n_common, max(la, lb)):
            lyr = extra_net.layers[i]
            print(f"+ Layer {i:2d} ({lyr.kind:20s}): only in {extra_side}")

    print(f"\n{'=' * 80}\n")
    return 0


def main():
    """Main CLI entry point for ACT Back-End."""
    parser = argparse.ArgumentParser(
        description="ACT Back-End: Core Verification Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ============================================================================
  # NETWORK FACTORY - Generate example networks
  # ============================================================================
  
  # Generate all example networks from default config
  python -m act.back_end --generate
  
  # Generate with custom config
  python -m act.back_end --generate --config my_config.yaml --output ./networks
  
  # List available example networks
  python -m act.back_end --list-examples
  
  # ============================================================================
  # VERIFICATION - Run verification on networks
  # ============================================================================
  
  # Single-shot verification
  python -m act.back_end --verify --network act/back_end/examples/nets/mnist_robust_easy.json
  
  # Branch-and-bound verification
  python -m act.back_end --verify --network mnist_robust_hard.json --bab
  
  # Custom BaB parameters
  python -m act.back_end --verify --network control_strict.json \\
    --bab --bab-max-depth 10 --bab-max-subproblems 1000 --timeout 300
  
  # Use specific solver
  python -m act.back_end --verify --network cifar_margin_tight.json \\
    --solver gurobi --timeout 60
  
  # ============================================================================
  # NETWORK INSPECTION - Analyze network structure
  # ============================================================================
  
  # Show network information
  python -m act.back_end --info --network mnist_robust_easy.json
  
  # Detailed layer information
  python -m act.back_end --info --network control_balanced.json --verbose
  
  # ============================================================================
  # TESTING - Run internal tests
  # ============================================================================
  
  # Test serialization (save/load round-trip)
  python -m act.back_end --test-serialization
  
  # ============================================================================
  # BENCHMARKING - Time analyze() on example nets
  # ============================================================================
  
  # Benchmark CNN analyze() on all cnn2d_plain_* example nets
  python -m act.back_end --bench cnn
  
  # Benchmark HybridZ analyze() on a synthetic MLP
  python -m act.back_end --bench hybridz
  
  # Run both benchmarks and write JSON output
  python -m act.back_end --bench all
  python -m act.back_end --bench cnn --bench-out /tmp/my_cnn_timing.json
  
  # ============================================================================
  # NET DIFF - Compare two network JSON files
  # ============================================================================
  
  # Compare layer count, kinds, variable widths, and param keys
  python -m act.back_end --diff-nets act/back_end/examples/nets/net_a.json \\
                                      act/back_end/examples/nets/net_b.json
  
  # ============================================================================
  # DEVICE CONFIGURATION
  # ============================================================================
  
  # Use CPU with float32
  python -m act.back_end --verify --network mnist.json --device cpu --dtype float32
  
  # Use GPU with float64
  python -m act.back_end --verify --network cifar.json --device cuda --dtype float64
        """,
    )

    # Command groups
    cmd_group = parser.add_mutually_exclusive_group(required=True)

    cmd_group.add_argument(
        "--generate",
        "-g",
        action="store_true",
        help="Generate example networks from YAML configuration",
    )
    cmd_group.add_argument(
        "--verify", "-v", action="store_true", help="Run verification on a network"
    )
    cmd_group.add_argument(
        "--info", "-i", action="store_true", help="Display network information"
    )
    cmd_group.add_argument(
        "--list-examples",
        "-l",
        action="store_true",
        dest="list_examples",
        help="List available example networks",
    )
    cmd_group.add_argument(
        "--test-serialization",
        action="store_true",
        dest="test_serialization",
        help="Run serialization tests",
    )
    cmd_group.add_argument(
        "--bench",
        type=str,
        choices=["cnn", "hybridz", "all"],
        metavar="{cnn,hybridz,all}",
        dest="bench",
        help="Run analyze() timing benchmarks: cnn nets, hybridz synthetic MLP, or all",
    )
    cmd_group.add_argument(
        "--diff-nets",
        nargs=2,
        metavar=("NET_A", "NET_B"),
        dest="diff_nets",
        help="Load two ACT Net JSON files and print a layer-level diff summary",
    )
    cmd_group.add_argument(
        "--verify-all",
        type=str,
        metavar="NETS_DIR",
        dest="verify_all",
        help=(
            "Verify every .json net under NETS_DIR in ONE Python process "
            "(amortizes ~5s of import startup per net; replaces per-net shell loops in CI)."
        ),
    )

    # Bench options
    bench_group = parser.add_argument_group("Bench Options")
    bench_group.add_argument(
        "--bench-out",
        type=str,
        default=None,
        dest="bench_out",
        help=(
            "Output JSON path for bench results "
            "(default: act/pipeline/log/bench_<kind>_<timestamp>.json)"
        ),
    )

    # Network factory options
    factory_group = parser.add_argument_group("Network Factory Options")
    factory_group.add_argument(
        "--config", "-c", type=str, help="Path to YAML configuration file"
    )
    factory_group.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory for generated networks (default: act/back_end/examples/nets)",
    )
    factory_group.add_argument(
        "--num", type=int, help="Number of networks to generate (generate mode)"
    )
    factory_group.add_argument(
        "--base-seed",
        type=int,
        dest="base_seed",
        help="Base seed for reproducible generation",
    )
    factory_group.add_argument(
        "--name-prefix",
        type=str,
        dest="name_prefix",
        help="Filename prefix for generated networks",
    )
    factory_group.add_argument(
        "--tf-targets",
        type=str,
        nargs="+",
        dest="tf_targets",
        choices=["interval", "hybridz", "dual"],
        help="Target TFs for layer filtering (generate mode)",
    )
    factory_group.add_argument(
        "--registry-mode",
        type=str,
        dest="registry_mode",
        choices=["intersection", "union"],
        default="intersection",
        help="How to combine TF layer sets: 'intersection' (default) or 'union'",
    )

    # Verification options
    verify_group = parser.add_argument_group("Verification Options")
    verify_group.add_argument(
        "--network", "-n", type=str, help="Path to network file (JSON format)"
    )
    verify_group.add_argument(
        "--solver",
        "-s",
        type=str,
        choices=["auto", "gurobi", "torch"],
        default=None,
        help="Solver backend (default: from config.yaml / $ACT_SOLVER / 'auto')",
    )
    verify_group.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=None,
        help="Verification timeout in seconds (default: from config.yaml)",
    )

    # BaB mode: --bab enables, --no-bab disables, absent = from config.yaml
    bab_toggle = verify_group.add_mutually_exclusive_group()
    bab_toggle.add_argument(
        "--bab",
        action="store_true",
        default=None,
        dest="bab",
        help="Enable branch-and-bound verification",
    )
    bab_toggle.add_argument(
        "--no-bab",
        action="store_false",
        dest="bab",
        help="Disable branch-and-bound (single-shot)",
    )

    # BaB algorithm parameters
    verify_group.add_argument(
        "--bab-max-depth",
        type=int,
        default=None,
        dest="bab_max_depth",
        help="Maximum BaB tree depth (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-max-subproblems",
        type=int,
        default=None,
        dest="bab_max_subproblems",
        help="Maximum number of BaB subproblems (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-branching",
        type=str,
        default=None,
        dest="bab_branching",
        help="Branching strategy (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-bounding",
        type=str,
        default=None,
        dest="bab_bounding",
        help="Bounding strategy (default: from config.yaml)",
    )

    # Backend config file
    verify_group.add_argument(
        "--backend-config",
        type=str,
        default=None,
        dest="backend_config",
        help="Path to backend YAML config (default: act/back_end/config.yaml)",
    )

    # Common options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Add standard device/dtype arguments (shared across all ACT CLIs)
    add_device_args(parser)

    # Detect user-provided flags BEFORE parsing so env vars / config.yaml
    # can serve as fallbacks without overriding explicit CLI flags.
    argv = sys.argv[1:]
    _user_set = lambda flag: any(  # noqa: E731
        a == flag or a.startswith(flag + "=") for a in argv
    )

    args = parser.parse_args()

    # Validate arguments based on command
    if args.verify or args.info:
        if not args.network:
            parser.error("--network is required for --verify and --info")

    if args.diff_nets:
        return run_diff_nets(args)

    # ── Build BackendConfig ──────────────────────────────────────────────
    # Load YAML as baseline, then overlay env vars and CLI flags on top.
    # Precedence: CLI flag > env var > config.yaml > dataclass default
    from act.back_end.config import BackendConfig

    backend_cfg = BackendConfig.from_yaml(
        config_path=args.backend_config,
        **_collect_backend_overrides(args, _user_set),
    )

    # Initialize device manager from the resolved config
    import argparse as _ap

    initialize_from_args(
        _ap.Namespace(device=backend_cfg.device, dtype=backend_cfg.dtype)
    )

    # Execute command
    try:
        if args.generate:
            return run_network_factory(args, backend_cfg)
        elif args.verify:
            return run_verification(args, backend_cfg)
        elif args.verify_all:
            return run_verify_all(args, backend_cfg)
        elif args.info:
            return run_network_info(args)
        elif args.list_examples:
            return list_examples(args)
        elif args.test_serialization:
            return run_serialization_test(args)
        elif args.bench:
            return run_bench(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def _collect_backend_overrides(args, _user_set) -> dict:
    """Build overrides dict from CLI flags + env vars.

    Only includes keys the user explicitly provided (CLI flag) or that are
    set in the environment.  Everything else falls through to config.yaml.

    Prefix conventions: ``bab_<field>`` → BaBConfig, ``gen_<field>`` → GenerationConfig.
    """
    overrides: dict = {}

    # ── Runtime selectors: CLI > env > config.yaml ──
    if args.solver is not None:
        overrides["solver"] = args.solver
    elif os.environ.get("ACT_SOLVER"):
        overrides["solver"] = os.environ["ACT_SOLVER"]

    if _user_set("--device"):
        overrides["device"] = args.device
    elif os.environ.get("ACT_DEVICE"):
        overrides["device"] = os.environ["ACT_DEVICE"]

    if _user_set("--dtype"):
        overrides["dtype"] = args.dtype
    elif os.environ.get("ACT_DTYPE"):
        overrides["dtype"] = os.environ["ACT_DTYPE"]

    if args.verbose:
        overrides["verbose"] = True

    # ── Verification ──
    if args.timeout is not None:
        overrides["timeout"] = args.timeout

    # bab enabled: --bab / --no-bab (None = defer to config.yaml)
    if args.bab is not None:
        overrides["bab_enabled"] = args.bab

    if args.bab_max_depth is not None:
        overrides["bab_max_depth"] = args.bab_max_depth
    if args.bab_max_subproblems is not None:
        overrides["bab_max_nodes"] = args.bab_max_subproblems
    if args.bab_branching is not None:
        overrides["bab_branching_method"] = args.bab_branching
    if args.bab_bounding is not None:
        overrides["bab_bounding_method"] = args.bab_bounding

    # ── Generation: CLI > env > config.yaml ──
    config_flag = getattr(args, "config", None)
    if config_flag is not None:
        overrides["gen_gen_config_path"] = config_flag

    output_flag = getattr(args, "output", None)
    if output_flag is not None:
        overrides["gen_output_dir"] = output_flag
    elif os.environ.get("ACT_GEN_OUTPUT"):
        overrides["gen_output_dir"] = os.environ["ACT_GEN_OUTPUT"]

    num_flag = getattr(args, "num", None)
    if num_flag is not None:
        overrides["gen_num_instances"] = num_flag
    elif os.environ.get("ACT_GEN_NUM"):
        overrides["gen_num_instances"] = int(os.environ["ACT_GEN_NUM"])

    seed_flag = getattr(args, "base_seed", None)
    if seed_flag is not None:
        overrides["gen_base_seed"] = seed_flag
    elif os.environ.get("ACT_GEN_SEED"):
        overrides["gen_base_seed"] = int(os.environ["ACT_GEN_SEED"])

    prefix_flag = getattr(args, "name_prefix", None)
    if prefix_flag is not None:
        overrides["gen_name_prefix"] = prefix_flag

    tf_flag = getattr(args, "tf_targets", None)
    if tf_flag is not None:
        overrides["gen_tf_targets"] = tf_flag

    reg_flag = getattr(args, "registry_mode", None)
    if reg_flag is not None and _user_set("--registry-mode"):
        overrides["gen_registry_mode"] = reg_flag

    return overrides


if __name__ == "__main__":
    sys.exit(main())
