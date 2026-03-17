from __future__ import annotations

import argparse
import importlib
import json
import random
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from act.front_end.model_synthesis import synthesize_models_from_specs
from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator
from act.front_end.torchvision_loader.create_specs import TorchVisionSpecCreator
from act.pipeline.fuzzing.actfuzzer import ACTFuzzer, FuzzingConfig
from act.pipeline.fuzzing.checker import Counterexample
from act.pipeline.script.rq1_config import BENCHMARKS, EXPERIMENT_DEFAULTS, METHODS
from act.util.cli_utils import initialize_from_args
from act.util.path_config import get_project_root


SpecResult = Tuple[str, str, Any, List[Any], List[Tuple[Any, Any]]]


def _load_custom_mnist(
    bench: Dict[str, Any], max_instances: Optional[int]
) -> List[SpecResult]:
    torch = importlib.import_module("torch")
    torchvision = importlib.import_module("torchvision")

    num_samples = max_instances or bench.get("num_samples", 30)
    eps = bench.get("epsilon", 0.02)

    weights_path = Path(get_project_root()) / bench["weights_path"]
    if not weights_path.exists():
        raise RuntimeError(f"MNIST weights not found: {weights_path}")

    from act.front_end.specs import InputSpec, OutputSpec
    from act.front_end.spec_creator_base import LabeledInputTensor

    # Load model
    model_file = Path(get_project_root()) / "data/torchvision/MNIST/models/fnn_6x256.py"
    model_cls_code = model_file.read_text()
    ns: Dict[str, Any] = {"__file__": str(model_file)}
    exec(model_cls_code, ns)
    pytorch_model = ns["model"]
    pytorch_model.eval()

    # Load MNIST test set (raw 1×28×28)
    mnist_root = str(Path(get_project_root()) / "data/torchvision/MNIST/raw/MNIST")
    transform = torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(
        root=mnist_root, train=False, download=False, transform=transform
    )

    labeled_tensors = []
    spec_pairs = []
    for i in range(min(num_samples, len(dataset))):
        img, label = dataset[i]
        img = img.unsqueeze(0).to(dtype=torch.get_default_dtype())
        label_t = torch.tensor([label], dtype=torch.long)
        labeled_tensors.append(LabeledInputTensor(tensor=img, label=label_t))

        lb = (img - eps).clamp(0.0, 1.0)
        ub = (img + eps).clamp(0.0, 1.0)
        in_spec = InputSpec(kind="BOX", lb=lb, ub=ub)
        out_spec = OutputSpec(kind="TOP1_ROBUST", y_true=label_t)
        spec_pairs.append((in_spec, out_spec))

    return [("MNIST", "fnn_6x256", pytorch_model, labeled_tensors, spec_pairs)]


def load_specs(benchmark_key: str, max_instances: Optional[int]) -> List[SpecResult]:
    if benchmark_key not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark '{benchmark_key}'. Available: {sorted(BENCHMARKS.keys())}"
        )

    bench = BENCHMARKS[benchmark_key]
    creator_type = bench.get("creator", "vnnlib")

    if creator_type == "custom_mnist":
        spec_results = _load_custom_mnist(bench, max_instances)
    elif creator_type == "torchvision":
        dataset_name = bench["dataset"]
        model_name = bench["model"]
        num_samples = max_instances or bench.get("num_samples", 30)

        try:
            creator = TorchVisionSpecCreator()
            spec_results = creator.create_specs_for_data_model_pairs(
                dataset_names=[dataset_name],
                model_names=[model_name],
                num_samples=num_samples,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load TorchVision specs for {dataset_name}+{model_name}.\n"
                f"Download with: python -m act.pipeline --download {dataset_name} --creator torchvision"
            ) from exc
    else:
        category = bench["category"]
        try:
            creator = VNNLibSpecCreator()
            spec_results = creator.create_specs_for_data_model_pairs(
                categories=[category],
                max_instances=max_instances,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load VNNLib specs. Ensure the benchmark is downloaded.\n"
                f"Download with: python -m act.front_end --download {category}"
            ) from exc

    if not spec_results:
        raise RuntimeError(
            f"No spec results generated for '{benchmark_key}'. Ensure data is downloaded."
        )

    spec_filter = bench.get("spec_filter")
    if spec_filter:
        in_kind = spec_filter.get("input_kind")
        out_kind = spec_filter.get("output_kind")
        filtered = []
        for ds, mn, model, lts, sp in spec_results:
            kept = [
                (i, o)
                for i, o in sp
                if (in_kind is None or i.kind == in_kind)
                and (out_kind is None or o.kind == out_kind)
            ]
            if kept:
                filtered.append((ds, mn, model, lts, kept))
        spec_results = filtered

    if not spec_results:
        raise RuntimeError(
            f"No spec results after filtering for '{benchmark_key}'. Check spec_filter config."
        )

    return spec_results


def _extract_initial_seeds(spec_results: List[SpecResult]) -> List[Any]:
    initial_seeds: List[Any] = []
    for _, _, _, labeled_tensors, _ in spec_results:
        initial_seeds.extend(labeled_tensors)
    return initial_seeds


def _format_model_key(model_key: Tuple[str, str, str, str]) -> str:
    data_source, model_name, input_kind, output_kind = model_key
    return f"{data_source}::{model_name}::{input_kind}::{output_kind}"


def _compute_ttfv(
    counterexamples: Iterable[Counterexample], start_time: float
) -> Optional[float]:
    timestamps = [ce.timestamp for ce in counterexamples]
    if not timestamps:
        return None
    return min(timestamps) - start_time


def _resolve_output_dir(output_dir: str) -> Path:
    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = Path(get_project_root()) / out_path
    return out_path


def run_single_experiment(
    benchmark_key: str,
    method_key: str,
    run_id: int,
    timeout: float,
    max_instances: Optional[int],
    output_dir: str,
) -> None:
    torch = importlib.import_module("torch")
    np = importlib.import_module("numpy")

    torch.manual_seed(run_id)
    np.random.seed(run_id)
    random.seed(run_id)

    method = METHODS.get(method_key)
    if method is None:
        raise ValueError(
            f"Unknown method '{method_key}'. Available: {sorted(METHODS.keys())}"
        )

    batch_mode = method["batch_mode"]
    if batch_mode not in {"all", "sequential"}:
        raise ValueError(f"Unsupported batch_mode '{batch_mode}'")

    try:
        spec_results = load_specs(benchmark_key, max_instances)
    except Exception as exc:
        print(f"❌ Spec loading failed for {benchmark_key}: {exc}")
        traceback.print_exc()
        return

    output_base = _resolve_output_dir(output_dir)
    output_path = output_base / benchmark_key / method_key
    output_path.mkdir(parents=True, exist_ok=True)

    experiment_start = time.time()
    all_counterexamples: List[Counterexample] = []
    per_group_results: List[Dict[str, Any]] = []
    total_iterations = 0
    total_violations = 0
    coverage_values: List[float] = []

    if batch_mode == "all":
        print(
            f"🔬 Benchmark={benchmark_key} Method={method_key} Run={run_id} Mode=B=all"
        )
        try:
            wrapped_models = synthesize_models_from_specs(spec_results)
        except Exception as exc:
            print(f"❌ Model synthesis failed: {exc}")
            traceback.print_exc()
            return

        if not wrapped_models:
            print("❌ No wrapped models synthesized.")
            return

        # Build per-model-id seed mapping for correct per-group seed selection
        # Each pytorch_model object has a unique id, and we map it to its seeds
        seeds_by_model_id: Dict[int, List[Any]] = {}
        for _, _, pytorch_model, labeled_tensors, _ in spec_results:
            mid = id(pytorch_model)
            if mid not in seeds_by_model_id:
                seeds_by_model_id[mid] = []
            seeds_by_model_id[mid].extend(labeled_tensors)

        if not seeds_by_model_id:
            print("❌ No initial seeds extracted.")
            return

        for idx, (model_key, wrapped_model) in enumerate(
            wrapped_models.items(), start=1
        ):
            group_start = time.time()
            model_key_str = _format_model_key(model_key)
            print(f"➡️  [{idx}/{len(wrapped_models)}] Fuzzing group {model_key_str}")

            # Extract the pytorch_model from wrapped_model to get its id
            # VerifiableModel structure: InputLayer → InputSpecLayer → pytorch_model → OutputSpecLayer
            from act.front_end.verifiable_model import (
                InputLayer,
                InputSpecLayer,
                OutputSpecLayer,
            )

            inner_model = None
            for child in wrapped_model.children():
                if not isinstance(child, (InputLayer, InputSpecLayer, OutputSpecLayer)):
                    inner_model = child
                    break

            # Look up seeds for this model group
            group_seeds = (
                seeds_by_model_id.get(id(inner_model), []) if inner_model else []
            )
            if not group_seeds:
                print(f"⚠️  No seeds found for group {model_key_str}, skipping")
                continue

            try:
                config = FuzzingConfig.from_yaml(
                    max_iterations=EXPERIMENT_DEFAULTS["max_iterations"],
                    timeout_seconds=timeout,
                    perturb_mode=method["perturb_mode"],
                    mutation_weights=method["mutation_weights"],
                    coverage_strategy=method["coverage_strategy"],
                    perturb_scale=EXPERIMENT_DEFAULTS["perturb_scale"],
                    activation_threshold=EXPERIMENT_DEFAULTS["activation_threshold"],
                    report_interval=EXPERIMENT_DEFAULTS["report_interval"],
                    verbose=EXPERIMENT_DEFAULTS["verbose"],
                    save_counterexamples=False,
                    output_dir=output_path,
                    trace_level=0,
                )
                fuzzer = ACTFuzzer(
                    wrapped_model=wrapped_model,
                    initial_seeds=group_seeds,
                    config=config,
                )
                report = fuzzer.fuzz()
            except Exception as exc:
                print(f"❌ Fuzzing failed for group {model_key_str}: {exc}")
                traceback.print_exc()
                continue

            group_time = time.time() - group_start
            group_violations = len(report.counterexamples)
            group_ttfv = _compute_ttfv(report.counterexamples, group_start)
            group_throughput = (
                report.total_iterations / report.total_time
                if report.total_time > 0
                else 0.0
            )

            all_counterexamples.extend(report.counterexamples)
            total_iterations += report.total_iterations
            total_violations += group_violations
            coverage_values.append(report.neuron_coverage)

            violated_indices = sorted(
                set(
                    ce.seed_index
                    for ce in report.counterexamples
                    if ce.seed_index is not None
                )
            )

            per_group_results.append(
                {
                    "model_key": model_key_str,
                    "batch_size": fuzzer.batch_size,
                    "violations": group_violations,
                    "coverage": report.neuron_coverage,
                    "ttfv": group_ttfv,
                    "throughput": group_throughput,
                    "time": group_time,
                    "total_iterations": report.total_iterations,
                    "violated_seed_indices": violated_indices,
                }
            )
            print(f"✅ Completed group {model_key_str} in {group_time:.1f}s")

        num_model_groups = len(wrapped_models)
        instances_processed = sum(len(lts) for _, _, _, lts, _ in spec_results)

    else:
        print(f"🔬 Benchmark={benchmark_key} Method={method_key} Run={run_id} Mode=B=1")
        num_instances = len(spec_results)
        instances_processed = 0

        for idx, spec_result in enumerate(spec_results, start=1):
            group_start = time.time()
            single_spec = [spec_result]
            data_source, model_name, _, labeled_tensors, _ = spec_result
            instances_processed += len(labeled_tensors)

            try:
                wrapped_models = synthesize_models_from_specs(single_spec)
            except Exception as exc:
                print(f"❌ Model synthesis failed for instance {idx}: {exc}")
                traceback.print_exc()
                continue

            if not wrapped_models:
                print(f"❌ No wrapped model synthesized for instance {idx}.")
                continue

            model_key, wrapped_model = next(iter(wrapped_models.items()))
            model_key_str = _format_model_key(model_key)
            print(f"➡️  [{idx}/{num_instances}] Fuzzing instance {model_key_str}")

            try:
                config = FuzzingConfig.from_yaml(
                    max_iterations=EXPERIMENT_DEFAULTS["max_iterations"],
                    timeout_seconds=timeout,
                    perturb_mode=method["perturb_mode"],
                    mutation_weights=method["mutation_weights"],
                    coverage_strategy=method["coverage_strategy"],
                    perturb_scale=EXPERIMENT_DEFAULTS["perturb_scale"],
                    activation_threshold=EXPERIMENT_DEFAULTS["activation_threshold"],
                    report_interval=EXPERIMENT_DEFAULTS["report_interval"],
                    verbose=EXPERIMENT_DEFAULTS["verbose"],
                    save_counterexamples=False,
                    output_dir=output_path,
                    trace_level=0,
                )
                initial_seeds = list(labeled_tensors)
                fuzzer = ACTFuzzer(
                    wrapped_model=wrapped_model,
                    initial_seeds=initial_seeds,
                    config=config,
                )
                report = fuzzer.fuzz()
            except Exception as exc:
                print(f"❌ Fuzzing failed for instance {idx}: {exc}")
                traceback.print_exc()
                continue

            group_time = time.time() - group_start
            group_violations = len(report.counterexamples)
            group_ttfv = _compute_ttfv(report.counterexamples, group_start)
            group_throughput = (
                report.total_iterations / report.total_time
                if report.total_time > 0
                else 0.0
            )

            all_counterexamples.extend(report.counterexamples)
            total_iterations += report.total_iterations
            total_violations += group_violations
            coverage_values.append(report.neuron_coverage)

            violated_indices = sorted(
                set(
                    ce.seed_index
                    for ce in report.counterexamples
                    if ce.seed_index is not None
                )
            )

            per_group_results.append(
                {
                    "model_key": model_key_str,
                    "data_source": data_source,
                    "model_name": model_name,
                    "batch_size": fuzzer.batch_size,
                    "violations": group_violations,
                    "coverage": report.neuron_coverage,
                    "ttfv": group_ttfv,
                    "throughput": group_throughput,
                    "time": group_time,
                    "total_iterations": report.total_iterations,
                    "violated_seed_indices": violated_indices,
                }
            )
            print(f"✅ Completed instance {idx} in {group_time:.1f}s")

        num_model_groups = len(spec_results)

    total_time = time.time() - experiment_start
    coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
    ttfv = _compute_ttfv(all_counterexamples, experiment_start)
    throughput = total_iterations / total_time if total_time > 0 else 0.0

    violation_timestamps = sorted(
        ce.timestamp - experiment_start for ce in all_counterexamples
    )

    result = {
        "benchmark": benchmark_key,
        "method": method_key,
        "run_id": run_id,
        "violations": total_violations,
        "coverage": coverage,
        "ttfv": ttfv,
        "throughput": throughput,
        "total_time": total_time,
        "num_model_groups": num_model_groups,
        "batch_mode": batch_mode,
        "instances_processed": instances_processed,
        "per_group_results": per_group_results,
        "violation_timestamps": violation_timestamps,
    }
    all_violated_indices = sorted(
        set(ce.seed_index for ce in all_counterexamples if ce.seed_index is not None)
    )

    result["all_violated_indices"] = all_violated_indices

    output_file = output_path / f"run_{run_id}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"📁 Saved results to {output_file}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ1 experiment runner")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(BENCHMARKS.keys()),
        help="Benchmark key from rq1_config.BENCHMARKS",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(METHODS.keys()),
        help="Method key from rq1_config.METHODS",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=EXPERIMENT_DEFAULTS["runs"],
        help="Number of independent runs",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=EXPERIMENT_DEFAULTS["timeout"],
        help="Per-run timeout in seconds",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "gpu"],
        help="Device to use (cpu/cuda)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Default dtype",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Limit number of instances loaded (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/rq1",
        help="Output directory for results",
    )
    parser.add_argument(
        "--start-run",
        type=int,
        default=0,
        help="Starting run ID (used by run_rq1.sh for resume)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    initialize_from_args(args)

    for run_id in range(args.start_run, args.start_run + args.runs):
        try:
            run_single_experiment(
                benchmark_key=args.benchmark,
                method_key=args.method,
                run_id=run_id,
                timeout=args.timeout,
                max_instances=args.max_instances,
                output_dir=args.output_dir,
            )
        except Exception as exc:
            print(f"❌ Run {run_id} failed: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
