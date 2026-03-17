"""
RQ1 Experiment Configuration.

Defines all 20 configurations (5 benchmarks × 4 methods) for the
"Batched Fuzzing Speedup" research question.

Usage:
    from act.pipeline.script.rq1_config import BENCHMARKS, METHODS, RQ1_CONFIGS

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

# ---------------------------------------------------------------------------
# Benchmarks: short key → VNNLib category + metadata
#
# NAMING CONVENTION:
#   - CLI uses SHORT KEYS (e.g., "acasxu") for --benchmark argument
#   - VNNLib category strings (e.g., "acasxu_2023") are only used
#     when calling VNNLibSpecCreator internally
#   - Output paths use short keys: experiments/rq1/acasxu/batch_aniso/run_0.json
# ---------------------------------------------------------------------------
BENCHMARKS = {
    "acasxu": {
        "category": "acasxu_2023",
        "expected_instances": 185,
        "description": "ACAS Xu collision avoidance (45 FC networks, 5D input)",
    },
    "cifar100": {
        "category": "cifar100_2024",
        "expected_instances": 199,
        "description": "CIFAR-100 ResNet (3×32×32, 100 classes)",
    },
    "tinyimagenet": {
        "category": "tinyimagenet_2024",
        "expected_instances": 199,
        "description": "TinyImageNet (3×64×64, 200 classes)",
    },
    "mnist": {
        "creator": "custom_mnist",
        "onnx_path": "data/torchvision/MNIST/models/fnn_6x256.onnx",
        "weights_path": "data/torchvision/MNIST/models/fnn_6x256.pt",
        "num_samples": 30,
        "epsilon": 0.1,
        "expected_instances": 30,
        "description": "MNIST 6×256 ReLU FNN (1×28×28, 10 classes, 98.2% acc)",
    },
    "trafficsigns": {
        "category": "traffic_signs_recognition_2023",
        "expected_instances": 44,
        "description": "Traffic sign recognition (3×32×32, 43 classes)",
    },
}

# ---------------------------------------------------------------------------
# Mutation Weights — Paper Table 2
#
# CRITICAL MAPPING:
#   Paper "Gradient" = code "pgd" (iterative PGD, the primary gradient strategy)
#   Paper does NOT use FGSM (single-step) or Activation separately
#   Code has 5 keys: gradient(FGSM), pgd(PGD), activation, boundary, random
#   Paper specifies: Gradient:0.5, Boundary:0.2, Random:0.3
#   → pgd:0.5, boundary:0.2, random:0.3, gradient:0.0, activation:0.0
# ---------------------------------------------------------------------------
_PAPER_WEIGHTS = {
    "gradient": 0.0,  # FGSM (single-step) — not used in paper
    "pgd": 0.5,  # PGD (iterative) — paper's "Gradient" strategy
    "activation": 0.0,  # DeepXplore activation-guided — not used in paper
    "boundary": 0.2,  # Boundary exploration
    "random": 0.3,  # Random perturbation
}

# ---------------------------------------------------------------------------
# Methods: 4 fuzzing configurations for RQ1
#
# batch_mode:
#   "all"        → B = all instances per model group (batched execution)
#   "sequential" → B = 1, loop over instances one at a time
# ---------------------------------------------------------------------------
METHODS = {
    "batch_aniso": {
        "perturb_mode": "adaptive_perdim",
        "mutation_weights": _PAPER_WEIGHTS,
        "coverage_strategy": "GlobalCov",
        "batch_mode": "all",
        "paper_name": r"\textbf{\sysname}",
        "description": "Batch fuzzing with anisotropic perturbation (our method)",
    },
    "batch_iso": {
        "perturb_mode": "adaptive_scalar",
        "mutation_weights": _PAPER_WEIGHTS,
        "coverage_strategy": "GlobalCov",
        "batch_mode": "all",
        "paper_name": r"\textsc{Batch\text{-}Iso}",
        "description": "Batch fuzzing with isotropic perturbation",
    },
    "seq_aniso": {
        "perturb_mode": "adaptive_perdim",
        "mutation_weights": _PAPER_WEIGHTS,
        "coverage_strategy": "GlobalCov",
        "batch_mode": "sequential",
        "paper_name": r"\textsc{Seq\text{-}Aniso}",
        "description": "Sequential (B=1) with anisotropic perturbation",
    },
    "seq_rand": {
        "perturb_mode": "fixed",
        "mutation_weights": _PAPER_WEIGHTS,
        "coverage_strategy": "GlobalCov",
        "batch_mode": "sequential",
        "paper_name": r"\textsc{Seq\text{-}Rand}",
        "description": "Sequential (B=1) with fixed perturbation (baseline)",
    },
}

# ---------------------------------------------------------------------------
# Experiment Defaults
# ---------------------------------------------------------------------------
EXPERIMENT_DEFAULTS = {
    "timeout": 60,  # 60 second per-instance/per-group budget
    "runs": 5,  # 5 independent runs per configuration
    "perturb_scale": 0.1,  # 10% of feasible range per perturbation
    "report_interval": 500,  # Print progress every 500 iterations
    "max_iterations": 999999,  # Effectively unlimited (timeout is the real limit)
    "activation_threshold": 0.1,  # Neuron activation threshold for GlobalCov
    "verbose": 1,  # Report violations in progress only
}

# ---------------------------------------------------------------------------
# RQ1 Configurations: 5 benchmarks × 4 methods = 20 configs
# ---------------------------------------------------------------------------
RQ1_CONFIGS = [
    (benchmark_key, method_key)
    for benchmark_key in BENCHMARKS
    for method_key in METHODS
]

# Sanity checks
assert len(BENCHMARKS) == 5, f"Expected 5 benchmarks, got {len(BENCHMARKS)}"
assert len(METHODS) == 4, f"Expected 4 methods, got {len(METHODS)}"
assert len(RQ1_CONFIGS) == 20, f"Expected 20 configs, got {len(RQ1_CONFIGS)}"
assert abs(sum(_PAPER_WEIGHTS.values()) - 1.0) < 1e-6, (
    "Mutation weights must sum to 1.0"
)
