"""
RQ3 Experiment Configuration: Scale Factor Sensitivity.

Defines 12 configurations (2 benchmarks × 6 scale factors) for the
"Sensitivity to perturbation scale factor s" research question.

Usage:
    from act.pipeline.script.rq3_config import SCALE_FACTORS, RQ3_BENCHMARKS, RQ3_CONFIGS

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from act.pipeline.script.rq1_config import BENCHMARKS, METHODS, EXPERIMENT_DEFAULTS

# ---------------------------------------------------------------------------
# Scale factors from paper Table tab:rq3-scale
# ---------------------------------------------------------------------------
SCALE_FACTORS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

# ---------------------------------------------------------------------------
# RQ3 benchmarks — subset of RQ1 benchmarks
# Paper Table tab:rq3-scale tests Cifar100 and TinyImageNet
# ---------------------------------------------------------------------------
RQ3_BENCHMARKS = {
    "cifar100": BENCHMARKS["cifar100"],
    "tinyimagenet": BENCHMARKS["tinyimagenet"],
}

# ---------------------------------------------------------------------------
# RQ3 methods — both aniso and iso are affected by scale factor
# ---------------------------------------------------------------------------
RQ3_METHODS = ["batch_aniso", "batch_iso"]

RQ3_DEFAULTS = {
    "timeout": EXPERIMENT_DEFAULTS["timeout"],
    "runs": EXPERIMENT_DEFAULTS["runs"],
    "coverage_strategy": "GlobalCov",
    "max_iterations": EXPERIMENT_DEFAULTS["max_iterations"],
    "activation_threshold": EXPERIMENT_DEFAULTS["activation_threshold"],
    "report_interval": EXPERIMENT_DEFAULTS["report_interval"],
    "verbose": EXPERIMENT_DEFAULTS["verbose"],
}

# ---------------------------------------------------------------------------
# RQ3 Configurations: 2 benchmarks × 6 scale factors × 2 methods = 24 configs
# ---------------------------------------------------------------------------
RQ3_CONFIGS = [
    (benchmark_key, scale_factor, method_key)
    for benchmark_key in RQ3_BENCHMARKS
    for scale_factor in SCALE_FACTORS
    for method_key in RQ3_METHODS
]

assert len(SCALE_FACTORS) == 6
assert len(RQ3_BENCHMARKS) == 2
assert len(RQ3_METHODS) == 2
assert len(RQ3_CONFIGS) == 24, f"Expected 24 configs, got {len(RQ3_CONFIGS)}"
for m in RQ3_METHODS:
    assert m in METHODS, f"RQ3 method '{m}' not found in METHODS"
