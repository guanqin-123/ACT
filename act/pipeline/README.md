# ACT Pipeline Module

Testing and integration framework for the Abstract Constraint Transformer (ACT). This module provides automatic PyTorch→ACT conversion, whitebox fuzzing, and comprehensive verifier validation.

## Overview

The ACT Pipeline bridges the front-end data processing and back-end verification core. It includes tools for:
- **PyTorch ↔ ACT Conversion**: Seamlessly convert between PyTorch `nn.Module` and ACT `Net` representations.
- **Inference-based Fuzzing**: Rapidly find counterexamples using gradient-guided mutations and coverage tracking.
- **Verifier Validation**: Rigorous soundness and numerical correctness checks for verification backends.
- **Benchmark Management**: Automated downloading and listing of VNNLIB benchmarks and TorchVision data-model pairs.

## Architecture

```
act/pipeline/
├── cli.py                # Main pipeline CLI
├── __main__.py           # Package entry point
├── verification/         # Verification utilities submodule
│   ├── torch2act.py      # Automatic PyTorch→ACT conversion
│   ├── act2torch.py      # ACT→PyTorch conversion utilities
│   ├── validate_verifier.py # Verifier correctness validation
│   ├── model_factory.py  # ACT Net factory for test networks
│   ├── utils.py          # Shared utilities and profiling
│   ├── llm_probe.py      # LLM-guided closed-loop BaB branching/scheduling controller
│   └── per_neuron_bounds.py # Per-neuron activation checking
├── fuzzing/              # Whitebox fuzzing framework
│   ├── actfuzzer.py      # Main fuzzing engine
│   └── ...
└── log/                  # Centralized execution logs
```

## Command-Line Interface

The pipeline is accessible via `python -m act.pipeline`. It provides commands for benchmark management, fuzzing, and verification.

### Benchmark Management

```bash
# List available VNNLIB categories
python -m act.pipeline --list

# Search for specific benchmarks
python -m act.pipeline --search acas

# Get detailed information about a category
python -m act.pipeline --info acasxu_2023

# Download a VNNLIB category
python -m act.pipeline --download acasxu_2023

# List downloaded data-model pairs
python -m act.pipeline --list-downloaded
```

### Whitebox Fuzzing

Run ACTFuzzer on VNNLIB or TorchVision targets:

```bash
# Fuzz a VNNLIB benchmark
python -m act.pipeline --fuzz --category acasxu_2023 --iterations 5000

# Fuzz a TorchVision dataset
python -m act.pipeline --fuzz --creator torchvision --dataset MNIST
```

### Verifier Validation

Ensure verifier soundness and numerical precision:

```bash
# Run comprehensive validation (Level 1 + Level 2)
python -m act.pipeline --validate-verifier --device cpu --dtype float64

# Run specific validation modes
python -m act.pipeline --validate-verifier --mode counterexample
python -m act.pipeline --validate-verifier --mode bounds --input-samples 20
```

### Conversion Tests

```bash
# Run PyTorch→ACT conversion tests
python -m act.pipeline --verify torch2act

# Run ACT→PyTorch conversion tests
python -m act.pipeline --verify act2torch
```

## Key Components

### Torch2ACT Converter (`verification/torch2act.py`)
Automatically converts PyTorch models to ACT's intermediate representation. It preserves verification constraints embedded in `VerifiableModel` wrappers and ensures weight equivalence.

### Verifier Validator (`verification/validate_verifier.py`)
Implements multi-level validation:
1. **Level 1 (Soundness)**: Verifies that the verifier does not report CERTIFIED when concrete counterexamples exist.
2. **Level 2 (Numerical Precision)**: Checks that abstract bounds correctly overapproximate concrete activation values across all layers.

### ACTFuzzer (`fuzzing/actfuzzer.py`)
A fast, GPU-accelerated fuzzer that uses:
- **Gradient Mutations**: FGSM/PGD-style perturbations.
- **Coverage Tracking**: DeepXplore-style neuron coverage.
- **Property Checking**: Automated detection of `OutputSpec` violations.

### LLM-Guided Branching Controller (`verification/llm_probe.py`)
Closed-loop LLM controller for the dual-batched BaB verifier (paper: *LEAPS — LLM-Guided
Branch-and-Bound for Scalable Neural Network Verification*). **Soundness boundary**: the LLM only
*proposes* search-scheduling decisions (split depth/group, wave width, refinement effort) — the BaB
verifier alone computes bounds and certifies/falsifies counterexamples. Invalid, missing, or
unavailable guidance always falls back to the verifier's own baseline behavior (disabled probe ==
baseline, bit-identical).

**Usage** — enable via `python -m act.back_end --verify ... --bab-llm-probe-enabled`:
```bash
--bab-llm-probe-enabled
--bab-llm-probe-backend {mock,openrouter,openai,glm,minimax}   # default: mock (offline, no network)
--bab-llm-probe-model <model-string>                            # e.g. "openai/gpt-4o" via openrouter
--bab-llm-probe-decisions split,frontier,refine,neuron          # "neuron" enables joint group selection
--multi-split-levels <k>                                        # k>1 required for neuron-group splitting
```
Remaining tunables: `--bab-llm-probe-{base-url,cadence,api-key-env,temperature,max-candidates,
max-candidates-total,history,max-failures,log}`.

**Providers** (`_PROVIDER_PRESETS`; one OpenAI-compatible HTTP backend, stdlib `urllib` only, no
hard dependency on `openai`/`litellm`):
| backend | base_url | API key env var |
|---|---|---|
| `mock` | — (offline, deterministic, no network) | — |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `glm` | `https://open.bigmodel.cn/api/paas/v4` | `ZHIPUAI_API_KEY` |
| `minimax` | `https://api.minimaxi.com/v1` | `MINIMAX_API_KEY` |
`--bab-llm-probe-base-url` / `--bab-llm-probe-api-key-env` override the preset.

**Neuron-group selection** requires `solver_tier ∈ {dual_alpha, dual_alpha_eta}` and
`branching_method ∈ {gain, babsr, fsb}`. The LLM proposes a per-lane neuron group; the verifier turns
it into the existing sound `2^k` sign-combination cube (`_multi_split_from_groups` in
`back_end/bab/branching/branching.py`) — the identical soundness argument as the static BaBSR
multi-split path. Any illegal, empty, or oversized group falls back to FSB/gain.

**Benchmark harness** (`scripts/bench_llm_branching.py`, offline by default via the `mock` backend —
no API key required):
```bash
# Compare all four configs (fsb / babsr / gain / gain+llm) on the factory example nets
python scripts/bench_llm_branching.py --configs fsb babsr gain gain+llm --out results.csv

# Paper-time run with a real provider for the gain+llm config
python scripts/bench_llm_branching.py --configs gain+llm --llm-backend openrouter
```
Metrics per (net, config): status, wall-time, #waves, #subproblems, #LLM calls, #fallbacks. Offline
plumbing smoke test: `tests/test_bench_llm_smoke.py` (MockBackend only, no network).

## Logging and Diagnostics

All execution logs, including detailed transfer function analysis, are stored in `act/pipeline/log/`.
- `pipeline_tests.log`: General test execution logs.
- `act_debug_tf.log`: Layer-by-layer transfer function analysis (enabled via `PerformanceOptions`).
- `fuzzing_results/`: Summary and counterexamples from fuzzing runs.

## License

ACT is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

