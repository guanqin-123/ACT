# ACTFuzzer

Inference-based whitebox fuzzing for neural network verification.

## Overview

ACTFuzzer is a lightweight, inference-based fuzzing tool that finds counterexamples by:
1. **Gradient-guided mutations**: FGSM-style adversarial perturbations
2. **Coverage tracking**: Neuron coverage (DeepXplore-style)
3. **Energy-based scheduling**: AFL-style seed prioritization
4. **Property checking**: Automatic OutputSpec violation detection

Unlike formal verification, ACTFuzzer runs fast GPU-accelerated inference to quickly explore the input space and find violations.

## Features

- ✅ **Fast**: ~1000x faster than formal verification
- ✅ **Practical**: Finds counterexamples without soundness guarantees
- ✅ **Flexible**: Works with VNNLib benchmarks and TorchVision datasets
- ✅ **Integrated**: Seamless integration with ACT's spec creators and model synthesis

## Architecture

```
ACTFuzzer
├── MutationEngine      # Gradient/activation/boundary/random mutations
├── CoverageTracker     # Neuron coverage tracking
├── SeedCorpus          # AFL-style seed management
└── PropertyChecker     # OutputSpec violation detection
```

## Usage

### Quick Start

```bash
# 1. Download benchmark
python -m act.pipeline --download acasxu_2023

# 2. Fuzz it
python -m act.pipeline --fuzz --category acasxu_2023 --iterations 5000

# 3. Check results
ls fuzzing_results/
cat fuzzing_results/summary.json
```

### Python API

```python
from act.pipeline.fuzzing import ACTFuzzer, FuzzingConfig
from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator
from act.front_end.model_synthesis import synthesize_models_from_specs

# Create specs
creator = VNNLibSpecCreator()
spec_results = creator.create_specs_for_data_model_pairs(
    categories=["acasxu_2023"],
    max_instances=10
)

# Synthesize models
wrapped_models, reports, input_data = synthesize_models_from_specs(spec_results)

# Extract seeds
initial_seeds = []
for _, _, _, labeled_tensors, _ in spec_results:
    initial_seeds.extend(labeled_tensors)

# Fuzz
config = FuzzingConfig(max_iterations=5000, device="cuda")
fuzzer = ACTFuzzer(
    wrapped_model=list(wrapped_models.values())[0],
    initial_seeds=initial_seeds,
    config=config
)

report = fuzzer.fuzz()
print(f"Found {len(report.counterexamples)} counterexamples")
```

## Configuration

Edit `config.yaml` to customize:

```yaml
fuzzing:
  max_iterations: 10000
  mutation_weights:
    gradient: 0.4      # FGSM-style
    activation: 0.3    # DeepXplore
    boundary: 0.2      # Edge cases
    random: 0.1        # Baseline
```

## Mutation Strategies

### 1. Gradient-Guided (40%)
FGSM-style adversarial perturbations:
```
x' = x + ε * sign(∇_x Loss(x))
```

### 2. Activation-Guided (30%)
Targets neurons with low activation (DeepXplore):
```
Maximize: Σ inactive_neurons
```

### 3. Boundary Exploration (20%)
Samples near InputSpec boundaries:
```
x' = x + ε * direction_to_boundary
```

### 4. Random (10%)
Gaussian noise baseline:
```
x' = x + N(0, ε²)
```

## Coverage Metrics

ACTFuzzer tracks **neuron coverage**:
```
Coverage = |{neurons that fired}| / |{total neurons}|
```

A neuron "fired" if `activation > threshold` (default: 0.1).

## Output

Fuzzing produces:
- `summary.json`: Statistics (iterations, time, coverage, violations)
- `counterexample_*.pt`: PyTorch tensors with input/output/label

Example `summary.json`:
```json
{
  "iterations": 5000,
  "time_seconds": 125.3,
  "counterexamples_found": 12,
  "neuron_coverage": 0.87,
  "mutations": 5000,
  "seeds_explored": 342
}
```

## Performance

Typical performance on NVIDIA RTX 3090:
- **ACAS Xu**: ~500 iterations/sec
- **MNIST CNN**: ~800 iterations/sec
- **CIFAR10 ResNet**: ~300 iterations/sec

## Comparison with Formal Verification

| Aspect | ACTFuzzer | Formal Verification |
|--------|-----------|---------------------|
| **Speed** | ~500 it/s | ~0.5 it/s (1000x slower) |
| **Soundness** | No (heuristic) | Yes (complete) |
| **Counterexamples** | Yes | Yes |
| **Proof** | No | Yes (if UNSAT) |
| **Use Case** | Bug finding | Certification |

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size (currently 1)
- Use `--device cpu`
- Lower `max_iterations`

### No Counterexamples Found
- Increase `max_iterations`
- Check InputSpec constraints (too restrictive?)
- Try different mutation weights

### Low Coverage
- Increase `max_iterations`
- Use gradient-guided mutations (set weight to 0.8)

## Citation

```bibtex
@software{actfuzzer2025,
  title = {ACTFuzzer: Inference-based Whitebox Fuzzing},
  author = {SVF-tools},
  year = {2025},
  url = {https://github.com/SVF-tools/ACT}
}
```

## License

AGPLv3+ - Copyright (C) 2025 SVF-tools/ACT
