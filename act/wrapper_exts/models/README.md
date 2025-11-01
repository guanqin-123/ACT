# Benchmark Models

This directory contains small ONNX models used for testing and benchmarking the ACT verification framework.

## Directory Structure

```
models/
└── vnnmodels/
    ├── MNIST/          # MNIST models (28x28 grayscale)
    │   ├── small_relu_mnist_cnn_model_1.onnx
    │   ├── small_sigmoid_mnist_cnn_model_1.onnx
    │   └── small_tanh_mnist_cnn_model_1.onnx
    └── CIFAR10/        # CIFAR10 models (32x32 RGB)
        ├── small_relu_cifar10_cnn_model_1.onnx
        ├── small_sigmoid_cifar10_cnn_model_1.onnx
        └── small_tanh_cifar10_cnn_model_1.onnx
```

## Purpose

These models are used for:

1. **CI/CD Testing**: GitHub Actions workflows test ERAN and αβ-CROWN integrations
2. **Quick Verification**: Small models for rapid testing during development
3. **Documentation Examples**: Referenced in README and copilot-instructions

## Usage

```bash
# Example: Run verification with MNIST model
python act/wrapper_exts/ext_runner.py \
  --model_path act/wrapper_exts/models/vnnmodels/MNIST/small_relu_mnist_cnn_model_1.onnx \
  --dataset mnist --spec_type local_lp \
  --start 0 --end 1 --epsilon 0.03 --norm inf \
  --mean 0.1307 --std 0.3081
```

## Model Properties

- **Architecture**: Simple CNN (Conv → ReLU/Sigmoid/Tanh → FC)
- **Size**: Small (~10-50KB each)
- **Purpose**: Fast verification benchmarks, not accuracy-focused

## Location

These models are placed under `wrapper_exts/` because they are primarily used by external verifier wrappers (ERAN, αβ-CROWN) in CI/CD pipelines.
