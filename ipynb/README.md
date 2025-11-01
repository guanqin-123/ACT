# Jupyter Notebooks

This directory contains Jupyter notebooks for demonstrating and visualizing ACT's capabilities.

## Available Notebooks

### `torchvision_visualization.ipynb`

**Purpose**: Demonstrates ACT's TorchVision loader with custom perturbation visualization.

**Contents**:
1. **TorchVision MNIST Visualization**: Load MNIST dataset with ACT's TorchVision loader, create input specifications, and visualize perturbed images with model predictions
2. **Creating Custom Verification Bounds**: Tutorial on creating L∞ perturbation bounds for images

**Key Features**:
- Interactive visualization of MNIST input perturbations
- Side-by-side comparison of original and perturbed images
- Model inference on perturbed inputs with color-coded predictions (green=correct, red=incorrect)
- Flexible custom specification creation
- Educational examples for understanding verification concepts

**Usage**:
```bash
# Open in Jupyter
jupyter notebook ipynb/torchvision_visualization.ipynb

# Or use VS Code's notebook interface
code ipynb/torchvision_visualization.ipynb
```

### `vnnlib_visualization.ipynb`

**Purpose**: Demonstrates ACT's VNNLib loader with ACAS Xu network visualization.

**Contents**:
1. **VNNLib ACAS Xu Visualization**: Load ACAS Xu networks from VNNLib benchmarks, visualize input bounds, and test network behavior on sample points
2. **Understanding VNNLib Specifications**: Tutorial on SMT-LIB format constraints and standardized benchmarks

**Key Features**:
- ACAS Xu input specification visualization as bar charts
- Network testing on boundary points (lower/center/upper)
- Collision avoidance action interpretation
- SMT-LIB constraint parsing explanation
- VNN-COMP benchmark workflow demonstration

**Usage**:
```bash
# Open in Jupyter
jupyter notebook ipynb/vnnlib_visualization.ipynb

# Or use VS Code's notebook interface
code ipynb/vnnlib_visualization.ipynb
```

## Running Notebooks

### Prerequisites
Ensure you have the `act-py312` (or your ACT environment) conda environment activated with:
- `ipykernel`
- `matplotlib`
- `numpy`
- `torch`
- `torchvision`

### Environment Setup
```bash
# Activate ACT environment
conda activate act-py312

# Install Jupyter if needed
conda install jupyter ipykernel matplotlib -y

# Register kernel
python -m ipykernel install --user --name act-py312 --display-name "Python (ACT)"
```

## Adding New Notebooks

When creating new notebooks for ACT:

1. **Place in this directory**: Keep all notebooks organized in `ipynb/` at the project root
2. **Document purpose**: Add a section to this README describing the notebook
3. **Use relative imports**: Import ACT modules with proper path handling:
   ```python
   import sys
   import os
   act_root = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
   if act_root not in sys.path:
       sys.path.insert(0, act_root)
   ```
4. **Test with clean kernel**: Restart kernel and run all cells to ensure reproducibility

## Notebook Categories

Future notebooks might include:
- **Tutorials**: Step-by-step guides for using ACT features
- **Benchmarks**: Performance analysis and comparison visualizations
- **Examples**: Real-world verification case studies
- **Debugging**: Diagnostic tools for troubleshooting verification issues
