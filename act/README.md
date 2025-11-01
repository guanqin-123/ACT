# ACT Directory

This directory contains the core verification framework for the Abstract Constraint Transformer (ACT) system. It implements a modern three-tier architecture: Front-End (data/model/spec processing), Back-End (verification core), and Pipeline (testing/integration) with PyTorch-native verification capabilities.

## Recent Development Updates

### Debugging and Performance Framework (October 2025)
- **PerformanceOptions**: Global debugging flags with `debug_tf`, `validate_constraints`, and configurable logging
- **Transfer Function Logging**: Detailed layer-by-layer analysis logging to `act/pipeline/log/act_debug_tf.log`
- **Constraint Validation**: Targeted validation framework that checks only referenced variables
- **ConSet Improvements**: Added `__iter__` and `__len__` methods for Pythonic container usage
- **Path Management**: Centralized logging to `act/pipeline/log/` using `path_config.py`

### Code Quality Improvements
- **Batch Dimension Fix**: Fixed `affine_bounds()` with proper batch dimension handling and squeeze operations
- **Cleaner Syntax**: Updated all code to use ConSet wrappers (`for con in cons` instead of `.S.values()`)
- **Guarded File I/O**: All debug file operations protected by feature flags
- **Architecture Cleanup**: Removed legacy loaders and raw_processors, consolidated into spec creator system

## Directory Structure

```
act/
├── main.py                         # Main unified verification interface
├── __init__.py                     # Package initialization
│
├── front_end/                      # Front-End: User-facing data processing
│   ├── torchvision/                # TorchVision integration
│   │   ├── create_specs.py         # TorchVisionSpecCreator for dataset-model pairs
│   │   ├── data_model_loader.py    # TorchVision dataset and model loading
│   │   └── data_model_mapping.py   # Dataset-model compatibility mappings
│   ├── vnnlib/                     # VNNLIB integration
│   │   └── create_specs.py         # VNNLibSpecCreator for VNNLIB specs
│   ├── specs.py                    # InputSpec/OutputSpec with InKind/OutKind enums
│   ├── spec_creator_base.py        # Base spec creator interface
│   ├── verifiable_model.py         # PyTorch verification wrapper modules
│   ├── model_synthesis.py          # Model synthesis using spec creators
│   └── README.md                   # Front-end documentation
│
├── back_end/                       # Back-End: Core verification engine
│   ├── core.py                     # Net, Layer, Bounds, Con, ConSet data structures
│   ├── verifier.py                 # Spec-free verification: verify_once(), verify_bab()
│   ├── layer_schema.py             # Layer type definitions and validation rules
│   ├── layer_util.py               # Layer validation and creation utilities
│   ├── bab.py                      # Branch-and-bound refinement with CE validation
│   ├── utils.py                    # Backend utilities (affine_bounds, validate_constraints)
│   ├── analyze.py                  # Network analysis and bounds propagation
│   ├── cons_exportor.py            # Constraint export to solvers
│   ├── transfer_functions.py       # Transfer function interface and dispatch
│   ├── net_factory.py              # YAML-driven network factory for examples
│   ├── solver/                     # MILP/LP optimization solvers
│   │   ├── solver_base.py          # Base solver interface
│   │   ├── solver_gurobi.py        # Gurobi MILP solver integration
│   │   └── solver_torch.py         # PyTorch-based LP solver
│   ├── interval_tf/                # Interval-based transfer functions
│   │   ├── interval_tf.py          # Interval TF implementation
│   │   ├── tf_mlp.py               # MLP layer interval analysis
│   │   ├── tf_cnn.py               # CNN layer interval analysis
│   │   ├── tf_rnn.py               # RNN layer interval analysis
│   │   └── tf_transformer.py       # Transformer interval analysis
│   ├── hybridz_tf/                 # HybridZ zonotope transfer functions
│   │   ├── hybridz_tf.py           # HybridZ TF implementation
│   │   ├── tf_mlp.py               # MLP layer zonotope analysis
│   │   ├── tf_cnn.py               # CNN layer zonotope analysis
│   │   ├── tf_rnn.py               # RNN layer zonotope analysis
│   │   └── tf_transformer.py       # Transformer zonotope analysis
│   ├── serialization/              # Net serialization and deserialization
│   │   ├── serialization.py        # NetSerializer with tensor encoding
│   │   └── test_serialization.py   # Serialization correctness tests
│   ├── examples/                   # Example networks and configurations
│   │   ├── examples_config.yaml    # YAML network definitions
│   │   ├── nets/                   # Generated ACT Net JSON files
│   │   └── README.md               # Examples documentation
│   └── README.md                   # Back-end documentation
│
├── pipeline/                       # Pipeline: Testing framework and integration
│   ├── torch2act.py                # Automatic PyTorch→ACT Net conversion
│   ├── validate_verifier.py        # Verifier correctness validation with concrete tests
│   ├── correctness.py              # Correctness validation utilities
│   ├── regression.py               # Baseline capture and regression testing
│   ├── integration.py              # Front-end integration bridge
│   ├── model_factory.py            # ACT Net factory for test networks
│   ├── config.py                   # YAML-based test scenario management
│   ├── reporting.py                # Results analysis and report generation
│   ├── utils.py                    # Shared utilities and performance profiling
│   ├── run_tests.py                # Command-line testing interface
│   ├── configs/                    # Configuration files
│   │   ├── mock_inputs.yaml        # Mock data generation templates
│   │   ├── test_scenarios.yaml     # Complete test scenario definitions
│   │   ├── solver_settings.yaml    # Solver configuration options
│   │   └── baselines.json          # Performance baseline storage
│   ├── examples/                   # Example usage and quick tests
│   ├── log/                        # Test execution logs (includes act_debug_tf.log)
│   └── README.md                   # Pipeline documentation
│
├── util/                           # Shared Utilities
│   ├── device_manager.py           # GPU-first CUDA device handling
│   ├── path_config.py              # Project path configuration and management
│   ├── options.py                  # Command-line arguments and PerformanceOptions
│   └── stats.py                    # Statistics and performance tracking
│
└── wrapper_exts/                   # External verifier integrations
    ├── abcrown/                    # αβ-CROWN integration module
    │   ├── abcrown_verifier.py     # αβ-CROWN wrapper and interface
    │   └── abcrown_runner.py       # αβ-CROWN backend execution script
    └── eran/                       # ERAN integration module
        └── eran_verifier.py        # ERAN wrapper and interface
```

## Module Documentation

### **Main Interface**
- **`main.py`**: Primary entry point for all verification tasks
  - Unified command-line interface supporting all verifiers
  - Parameter parsing, validation, and backend routing
  - Comprehensive argument compatibility across different verification tools
  - Integration with configuration defaults from `../modules/configs/`

### **`front_end/` - User-Facing Data Processing**
- **Spec Creator System**: Unified framework for creating specifications from various sources
  - **`TorchVisionSpecCreator`**: Creates specs from TorchVision datasets and models
  - **`VNNLibSpecCreator`**: Creates specs from VNNLIB files
  - **`BaseSpecCreator`**: Abstract interface for spec creators

- **`specs.py`**: Specification data structures and enums
  - `InputSpec`/`OutputSpec` classes with `InKind`/`OutKind` type safety
  - Support for BOX, L_INF, LIN_POLY input constraints and SAFETY, ASSERT output properties

- **`verifiable_model.py`**: PyTorch verification wrapper modules
  - `InputLayer`: Declares symbolic input blocks for verification
  - `InputSpecLayer`: Wraps ACT InputSpec as nn.Module for seamless integration
  - `OutputSpecLayer`: Wraps ACT OutputSpec as nn.Module for property specification

- **`model_synthesis.py`**: Model synthesis using spec creators
  - Unified synthesis pipeline using spec creator system
  - Automatic wrapped model generation from dataset-model pairs

- **Preprocessors**: Modular preprocessing pipeline
  - **`preprocessor_image.py`**: Image normalization, augmentation, and format conversion
  - **`preprocessor_text.py`**: Text preprocessing utilities
  - **`preprocessor_base.py`**: Base preprocessor interface and common functionality

### **`back_end/` - Core Verification Engine**
- **`core.py`**: Fundamental ACT data structures
  - `Net`: Network representation with layers and graph connectivity
  - `Layer`: Individual layer with params, metadata, and variable mappings
  - `Bounds`: Box constraints with lb/ub tensors for variable ranges
  - `Con`/`ConSet`: Constraint representation with Pythonic iteration support (`__iter__`, `__len__`)

- **`verifier.py`**: Spec-free verification engine
  - `verify_once()`: Single-shot verification using embedded ACT constraints
  - `verify_bab()`: Branch-and-bound refinement with counterexample validation
  - Integrated constraint validation with targeted variable checking
  - No external input specs required - all constraints extracted from ACT Net

- **`utils.py`**: Backend utility functions
  - `affine_bounds()`: Affine transformation with proper batch dimension handling
  - `validate_constraints()`: Targeted constraint validation (only checks referenced variables)
  - Debug logging support with guarded file operations

- **`transfer_functions.py`**: Transfer function interface and dispatch
  - Abstract `TransferFunction` base class with `supports_layer()` and `apply()` methods
  - Global transfer function registry with mode selection (interval/hybridz)
  - `dispatch_tf()`: Main entry point with optional debug logging
  - Configurable constraint logging (up to 50 constraints by default via `PerformanceOptions`)

- **`net_factory.py`**: YAML-driven network factory
  - Generates example ACT networks from YAML configurations
  - Automatic parameter generation for INPUT_SPEC and ASSERT layers
  - Comprehensive ASSERT specification guide (TOP1_ROBUST, MARGIN_ROBUST, LINEAR_LE, RANGE)
  - Proper tensor serialization using NetSerializer

- **`layer_schema.py`**: Layer type definitions and validation rules
  - Comprehensive schema definitions for all supported layer types
  - Parameter validation and metadata requirements

- **`bab.py`**: Branch-and-bound refinement implementation
  - BaB tree management with priority queues
  - Counterexample validation and refinement strategies
  - Configurable depth limits and timeout handling

- **`solver/`**: MILP/LP optimization backend
  - **`solver_gurobi.py`**: Gurobi MILP solver integration with license management
  - **`solver_torch.py`**: PyTorch-based LP solver for lightweight optimization
  - **`solver_base.py`**: Unified solver interface and status handling

- **Transfer Function Implementations**: Two precision/performance modes
  - **`interval_tf/`**: Fast interval-based bounds propagation
    - `IntervalTF`: Main implementation with layer-specific modules
    - Separate modules for MLP, CNN, RNN, and Transformer layers
  - **`hybridz_tf/`**: High-precision zonotope-based analysis
    - `HybridzTF`: Enhanced precision with zonotope domains
    - Separate modules for MLP, CNN, RNN, and Transformer layers

- **`serialization/`**: Net persistence and loading
  - **`serialization.py`**: `NetSerializer` with proper tensor encoding/decoding
  - **`test_serialization.py`**: Serialization correctness validation

- **`examples/`**: Example networks and test cases
  - **`examples_config.yaml`**: YAML definitions for example networks
  - **`nets/`**: Generated ACT Net JSON files (MNIST, CIFAR, control, reachability)
  - Networks include embedded INPUT_SPEC and ASSERT layers for spec-free verification

### **`pipeline/` - Testing Framework and Integration**
- **`torch2act.py`**: Automatic PyTorch→ACT Net conversion
  - Seamless conversion from PyTorch nn.Module to ACT Net representation
  - Preserves all verification constraints and model semantics
  - Support for complex wrapper layer patterns
  - Debug logging support for conversion process

- **`validate_verifier.py`**: Comprehensive verifier validation framework
  - Tests 12 networks (MNIST, CIFAR, control, reachability) with 2 solvers (Gurobi, TorchLP)
  - Concrete counterexample generation and validation
  - Formal verification result checking (SAT/UNSAT/CERTIFIED)
  - Detailed test reporting with pass/fail/inconclusive status

- **`model_factory.py`**: ACT Net factory for test networks
  - Pre-loads networks from `act/back_end/examples/nets/`
  - PyTorch model generation from ACT Nets
  - Integration with VerifiableModel wrapper layers

- **Testing Framework**: Comprehensive validation and regression testing
  - **`correctness.py`**: Verifier correctness validation with property-based testing
  - **`regression.py`**: Baseline capture and performance regression detection
  - **`integration.py`**: Front-end integration bridge for real ACT component testing

- **`config.py`**: YAML-based test scenario management
  - Configuration loading and validation
  - Test scenario composition and parameter management

- **Performance and Reporting**:
  - **`utils.py`**: Performance profiling, memory tracking, and optimization utilities
  - **`reporting.py`**: Results analysis and comprehensive report generation
  - **`run_tests.py`**: Command-line testing interface with parallel execution

- **`log/`**: Centralized logging directory
  - **`act_debug_tf.log`**: Transfer function debug output (layer-by-layer analysis, bounds, constraints)
  - Test execution logs and validation results

### **`util/` - Shared Utilities**

- **`device_manager.py`**: GPU-first CUDA device handling
  - Automatic device detection and management
  - GPU memory optimization and fallback strategies
  - Global PyTorch device and dtype configuration

- **`path_config.py`**: Project path configuration and management
  - `get_project_root()`, `get_data_root()`, `get_config_root()`
  - `get_pipeline_log_dir()`: Returns absolute path to `act/pipeline/log/`
  - `ensure_gurobi_license()`: Automatic Gurobi license detection
  - Centralized path management for all ACT modules

- **`options.py`**: Command-line arguments and performance configuration
  - Centralized CLI option definitions for all verifiers
  - **`PerformanceOptions`**: Global debugging and performance flags
    - `debug_tf`: Enable/disable transfer function debug logging (default: True)
    - `validate_constraints`: Enable/disable constraint validation (default: True)
    - `debug_output_file`: Path to debug log (default: `act/pipeline/log/act_debug_tf.log`)
    - `debug_tf_max_constraints`: Max constraints to log per layer (default: 50)
    - Methods: `enable_debug_tf()`, `disable_all()`, `set_debug_output_file()`
  - Parameter validation and default value management

- **`stats.py`**: Statistics and performance tracking
  - Verification result logging and analysis
  - Performance metrics collection and reporting

### **`wrapper_exts/` - External Verifier Integrations**

#### **`abcrown/` - αβ-CROWN Integration**
- **`abcrown_verifier.py`**: αβ-CROWN wrapper and interface
  - Translates ACT parameters to αβ-CROWN format
  - Manages conda environment isolation for αβ-CROWN execution
  - Handles subprocess communication and result parsing
  - Provides error handling and comprehensive logging

- **`abcrown_runner.py`**: αβ-CROWN backend execution script
  - Contains code adapted from the open-source αβ-CROWN project with enhancements for ACT framework integration
  - Executes within isolated `act-abcrown` conda environment
  - Direct interface to αβ-CROWN complete verification engine
  - Independent execution without ACT path dependencies

#### **`eran/` - ERAN Integration**
- **`eran_verifier.py`**: ERAN wrapper and interface
  - Integration with ERAN abstract interpretation methods
  - Support for DeepPoly, DeepZono, and other ERAN domains
  - Parameter translation for ERAN backend compatibility


## Architecture Benefits

The three-tier modular architecture provides several key advantages:

### **Three-Tier Design**
- **Front-End Separation**: User-facing data processing isolated from core verification logic
- **Back-End Focus**: Pure verification engine with PyTorch-native analysis and optimization
- **Pipeline Integration**: Comprehensive testing framework and Torch→ACT conversion bridge
- **Clean Boundaries**: Clear interfaces between data processing, verification, and testing

### **Modern Verification Features**
- **Spec-Free Verification**: All constraints embedded in PyTorch models via wrapper layers
- **PyTorch-Native**: Verification engine operates directly on PyTorch tensors for performance
- **Automatic Conversion**: Seamless PyTorch→ACT Net conversion preserving all semantics
- **GPU-First**: Optimized CUDA device management with automatic fallback strategies
- **Debug Infrastructure**: Comprehensive debugging with transfer function logging and constraint validation

### **Code Quality and Maintainability**
- **Pythonic Containers**: ConSet with `__iter__` and `__len__` for natural iteration
- **Type Safety**: Proper type hints and validation throughout codebase
- **Guarded Operations**: Debug file I/O protected by feature flags to prevent performance impact
- **Centralized Logging**: All debug output to `act/pipeline/log/` with configurable detail levels
- **Batch Handling**: Proper tensor dimension management with assertions and squeeze operations

### **Modular Design**
- **Clear Separation**: Front-end, back-end, and pipeline modules have distinct responsibilities
- **Independent Development**: Modules can be developed, tested, and maintained separately
- **Easy Extension**: Add new verifiers by creating new modules in `wrapper_exts/`
- **Reusable Components**: Shared utilities and interfaces enable code reuse
- **Transfer Function Modes**: Pluggable TF implementations (interval vs. hybridz) via global registry

### **Testing and Validation**
- **Comprehensive Testing**: Pipeline framework provides correctness, regression, and performance testing
- **Validation Framework**: `validate_verifier.py` tests 12 networks across 2 solvers (24 test cases)
- **Concrete Counterexamples**: Real input generation to validate formal verification results
- **Integration Testing**: Real ACT component testing with front-end bridge
- **Continuous Validation**: Baseline capture and regression detection for quality assurance
- **Constraint Validation**: Targeted validation checks only variables referenced in constraints

### **Configuration Management**
- **Centralized Defaults**: Configuration files in `../modules/configs/` provide optimal parameters
- **Device Management**: Intelligent GPU/CPU device selection and memory optimization
- **Environment Isolation**: Different verifiers can use separate conda environments
- **Parameter Management**: Unified command-line interface with type validation
- **Path Configuration**: Centralized path management via `path_config.py`

### **Debugging and Development**
- **PerformanceOptions**: Global flags for enabling/disabling debug features
- **Transfer Function Logging**: Layer-by-layer analysis with bounds, parameters, and constraints
- **Configurable Detail**: Control constraint logging depth (default: 50 per layer)
- **Targeted Validation**: Efficient constraint validation focusing on referenced variables
- **Guarded I/O**: All debug operations protected to minimize production overhead

### **Integration Flexibility**
- **Unified Interface**: Single entry point (`main.py`) for all verification tasks
- **Backend Abstraction**: Consistent API regardless of underlying verification method
- **Parameter Translation**: Automatic conversion between ACT and backend-specific formats
- **Result Standardization**: Uniform output format across all verification backends

### **Performance Optimization**
- **Memory Management**: Comprehensive memory tracking and optimization throughout pipeline
- **Utility Reuse**: Common operations centralized to eliminate code duplication
- **Efficient Imports**: Modular structure reduces import overhead and circular dependencies
- **GPU Acceleration**: PyTorch-native verification leverages GPU computation where beneficial
- **Configurable Logging**: Disable debug features in production for optimal performance

## Examples and Network Generation
Example ACT networks are stored as JSON under `act/back_end/examples/nets/`.
These files are generated from the YAML configuration `act/back_end/examples/examples_config.yaml`
using the YAML-driven network factory. The test suite and serializer load networks
from the `examples/nets` directory. When authoring new example networks prefer the
YAML configuration and the factory rather than hand-editing the JSON files.