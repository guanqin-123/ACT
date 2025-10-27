#===- act/pipeline/__init__.py - ACT Pipeline Module -------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   ACT Pipeline module for PyTorch model generation and testing utilities.
#   Provides tools for converting between PyTorch models and ACT Nets,
#   performance regression testing, and utility functions.
#
#===---------------------------------------------------------------------===#

"""ACT Pipeline Module - Model Generation and Testing Utilities.

This module provides utilities for PyTorch model generation, ACT conversion,
regression testing, and performance analysis.

Key Components:
    - ModelFactory: Create PyTorch models from YAML configurations
    - TorchToACT: Convert PyTorch models to ACT representation
    - RegressionTester: Track performance baselines and detect regressions
    - ReportGenerator: Generate comprehensive test reports with visualization
    - PerformanceProfiler: Profile execution time and memory usage

Example:
    # Create PyTorch model from examples_config.yaml
    factory = ModelFactory()
    model = factory.create_model("mnist_mlp_small", load_weights=True)
    
    # Convert to ACT format
    converter = TorchToACT()
    act_net = converter.convert(model, input_shape=(1, 784))
    
    # Run regression tests
    tester = RegressionTester()
    result = tester.compare_to_baseline("v1.0", (val_results, perf_results))
"""

# Core imports
from act.pipeline.model_factory import ModelFactory
from act.pipeline.torch2act import TorchToACT

# Conditionally import regression/reporting modules (optional dependencies)
try:
    from act.pipeline.regression import (
        ValidationResult,
        PerformanceResult,
        VerifyResult,
        BaselineManager,
        RegressionTester,
        RegressionResult,
        TrendData,
        BaselineMetrics,
    )
    REGRESSION_AVAILABLE = True
except ImportError:
    REGRESSION_AVAILABLE = False
    ValidationResult = None
    PerformanceResult = None
    VerifyResult = None
    BaselineManager = None
    RegressionTester = None
    RegressionResult = None
    TrendData = None
    BaselineMetrics = None

try:
    from act.pipeline.reporting import (
        ReportGenerator,
        ReportConfig,
        TestSummary,
        PerformanceSummary,
        RegressionSummary,
    )
    REPORTING_AVAILABLE = True
except ImportError:
    REPORTING_AVAILABLE = False
    ReportGenerator = None
    ReportConfig = None
    TestSummary = None
    PerformanceSummary = None
    RegressionSummary = None

try:
    from act.pipeline.utils import (
        PerformanceProfiler,
        ParallelExecutor,
        print_memory_usage,
        clear_torch_cache,
        setup_logging,
        ProgressTracker,
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    PerformanceProfiler = None
    ParallelExecutor = None
    print_memory_usage = None
    clear_torch_cache = None
    setup_logging = None
    ProgressTracker = None


__all__ = [
    # Core model factory and conversion
    'ModelFactory',
    'TorchToACT',
    
    # Data classes
    'ValidationResult',
    'PerformanceResult',
    'VerifyResult',
    
    # Regression testing (optional)
    'BaselineManager',
    'RegressionTester', 
    'RegressionResult',
    'TrendData',
    'BaselineMetrics',
    
    # Reporting (optional)
    'ReportGenerator',
    'ReportConfig',
    'TestSummary',
    'PerformanceSummary',
    'RegressionSummary',
    
    # Utilities (optional)
    'PerformanceProfiler',
    'ParallelExecutor',
    'print_memory_usage',
    'clear_torch_cache',
    'setup_logging',
    'ProgressTracker',
    
    # Availability flags
    'REGRESSION_AVAILABLE',
    'REPORTING_AVAILABLE',
    'UTILS_AVAILABLE',
]
