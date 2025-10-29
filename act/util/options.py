#===- util.options.py ----ACT Native Parameters & Option Definitions---------#
#
#                 ACT: Abstract Constraints Transformer
#
# Copyright (C) <2025->  ACT Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Authors: ACT Team   
#                                                                         
# External tool compatibility parameters are adapted from:               
#   - α,β-CROWN: https://github.com/Verified-Intelligence/alpha-beta-CROWN 
#     Copyright (C) 2021-2025 The α,β-CROWN Team                           
#     Licensed under BSD 3-Clause License                                  
#   - ERAN: https://github.com/eth-sri/eran                                
#     Copyright ETH Zurich, Licensed under Apache 2.0 License     
#   - This integration enables unified command-line interface across        
#     different verification backends while maintaining ACT's native  
#     capabilities and novel contributions.   
#
# Purpose:
#   Centralised definition of native parameters and command-line options for
#   the Abstract Constraints Transformer (ACT), including defaults, validation,
#   and help text used across the toolchain.
#
#===----------------------------------------------------------------------===#
    
import argparse
import os
from act.util.path_config import get_pipeline_log_dir

def get_parser():

    parser = argparse.ArgumentParser(description='Abstract Constraint Transformer (ACT) - Unified Neural Network Verification Framework')
    
    # ACT Core Verifier Selection
    parser.add_argument('--verifier', type=str, default=None, 
                        choices=['act', 'eran', 'abcrown'],
                        help='Backend verification engine. "eran": ERAN external verifier, "abcrown": αβ-CROWN external verifier, "act": ACT torch-native abstraction framework')
    parser.add_argument('--method', type=str, default=None, 
                        help='Verification method. ERAN: [deepzono, refinezono, deeppoly, refinepoly], αβ-CROWN: [alpha, beta, alpha_beta], ACT-Native: [torch-native]')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'gpu'],
                        help='Computation device (cpu, cuda, or gpu). Note: "cuda" and "gpu" are equivalent')
    parser.add_argument('--dtype', type=str, default='float64', choices=['float32', 'float64'],
                        help='Default PyTorch data type (float32 or float64)')
    parser.add_argument('--solver', type=str, default='auto', 
                        choices=['auto', 'gurobi', 'torch', 'both'],
                        help='Solver backend for constraint solving. "auto": try Gurobi first, fallback to PyTorch, "gurobi": Gurobi MILP solver only, "torch": PyTorch LP solver only, "both": test both solvers')

    # ACT CI/CD Environment Configuration
    parser.add_argument('--ci', action='store_true', default=False,
                        help='ACT CI mode: Use scipy.linprog instead of Gurobi for LP solving (no commercial license required). Automatically enables fallback to open-source solvers when Gurobi license is unavailable')

    # ACT Specification Refinement (Branch-and-Bound) Framework
    parser.add_argument('--enable_spec_refinement', action='store_true', default=False,
                        help='ACT innovation: Enable specification refinement BaB verification. Automatically triggers when initial abstract verification returns UNKNOWN/UNSAT')

    # ================================================================================
    # EXTERNAL TOOL COMPATIBILITY PARAMETERS
    # ================================================================================
    # The following parameters are designed for compatibility with external verification
    # tools (ERAN and αβ-CROWN) to enable unified interface calls. Parameter names and
    # descriptions are adapted from:
    # 
    # - αβ-CROWN (α,β-CROWN): https://github.com/Verified-Intelligence/alpha-beta-CROWN
    #   Copyright (C) 2021-2025 The α,β-CROWN Team
    #   Licensed under BSD 3-Clause License
    #   Primary authors: Huan Zhang, Zhouxing Shi, Xiangru Zhong
    #
    # - ERAN: https://github.com/eth-sri/eran
    #   Copyright ETH Zurich
    #   Licensed under Apache 2.0 License
    # ================================================================================

    # Model Configuration (adapted from αβ-CROWN model hierarchy)
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to neural network model file (ONNX format supported)')

    # Data Configuration (adapted from αβ-CROWN data hierarchy)
    parser.add_argument("--start", type=int, default=0, 
                        help='Start from the i-th property in specified dataset')
    parser.add_argument("--end", type=int, default=10000, 
                        help='End with the (i-1)-th property in the dataset')
    parser.add_argument('--num_outputs', type=int, default=10,
                        help="Number of output classes for classification problems")
    parser.add_argument("--mean", nargs='+', type=float, default=None,
                        help='Mean values for data preprocessing normalisation (single value or per-channel list)')
    parser.add_argument("--std", nargs='+', type=float, default=None,
                        help='Standard deviation values for data preprocessing normalisation (single value or per-channel list)')
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (mnist, cifar10, etc.) or path to CSV file")
    parser.add_argument("--anchor", type=str, default=None,
                        help="Anchor dataset path for data point anchoring in specifications")

    # Specification Configuration (adapted from αβ-CROWN specification hierarchy)
    parser.add_argument("--spec_type", type=str, default=None, 
                        choices=['local_lp', 'local_vnnlib', 'set_vnnlib', "set_box"],
                        help='Verification specification type: "local_lp"=Lp norm around data points, "local_vnnlib"=VNNLIB with anchor points, "set_vnnlib"=set-based VNNLIB (e.g., AcasXu), "set_box"=box constraints')
    parser.add_argument("--norm", type=str, default=None, choices=['1', '2', 'inf'],
                        help='Lp-norm for epsilon perturbation in robustness verification')
    parser.add_argument("--epsilon", type=float, default=None,
                        help='Perturbation bound (Lp norm). If unset, dataset-specific defaults may apply')
    parser.add_argument("--vnnlib_path", type=str, default=None,
                        help='Path to VNNLIB specification file (overrides Lp/robustness verification arguments)')
    # ================================================================================
    
    # ================================================================================
    return parser


class PerformanceOptions:
    """Global performance and debugging options.
    
    Both options default to True (debugging enabled) to help catch bugs during development.
    Use disable_all() for production/performance runs.
    
    Attributes:
        debug_tf: Enable debug logging in transfer functions (default: True)
        validate_constraints: Enable constraint validation (default: True)
        debug_output_file: Path to debug log file (default: "act/pipeline/log/act_debug_tf.log")
        debug_tf_max_constraints: Maximum number of constraints to log per layer (default: 50)
    
    Example:
        # Default: debugging enabled
        from act.util.options import PerformanceOptions
        
        # Performance mode
        PerformanceOptions.disable_all()
        
        # Custom debug file
        PerformanceOptions.set_debug_output_file("my_debug.log")
        
        # Show more/fewer constraints in debug log
        PerformanceOptions.debug_tf_max_constraints = 100
    """
    debug_tf: bool = True
    validate_constraints: bool = True
    debug_output_file: str = os.path.join(get_pipeline_log_dir(), "act_debug_tf.log")
    debug_tf_max_constraints: int = 50
    
    @classmethod
    def enable_debug_tf(cls) -> None:
        """Enable transfer function debug logging."""
        cls.debug_tf = True
    
    @classmethod
    def enable_validate_constraints(cls) -> None:
        """Enable constraint validation."""
        cls.validate_constraints = True
    
    @classmethod
    def disable_all(cls) -> None:
        """Disable all debugging features for performance runs."""
        cls.debug_tf = False
        cls.validate_constraints = False
    
    @classmethod
    def set_debug_output_file(cls, filepath: str) -> None:
        """Set the debug output file path.
        
        Args:
            filepath: Path to the debug output file
        """
        cls.debug_output_file = filepath