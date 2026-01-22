#===- act/front_end/__init__.py - ACT Frontend Preprocessing Module ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   ACT Front-End module providing unified specification system and utilities
#   for DNN verification. Integrates with spec creators for data/model/spec
#   loading and synthesis.
#
#===---------------------------------------------------------------------===#

"""
ACT Front-End Module

Key Features:
- Unified specification system (InputSpec/OutputSpec)
- Native batch support in VerifiableModel (works with batch=1 or batch>1)
- Batched spec helpers (BatchedInputSpec/BatchedOutputSpec) for multi-sample specs
- Spec creators (TorchVision, VNNLib)
- Device-aware tensor management

Usage:
    >>> from act.front_end import InputSpec, OutputSpec, InKind, OutKind
    >>> from act.front_end import VerifiableModel, InputSpecLayer, OutputSpecLayer
    >>> 
    >>> # Create specs (single-sample structure)
    >>> input_spec = InputSpec(kind=InKind.LINF_BALL, center=data, eps=0.1)
    >>> output_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=5)
    >>> 
    >>> # VerifiableModel now supports batched inference natively
    >>> model = VerifiableModel(InputSpecLayer(input_spec), nn_model, OutputSpecLayer(output_spec))
    >>> results = model(batched_input)  # Works with batch_size > 1
    >>> # results['input_satisfied'] is Tensor[B], results['output_satisfied'] is Tensor[B]
"""

# Core specification system (single-sample + batched) - ALL FROM specs.py
from act.front_end.specs import (
    InputSpec, OutputSpec, InKind, OutKind,
    BatchedInputSpec, BatchedOutputSpec,
)

# Verifiable model components (now with native batch support) - ALL FROM verifiable_model.py
from act.front_end.verifiable_model import (
    InputLayer,
    InputSpecLayer,
    OutputSpecLayer,
    VerifiableModel,
)

# Model synthesis (single-sample + batched) - ALL FROM model_synthesis.py
from act.front_end.model_synthesis import (
    # Single-sample
    WrapReport,
    synthesize_single_model_from_spec,
    synthesize_models_from_specs,
    model_synthesis,
    # Batched
    BatchedWrapReport,
    synthesize_batched_model,
    synthesize_batched_model_from_loader,
    synthesize_batched_models_from_specs,
)

# Batched data loaders (utility class) - KEEP batched_loader.py for now
from act.front_end.batched_loader import BatchedSpecLoader, load_batched

# Device management - import only when needed to avoid triggering argparse at import time
# from act.util.device_manager import get_default_device, get_default_dtype, get_current_settings

__all__ = [
    # Specifications (single-sample structure, but layers support batched inference)
    'InputSpec', 'OutputSpec', 'InKind', 'OutKind',
    
    # Batched specifications (for creating batched specs from multiple samples)
    'BatchedInputSpec', 'BatchedOutputSpec',
    
    # Verifiable model components (native batch support - works with batch=1 or batch>1)
    'InputLayer', 'InputSpecLayer', 'OutputSpecLayer', 'VerifiableModel',
    
    # Synthesis
    'WrapReport', 'synthesize_single_model_from_spec', 
    'synthesize_models_from_specs', 'model_synthesis',
    
    # Batched synthesis
    'BatchedWrapReport', 'synthesize_batched_model', 
    'synthesize_batched_model_from_loader', 'synthesize_batched_models_from_specs',
    
    # Batched data loaders
    'BatchedSpecLoader', 'load_batched',
]
