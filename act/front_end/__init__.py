#===- act/front_end/__init__.py - ACT Frontend Preprocessing Module ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
"""
ACT Front-End Module

Primary API: SpecLoader
=======================
SpecLoader is the unified entry point for all data loading and spec creation.

Quick Start:
    >>> from act.front_end import SpecLoader
    >>> 
    >>> # Load data + model + create specs in one line
    >>> loader = SpecLoader.from_torchvision("MNIST", 32, eps=0.1, model_name="resnet18")
    >>> 
    >>> # Quick access to data
    >>> images, labels, model = loader.data
    >>> 
    >>> # Create VerifiableModel for verification
    >>> wrapped, report = loader.synthesize()
    >>> 
    >>> # Or get specs manually
    >>> input_spec, output_spec = loader.get_specs()

VNNLib Support:
    >>> loader = SpecLoader.from_vnnlib("mnist_fc", 50)
    >>> wrapped, report = loader.synthesize()

Direct Tensor Input:
    >>> loader = SpecLoader.from_tensors(images, labels, eps=0.1, model=my_model)
    >>> wrapped, report = loader.synthesize()

Low-Level API:
    - InputSpec, OutputSpec: Specification dataclasses
    - InKind, OutKind: Spec type enums  
    - VerifiableModel: Wrapped model with embedded constraints
    - synthesize_model(): Direct model wrapping function
"""

from act.front_end.specs import (
    InputSpec, OutputSpec, InKind, OutKind,
)

from act.front_end.verifiable_model import (
    InputLayer,
    InputSpecLayer,
    OutputSpecLayer,
    VerifiableModel,
)

from act.front_end.model_synthesis import (
    WrapReport,
    run_synthesis_pipeline,
    synthesize_model,
    synthesize_models_grouped,
)

from act.front_end.spec_loader import SpecLoader

__all__ = [
    'SpecLoader',
    'InputSpec', 'OutputSpec', 'InKind', 'OutKind',
    'InputLayer', 'InputSpecLayer', 'OutputSpecLayer', 'VerifiableModel',
    'WrapReport', 'run_synthesis_pipeline',
    'synthesize_model', 'synthesize_models_grouped',
]
