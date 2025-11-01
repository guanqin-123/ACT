#===- act/front_end/model_synthesis.py - Model Synthesis Framework -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Model Synthesis and Generation Framework. Advanced neural network synthesis,
#   optimization, and domain-specific model generation. Single-file implementation
#   for ACT-compatible model synthesis pipeline.
#
#===---------------------------------------------------------------------===#

# Detect if running as script (not as module) and exit with helpful message
if __name__ == "__main__" and __package__ is None:
    import sys
    print("\n" + "="*80)
    print("⚠️  ERROR: Cannot run as script due to import conflicts!")
    print("Please run as a module instead:")
    print("  python -m act.front_end.model_synthesis")
    print("="*80 + "\n")
    sys.exit(1)

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union

# Import ACT components
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.front_end.verifiable_model import (
    InputLayer,
    InputSpecLayer,
    OutputSpecLayer,
    VerifiableModel,
)


# -----------------------------------------------------------------------------
# 2) Small utilities
# -----------------------------------------------------------------------------
def prod(seq: Tuple[int, ...]) -> int:
    """Calculate product of sequence elements."""
    p = 1
    for s in seq:
        p *= s
    return p


def infer_layout_from_tensor(x: torch.Tensor) -> str:
    """Infer tensor layout (HWC, CHW, or FLAT) from shape."""
    if x.dim() == 4 and x.shape[-1] in (1, 3, 4):
        return "HWC"
    elif x.dim() == 4:
        return "CHW"
    return "FLAT"


def needs_flatten_before_model(model: nn.Module) -> bool:
    """Check if model needs flattening layer before first Linear layer."""
    children = list(model.children())
    if not children:
        return isinstance(model, nn.Linear)
    first = children[0]
    return isinstance(first, nn.Linear)


# -----------------------------------------------------------------------------
# 3) Model synthesis from spec creators
# -----------------------------------------------------------------------------
@dataclass
class WrapReport:
    """Report metadata for wrapped model."""
    input_shape: Tuple[int, ...]
    in_spec_kind: str
    out_spec_kind: str
    data_source: str
    model_name: str


def synthesize_models_from_specs(
    spec_results: List[Tuple[str, str, nn.Module, List[torch.Tensor], List[Tuple[InputSpec, OutputSpec]]]]
) -> Tuple[Dict[str, nn.Sequential], Dict[str, WrapReport], Dict[str, Dict[str, torch.Tensor]]]:
    """
    Synthesize wrapped models directly from spec creator results.
    
    Aligned with TorchVisionSpecCreator and VNNLibSpecCreator output format.
    Processes each (dataset, model) pair with its associated spec pairs.
    
    Args:
        spec_results: Output from create_specs_for_data_model_pairs()
            List of (data_source, model_name, pytorch_model, input_tensors, spec_pairs)
            where:
            - data_source: Dataset/category name (e.g., "MNIST", "mnist_fc")
            - model_name: Model name (e.g., "simple_cnn", "instance_0")
            - pytorch_model: torch.nn.Module
            - input_tensors: List of input sample tensors
            - spec_pairs: List of (InputSpec, OutputSpec) tuples
    
    Returns:
        wrapped_models: Dict[combo_id, nn.Sequential] - Synthesized VerifiableModel instances
        reports: Dict[combo_id, WrapReport] - Metadata for each wrapped model
        input_data: Dict[data_source, data_pack] - Input data for testing
        
    combo_id format: "m:<model_name>|x:<data_source>|s:<spec_index>|is:<input_kind>|os:<output_kind>_m<margin>"
    """
    wrapped_models: Dict[str, nn.Sequential] = {}
    reports: Dict[str, WrapReport] = {}
    input_data: Dict[str, Dict[str, torch.Tensor]] = {}
    
    print(f"\n🧬 Synthesizing models from {len(spec_results)} spec result(s)...")
    
    for data_source, model_name, pytorch_model, input_tensors, spec_pairs in spec_results:
        if not input_tensors:
            print(f"⚠️  Skipping {data_source} + {model_name}: No input tensors")
            continue
        
        if not spec_pairs:
            print(f"⚠️  Skipping {data_source} + {model_name}: No spec pairs")
            continue
        
        # Calculate specs per sample (assumes uniform distribution)
        specs_per_sample = len(spec_pairs) // len(input_tensors) if input_tensors else 0
        
        # Create wrapped models for each spec pair
        for spec_idx, (input_spec, output_spec) in enumerate(spec_pairs):
            # Determine which input tensor this spec corresponds to
            sample_idx = spec_idx // specs_per_sample if specs_per_sample > 0 else 0
            sample_idx = min(sample_idx, len(input_tensors) - 1)  # Clamp to valid range
            input_tensor = input_tensors[sample_idx]
            
            # Ensure batch dimension
            if input_tensor.dim() == len(input_tensor.shape):
                # Already has proper dimensions, add batch if needed
                if input_tensor.shape[0] != 1:
                    x = input_tensor.unsqueeze(0)
                else:
                    x = input_tensor
            else:
                x = input_tensor.unsqueeze(0)
            
            input_shape = tuple(x.shape)
            
            # Infer metadata from tensor
            layout = infer_layout_from_tensor(x)
            center_opt = x.squeeze(0).reshape(-1)  # For LINF_BALL specs
            dtype = x.dtype
            
            # Infer domain and channels
            if x.dim() == 4:
                domain = "vision"
                channels = x.shape[1]  # (B,C,H,W)
            else:
                domain = "tabular"
                channels = None
            
            # Compute value range
            value_range = (float(x.min().item()), float(x.max().item())) if x.numel() > 0 else None
            
            # Store in input_data for testing (use first sample as representative)
            if data_source not in input_data:
                first_tensor = input_tensors[0]
                first_x = first_tensor.unsqueeze(0) if first_tensor.dim() == 3 else first_tensor
                input_data[data_source] = {
                    "x": first_x,
                    "layout": infer_layout_from_tensor(first_x),
                    "center": first_tensor.reshape(-1),
                    "labels": None,  # Specs contain label info if needed
                    "scale_hint": "normalized" if domain == "vision" else "unknown",
                }
            
            # Create unique combo_id with spec index to avoid overwrites
            margin_str = f"m{output_spec.margin:.1f}" if hasattr(output_spec, 'margin') and output_spec.margin is not None else "m0.0"
            combo_id = f"m:{model_name}|x:{data_source}|s:{spec_idx}|is:{input_spec.kind}|os:{output_spec.kind}_{margin_str}"
            
            # Extract label from output spec if available
            label_tensor = None
            if hasattr(output_spec, 'y_true') and output_spec.y_true is not None:
                label_tensor = torch.tensor([output_spec.y_true])
            
            # Build layer stack
            layers: List[nn.Module] = [
                InputLayer(
                    shape=input_shape,
                    dtype=dtype,
                    center=center_opt,
                    layout=layout,
                    dataset_name=data_source,
                    num_classes=None,
                    value_range=value_range,
                    scale_hint="normalized" if domain == "vision" else "unknown",
                    distribution="normalized" if domain == "vision" else "unknown",
                    label=label_tensor,
                    sample_id=None,
                    domain=domain,
                    channels=channels,
                ),
                InputSpecLayer(spec=input_spec),
            ]
            
            # Add flatten if needed
            if needs_flatten_before_model(pytorch_model) and len(input_shape) > 2:
                layers.append(nn.Flatten())
            
            # Add model and output spec
            layers.append(pytorch_model)
            layers.append(OutputSpecLayer(spec=output_spec))
            
            # Create VerifiableModel
            wrapped = VerifiableModel(*layers)
            wrapped_models[combo_id] = wrapped
            
            # Create report
            reports[combo_id] = WrapReport(
                input_shape=input_shape,
                in_spec_kind=input_spec.kind,
                out_spec_kind=output_spec.kind,
                data_source=data_source,
                model_name=model_name,
            )
        
        print(f"✓ {data_source} + {model_name}: Created {len(spec_pairs)} wrapped model(s)")
    
    print(f"\n🎉 Synthesized {len(wrapped_models)} wrapped models from specs!")
    return wrapped_models, reports, input_data


# -----------------------------------------------------------------------------
# 4) Model synthesis main function
# -----------------------------------------------------------------------------
def model_synthesis() -> Tuple[Dict[str, nn.Sequential], Dict[str, Dict[str, torch.Tensor]]]:
    """
    Main model synthesis function using new spec creators.
    
    Simplified implementation that delegates spec creation to TorchVisionSpecCreator
    or VNNLibSpecCreator, then synthesizes wrapped models directly.
    
    Returns:
        wrapped_models: Dict[combo_id, nn.Sequential] - All synthesized wrapped models
        input_data: Dict[dataset_name, data_pack] - Input data for testing
        
    Raises:
        RuntimeError: If no spec creator can load data-model pairs or create specs
    """
    from act.front_end.torchvision.create_specs import TorchVisionSpecCreator
    
    print(f"\n{'='*80}")
    print(f"MODEL SYNTHESIS: Using New Spec Creators")
    print(f"{'='*80}")
    
    # Try TorchVision spec creator
    print(f"\n📊 Attempting to use TorchVisionSpecCreator...")
    creator = TorchVisionSpecCreator(config_name="torchvision_classification")
    
    # Create specs for all downloaded dataset-model pairs
    spec_results = creator.create_specs_for_data_model_pairs(
        num_samples=3,  # Use 3 sample per pair for synthesis
        validate_shapes=True
    )
    
    # Validate results
    if not spec_results:
        raise RuntimeError(
            "No dataset-model pairs found! Please download datasets first.\n\n"
            "Examples:\n"
            "  python -m act.front_end.torchvision --download MNIST simple_cnn\n"
            "  python -m act.front_end.torchvision --download CIFAR10 lenet5\n"
            "  python -m act.front_end.torchvision --list  # Show available pairs\n"
        )
    
    print(f"✓ Successfully created specs using TorchVisionSpecCreator")
    print(f"  Found {len(spec_results)} dataset-model pair(s)")
    
    # Synthesize wrapped models from spec results
    wrapped_models, reports, input_data = synthesize_models_from_specs(spec_results)
    
    # Validate synthesis results
    if not wrapped_models:
        raise RuntimeError(
            "Failed to synthesize any wrapped models! "
            "Spec results were loaded but model synthesis failed. "
            "Check spec_results format and synthesize_models_from_specs() logic."
        )
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SYNTHESIS COMPLETE")
    print(f"{'='*80}")
    print(f"  • Wrapped models: {len(wrapped_models)}")
    print(f"  • Data sources: {len(input_data)}")
    print(f"  • Unique dataset-model pairs: {len(set((r.data_source, r.model_name) for r in reports.values()))}")
    
    # Calculate statistics from spec_results for breakdown
    total_samples = sum(len(input_tensors) for _, _, _, input_tensors, _ in spec_results)
    total_spec_pairs = sum(len(spec_pairs) for _, _, _, _, spec_pairs in spec_results)
    
    # Print detailed breakdown
    if total_samples > 0 and total_spec_pairs > 0:
        specs_per_sample = total_spec_pairs // total_samples if total_samples else 0
        print(f"\n📊 Breakdown:")
        print(f"  • Input samples: {total_samples}")
        print(f"  • Spec pairs per sample: {specs_per_sample}")
        print(f"    (= 2 input kinds × 4 epsilons × 3 output specs)")
        print(f"    (= BOX, LINF_BALL × 0.01,0.03,0.05,0.1 × MARGIN_ROBUST(m=0.0,0.5), TOP1_ROBUST)")
        print(f"  • Total spec pairs: {total_spec_pairs}")
        print(f"  • Calculation: {total_samples} samples × {specs_per_sample} specs/sample = {total_spec_pairs} wrapped models")
    
    return wrapped_models, input_data


if __name__ == "__main__":
    from act.util.model_inference import model_inference
    
    # Step 1: Synthesize all wrapped models using new spec creators
    wrapped_models, input_data = model_synthesis()
    
    # Step 2: Test all models with inference
    successful_models = model_inference(wrapped_models, input_data)
    
    print(f"\n✅ Successfully inferred {len(successful_models)} out of {len(wrapped_models)} models")
    print(f"\n🎯 NEW SPEC CREATOR INTEGRATION: COMPLETE ✅")
