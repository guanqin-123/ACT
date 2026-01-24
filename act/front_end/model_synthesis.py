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
    """Report metadata for wrapped model. Works for any batch size (B>=1)."""
    input_shape: Tuple[int, ...]
    in_spec_kind: str
    out_spec_kind: str
    data_source: str
    model_name: str
    batch_size: int = 1
    unique_labels: int = 1


# -----------------------------------------------------------------------------
# 3) Synthesis pipeline (CLI workflow)
# -----------------------------------------------------------------------------
def run_synthesis_pipeline(creator: str = 'torchvision') -> Dict[str, nn.Module]:
    """
    CLI workflow: run full synthesis pipeline with a spec creator.
    
    This is a high-level orchestration function that:
    1. Loads the appropriate spec creator (TorchVision or VNNLIB)
    2. Creates specs for all available data-model pairs
    3. Synthesizes wrapped models via synthesize_models_grouped()
    4. Handles memory cleanup and prints summary
    
    For programmatic use, prefer synthesize_model() or synthesize_models_grouped().
    
    Args:
        creator: Creator to use ('torchvision' or 'vnnlib'). Defaults to 'torchvision'.
    
    Returns:
        wrapped_models: Dict[combo_id, nn.Module] - All synthesized wrapped models
        
    Raises:
        RuntimeError: If no spec creator can load data-model pairs or create specs
    """
    print(f"\n{'='*80}")
    print(f"MODEL SYNTHESIS: Using New Spec Creators ({creator.upper()})")
    print(f"{'='*80}")
    
    # Select creator based on parameter
    if creator == 'vnnlib':
        from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator
        
        print(f"\n📊 Attempting to use VNNLibSpecCreator...")
        spec_creator = VNNLibSpecCreator(config_name="vnnlib_default")
        
        # Create specs for all downloaded VNNLIB instances
        # Use max_instances=3 to limit for testing (185 total instances available)
        spec_results = spec_creator.create_specs_for_data_model_pairs(
            categories=None,  # All downloaded categories
            max_instances=3,  # Limit to 3 instances per category for synthesis
            validate_shapes=True
        )
    
    elif creator == 'torchvision':
        from act.front_end.torchvision_loader.create_specs import TorchVisionSpecCreator
        
        print(f"\n📊 Attempting to use TorchVisionSpecCreator...")
        spec_creator = TorchVisionSpecCreator(config_name="torchvision_classification")
        
        # Create specs for all downloaded dataset-model pairs
        spec_results = spec_creator.create_specs_for_data_model_pairs(
            num_samples=1,  # Use 1 sample per pair for synthesis
            validate_shapes=True
        )
    
    else:
        raise ValueError(f"Unknown creator: {creator}. Use 'torchvision' or 'vnnlib'.")
    
    # Validate results
    if not spec_results:
        if creator == 'vnnlib':
            raise RuntimeError(
                "No VNNLIB instances found! Please download VNNLIB benchmarks first.\n\n"
                "Examples:\n"
                "  python -m act.front_end --download acasxu_2023      # ACAS Xu collision avoidance\n"
                "  python -m act.front_end --download vit_2023          # Vision Transformer\n"
                "  python -m act.front_end --list-downloads             # Show what's downloaded\n"
            )
        else:
            raise RuntimeError(
                "No dataset-model pairs found! Please download datasets first.\n\n"
                "Examples:\n"
                "  python -m act.front_end --download MNIST              # Downloads MNIST + all models\n"
                "  python -m act.front_end --download CIFAR10            # Downloads CIFAR10 + all models\n"
                "  python -m act.front_end --list                        # Show all available datasets\n"
                "  python -m act.front_end --list-downloads              # Show what's already downloaded\n"
            )
    
    print(f"[Pipeline] Created specs using {creator.upper()} spec creator")
    print(f"           Found {len(spec_results)} dataset-model pair(s)")
    
    total_samples = sum(images.shape[0] for _, _, _, images, _, _ in spec_results)
    total_spec_pairs = sum(len(spec_pairs) for _, _, _, _, _, spec_pairs in spec_results)
    specs_per_sample = total_spec_pairs // total_samples if total_samples else 0
    
    wrapped_models, reports = synthesize_models_grouped(spec_results)
    
    import gc
    del spec_results
    gc.collect()
    
    # Validate synthesis results
    if not wrapped_models:
        raise RuntimeError(
            "Failed to synthesize any wrapped models! "
            "Spec results were loaded but model synthesis failed. "
            "Check spec_results format and synthesize_models_grouped() logic."
        )
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SYNTHESIS COMPLETE")
    print(f"{'='*80}")
    print(f"  • Wrapped models: {len(wrapped_models)}")
    print(f"  • Unique dataset-model pairs: {len(set((r.data_source, r.model_name) for r in reports.values()))}")
    
    # Print detailed breakdown (using pre-calculated stats)
    if total_samples > 0 and total_spec_pairs > 0:
        print(f"\n📊 Breakdown:")
        print(f"  • Input samples: {total_samples}")
        print(f"  • Spec pairs per sample: {specs_per_sample}")
        print(f"    (= 2 input kinds × 4 epsilons × 3 output specs)")
        print(f"    (= BOX, LINF_BALL × 0.01,0.03,0.05,0.1 × MARGIN_ROBUST(m=0.0,0.5), TOP1_ROBUST)")
        print(f"  • Total spec pairs: {total_spec_pairs}")
        print(f"  • Calculation: {total_samples} samples × {specs_per_sample} specs/sample = {total_spec_pairs} wrapped models")
    
    return wrapped_models


# =============================================================================
# Model Synthesis (Batch-Native)
# =============================================================================

def _get_spec_batch_size(spec: Union[InputSpec, OutputSpec]) -> int:
    """
    Safely extract batch size from a spec.
    
    Returns:
        Batch size (B dimension), or 1 if cannot be determined.
    """
    if isinstance(spec, InputSpec):
        tensor = spec._get_tensor()
        return tensor.shape[0] if tensor is not None and tensor.dim() > 0 else 1
    elif isinstance(spec, OutputSpec):
        if spec.y_true is not None:
            if isinstance(spec.y_true, torch.Tensor):
                return spec.y_true.shape[0] if spec.y_true.dim() > 0 else 1
            return 1  # scalar label
        if spec.c is not None and isinstance(spec.c, torch.Tensor):
            return spec.c.shape[0] if spec.c.dim() > 0 else 1
        if spec.lb is not None and isinstance(spec.lb, torch.Tensor):
            return spec.lb.shape[0] if spec.lb.dim() > 0 else 1
    return 1


def _validate_and_cap_batch(
    input_spec: InputSpec,
    output_spec: OutputSpec,
    max_batch_size: Optional[int] = None,
) -> Tuple[InputSpec, OutputSpec, int]:
    """
    Validate input/output specs have consistent batch sizes and optionally cap.
    
    Args:
        input_spec: Input specification
        output_spec: Output specification  
        max_batch_size: Maximum batch size (None = no cap, use spec's batch size)
        
    Returns:
        (input_spec, output_spec, batch_size) - specs may be sliced if capped
        
    Raises:
        ValueError: If input/output batch sizes are inconsistent
    """
    in_batch = _get_spec_batch_size(input_spec)
    out_batch = _get_spec_batch_size(output_spec)
    
    if in_batch != out_batch:
        raise ValueError(
            f"Batch size mismatch: InputSpec has B={in_batch}, "
            f"OutputSpec has B={out_batch}. They must match."
        )
    
    actual_batch = in_batch
    
    # Cap to max_batch_size if specified
    if max_batch_size is not None and actual_batch > max_batch_size:
        input_spec = _slice_input_spec(input_spec, max_batch_size)
        output_spec = _slice_output_spec(output_spec, max_batch_size)
        actual_batch = max_batch_size
    
    return input_spec, output_spec, actual_batch


def _slice_input_spec(spec: InputSpec, max_batch: int) -> InputSpec:
    """Slice InputSpec to max_batch samples."""
    if spec.kind == InKind.BOX:
        return InputSpec(kind=InKind.BOX, lb=spec.lb[:max_batch], ub=spec.ub[:max_batch])
    elif spec.kind == InKind.LINF_BALL:
        center = spec.center[:max_batch]
        eps = spec.eps[:max_batch] if isinstance(spec.eps, torch.Tensor) and spec.eps.dim() > 0 else spec.eps
        return InputSpec(kind=InKind.LINF_BALL, center=center, eps=eps)
    elif spec.kind == InKind.LIN_POLY:
        return InputSpec(kind=InKind.LIN_POLY, A=spec.A[:max_batch], b=spec.b[:max_batch])
    return spec


def _slice_output_spec(spec: OutputSpec, max_batch: int) -> OutputSpec:
    """Slice OutputSpec to max_batch samples."""
    y_true = spec.y_true
    if isinstance(y_true, torch.Tensor) and y_true.dim() > 0:
        y_true = y_true[:max_batch]
    
    margin = spec.margin
    if isinstance(margin, torch.Tensor) and margin.dim() > 0:
        margin = margin[:max_batch]
    
    c = spec.c
    if isinstance(c, torch.Tensor) and c.dim() > 0:
        c = c[:max_batch]
    
    d = spec.d
    if isinstance(d, torch.Tensor) and d.dim() > 0:
        d = d[:max_batch]
    
    lb = spec.lb
    if isinstance(lb, torch.Tensor) and lb.dim() > 0:
        lb = lb[:max_batch]
    
    ub = spec.ub
    if isinstance(ub, torch.Tensor) and ub.dim() > 0:
        ub = ub[:max_batch]
    
    return OutputSpec(
        kind=spec.kind, y_true=y_true, margin=margin,
        c=c, d=d, lb=lb, ub=ub, meta=spec.meta
    )


def _wrap_model(
    pytorch_model: nn.Module,
    input_spec: InputSpec,
    output_spec: OutputSpec,
    data_source: str = "unknown",
    model_name: str = "unknown",
    max_batch_size: Optional[int] = None,
) -> Tuple[VerifiableModel, WrapReport]:
    """
    Core helper: wrap a model with specs and create report.
    
    This is the single source of truth for VerifiableModel creation.
    
    Args:
        pytorch_model: The neural network to wrap
        input_spec: Input specification (batched)
        output_spec: Output specification (batched)
        data_source: Dataset name for reporting
        model_name: Model name for reporting
        max_batch_size: Maximum batch size (None = use spec's batch size)
        
    Returns:
        (VerifiableModel, WrapReport)
        
    Raises:
        ValueError: If input/output batch sizes are inconsistent
    """
    # Validate and optionally cap batch size
    input_spec, output_spec, batch_size = _validate_and_cap_batch(
        input_spec, output_spec, max_batch_size
    )
    
    wrapped = VerifiableModel(
        InputSpecLayer(input_spec),
        pytorch_model,
        OutputSpecLayer(output_spec),
    )
    
    # Get labels for unique count
    labels = output_spec.y_true
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor([labels])
    
    return wrapped, WrapReport(
        input_shape=tuple(input_spec._get_tensor().shape),
        in_spec_kind=input_spec.kind,
        out_spec_kind=output_spec.kind,
        data_source=data_source,
        model_name=model_name,
        batch_size=batch_size,
        unique_labels=len(labels.unique()),
    )


def _make_input_spec(
    images: torch.Tensor,
    eps: Union[float, torch.Tensor],
    input_kind: str,
) -> InputSpec:
    """Create InputSpec from images and epsilon."""
    if input_kind == InKind.LINF_BALL:
        eps_scalar = eps if not isinstance(eps, torch.Tensor) else eps[0].item()
        return InputSpec(kind=InKind.LINF_BALL, center=images, eps=eps_scalar)
    elif input_kind == InKind.BOX:
        if isinstance(eps, torch.Tensor):
            eps_exp = eps.view(-1, *([1] * (images.dim() - 1)))
        else:
            eps_exp = eps
        return InputSpec(kind=InKind.BOX, lb=(images - eps_exp).clamp(0, 1), ub=(images + eps_exp).clamp(0, 1))
    else:
        raise ValueError(f"Unsupported input_kind: {input_kind}")


def synthesize_model(
    pytorch_model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: Union[float, torch.Tensor] = 0.1,
    input_kind: str = InKind.LINF_BALL,
    output_kind: str = OutKind.TOP1_ROBUST,
    batch_size: Optional[int] = None,
    data_source: str = "unknown",
    model_name: str = "unknown",
) -> Tuple[VerifiableModel, WrapReport]:
    """
    Synthesize VerifiableModel from batched tensors.
    
    All inputs are batch-native: images[B, C, H, W], labels[B].
    
    Args:
        pytorch_model: The neural network to wrap
        images: Input images tensor [B, C, H, W]
        labels: Ground truth labels [B]
        eps: Perturbation epsilon (scalar or [B] tensor)
        input_kind: Input spec type (LINF_BALL, BOX)
        output_kind: Output spec type (TOP1_ROBUST, MARGIN_ROBUST, etc.)
        batch_size: Expected batch size (validates images.shape[0] if provided)
        data_source: Dataset name for reporting
        model_name: Model name for reporting
        
    Returns:
        (VerifiableModel, WrapReport)
    """
    B = images.shape[0]
    if batch_size is not None and batch_size != B:
        raise ValueError(f"batch_size={batch_size} but images has {B} samples")
    
    input_spec = _make_input_spec(images, eps, input_kind)
    output_spec = OutputSpec(kind=output_kind, y_true=labels)
    
    return _wrap_model(pytorch_model, input_spec, output_spec, data_source, model_name)


def synthesize_models_grouped(
    spec_results: List[Tuple[str, str, nn.Module, torch.Tensor, torch.Tensor, List[Tuple[InputSpec, OutputSpec]]]],
    max_batch_size: Optional[int] = None,
) -> Tuple[Dict[str, VerifiableModel], Dict[str, WrapReport]]:
    """
    Synthesize models from spec creator results.
    
    BATCH-NATIVE: Spec creators now produce batched specs directly (B>1).
    Each spec pair is already batched - just wrap directly.
    
    Args:
        spec_results: List of (data_source, model_name, model, images[B], labels[B], spec_pairs)
                      where each spec_pair has InputSpec[B] and OutputSpec[B]
        max_batch_size: Optional cap on batch size (slices specs if larger)
        
    Returns:
        (models, reports) dicts keyed by model_id
    """
    models: Dict[str, VerifiableModel] = {}
    reports: Dict[str, WrapReport] = {}
    
    print(f"\n[Synthesis] Processing {len(spec_results)} spec result(s)...")
    
    for data_source, model_name, pytorch_model, _images, _labels, spec_pairs in spec_results:
        if not spec_pairs:
            continue
        
        for idx, (input_spec, output_spec) in enumerate(spec_pairs):
            wrapped, report = _wrap_model(
                pytorch_model, input_spec, output_spec,
                data_source, model_name, max_batch_size
            )
            
            model_id = f"m:{model_name}|x:{data_source}|in:{input_spec.kind}|out:{output_spec.kind}|i:{idx}"
            models[model_id] = wrapped
            reports[model_id] = report
        
        print(f"  {data_source} + {model_name}: {len(spec_pairs)} specs -> {len(spec_pairs)} models (B={report.batch_size})")
    
    print(f"[Synthesis] Done: {len(models)} models")
    return models, reports


if __name__ == "__main__":
    from act.util.model_inference import model_inference
    from act.util.device_manager import initialize_device
    
    # Initialize device/dtype before synthesis (models typically use float32)
    initialize_device(device='cuda', dtype='float32')
    
    # Step 1: Synthesize all wrapped models using new spec creators
    wrapped_models = run_synthesis_pipeline()
    
    # Step 2: Test all models with inference (input data extracted from wrapped models)
    successful_models = model_inference(wrapped_models)
    
    print(f"\n✅ Successfully inferred {len(successful_models)} out of {len(wrapped_models)} models")
    print(f"\n🎯 NEW SPEC CREATOR INTEGRATION: COMPLETE ✅")
