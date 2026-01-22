#===- act/front_end/batched_loader.py - Batched Data Loading Utils ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Thin wrapper around existing loaders to collect samples into batches.
#   Reuses torchvision_loader and vnnlib_loader - minimal code.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from typing import Optional, Union, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn as nn

from act.front_end.specs import InKind, OutKind, BatchedInputSpec, BatchedOutputSpec


@dataclass
class BatchedSpecLoader:
    """Holds batched data: images [B,...], labels [B], eps, and optional model."""
    images: torch.Tensor
    labels: torch.Tensor
    eps: Union[float, torch.Tensor] = 0.0
    model: Optional[nn.Module] = None
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def batch_size(self) -> int:
        return self.images.shape[0]
    
    def __len__(self) -> int:
        return self.batch_size
    
    def __repr__(self) -> str:
        eps_str = f"{self.eps:.4f}" if isinstance(self.eps, float) else f"[B]"
        return f"BatchedSpecLoader(src={self.source}, B={self.batch_size}, eps={eps_str})"
    
    # =========================================================================
    # From TorchVision - use dataloader directly
    # =========================================================================
    
    @classmethod
    def from_torchvision(
        cls,
        dataset_name: str,
        num_samples: int,
        eps: float = 0.0,
        model_name: Optional[str] = None,
        split: str = "test",
    ) -> 'BatchedSpecLoader':
        """
        Load B samples from TorchVision via existing loader.
        Uses batch_size=num_samples to get all data in one batch.
        """
        from act.front_end.torchvision_loader import load_dataset_model_pair
        
        result = load_dataset_model_pair(
            dataset_name=dataset_name,
            model_name=model_name,
            split=split,
            batch_size=num_samples,  # Get all at once
            shuffle=False,
        )
        
        # Get first (and only needed) batch from dataloader
        images, labels = next(iter(result['dataloader']))
        
        return cls(
            images=images,
            labels=labels,
            eps=eps,
            model=result.get('model'),
            source=f"torchvision:{dataset_name}",
            metadata={'split': split, 'model_name': model_name},
        )
    
    # =========================================================================
    # From VNNLib - load model ONCE, parse specs in loop
    # =========================================================================
    
    @classmethod
    def from_vnnlib(
        cls,
        category: str,
        num_samples: int,
        onnx_model: Optional[str] = None,
        eps: Optional[float] = None,
    ) -> 'BatchedSpecLoader':
        """
        Load B samples from VNNLib category.
        Optimized: converts ONNX model only once, then parses VNNLIB specs.
        """
        from act.front_end.vnnlib_loader import (
            list_downloaded_pairs, download_vnnlib_category,
            convert_onnx_to_pytorch, parse_vnnlib_to_tensors, get_onnx_input_shape
        )
        from act.front_end.vnnlib_loader.vnnlib_parser import extract_label_from_vnnlib
        
        download_vnnlib_category(category)
        
        pairs = [p for p in list_downloaded_pairs() if p['category'] == category]
        if not pairs:
            raise ValueError(f"No instances for category '{category}'")
        
        # Model consistency check
        unique_models = sorted(set(Path(p['onnx_model']).name for p in pairs))
        if len(unique_models) > 1:
            if onnx_model is None:
                raise ValueError(f"Category '{category}' has {len(unique_models)} models: {unique_models}. Specify onnx_model=.")
            pairs = [p for p in pairs if Path(p['onnx_model']).name == Path(onnx_model).name]
            if not pairs:
                raise ValueError(f"Model '{onnx_model}' not found in {unique_models}")
        
        # Check if requested samples exceeds available
        available = len(pairs)
        if num_samples > available:
            print(f"Warning: Requested {num_samples} samples but only {available} available for "
                  f"'{onnx_model or category}'. Loading {available} samples.")
        
        pairs = pairs[:num_samples]
        
        # Load model ONCE
        onnx_path = Path(pairs[0]['paths']['onnx'])
        model = convert_onnx_to_pytorch(onnx_path, simplify=True)
        model.eval()
        input_shape = get_onnx_input_shape(onnx_path)
        
        # Parse VNNLIB specs (no model conversion in loop)
        images_list, labels_list = [], []
        for p in pairs:
            try:
                vnnlib_path = Path(p['paths']['vnnlib'])
                tensor, meta = parse_vnnlib_to_tensors(vnnlib_path, input_shape)
                images_list.append(tensor.squeeze(0))
                labels_list.append(extract_label_from_vnnlib(vnnlib_path) or 0)
            except Exception as e:
                print(f"Warning: skipping {p['vnnlib_spec']}: {e}")
        
        if not images_list:
            raise RuntimeError(f"Failed to load any samples from '{category}'")
        
        images = torch.stack(images_list)
        labels = torch.tensor(labels_list, dtype=torch.long)
        
        # Infer eps from first spec if not provided
        if eps is None:
            _, meta = parse_vnnlib_to_tensors(Path(pairs[0]['paths']['vnnlib']), input_shape)
            if 'input_lb' in meta and 'input_ub' in meta:
                lb, ub = meta['input_lb'], meta['input_ub']
                eps = ((ub - lb) / 2).flatten().max().item()
            elif 'input_bounds' in meta and isinstance(meta['input_bounds'], dict):
                # input_bounds is dict {pixel_idx: (lb, ub)}
                perturbations = [(ub - lb) / 2 for lb, ub in meta['input_bounds'].values()]
                eps = max(perturbations) if perturbations else 0.0
        
        return cls(
            images=images, labels=labels, eps=eps or 0.0, model=model,
            source=f"vnnlib:{category}",
            metadata={'category': category, 'onnx_model': pairs[0]['onnx_model']},
        )
    
    @staticmethod
    def list_vnnlib_models(category: str) -> List[str]:
        """List ONNX models in a VNNLib category."""
        from act.front_end.vnnlib_loader import list_downloaded_pairs, download_vnnlib_category
        download_vnnlib_category(category)
        pairs = [p for p in list_downloaded_pairs() if p['category'] == category]
        return sorted(set(Path(p['onnx_model']).name for p in pairs))
    
    @staticmethod
    def count_vnnlib_instances(category: str, onnx_model: Optional[str] = None) -> Dict[str, int]:
        """
        Count available VNNLib instances per model in a category.
        
        Args:
            category: VNNLib category name
            onnx_model: Optional specific model to count
        
        Returns:
            Dict mapping model name to instance count
            
        Example:
            >>> BatchedSpecLoader.count_vnnlib_instances('cifar100_2024')
            {'CIFAR100_resnet_large.onnx': 12, 'CIFAR100_resnet_medium.onnx': 10}
        """
        from act.front_end.vnnlib_loader import list_downloaded_pairs, download_vnnlib_category
        from collections import Counter
        
        download_vnnlib_category(category)
        pairs = [p for p in list_downloaded_pairs() if p['category'] == category]
        
        if onnx_model:
            pairs = [p for p in pairs if Path(p['onnx_model']).name == Path(onnx_model).name]
        
        models = [Path(p['onnx_model']).name for p in pairs]
        return dict(Counter(models))
    
    # =========================================================================
    # From tensors (direct)
    # =========================================================================
    
    @classmethod
    def from_tensors(
        cls,
        images: torch.Tensor,
        labels: torch.Tensor,
        eps: Union[float, torch.Tensor] = 0.0,
        model: Optional[nn.Module] = None,
    ) -> 'BatchedSpecLoader':
        """Create directly from tensors."""
        return cls(images=images, labels=labels, eps=eps, model=model, source="tensors")
    
    # =========================================================================
    # Get specs
    # =========================================================================
    
    def get_batched_specs(
        self,
        input_kind: str = InKind.LINF_BALL,
        output_kind: str = OutKind.TOP1_ROBUST,
    ) -> Tuple[BatchedInputSpec, BatchedOutputSpec]:
        """Create BatchedInputSpec and BatchedOutputSpec."""
        if input_kind == InKind.LINF_BALL:
            input_spec = BatchedInputSpec(kind=InKind.LINF_BALL, center=self.images, eps=self.eps)
        elif input_kind == InKind.BOX:
            eps = self.eps
            if isinstance(eps, torch.Tensor):
                eps = eps.view(-1, *([1] * (self.images.dim() - 1)))
            input_spec = BatchedInputSpec(
                kind=InKind.BOX,
                lb=(self.images - eps).clamp(0, 1),
                ub=(self.images + eps).clamp(0, 1),
            )
        else:
            raise ValueError(f"Unsupported input_kind: {input_kind}")
        
        output_spec = BatchedOutputSpec(kind=output_kind, y_true=self.labels)
        return input_spec, output_spec
    
    def get_batched_model(self):
        """Create VerifiableModel for batched inference from loaded data and model."""
        from act.front_end.verifiable_model import VerifiableModel, InputSpecLayer, OutputSpecLayer
        from act.front_end.specs import InputSpec, OutputSpec
        
        if self.model is None:
            raise ValueError("No model. Use from_vnnlib() or from_torchvision(model_name=...)")
        
        batched_in, batched_out = self.get_batched_specs()
        
        input_spec = InputSpec(
            kind=batched_in.kind,
            lb=batched_in.lb, ub=batched_in.ub,
            center=batched_in.center, 
            eps=batched_in.eps if not isinstance(batched_in.eps, torch.Tensor) else batched_in.eps[0].item() if batched_in.eps is not None else None,
            A=batched_in.A, b=batched_in.b
        )
        output_spec = OutputSpec(
            kind=batched_out.kind,
            y_true=batched_out.y_true,
            margin=batched_out.margin if not isinstance(batched_out.margin, torch.Tensor) else batched_out.margin[0].item(),
        )
        
        return VerifiableModel(
            InputSpecLayer(input_spec),
            self.model,
            OutputSpecLayer(output_spec),
        )


# =========================================================================
# Convenience function
# =========================================================================

def load_batched(
    source: str,
    num_samples: int,
    eps: float = 0.0,
    **kwargs,
) -> Tuple[BatchedInputSpec, BatchedOutputSpec]:
    """
    Universal batched loader.
    
    Args:
        source: 'mnist', 'cifar10', 'vnnlib:<category>', or 'vnnlib:<category>:<model>'
        num_samples: Number of samples (B)
        eps: Perturbation epsilon
    
    Examples:
        >>> load_batched('mnist', 100, eps=0.1)
        >>> load_batched('vnnlib:mnist_fc', 50)
        >>> load_batched('vnnlib:cifar100_2024:CIFAR100_resnet_medium.onnx', 50)
    """
    if source.startswith('vnnlib:'):
        parts = source.split(':')
        category = parts[1]
        onnx_model = parts[2] if len(parts) > 2 else kwargs.pop('onnx_model', None)
        loader = BatchedSpecLoader.from_vnnlib(category, num_samples, onnx_model=onnx_model, eps=eps or None)
    else:
        # Treat as TorchVision dataset (case-insensitive lookup handled by loader)
        loader = BatchedSpecLoader.from_torchvision(source, num_samples, eps=eps, **kwargs)
    
    return loader.get_batched_specs()
