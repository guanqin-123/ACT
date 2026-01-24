#===- act/front_end/spec_loader.py - Unified Specification Loader -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations
from typing import Optional, Union, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn as nn

from act.front_end.specs import InKind, OutKind, InputSpec, OutputSpec


@dataclass
class SpecLoader:
    """
    Unified entry point for loading data, models, and creating specs.
    
    This is THE primary API for ACT front-end. All data loading flows through here.
    
    Quick Start:
        >>> loader = SpecLoader.from_torchvision("MNIST", 32, eps=0.1)
        >>> images, labels, model = loader.data  # Quick access
        >>> wrapped = loader.synthesize()         # Get VerifiableModel
        
    For More Control:
        >>> input_spec, output_spec = loader.get_specs()
        >>> wrapped = loader.get_model()
        
    VNNLib (no eps needed - bounds from file):
        >>> loader = SpecLoader.from_vnnlib("mnist_fc", 32)
        >>> wrapped = loader.synthesize()  # Uses lb/ub from VNNLIB directly
    """
    images: torch.Tensor
    labels: torch.Tensor
    eps: Union[float, torch.Tensor] = 0.0
    model: Optional[nn.Module] = None
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    # For VNNLib: store bounds directly (no eps needed)
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    
    @property
    def batch_size(self) -> int:
        return self.images.shape[0]
    
    @property
    def B(self) -> int:
        return self.batch_size
    
    @property
    def shape(self) -> torch.Size:
        return self.images.shape
    
    @property
    def num_classes(self) -> int:
        return self.metadata.get('num_classes', int(self.labels.max().item()) + 1)
    
    @property
    def data(self) -> Tuple[torch.Tensor, torch.Tensor, nn.Module]:
        """Quick access: (images, labels, model) tuple."""
        if self.model is None:
            raise ValueError("No model loaded. Use model_name parameter.")
        return self.images, self.labels, self.model
    
    def __len__(self) -> int:
        return self.batch_size
    
    def __repr__(self) -> str:
        model_str = type(self.model).__name__ if self.model else "None"
        if self.lb is not None and self.ub is not None:
            return f"SpecLoader({self.source}, B={self.B}, bounds=lb/ub, model={model_str})"
        eps_str = f"{self.eps:.4f}" if isinstance(self.eps, float) else "[B]"
        return f"SpecLoader({self.source}, B={self.B}, eps={eps_str}, model={model_str})"
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def from_torchvision(
        cls,
        dataset_name: str,
        num_samples: int = 32,
        eps: float = 0.0,
        model_name: Optional[str] = None,
        split: str = "test",
        verbose: bool = False,
    ) -> 'SpecLoader':
        """
        Load from TorchVision dataset.
        
        Args:
            dataset_name: Dataset name (e.g., "MNIST", "CIFAR10")
            num_samples: Batch size to load
            eps: Perturbation epsilon for specs
            model_name: Model to load (e.g., "resnet18"). None = dataset default.
            split: Dataset split ("train" or "test")
            verbose: Print loading progress
            
        Returns:
            SpecLoader with images[B,C,H,W], labels[B], and model
            
        Example:
            >>> loader = SpecLoader.from_torchvision("MNIST", 32, eps=0.1)
            >>> loader = SpecLoader.from_torchvision("CIFAR10", 64, model_name="resnet18")
        """
        from act.front_end.torchvision_loader import load_dataset_model_pair
        
        result = load_dataset_model_pair(
            dataset_name=dataset_name,
            model_name=model_name,
            split=split,
            batch_size=num_samples,
            shuffle=False,
            verbose=verbose,
        )
        
        images, labels = next(iter(result['dataloader']))
        
        return cls(
            images=images,
            labels=labels,
            eps=eps,
            model=result.get('model'),
            source=f"torchvision:{dataset_name}",
            metadata={
                'split': split,
                'model_name': model_name or result.get('metadata', {}).get('model'),
                'num_classes': result.get('num_classes', 10),
                'dataset_name': dataset_name,
            },
        )
    
    @classmethod
    def from_vnnlib(
        cls,
        category: str,
        num_samples: int = 32,
        onnx_model: Optional[str] = None,
        eps: Optional[float] = None,
        verbose: bool = False,
    ) -> 'SpecLoader':
        """
        Load from VNNLib category.
        
        Args:
            category: VNNLib category name
            num_samples: Number of instances to load
            onnx_model: Specific ONNX model (required if category has multiple)
            eps: Override epsilon (None = infer from VNNLIB)
            verbose: Print loading progress
            
        Returns:
            SpecLoader with images, labels, and converted PyTorch model
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
        
        unique_models = sorted(set(Path(p['onnx_model']).name for p in pairs))
        if len(unique_models) > 1:
            if onnx_model is None:
                raise ValueError(f"Category '{category}' has {len(unique_models)} models: {unique_models}. Specify onnx_model=.")
            pairs = [p for p in pairs if Path(p['onnx_model']).name == Path(onnx_model).name]
            if not pairs:
                raise ValueError(f"Model '{onnx_model}' not found in {unique_models}")
        
        available = len(pairs)
        if num_samples > available:
            if verbose:
                print(f"Note: Requested {num_samples} but only {available} available. Loading {available}.")
            num_samples = available
        
        pairs = pairs[:num_samples]
        
        onnx_path = Path(pairs[0]['paths']['onnx'])
        model = convert_onnx_to_pytorch(onnx_path, simplify=True)
        model.eval()
        input_shape = get_onnx_input_shape(onnx_path)
        
        images_list, labels_list, lb_list, ub_list = [], [], [], []
        for p in pairs:
            try:
                vnnlib_path = Path(p['paths']['vnnlib'])
                tensor, meta = parse_vnnlib_to_tensors(vnnlib_path, input_shape)
                images_list.append(tensor.squeeze(0))
                labels_list.append(extract_label_from_vnnlib(vnnlib_path) or 0)
                
                # Extract lb/ub bounds from VNNLIB
                if 'input_bounds' in meta and isinstance(meta['input_bounds'], dict):
                    num_inputs = len(meta['input_bounds'])
                    lb_vals = [meta['input_bounds'][i][0] for i in range(num_inputs)]
                    ub_vals = [meta['input_bounds'][i][1] for i in range(num_inputs)]
                    lb_tensor = torch.tensor(lb_vals).view(input_shape[1:])  # Remove batch dim
                    ub_tensor = torch.tensor(ub_vals).view(input_shape[1:])
                    lb_list.append(lb_tensor)
                    ub_list.append(ub_tensor)
            except Exception as e:
                if verbose:
                    print(f"Warning: skipping {p['vnnlib_spec']}: {e}")
        
        if not images_list:
            raise RuntimeError(f"Failed to load any samples from '{category}'")
        
        images = torch.stack(images_list)
        labels = torch.tensor(labels_list, dtype=torch.long)
        
        # Stack bounds if available
        lb = torch.stack(lb_list) if lb_list else None
        ub = torch.stack(ub_list) if ub_list else None
        
        return cls(
            images=images, 
            labels=labels, 
            eps=0.0,  # Not used for VNNLib
            model=model,
            source=f"vnnlib:{category}",
            metadata={'category': category, 'onnx_model': pairs[0]['onnx_model']},
            lb=lb,
            ub=ub,
        )
    
    @classmethod
    def from_tensors(
        cls,
        images: torch.Tensor,
        labels: torch.Tensor,
        eps: Union[float, torch.Tensor] = 0.0,
        model: Optional[nn.Module] = None,
    ) -> 'SpecLoader':
        """Create directly from tensors."""
        return cls(images=images, labels=labels, eps=eps, model=model, source="tensors")
    
    # =========================================================================
    # Spec Creation
    # =========================================================================
    
    @property
    def has_bounds(self) -> bool:
        """True if lb/ub bounds are available (VNNLib source)."""
        return self.lb is not None and self.ub is not None
    
    def get_specs(
        self,
        input_kind: Optional[str] = None,
        output_kind: str = OutKind.TOP1_ROBUST,
        eps: Optional[float] = None,
    ) -> Tuple[InputSpec, OutputSpec]:
        """
        Create InputSpec and OutputSpec.
        
        Args:
            input_kind: InKind.LINF_BALL or InKind.BOX (None = auto-detect)
            output_kind: OutKind.TOP1_ROBUST, MARGIN_ROBUST, etc.
            eps: Override epsilon (None = use loader's eps, ignored for VNNLib)
            
        Note:
            For VNNLib sources (has_bounds=True), lb/ub from file are used directly.
            The eps parameter is ignored.
        """
        # VNNLib: use lb/ub directly (ignore eps)
        if self.has_bounds:
            input_spec = InputSpec(kind=InKind.BOX, lb=self.lb, ub=self.ub)
            output_spec = OutputSpec(kind=output_kind, y_true=self.labels)
            return input_spec, output_spec
        
        # TorchVision: use eps to create specs
        input_kind = input_kind or InKind.LINF_BALL
        eps = eps if eps is not None else self.eps
        
        if input_kind == InKind.LINF_BALL:
            input_spec = InputSpec(kind=InKind.LINF_BALL, center=self.images, eps=eps)
        elif input_kind == InKind.BOX:
            if isinstance(eps, torch.Tensor):
                eps_exp = eps.view(-1, *([1] * (self.images.dim() - 1)))
            else:
                eps_exp = eps
            input_spec = InputSpec(
                kind=InKind.BOX,
                lb=(self.images - eps_exp).clamp(0, 1),
                ub=(self.images + eps_exp).clamp(0, 1),
            )
        else:
            raise ValueError(f"Unsupported input_kind: {input_kind}")
        
        output_spec = OutputSpec(kind=output_kind, y_true=self.labels)
        return input_spec, output_spec
    
    # =========================================================================
    # Model Wrapping
    # =========================================================================
    
    def get_model(
        self,
        input_kind: str = InKind.LINF_BALL,
        output_kind: str = OutKind.TOP1_ROBUST,
        eps: Optional[float] = None,
    ) -> 'VerifiableModel':
        """
        Create VerifiableModel with specs embedded.
        
        Returns:
            VerifiableModel ready for verification
        """
        from act.front_end.verifiable_model import VerifiableModel, InputSpecLayer, OutputSpecLayer
        
        if self.model is None:
            raise ValueError("No model. Use from_vnnlib() or from_torchvision(model_name=...)")
        
        input_spec, output_spec = self.get_specs(input_kind, output_kind, eps)
        
        return VerifiableModel(
            InputSpecLayer(input_spec),
            self.model,
            OutputSpecLayer(output_spec),
        )
    
    def synthesize(
        self,
        eps: Optional[float] = None,
        input_kind: Optional[str] = None,
        output_kind: str = OutKind.TOP1_ROBUST,
    ) -> Tuple['VerifiableModel', 'WrapReport']:
        """
        Synthesize VerifiableModel with full report.
        
        This is the recommended way to create a model for verification.
        
        Args:
            eps: Override epsilon (None = use loader's eps, ignored for VNNLib)
            input_kind: Input spec type (None = auto: BOX for VNNLib, LINF_BALL for TorchVision)
            output_kind: Output spec type
            
        Returns:
            (VerifiableModel, WrapReport) tuple
            
        Example:
            >>> # TorchVision (needs eps)
            >>> loader = SpecLoader.from_torchvision("MNIST", 32, eps=0.1)
            >>> wrapped, report = loader.synthesize()
            >>> 
            >>> # VNNLib (no eps needed - uses lb/ub from file)
            >>> loader = SpecLoader.from_vnnlib("mnist_fc", 32)
            >>> wrapped, report = loader.synthesize()
        """
        from act.front_end.model_synthesis import _wrap_model
        
        if self.model is None:
            raise ValueError("No model. Use from_torchvision(model_name=...) or from_vnnlib()")
        
        # Get specs (handles VNNLib lb/ub automatically)
        input_spec, output_spec = self.get_specs(
            input_kind=input_kind,
            output_kind=output_kind,
            eps=eps,
        )
        
        return _wrap_model(
            self.model,
            input_spec,
            output_spec,
            self.source,
            self.metadata.get('model_name', 'unknown'),
        )
    
    # =========================================================================
    # Slicing & Batching
    # =========================================================================
    
    def slice(self, start: int = 0, end: Optional[int] = None) -> 'SpecLoader':
        """Return a new SpecLoader with sliced data."""
        end = end or self.batch_size
        return SpecLoader(
            images=self.images[start:end],
            labels=self.labels[start:end],
            eps=self.eps[start:end] if isinstance(self.eps, torch.Tensor) else self.eps,
            model=self.model,
            source=self.source,
            metadata=self.metadata,
            lb=self.lb[start:end] if self.lb is not None else None,
            ub=self.ub[start:end] if self.ub is not None else None,
        )
    
    def __getitem__(self, idx: Union[int, slice]) -> 'SpecLoader':
        """Index or slice the loader."""
        if isinstance(idx, int):
            return SpecLoader(
                images=self.images[idx:idx+1],
                labels=self.labels[idx:idx+1],
                eps=self.eps[idx:idx+1] if isinstance(self.eps, torch.Tensor) else self.eps,
                model=self.model,
                source=self.source,
                metadata=self.metadata,
                lb=self.lb[idx:idx+1] if self.lb is not None else None,
                ub=self.ub[idx:idx+1] if self.ub is not None else None,
            )
        else:
            return SpecLoader(
                images=self.images[idx],
                labels=self.labels[idx],
                eps=self.eps[idx] if isinstance(self.eps, torch.Tensor) else self.eps,
                model=self.model,
                source=self.source,
                metadata=self.metadata,
                lb=self.lb[idx] if self.lb is not None else None,
                ub=self.ub[idx] if self.ub is not None else None,
            )
    
    # =========================================================================
    # Static Utilities
    # =========================================================================
    
    @staticmethod
    def list_torchvision() -> List[Dict[str, Any]]:
        """List all downloaded TorchVision dataset-model pairs."""
        from act.front_end.torchvision_loader import list_downloaded_pairs
        return list_downloaded_pairs()
    
    @staticmethod
    def list_vnnlib() -> List[Dict[str, Any]]:
        """List all downloaded VNNLib instances."""
        from act.front_end.vnnlib_loader import list_downloaded_pairs
        return list_downloaded_pairs()
    
    @staticmethod
    def list_vnnlib_models(category: str) -> List[str]:
        """List ONNX models in a VNNLib category."""
        from act.front_end.vnnlib_loader import list_downloaded_pairs, download_vnnlib_category
        download_vnnlib_category(category)
        pairs = [p for p in list_downloaded_pairs() if p['category'] == category]
        return sorted(set(Path(p['onnx_model']).name for p in pairs))
    
    @staticmethod
    def count_vnnlib_instances(category: str, onnx_model: Optional[str] = None) -> Dict[str, int]:
        """Count available VNNLib instances per model."""
        from act.front_end.vnnlib_loader import list_downloaded_pairs, download_vnnlib_category
        from collections import Counter
        
        download_vnnlib_category(category)
        pairs = [p for p in list_downloaded_pairs() if p['category'] == category]
        
        if onnx_model:
            pairs = [p for p in pairs if Path(p['onnx_model']).name == Path(onnx_model).name]
        
        models = [Path(p['onnx_model']).name for p in pairs]
        return dict(Counter(models))

