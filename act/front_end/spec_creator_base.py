"""
Base class for specification creators with shape validation.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Callable
import yaml
import torch
from pathlib import Path

from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.util.path_config import get_spec_config_path, get_default_spec_config_path


class BaseSpecCreator(ABC):
    """
    Abstract base class for creating InputSpec/OutputSpec pairs.
    
    Provides shared configuration, validation, and spec generation utilities
    for different spec sources (TorchVision datasets, VNNLIB files, etc.)
    """
    
    def __init__(
        self,
        config_name: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize spec creator with configuration.
        
        Args:
            config_name: Named config from configs/specs/ (e.g., 'torchvision_classification')
            config_dict: Runtime configuration overrides
        """
        self.config = self._load_config(config_name)
        if config_dict:
            self.config.update(config_dict)
    
    def _load_config(self, config_name: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if config_name:
                config_path = get_spec_config_path(config_name)
            else:
                config_path = get_default_spec_config_path()
            
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️  Could not load config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Fallback default configuration"""
        return {
            'input_spec_types': ['BOX', 'LINF_BALL'],
            'output_spec_types': ['MARGIN_ROBUST', 'TOP1_ROBUST'],
            'perturbation': {'epsilon_values': [0.01, 0.03], 'norm': 'inf'},
            'combination_strategy': 'balanced'
        }
    
    # ==================== SHAPE VALIDATION ==================== #
    
    def _get_model_io_shapes(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor
    ) -> Tuple[torch.Size, torch.Size]:
        """
        Get model input and output shapes by running inference.
        
        Args:
            model: PyTorch model
            sample_input: Sample input tensor
        
        Returns:
            (input_shape, output_shape) tuple
        """
        model.eval()
        with torch.no_grad():
            # Handle batch dimension
            if sample_input.dim() == 3:  # Single image without batch
                test_input = sample_input.unsqueeze(0)
            else:
                test_input = sample_input
            
            output = model(test_input)
        
        return sample_input.shape, output.shape
    
    def _validate_input_spec_shape(
        self,
        spec: InputSpec,
        expected_shape: torch.Size
    ) -> Tuple[bool, str]:
        """
        Validate InputSpec shape matches expected dimensions.
        
        Returns:
            (is_valid, error_message) tuple
        """
        if spec.kind == InKind.BOX:
            if spec.lb.shape != expected_shape:
                return False, f"BOX lb shape {spec.lb.shape} != expected {expected_shape}"
            if spec.ub.shape != expected_shape:
                return False, f"BOX ub shape {spec.ub.shape} != expected {expected_shape}"
        
        elif spec.kind == InKind.LINF_BALL:
            if spec.center.shape != expected_shape:
                return False, f"LINF_BALL center shape {spec.center.shape} != expected {expected_shape}"
        
        elif spec.kind == InKind.LIN_POLY:
            flat_size = expected_shape.numel()
            if spec.A.shape[1] != flat_size:
                return False, f"LIN_POLY A columns {spec.A.shape[1]} != flattened input {flat_size}"
        
        return True, ""
    
    def _validate_output_spec_shape(
        self,
        spec: OutputSpec,
        num_classes: int
    ) -> Tuple[bool, str]:
        """
        Validate OutputSpec is compatible with model output.
        
        Returns:
            (is_valid, error_message) tuple
        """
        if spec.kind in [OutKind.MARGIN_ROBUST, OutKind.TOP1_ROBUST]:
            if not (0 <= spec.y_true < num_classes):
                return False, f"Class label {spec.y_true} out of range [0, {num_classes})"
        
        elif spec.kind == OutKind.LINEAR_LE:
            if spec.c.numel() != num_classes:
                return False, f"LINEAR_LE coeff size {spec.c.numel()} != num_classes {num_classes}"
        
        elif spec.kind == OutKind.RANGE:
            # Range specs are always valid (can be per-output or global)
            pass
        
        return True, ""
    
    def validate_spec_pair_with_model(
        self,
        input_spec: InputSpec,
        output_spec: OutputSpec,
        model: torch.nn.Module,
        sample_input: torch.Tensor
    ) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation: spec shapes match model I/O.
        
        Returns:
            (is_valid, error_messages) tuple
        """
        errors = []
        
        # Get model I/O shapes
        try:
            input_shape, output_shape = self._get_model_io_shapes(model, sample_input)
            # Assume last dimension is classes for classification
            if len(output_shape) > 1:
                num_classes = output_shape[-1]
            else:
                num_classes = output_shape[0]
        except Exception as e:
            errors.append(f"Failed to infer model shapes: {e}")
            return False, errors
        
        # Validate input spec
        input_valid, input_err = self._validate_input_spec_shape(input_spec, input_shape)
        if not input_valid:
            errors.append(f"Input spec validation failed: {input_err}")
        
        # Validate output spec
        output_valid, output_err = self._validate_output_spec_shape(output_spec, num_classes)
        if not output_valid:
            errors.append(f"Output spec validation failed: {output_err}")
        
        return len(errors) == 0, errors
    
    # ==================== SPEC COMBINATION ==================== #
    
    def _create_spec_combinations(
        self,
        input_specs: List[InputSpec],
        output_specs: List[OutputSpec]
    ) -> List[Tuple[InputSpec, OutputSpec]]:
        """Create spec combinations based on strategy"""
        strategy = self.config.get('combination_strategy', 'balanced')
        
        if strategy == 'full':
            # Cartesian product
            return [(i, o) for i in input_specs for o in output_specs]
        elif strategy == 'minimal':
            # One-to-one pairing (truncate to min length)
            min_len = min(len(input_specs), len(output_specs))
            return list(zip(input_specs[:min_len], output_specs[:min_len]))
        else:  # 'balanced'
            # Balanced pairing (cycle shorter list)
            if not input_specs or not output_specs:
                return []
            pairs = []
            longer = input_specs if len(input_specs) >= len(output_specs) else output_specs
            shorter = output_specs if len(input_specs) >= len(output_specs) else input_specs
            is_input_longer = len(input_specs) >= len(output_specs)
            
            for i, item in enumerate(longer):
                paired_item = shorter[i % len(shorter)]
                if is_input_longer:
                    pairs.append((item, paired_item))
                else:
                    pairs.append((paired_item, item))
            return pairs
    
    @abstractmethod
    def create_specs_for_data_model_pairs(
        self,
        max_samples: Optional[int] = None,
        filter_fn: Optional[Callable] = None,
        validate_shapes: bool = True
    ) -> List[Tuple[str, str, torch.nn.Module, List[torch.Tensor], List[Tuple[InputSpec, OutputSpec]]]]:
        """
        Create specs for data-model pairs (must be implemented by subclasses).
        
        Args:
            max_samples: Maximum number of samples/instances to process
            filter_fn: Optional filter function (source, model) -> bool
            validate_shapes: Whether to validate spec shapes with model
        
        Returns:
            List of (data_source, model_name, pytorch_model, input_tensors, spec_pairs) tuples
            - data_source: Dataset/category identifier
            - model_name: Model identifier
            - pytorch_model: PyTorch nn.Module
            - input_tensors: List of input tensors
            - spec_pairs: List of (InputSpec, OutputSpec) tuples
        """
        pass
