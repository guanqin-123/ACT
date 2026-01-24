#===- act/front_end/torchvision/create_specs.py - TorchVision Specs ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Create InputSpec and OutputSpec from TorchVision dataset-model pairs.
#   Sample-based spec generation for image classification models.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
import torch
import torch.nn as nn

from act.front_end.spec_creator_base import BaseSpecCreator
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.front_end.torchvision_loader.data_model_loader import (
    list_downloaded_pairs,
    load_dataset_model_pair
)

logger = logging.getLogger(__name__)


class TorchVisionSpecCreator(BaseSpecCreator):
    """
    Create verification specifications from TorchVision dataset-model pairs.
    
    Generates InputSpec and OutputSpec based on actual data samples:
    - Input specs: BOX or LINF_BALL perturbations around sample images
    - Output specs: Classification robustness properties (MARGIN_ROBUST)
    
    Example:
        >>> creator = TorchVisionSpecCreator(config_name="torchvision_classification")
        >>> results = creator.create_specs_for_data_model_pairs(
        ...     dataset_names=["MNIST"],
        ...     model_names=["simple_cnn"],
        ...     num_samples=10
        ... )
        >>> 
        >>> for data_source, model_name, pytorch_model, images, labels, spec_pairs in results:
        ...     print(f"{data_source} + {model_name}: {len(spec_pairs)} spec pairs, B={images.shape[0]}")
    """
    
    def __init__(
        self,
        config_name: Optional[str] = "torchvision_classification",
        config_dict: Optional[Dict] = None
    ):
        """
        Initialize TorchVision spec creator.
        
        Args:
            config_name: Name of YAML config file (without .yaml extension)
            config_dict: Direct config dict (overrides config_name if provided)
        """
        super().__init__(config_name, config_dict)
    
    def create_specs_for_data_model_pairs(
        self,
        dataset_names: Optional[List[str]] = None,
        model_names: Optional[List[str]] = None,
        num_samples: int = 10,
        start_index: int = 0,
        split: str = "test",
        validate_shapes: bool = True
    ) -> List[Tuple[str, str, nn.Module, torch.Tensor, torch.Tensor, List[Tuple[InputSpec, OutputSpec]]]]:
        """
        Create specs for TorchVision dataset-model pairs.
        
        BATCH-NATIVE: Returns batched tensors directly (no LabeledInputTensor wrapper).
        
        Args:
            dataset_names: List of dataset names (None = all downloaded)
            model_names: List of model names (None = all for each dataset)
            num_samples: Number of samples to generate specs for
            start_index: Starting index in dataset
            split: Dataset split ('train' or 'test')
            validate_shapes: Whether to validate specs against model
            
        Returns:
            List of tuples:
            - data_source: Dataset name (e.g., "MNIST")
            - model_name: Model name (e.g., "simple_cnn")
            - pytorch_model: torch.nn.Module
            - images: Batched image tensor [B, C, H, W]
            - labels: Batched label tensor [B]
            - spec_pairs: List of (InputSpec, OutputSpec) tuples
            
        Example:
            >>> creator = TorchVisionSpecCreator()
            >>> results = creator.create_specs_for_data_model_pairs(
            ...     dataset_names=["MNIST"],
            ...     num_samples=5
            ... )
        """
        logger.info(
            f"Creating TorchVision specs: datasets={dataset_names}, "
            f"models={model_names}, samples={num_samples}"
        )
        
        # Get all downloaded pairs
        all_pairs = list_downloaded_pairs()
        
        if not all_pairs:
            logger.warning("No downloaded dataset-model pairs found")
            return []
        
        # Filter by dataset names if specified
        if dataset_names is not None:
            dataset_names_lower = [name.lower() for name in dataset_names]
            all_pairs = [
                p for p in all_pairs 
                if p['dataset'].lower() in dataset_names_lower
            ]
        
        # Filter by model names if specified
        if model_names is not None:
            model_names_lower = [name.lower() for name in model_names]
            all_pairs = [
                p for p in all_pairs 
                if p['model'].lower() in model_names_lower
            ]
        
        if not all_pairs:
            logger.warning("No pairs match the specified filters")
            return []
        
        logger.info(f"Processing {len(all_pairs)} dataset-model pairs")
        
        results = []
        
        for pair_info in all_pairs:
            dataset_name = pair_info['dataset']
            model_name = pair_info['model']
            
            try:
                # Load pair with batch_size=num_samples for efficiency
                logger.info(f"Loading pair: {dataset_name} + {model_name}")
                pair_data = load_dataset_model_pair(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    split=split,
                    batch_size=num_samples,  # Load all samples in one batch
                    shuffle=False,
                    auto_download=False  # Already filtered to downloaded pairs
                )
                
                pytorch_model = pair_data['model']
                dataloader = pair_data['dataloader']
                
                # Generate specs for this pair
                result = self._create_specs_for_single_pair(
                    data_source=dataset_name,
                    model_name=model_name,
                    pytorch_model=pytorch_model,
                    dataloader=dataloader,
                    num_samples=num_samples,
                    start_index=start_index,
                    validate_shapes=validate_shapes
                )
                
                if result is not None:
                    results.append(result)
                
                # Memory optimization: Free dataset/dataloader after extracting input_tensors
                # pair_data contains the dataset (476 MB for MNIST) which is no longer needed
                import gc
                del pair_data, dataloader
                gc.collect()
                
            except Exception as e:
                logger.error(
                    f"Failed to create specs for {dataset_name} + {model_name}: {e}"
                )
        
        logger.info(f"Successfully created specs for {len(results)} pairs")
        return results
    
    def _create_specs_for_single_pair(
        self,
        data_source: str,
        model_name: str,
        pytorch_model: nn.Module,
        dataloader,
        num_samples: int,
        start_index: int,
        validate_shapes: bool
    ) -> Optional[Tuple[str, str, nn.Module, torch.Tensor, torch.Tensor, List[Tuple[InputSpec, OutputSpec]]]]:
        """
        Create specs for a single dataset-model pair.
        
        BATCH-NATIVE: Creates batched specs directly with B=num_samples.
        Each spec pair has InputSpec[B] and OutputSpec[B] where B=num_samples.
        
        Returns:
            Tuple of (data_source, model_name, pytorch_model, images[B,C,H,W], labels[B], spec_pairs)
            or None if failed
        """
        logger.info(f"Generating specs for {data_source} + {model_name}")
        
        # Use iterator to get one batch directly
        data_iter = iter(dataloader)
        
        # Skip to start_index if needed
        for _ in range(start_index):
            try:
                next(data_iter)
            except StopIteration:
                logger.warning(f"start_index {start_index} exceeds dataset size for {data_source}")
                return None
        
        # Get one batch of num_samples
        try:
            batched_images, batched_targets = next(data_iter)
        except StopIteration:
            logger.warning(f"No samples available for {data_source}")
            return None
        
        actual_samples = batched_images.shape[0]
        logger.info(f"Loaded {actual_samples} samples in one batch")
        
        # BATCH-NATIVE: Create batched specs directly from tensors
        spec_pairs = self._create_batched_specs_from_tensors(batched_images, batched_targets)
        
        logger.info(f"Generated {len(spec_pairs)} batched spec combinations (B={actual_samples})")
        
        if validate_shapes and spec_pairs:
            validated_pairs = self._validate_and_filter_specs(
                spec_pairs,
                pytorch_model,
                batched_images  # Full batch for shape validation (specs are batched)
            )
            
            if len(validated_pairs) < len(spec_pairs):
                logger.warning(
                    f"Filtered {len(spec_pairs) - len(validated_pairs)} invalid specs"
                )
            
            spec_pairs = validated_pairs
        
        if not spec_pairs:
            logger.warning(f"No valid specs generated for {data_source} + {model_name}")
            return None
        
        return (data_source, model_name, pytorch_model, batched_images, batched_targets, spec_pairs)
    
    def _create_batched_specs_from_tensors(
        self,
        batched_images: torch.Tensor,
        batched_labels: torch.Tensor,
    ) -> List[Tuple[InputSpec, OutputSpec]]:
        """
        Create batched specs directly from batched tensors.
        
        BATCH-NATIVE: Creates K specs with B=batched_images.shape[0].
        
        Args:
            batched_images: Batched image tensor [B, C, H, W]
            batched_labels: Batched label tensor [B]
            
        Returns:
            List of (InputSpec[B], OutputSpec[B])
        """
        # Get config
        epsilons = self.config.get('epsilons', [0.01, 0.03, 0.05])
        input_kinds = self.config.get('input_kinds', ['BOX', 'LINF_BALL'])
        output_kinds = self.config.get('output_kinds', ['MARGIN_ROBUST'])
        margins = self.config.get('margins', [0.0])
        
        spec_pairs = []
        
        for in_kind in input_kinds:
            for eps in epsilons:
                # Create batched InputSpec
                if in_kind == 'BOX':
                    input_spec = InputSpec(
                        kind=InKind.BOX,
                        lb=torch.clamp(batched_images - eps, 0.0, 1.0),
                        ub=torch.clamp(batched_images + eps, 0.0, 1.0),
                    )
                elif in_kind == 'LINF_BALL':
                    input_spec = InputSpec(
                        kind=InKind.LINF_BALL,
                        center=batched_images.clone(),
                        eps=eps,
                    )
                else:
                    continue
                
                for out_kind in output_kinds:
                    for margin in margins:
                        # Create batched OutputSpec
                        if out_kind == 'MARGIN_ROBUST':
                            output_spec = OutputSpec(
                                kind=OutKind.MARGIN_ROBUST,
                                y_true=batched_labels.clone(),
                                margin=margin,
                            )
                        elif out_kind == 'TOP1_ROBUST':
                            output_spec = OutputSpec(
                                kind=OutKind.TOP1_ROBUST,
                                y_true=batched_labels.clone(),
                            )
                        else:
                            continue
                        
                        spec_pairs.append((input_spec, output_spec))
        
        return spec_pairs
    
    def _validate_and_filter_specs(
        self,
        spec_pairs: List[Tuple[InputSpec, OutputSpec]],
        pytorch_model: nn.Module,
        sample_input: torch.Tensor
    ) -> List[Tuple[InputSpec, OutputSpec]]:
        """
        Validate spec pairs against model and filter invalid ones.
        
        Args:
            spec_pairs: List of (InputSpec, OutputSpec) tuples
            pytorch_model: PyTorch model to validate against
            sample_input: Sample input tensor for shape inference
            
        Returns:
            Filtered list of valid spec pairs
        """
        valid_pairs = []
        
        for input_spec, output_spec in spec_pairs:
            try:
                is_valid = self.validate_spec_pair_with_model(
                    input_spec,
                    output_spec,
                    pytorch_model,
                    sample_input
                )
                
                if is_valid:
                    valid_pairs.append((input_spec, output_spec))
                else:
                    logger.debug(
                        f"Spec pair validation failed: "
                        f"{input_spec.kind}, {output_spec.kind}"
                    )
            
            except Exception as e:
                logger.debug(f"Spec validation error: {e}")
        
        return valid_pairs
