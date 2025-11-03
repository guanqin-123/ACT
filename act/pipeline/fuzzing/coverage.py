"""
Coverage tracking for ACTFuzzer.

Tracks neuron coverage (DeepXplore-style) during fuzzing to guide exploration.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from typing import Dict, Set
import torch
import torch.nn as nn


class CoverageTracker:
    """
    Track neuron coverage during fuzzing.
    
    Metrics:
    - Neuron coverage: % of neurons that have activated (output > threshold)
    - Total neurons: Count of all neurons in model
    - Covered neurons: Set of (layer_name, neuron_idx) tuples
    
    Example:
        >>> tracker = CoverageTracker(model)
        >>> activations = instrumentor.capture_activations(input_tensor)
        >>> coverage_delta = tracker.update(input_tensor, activations)
        >>> print(f"Coverage: {tracker.get_coverage():.2%}")
    """
    
    def __init__(self, model: nn.Module, threshold: float = 0.1):
        """
        Initialize coverage tracker.
        
        Args:
            model: Model to track coverage for
            threshold: Activation threshold (neuron is "active" if output > threshold)
        """
        self.model = model
        self.threshold = threshold
        
        # Track covered neurons as (layer_name, neuron_idx)
        self.covered_neurons: Set[tuple] = set()
        
        # Count total neurons
        self.total_neurons = self._count_neurons()
    
    def _count_neurons(self) -> int:
        """Count total neurons in model."""
        count = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                count += module.out_features
            elif isinstance(module, nn.Conv2d):
                # For Conv2d, count output channels
                count += module.out_channels
        
        return count
    
    def update(self, 
               input_tensor: torch.Tensor,
               activations: Dict[str, torch.Tensor]
              ) -> float:
        """
        Update coverage with new activations.
        
        Args:
            input_tensor: Input that was tested (unused, for future extensions)
            activations: Dict of layer activations from instrumentation
        
        Returns:
            Coverage delta (increase in coverage from 0.0 to 1.0)
        """
        old_count = len(self.covered_neurons)
        
        # Process activations
        for layer_name, activation in activations.items():
            # Check if this is a layer we track
            if 'relu' in layer_name.lower() or 'linear' in layer_name.lower() or 'conv' in layer_name.lower():
                # Find neurons that fired (activation > threshold)
                fired_mask = (activation.abs() > self.threshold)
                
                # Get indices of fired neurons
                # Handle different tensor shapes
                if fired_mask.dim() == 2:
                    # Linear layer: (batch, neurons)
                    fired_indices = fired_mask[0].nonzero(as_tuple=True)[0].tolist()
                elif fired_mask.dim() == 4:
                    # Conv layer: (batch, channels, height, width)
                    # Track by channel
                    fired_indices = fired_mask[0].any(dim=(1, 2)).nonzero(as_tuple=True)[0].tolist()
                else:
                    # Flatten and track
                    fired_indices = fired_mask.flatten().nonzero(as_tuple=True)[0].tolist()
                
                # Add to covered set
                for idx in fired_indices:
                    self.covered_neurons.add((layer_name, idx))
        
        # Compute coverage delta
        new_count = len(self.covered_neurons)
        delta = (new_count - old_count) / self.total_neurons if self.total_neurons > 0 else 0.0
        
        return delta
    
    def get_coverage(self) -> float:
        """
        Get current coverage percentage.
        
        Returns:
            Coverage from 0.0 to 1.0
        """
        if self.total_neurons == 0:
            return 0.0
        
        return len(self.covered_neurons) / self.total_neurons
    
    def get_stats(self) -> Dict[str, float]:
        """Get detailed coverage statistics."""
        coverage = self.get_coverage()
        
        # Count covered neurons per layer
        layer_coverage = {}
        for layer_name, _ in self.covered_neurons:
            layer_coverage[layer_name] = layer_coverage.get(layer_name, 0) + 1
        
        return {
            "coverage": coverage,
            "covered_neurons": len(self.covered_neurons),
            "total_neurons": self.total_neurons,
            "layers_with_coverage": len(layer_coverage),
            "avg_neurons_per_layer": (
                sum(layer_coverage.values()) / len(layer_coverage)
                if layer_coverage else 0
            ),
        }
    
    def reset(self):
        """Reset coverage tracking."""
        self.covered_neurons.clear()
