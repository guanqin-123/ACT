#!/usr/bin/env python3
#===- act/pipeline/act2torch.py - ACT to Torch Converter ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   ACT → PyTorch converter. Converts an ACT Net (abstract constraint graph)
#   into a concrete PyTorch nn.Module for inference, testing, and integration.
#   
#   This is the inverse operation of torch2act.py, enabling bidirectional
#   conversion between verification representations and executable models.
#
# Usage:
#   from act.pipeline.act2torch import ACTToTorch
#   
#   # Convert ACT Net to PyTorch model
#   converter = ACTToTorch(act_net)
#   model = converter.run()
#   
#   # Use the model
#   model.eval()
#   output = model(input_tensor)
#
# Design:
#   - Mirrors TorchToACT class structure for symmetry
#   - Single-purpose: pure ACT→Torch conversion
#   - Transfers weights from ACT layer params to PyTorch parameters
#   - Skips wrapper layers (INPUT, INPUT_SPEC, ASSERT)
#   - Returns nn.Sequential model ready for inference
#
#===---------------------------------------------------------------------===#

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import logging

from act.back_end.core import Net, Layer

logger = logging.getLogger(__name__)


class ACTToTorch:
    """
    Convert ACT Net to PyTorch nn.Module.
    
    This class provides the inverse transformation of TorchToACT, enabling
    bidirectional conversion between verification representations (ACT) and
    executable models (PyTorch).
    
    Usage:
        converter = ACTToTorch(act_net)
        model = converter.run()  # Returns nn.Sequential
    
    Args:
        act_net: ACT Net object containing layers with architecture and weights
    
    Returns:
        PyTorch nn.Sequential model ready for inference
    """
    
    def __init__(self, act_net: Net):
        """
        Initialize converter with ACT Net.
        
        Args:
            act_net: ACT Net object (contains architecture + weights)
        
        Raises:
            TypeError: If act_net is not a Net instance
        """
        if not isinstance(act_net, Net):
            raise TypeError(f"ACTToTorch expects a Net object, got {type(act_net)}")
        self.act_net = act_net
    
    def run(self) -> nn.Module:
        """
        Convert ACT Net to PyTorch nn.Module.
        
        Iterates through ACT layers, creates corresponding PyTorch layers,
        transfers weights, and assembles into nn.Sequential model.
        
        Returns:
            Sequential or custom nn.Module ready for inference
        
        Raises:
            ValueError: If no valid PyTorch layers can be created
        """
        torch_layers = []
        
        for i, act_layer in enumerate(self.act_net.layers):
            kind = act_layer.kind
            meta = act_layer.meta
            
            # Skip wrapper layers (INPUT, INPUT_SPEC, ASSERT)
            if kind in ['INPUT', 'INPUT_SPEC', 'ASSERT']:
                continue
            
            # Build PyTorch layer from ACT layer (includes weight transfer)
            torch_layer = self._create_torch_layer(kind, meta, act_layer)
            
            if torch_layer is not None:
                torch_layers.append(torch_layer)
        
        if not torch_layers:
            raise ValueError("No valid PyTorch layers found in ACT Net")
        
        # Return sequential model
        model = nn.Sequential(*torch_layers)
        model.eval()  # Set to evaluation mode by default
        
        return model
    
    def _transfer_weights(self, torch_layer: nn.Module, act_layer: Layer, 
                         weight_key: str = "W", bias_key: str = "b") -> None:
        """
        Transfer weights and biases from ACT layer to PyTorch layer.
        
        Args:
            torch_layer: PyTorch layer with weight/bias parameters
            act_layer: ACT layer containing parameter tensors
            weight_key: Key for weight parameter in act_layer.params ("W" or "weight")
            bias_key: Key for bias parameter in act_layer.params ("b" or "bias")
        """
        with torch.no_grad():
            # Transfer weights
            if weight_key in act_layer.params:
                torch_layer.weight.copy_(act_layer.params[weight_key])
            
            # Transfer bias (or zero it if not present in ACT layer)
            if hasattr(torch_layer, 'bias') and torch_layer.bias is not None:
                if bias_key in act_layer.params:
                    torch_layer.bias.copy_(act_layer.params[bias_key])
                else:
                    torch_layer.bias.zero_()
    
    def _create_torch_layer(self, kind: str, meta: Dict[str, Any], 
                           act_layer: Optional[Layer] = None) -> Optional[nn.Module]:
        """
        Create PyTorch layer from ACT layer kind and metadata.
        
        Args:
            kind: Layer kind string (DENSE, CONV2D, RELU, etc.)
            meta: Layer metadata dictionary
            act_layer: Optional ACT Layer to load weights from
            
        Returns:
            PyTorch nn.Module or None if layer should be skipped
        
        Raises:
            ValueError: If required metadata is missing for layer type
        """
        # Dense/Linear layers
        if kind == "DENSE":
            in_features = meta.get("in_features")
            out_features = meta.get("out_features")
            bias_enabled = meta.get("bias_enabled", True)
            
            if in_features is None:
                raise ValueError("DENSE layer requires 'in_features' in meta")
            if out_features is None:
                raise ValueError("DENSE layer requires 'out_features' in meta")
            
            layer = nn.Linear(in_features, out_features, bias=bias_enabled)
            
            # Transfer weights and bias from ACT layer
            if act_layer is not None:
                self._transfer_weights(layer, act_layer, weight_key="W", bias_key="b")
            
            return layer
        
        # Convolutional layers
        elif kind == "CONV2D":
            in_channels = meta.get("in_channels")
            out_channels = meta.get("out_channels")
            kernel_size = meta.get("kernel_size", 3)
            stride = meta.get("stride", 1)
            padding = meta.get("padding", 0)
            dilation = meta.get("dilation", 1)
            groups = meta.get("groups", 1)
            
            if in_channels is None:
                raise ValueError("CONV2D layer requires 'in_channels' in meta")
            if out_channels is None:
                raise ValueError("CONV2D layer requires 'out_channels' in meta")
            
            layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups
            )
            
            # Transfer weights and bias from ACT layer
            if act_layer is not None:
                self._transfer_weights(layer, act_layer, weight_key="weight", bias_key="bias")
            
            return layer
        
        elif kind == "CONV1D":
            in_channels = meta.get("in_channels")
            out_channels = meta.get("out_channels")
            kernel_size = meta.get("kernel_size", 3)
            stride = meta.get("stride", 1)
            padding = meta.get("padding", 0)
            
            if in_channels is None:
                raise ValueError("CONV1D layer requires 'in_channels' in meta")
            if out_channels is None:
                raise ValueError("CONV1D layer requires 'out_channels' in meta")
            
            layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
            
            # Transfer weights and bias from ACT layer
            if act_layer is not None:
                self._transfer_weights(layer, act_layer, weight_key="weight", bias_key="bias")
            
            return layer
        
        elif kind == "CONV3D":
            in_channels = meta.get("in_channels")
            out_channels = meta.get("out_channels")
            kernel_size = meta.get("kernel_size", 3)
            stride = meta.get("stride", 1)
            padding = meta.get("padding", 0)
            
            if in_channels is None:
                raise ValueError("CONV3D layer requires 'in_channels' in meta")
            if out_channels is None:
                raise ValueError("CONV3D layer requires 'out_channels' in meta")
            
            layer = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
            
            # Transfer weights and bias from ACT layer
            if act_layer is not None:
                self._transfer_weights(layer, act_layer, weight_key="weight", bias_key="bias")
            
            return layer
        
        # Pooling layers
        elif kind == "MAXPOOL2D":
            kernel_size = meta.get("kernel_size")
            stride = meta.get("stride")
            padding = meta.get("padding", 0)
            
            if kernel_size is None:
                raise ValueError("MAXPOOL2D layer requires 'kernel_size' in meta")
            
            return nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        
        elif kind == "MAXPOOL1D":
            kernel_size = meta.get("kernel_size")
            stride = meta.get("stride")
            padding = meta.get("padding", 0)
            
            if kernel_size is None:
                raise ValueError("MAXPOOL1D layer requires 'kernel_size' in meta")
            
            return nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
        
        elif kind == "MAXPOOL3D":
            kernel_size = meta.get("kernel_size")
            stride = meta.get("stride")
            padding = meta.get("padding", 0)
            
            if kernel_size is None:
                raise ValueError("MAXPOOL3D layer requires 'kernel_size' in meta")
            
            return nn.MaxPool3d(kernel_size, stride=stride, padding=padding)
        
        elif kind == "AVGPOOL2D":
            kernel_size = meta.get("kernel_size")
            stride = meta.get("stride")
            padding = meta.get("padding", 0)
            
            if kernel_size is None:
                raise ValueError("AVGPOOL2D layer requires 'kernel_size' in meta")
            
            return nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
        
        elif kind == "ADAPTIVEAVGPOOL2D":
            output_size = meta.get("output_size", 1)
            return nn.AdaptiveAvgPool2d(output_size)
        
        # Activation functions
        elif kind == "RELU":
            return nn.ReLU()
        
        elif kind == "LRELU":
            negative_slope = meta.get("negative_slope", 0.01)
            return nn.LeakyReLU(negative_slope)
        
        elif kind == "PRELU":
            return nn.PReLU()
        
        elif kind == "SIGMOID":
            return nn.Sigmoid()
        
        elif kind == "TANH":
            return nn.Tanh()
        
        elif kind == "SOFTPLUS":
            return nn.Softplus()
        
        elif kind == "SILU":
            return nn.SiLU()
        
        elif kind == "GELU":
            approximate = meta.get("approximate", "none")
            return nn.GELU(approximate=approximate)
        
        elif kind == "RELU6":
            return nn.ReLU6()
        
        elif kind == "HARDTANH":
            min_val = meta.get("min_val", -1.0)
            max_val = meta.get("max_val", 1.0)
            return nn.Hardtanh(min_val, max_val)
        
        elif kind == "HARDSIGMOID":
            return nn.Hardsigmoid()
        
        elif kind == "HARDSWISH":
            return nn.Hardswish()
        
        elif kind == "SOFTSIGN":
            return nn.Softsign()
        
        elif kind == "MISH":
            return nn.Mish()
        
        # Tensor operations
        elif kind == "FLATTEN":
            start_dim = meta.get("start_dim", 1)
            end_dim = meta.get("end_dim", -1)
            return nn.Flatten(start_dim, end_dim)
        
        elif kind == "DROPOUT":
            p = meta.get("p", 0.5)
            return nn.Dropout(p)
        
        elif kind == "BATCHNORM2D":
            num_features = meta.get("num_features")
            if num_features is None:
                raise ValueError("BATCHNORM2D requires 'num_features' in meta")
            return nn.BatchNorm2d(num_features)
        
        elif kind == "BATCHNORM1D":
            num_features = meta.get("num_features")
            if num_features is None:
                raise ValueError("BATCHNORM1D requires 'num_features' in meta")
            return nn.BatchNorm1d(num_features)
        
        elif kind == "LAYERNORM":
            normalized_shape = meta.get("normalized_shape")
            if normalized_shape is None:
                raise ValueError("LAYERNORM requires 'normalized_shape' in meta")
            return nn.LayerNorm(normalized_shape)
        
        # Embedding and sequence layers
        elif kind == "EMBEDDING":
            num_embeddings = meta.get("num_embeddings")
            embedding_dim = meta.get("embedding_dim")
            
            if num_embeddings is None:
                raise ValueError("EMBEDDING requires 'num_embeddings' in meta")
            if embedding_dim is None:
                raise ValueError("EMBEDDING requires 'embedding_dim' in meta")
            
            return nn.Embedding(num_embeddings, embedding_dim)
        
        elif kind == "RNN":
            input_size = meta.get("input_size")
            hidden_size = meta.get("hidden_size")
            num_layers = meta.get("num_layers", 1)
            bidirectional = meta.get("bidirectional", False)
            batch_first = meta.get("batch_first", False)
            
            if input_size is None:
                raise ValueError("RNN requires 'input_size' in meta")
            if hidden_size is None:
                raise ValueError("RNN requires 'hidden_size' in meta")
            
            return nn.RNN(input_size, hidden_size, num_layers, 
                         batch_first=batch_first, bidirectional=bidirectional)
        
        elif kind == "LSTM":
            input_size = meta.get("input_size")
            hidden_size = meta.get("hidden_size")
            num_layers = meta.get("num_layers", 1)
            bidirectional = meta.get("bidirectional", False)
            batch_first = meta.get("batch_first", False)
            
            if input_size is None:
                raise ValueError("LSTM requires 'input_size' in meta")
            if hidden_size is None:
                raise ValueError("LSTM requires 'hidden_size' in meta")
            
            return nn.LSTM(input_size, hidden_size, num_layers,
                          batch_first=batch_first, bidirectional=bidirectional)
        
        elif kind == "GRU":
            input_size = meta.get("input_size")
            hidden_size = meta.get("hidden_size")
            num_layers = meta.get("num_layers", 1)
            bidirectional = meta.get("bidirectional", False)
            batch_first = meta.get("batch_first", False)
            
            if input_size is None:
                raise ValueError("GRU requires 'input_size' in meta")
            if hidden_size is None:
                raise ValueError("GRU requires 'hidden_size' in meta")
            
            return nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=batch_first, bidirectional=bidirectional)
        
        elif kind == "SOFTMAX":
            axis = meta.get("axis", -1)
            return nn.Softmax(dim=axis)
        
        # Skip or warn about unsupported layers
        else:
            logger.warning(f"Unsupported layer kind '{kind}' - skipping")
            return None
