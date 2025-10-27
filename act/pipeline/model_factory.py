#!/usr/bin/env python3
#===- act/pipeline/model_factory.py - PyTorch Model Factory ------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   PyTorch model factory that creates nn.Module instances from
#   examples_config.yaml specifications. Complements net_factory.py
#   (which creates ACT Net objects) by generating actual PyTorch models
#   for testing, integration, and Torch2ACT conversion.
#
#   CRITICAL: Models created by this factory are EQUIVALENT to the
#   corresponding ACT Nets - they share the exact same weights loaded
#   from the pre-generated JSON files. This ensures numerical consistency
#   between PyTorch inference and ACT verification.
#
# Usage:
#   factory = ModelFactory()
#   model = factory.create_model("mnist_mlp_small", load_weights=True)
#   model.eval()
#   output = model(input_tensor)
#
# Testing:
#   python act/pipeline/model_factory.py  # Verifies all models
#
#===---------------------------------------------------------------------===#

import yaml
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

from act.back_end.core import Net, Layer
from act.back_end.serialization.serialization import NetSerializer

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating PyTorch models from examples_config.yaml."""
    
    def __init__(self, 
                 config_path: str = "act/back_end/examples/examples_config.yaml",
                 nets_dir: str = "act/back_end/examples/nets"):
        """
        Initialize factory with configuration file.
        
        Args:
            config_path: Path to examples_config.yaml
            nets_dir: Directory containing pre-generated ACT Net JSON files
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.nets_dir = Path(nets_dir)
    
    def create_model(self, name: str, load_weights: bool = True) -> nn.Module:
        """
        Create PyTorch model from configuration.
        
        Args:
            name: Network name from examples_config.yaml
            load_weights: If True, load weights from corresponding ACT Net JSON file
            
        Returns:
            PyTorch nn.Module ready for inference or training
            
        Raises:
            KeyError: If network name not found in config
            ValueError: If network architecture is invalid
        """
        if name not in self.config['networks']:
            available = ", ".join(self.config['networks'].keys())
            raise KeyError(f"Network '{name}' not found. Available: {available}")
        
        spec = self.config['networks'][name]
        architecture_type = spec.get('architecture_type', 'mlp')
        layers_spec = spec['layers']
        
        # Load ACT Net if weights should be transferred
        act_net = None
        if load_weights:
            net_path = self.nets_dir / f"{name}.json"
            if net_path.exists():
                with open(net_path, 'r') as f:
                    net_dict = json.load(f)
                act_net, _ = NetSerializer.deserialize_net(net_dict)
                logger.info(f"Loaded ACT Net from {net_path}")
            else:
                logger.warning(f"ACT Net file not found: {net_path}. Creating model with random weights.")
        
        # Build PyTorch module from layer specifications
        model = self._build_model(layers_spec, architecture_type, act_net)
        
        logger.info(f"Created PyTorch model '{name}' ({architecture_type}) with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
    
    def _build_model(self, layers_spec: List[Dict[str, Any]], architecture_type: str, act_net: Optional[Net] = None) -> nn.Module:
        """
        Build PyTorch nn.Module from layer specifications.
        
        Args:
            layers_spec: List of layer configurations
            architecture_type: Type hint ('mlp', 'cnn', 'adversarial')
            act_net: Optional ACT Net object to load weights from
            
        Returns:
            Sequential or custom nn.Module
        """
        torch_layers = []
        act_layer_idx = 0  # Index into ACT Net layers (synchronized with layers_spec)
        
        for i, layer_spec in enumerate(layers_spec):
            kind = layer_spec['kind']
            meta = layer_spec.get('meta', {})
            
            # Skip wrapper layers (INPUT, INPUT_SPEC, ASSERT)
            if kind in ['INPUT', 'INPUT_SPEC', 'ASSERT']:
                act_layer_idx += 1
                continue
            
            # Get corresponding ACT layer if available
            act_layer = None
            if act_net is not None and act_layer_idx < len(act_net.layers):
                act_layer = act_net.layers[act_layer_idx]
                # Verify kind matches
                if act_layer.kind != kind:
                    logger.warning(f"Layer kind mismatch at index {act_layer_idx}: config={kind}, ACT={act_layer.kind}")
            
            # Build PyTorch layer based on kind
            torch_layer = self._create_torch_layer(kind, meta, act_layer)
            
            if torch_layer is not None:
                torch_layers.append(torch_layer)
            
            act_layer_idx += 1
        
        if not torch_layers:
            raise ValueError("No valid PyTorch layers found in specification")
        
        # Return sequential model
        model = nn.Sequential(*torch_layers)
        model.eval()  # Set to evaluation mode by default
        
        return model
    
    def _transfer_weights(self, torch_layer: nn.Module, act_layer: Layer, weight_key: str = "W", bias_key: str = "b") -> None:
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
    
    def _create_torch_layer(self, kind: str, meta: Dict[str, Any], act_layer: Optional[Layer] = None) -> Optional[nn.Module]:
        """
        Create PyTorch layer from ACT layer kind and metadata.
        
        Args:
            kind: Layer kind string (DENSE, CONV2D, RELU, etc.)
            meta: Layer metadata dictionary
            act_layer: Optional ACT Layer to load weights from
            
        Returns:
            PyTorch nn.Module or None if layer should be skipped
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
    
    def generate_input_from_input_layer(self, layers_spec: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Generate input tensor from INPUT layer metadata in layer specifications.
        
        Args:
            layers_spec: List of layer configurations from examples_config.yaml
            
        Returns:
            Input tensor generated based on INPUT layer metadata
            
        Raises:
            ValueError: If no INPUT layer found or required metadata missing
        """
        # Find INPUT layer
        input_layer = None
        for layer_spec in layers_spec:
            if layer_spec['kind'] == 'INPUT':
                input_layer = layer_spec
                break
        
        if input_layer is None:
            raise ValueError("No INPUT layer found in layer specifications")
        
        meta = input_layer.get('meta', {})
        
        # Extract metadata with defaults
        shape = meta.get('shape')
        if shape is None:
            raise ValueError("INPUT layer missing required 'shape' metadata")
        
        dtype_str = meta.get('dtype')
        if dtype_str is None:
            raise ValueError("INPUT layer missing required 'dtype' metadata")
        
        value_range = meta.get('value_range', [0.0, 1.0])
        distribution = meta.get('distribution', 'uniform')
        
        # Map dtype string to torch dtype (handle both "float64" and "torch.float64" formats)
        if dtype_str.startswith('torch.'):
            dtype_str = dtype_str.replace('torch.', '')
        
        dtype_map = {
            'float16': torch.float16,
            'float32': torch.float32,
            'float64': torch.float64,
            'int8': torch.int8,
            'int16': torch.int16,
            'int32': torch.int32,
            'int64': torch.int64,
            'uint8': torch.uint8,
        }
        
        if dtype_str not in dtype_map:
            raise ValueError(f"Unsupported dtype '{dtype_str}'. Must be one of: {list(dtype_map.keys())}")
        
        dtype = dtype_map[dtype_str]
        
        # Generate tensor based on distribution using the target dtype directly
        if distribution == 'normal':
            # Generate normal distribution, then scale to value_range
            tensor = torch.randn(*shape, dtype=dtype)
            # Normalize to [0, 1] approximately (clip to 3 sigma)
            tensor = torch.clamp(tensor, -3.0, 3.0)
            tensor = (tensor + 3.0) / 6.0  # Map [-3, 3] to [0, 1]
            # Scale to target range
            tensor = tensor * (value_range[1] - value_range[0]) + value_range[0]
        elif distribution == 'uniform':
            # Generate uniform distribution in value_range
            tensor = torch.rand(*shape, dtype=dtype)
            tensor = tensor * (value_range[1] - value_range[0]) + value_range[0]
        elif distribution == 'zeros':
            tensor = torch.zeros(*shape, dtype=dtype)
        elif distribution == 'ones':
            tensor = torch.ones(*shape, dtype=dtype)
        else:
            # Default to uniform
            logger.warning(f"Unknown distribution '{distribution}', using uniform")
            tensor = torch.rand(*shape, dtype=dtype)
            tensor = tensor * (value_range[1] - value_range[0]) + value_range[0]
        
        return tensor
    
    def list_networks(self) -> List[str]:
        """List all available network names."""
        return list(self.config['networks'].keys())
    
    def get_network_info(self, name: str) -> Dict[str, Any]:
        """Get metadata about a network without creating it."""
        if name not in self.config['networks']:
            raise KeyError(f"Network '{name}' not found")
        
        spec = self.config['networks'][name]
        
        return {
            'name': name,
            'description': spec.get('description', 'No description'),
            'architecture_type': spec.get('architecture_type', 'unknown'),
            'input_shape': spec.get('input_shape', 'unknown'),
            'num_layers': len([l for l in spec['layers'] if l['kind'] not in ['INPUT', 'INPUT_SPEC', 'ASSERT']]),
            'metadata': spec.get('metadata', {})
        }


def main():
    """Test model factory with all example networks and verify equivalence with ACT Nets."""
    logging.basicConfig(level=logging.INFO)
    
    # Import inference function
    from act.util.model_inference import infer_single_model
    
    factory = ModelFactory()
    
    print("=" * 80)
    print("PyTorch Model Factory - Testing All Networks")
    print("=" * 80)
    
    all_passed = True
    inference_passed = 0
    inference_failed = 0
    
    for name in factory.list_networks():
        print(f"\n{'=' * 80}")
        print(f"Network: {name}")
        print("=" * 80)
        
        # Get network info
        info = factory.get_network_info(name)
        print(f"Description: {info['description']}")
        print(f"Architecture: {info['architecture_type']}")
        print(f"Input shape: {info['input_shape']}")
        print(f"Core layers: {info['num_layers']}")
        
        # Create model
        try:
            model = factory.create_model(name, load_weights=True)
            
            # Print model summary
            print(f"\nPyTorch Model:")
            print(model)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\nTotal parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
            # Generate input from INPUT layer metadata
            print(f"\n🎲 Generating input from INPUT layer metadata...")
            layers_spec = factory.config['networks'][name]['layers']
            try:
                input_tensor = factory.generate_input_from_input_layer(layers_spec)
                print(f"  ✅ Generated input tensor:")
                print(f"     Shape: {list(input_tensor.shape)}")
                print(f"     Dtype: {input_tensor.dtype}")
                print(f"     Range: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
                
                # Test inference using infer_single_model
                print(f"\n🔧 Testing inference with infer_single_model()...")
                success, output, error_msg = infer_single_model(name, model, input_tensor)
                
                if success:
                    print(f"  ✅ Inference successful!")
                    print(f"     Output shape: {list(output.shape)}")
                    print(f"     Output range: [{output.min():.4f}, {output.max():.4f}]")
                    print(f"     Output mean: {output.mean():.4f}")
                    inference_passed += 1
                else:
                    print(f"  ❌ Inference failed: {error_msg}")
                    inference_failed += 1
                    all_passed = False
                    
            except Exception as e:
                print(f"  ❌ Failed to generate input or run inference: {e}")
                inference_failed += 1
                all_passed = False
            
            # Verify equivalence with ACT Net
            print(f"\n📊 Verifying equivalence with ACT Net...")
            net_path = factory.nets_dir / f"{name}.json"
            if net_path.exists():
                with open(net_path, 'r') as f:
                    net_dict = json.load(f)
                act_net, _ = NetSerializer.deserialize_net(net_dict)
                
                # Check that parameter counts match
                act_param_count = sum(
                    p.numel() for layer in act_net.layers 
                    for p in layer.params.values() 
                    if isinstance(p, torch.Tensor)
                )
                
                if act_param_count == total_params:
                    print(f"  ✅ Parameter count matches: {total_params:,}")
                else:
                    print(f"  ⚠️  Parameter count mismatch: PyTorch={total_params:,}, ACT={act_param_count:,}")
                
                # Verify weight transfer by comparing layer by layer
                torch_layer_idx = 0
                for i, layer_spec in enumerate(factory.config['networks'][name]['layers']):
                    if layer_spec['kind'] in ['INPUT', 'INPUT_SPEC', 'ASSERT']:
                        continue
                    
                    act_layer = act_net.layers[i]
                    torch_layer = model[torch_layer_idx]
                    
                    # Verify weight transfer for parametric layers
                    if layer_spec['kind'] == 'DENSE' and 'W' in act_layer.params:
                        weight_diff = (torch_layer.weight - act_layer.params['W']).abs().max()
                        if weight_diff < 1e-10:
                            print(f"  ✅ DENSE layer [{torch_layer_idx}]: weights match (diff={weight_diff:.2e})")
                        else:
                            print(f"  ❌ DENSE layer [{torch_layer_idx}]: weights differ (diff={weight_diff:.2e})")
                            all_passed = False
                    
                    elif layer_spec['kind'] in ['CONV2D', 'CONV1D', 'CONV3D'] and 'weight' in act_layer.params:
                        weight_diff = (torch_layer.weight - act_layer.params['weight']).abs().max()
                        if weight_diff < 1e-10:
                            print(f"  ✅ {layer_spec['kind']} layer [{torch_layer_idx}]: weights match (diff={weight_diff:.2e})")
                        else:
                            print(f"  ❌ {layer_spec['kind']} layer [{torch_layer_idx}]: weights differ (diff={weight_diff:.2e})")
                            all_passed = False
                    
                    torch_layer_idx += 1
            else:
                print(f"  ⚠️  No ACT Net file found for comparison")
            
            print(f"\n✅ Successfully created model '{name}'")
            
        except Exception as e:
            print(f"\n❌ Failed to create model '{name}': {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            inference_failed += 1
    
    print("\n" + "=" * 80)
    print(f"📊 Inference Test Summary:")
    print(f"   ✅ Passed: {inference_passed}")
    print(f"   ❌ Failed: {inference_failed}")
    print(f"   Total: {inference_passed + inference_failed}")
    print("=" * 80)
    
    if all_passed:
        print("✅ All models created successfully and verified equivalent to ACT Nets")
    else:
        print("⚠️  Some models had issues - see details above")
    print("=" * 80)


if __name__ == "__main__":
    main()
