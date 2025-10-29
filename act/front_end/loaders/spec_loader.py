#===- act/front_end/loaders/spec_loader.py - Specification Loading -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Specification Loading for Front-End Integration. Clean torch tensor
#   specification generation and loading. Handles input/output specification
#   creation and VNNLIB constraint parsing using global device settings.
#
#===---------------------------------------------------------------------===#


from __future__ import annotations
import os
import torch
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.front_end.raw_processors.preprocessor_base import Preprocessor
from act.util.device_manager import get_current_settings
from act.util.path_config import get_project_root


class SpecLoader:
    """Clean torch tensor specification generation and loading with global device settings"""
    
    def __init__(self, preprocessor: Optional[Preprocessor] = None):
        """
        Initialize SpecLoader with global device settings
        
        Args:
            preprocessor: Optional preprocessor for model-space spec generation
        """
        self.preprocessor = preprocessor
        self.device, self.dtype = get_current_settings()
        
    def create_input_specs(self, samples: List[Any], 
                          spec_config: Dict[str, Any]) -> List[InputSpec]:
        """
        Generate input specifications for raw samples
        
        Args:
            samples: List of raw samples (torch.Tensors, numpy arrays, PIL Images, etc.)
            spec_config: Configuration dictionary with spec parameters
                - type: "linf_ball", "l2_ball", "box", "custom"
                - epsilon: perturbation radius (for ball specs)
                - bounds: custom bounds (for box specs)
                - norm_type: normalization before applying constraints
                
        Returns:
            List of InputSpec objects
        """
        spec_type = spec_config.get("type", "linf_ball")
        epsilon = spec_config.get("epsilon", 0.1)
        
        input_specs = []
        
        for i, sample in enumerate(samples):
            # Convert sample to torch tensor with global device settings
            if self.preprocessor:
                # Use preprocessor to convert to model space
                if isinstance(sample, torch.Tensor):
                    sample_tensor = sample.to(device=self.device, dtype=self.dtype)
                else:  # PIL Image or other format
                    sample_tensor = self.preprocessor.prepare_sample(sample)
                    sample_tensor = sample_tensor.to(device=self.device, dtype=self.dtype)
            else:
                # Raw sample processing - accept torch tensors or numpy arrays
                if isinstance(sample, torch.Tensor):
                    sample_tensor = sample.to(device=self.device, dtype=self.dtype)
                elif hasattr(sample, '__array__') or hasattr(sample, 'shape'):
                    # Handle numpy arrays or array-like objects
                    import numpy as np
                    if isinstance(sample, np.ndarray):
                        sample_tensor = torch.from_numpy(sample).to(device=self.device, dtype=self.dtype)
                    else:
                        # Convert other array-like objects to tensor
                        sample_tensor = torch.tensor(sample, device=self.device, dtype=self.dtype)
                else:
                    raise ValueError(f"Cannot process sample type {type(sample)} without preprocessor. Expected torch.Tensor or numpy.ndarray, got {type(sample)}")
                    
            # Generate specification based on type
            if spec_type == "linf_ball":
                input_spec = self._create_linf_ball_spec(sample_tensor, epsilon)
            elif spec_type == "l2_ball":
                # L2 ball represented as box constraints (approximation)
                radius = spec_config.get("radius", epsilon)
                input_spec = self._create_box_spec(sample_tensor, epsilon=radius)
            elif spec_type == "box":
                bounds = spec_config.get("bounds", None)
                input_spec = self._create_box_spec(sample_tensor, bounds, epsilon)
            elif spec_type == "custom":
                # Custom specification from provided bounds
                lb = spec_config.get("lower_bound")
                ub = spec_config.get("upper_bound")
                if lb is None or ub is None:
                    raise ValueError("Custom spec requires 'lower_bound' and 'upper_bound'")
                input_spec = InputSpec(kind=InKind.BOX, 
                                     lb=torch.tensor(lb, device=self.device, dtype=self.dtype), 
                                     ub=torch.tensor(ub, device=self.device, dtype=self.dtype))
            else:
                raise ValueError(f"Unsupported input spec type: {spec_type}")
                
            input_specs.append(input_spec)
            
        print(f"📋 Generated {len(input_specs)} input specs of type {spec_type}")
        return input_specs
        
    def _create_linf_ball_spec(self, center: torch.Tensor, epsilon: float) -> InputSpec:
        """Create L∞ ball specification around center point with global device settings"""
        return InputSpec(
            kind=InKind.LINF_BALL,
            center=center.clone().to(device=self.device, dtype=self.dtype),
            eps=epsilon
        )
        
    def _create_box_spec(self, center: torch.Tensor, 
                        bounds: Optional[Tuple[float, float]] = None,
                        epsilon: float = 0.1) -> InputSpec:
        """Create box specification around center point with global device settings"""
        if bounds:
            lb_val, ub_val = bounds
            lb = torch.full_like(center, lb_val, device=self.device, dtype=self.dtype)
            ub = torch.full_like(center, ub_val, device=self.device, dtype=self.dtype)
        else:
            # Create epsilon-box around center
            lb = (center - epsilon).to(device=self.device, dtype=self.dtype)
            ub = (center + epsilon).to(device=self.device, dtype=self.dtype)
            
        return InputSpec(kind=InKind.BOX, lb=lb, ub=ub)
        
    def create_output_specs(self, labels: List[int], 
                           spec_config: Dict[str, Any]) -> List[OutputSpec]:
        """
        Generate output specifications for labels
        
        Args:
            labels: List of ground truth labels
            spec_config: Configuration dictionary with spec parameters
                - output_type: "margin_robust", "top1_robust", "linear_le", "range"
                - margin: robustness margin (for margin_robust)
                - target_class: target class for adversarial specs
                
        Returns:
            List of OutputSpec objects
        """
        output_type = spec_config.get("output_type", "margin_robust")
        margin = spec_config.get("margin", 0.0)
        
        output_specs = []
        
        for label in labels:
            if output_type == "margin_robust":
                output_spec = OutputSpec(
                    kind=OutKind.MARGIN_ROBUST,
                    y_true=label,
                    margin=margin
                )
            elif output_type == "top1_robust":
                # Top-1 robustness is equivalent to margin robustness with margin=0
                output_spec = OutputSpec(
                    kind=OutKind.MARGIN_ROBUST,
                    y_true=label,
                    margin=0.0
                )
            elif output_type == "linear_le":
                # Linear constraint: coeffs^T * output <= bound
                coeffs = spec_config.get("coeffs")
                bound = spec_config.get("bound", 0.0)
                if coeffs is None:
                    raise ValueError("linear_le spec requires 'coeffs' parameter")
                output_spec = OutputSpec(
                    kind=OutKind.LINEAR_LE,
                    c=torch.tensor(coeffs, device=self.device, dtype=self.dtype),
                    d=bound
                )
            elif output_type == "range":
                # Output range constraints
                lower = spec_config.get("lower", -float('inf'))
                upper = spec_config.get("upper", float('inf'))
                output_spec = OutputSpec(
                    kind=OutKind.RANGE,
                    lb=torch.tensor([lower], device=self.device, dtype=self.dtype),
                    ub=torch.tensor([upper], device=self.device, dtype=self.dtype)
                )
            else:
                raise ValueError(f"Unsupported output spec type: {output_type}")
                
            output_specs.append(output_spec)
            
        print(f"📋 Generated {len(output_specs)} output specs of type {output_type}")
        return output_specs
        
    def discover_all_specs(self) -> Dict[str, List[str]]:
        """Comprehensively discover all specifications in the project"""
        specs = {
            "vnnlib": [],
            "json": [],
            "config": [],
            "other": []
        }
        
        # Search entire project for specification files
        project_root = Path(get_project_root())
        
        # Common specification directories
        spec_dirs = ["data", "configs", "specs", "properties","vnnlib"]
        
        for spec_dir in spec_dirs:
            spec_path = project_root / spec_dir
            if spec_path.exists():
                for spec_file in spec_path.rglob("*"):
                    if spec_file.is_file():
                        suffix = spec_file.suffix.lower()
                        if suffix == ".vnnlib":
                            specs["vnnlib"].append(str(spec_file))
                        elif suffix == ".json" and "spec" in spec_file.name.lower():
                            specs["json"].append(str(spec_file))
                        elif suffix in [".ini", ".yaml", ".yml", ".toml"]:
                            specs["config"].append(str(spec_file))
                        elif suffix in [".txt", ".prop", ".smt2"]:
                            specs["other"].append(str(spec_file))
                
        return specs
    
    def load_vnnlib_specs(self, vnnlib_path: str) -> List[Tuple[InputSpec, OutputSpec]]:
        """
        Load VNNLIB specification file and convert to InputSpec/OutputSpec pairs
        
        Parses SMT-LIB format VNNLIB files to extract:
        - Input variable bounds (X_*) 
        - Output constraints (Y_*) in various forms (DNF, linear, range)
        
        Returns:
            List of (InputSpec, OutputSpec) tuples
        """
        try:
            import re
            
            with open(vnnlib_path, 'r') as f:
                lines = f.readlines()
            
            print(f"📋 Loading VNNLIB {vnnlib_path}")
            
            # Remove comments and workaround for '<' vs '<='
            lines = [line.strip() for line in lines if not line.strip().startswith(';')]
            lines = [line.replace('< ', '<= ') if '< ' in line else line for line in lines]
            content = '\n'.join(lines)
            
            # ===== PARSE INPUT VARIABLES AND BOUNDS =====
            # Pattern: (declare-const X_i Real)
            input_vars = re.findall(r'\(declare-const\s+(X_\d+)\s+Real\)', content)
            
            # Extract bounds for each input variable
            input_bounds = {}
            for var in input_vars:
                # Find upper bound: (assert (<= X_i value))
                ub_match = re.search(rf'\(assert\s+\(\<=\s+{re.escape(var)}\s+(-?[\d.eE+-]+)\)\)', content)
                # Find lower bound: (assert (>= X_i value))
                lb_match = re.search(rf'\(assert\s+\(\>=\s+{re.escape(var)}\s+(-?[\d.eE+-]+)\)\)', content)
                
                if ub_match and lb_match:
                    input_bounds[var] = (float(lb_match.group(1)), float(ub_match.group(1)))
            
            if not input_bounds:
                print(f"⚠️  No input bounds found in VNNLIB")
                return []
            
            # Sort input variables by index and create tensors
            sorted_vars = sorted(input_bounds.keys(), key=lambda x: int(x.split('_')[1]))
            lb_values = [input_bounds[var][0] for var in sorted_vars]
            ub_values = [input_bounds[var][1] for var in sorted_vars]
            
            lb_tensor = torch.tensor(lb_values, device=self.device, dtype=self.dtype)
            ub_tensor = torch.tensor(ub_values, device=self.device, dtype=self.dtype)
            
            input_spec = InputSpec(kind=InKind.BOX, lb=lb_tensor, ub=ub_tensor)
            print(f"✅ Parsed {len(input_bounds)} input variables (X_0 to X_{len(input_bounds)-1})")
            
            # ===== PARSE OUTPUT CONSTRAINTS =====
            output_vars = re.findall(r'\(declare-const\s+(Y_\d+)\s+Real\)', content)
            num_outputs = len(output_vars)
            
            # Pattern 1: DNF robustness property - (assert (or (and (>= Y_i Y_j)) ...))
            # This represents: "at least one class i != j has higher score than true class j"
            dnf_pattern = r'\(assert\s+\(or\s+((?:\(and\s+\(\>=\s+Y_\d+\s+Y_\d+\)\)\s*)+)\)\)'
            dnf_match = re.search(dnf_pattern, content)
            
            if dnf_match:
                # Extract all comparisons of form (>= Y_i Y_j)
                comparisons = re.findall(r'\(\>=\s+(Y_\d+)\s+(Y_\d+)\)', dnf_match.group(0))
                
                if comparisons:
                    # Find the common Y_j that appears on the right side (true label)
                    right_side_vars = [comp[1] for comp in comparisons]
                    # The true label is the one that appears most frequently on right side
                    from collections import Counter
                    true_label_var = Counter(right_side_vars).most_common(1)[0][0]
                    true_label = int(true_label_var.split('_')[1])
                    
                    print(f"✅ Parsed DNF robustness property: adversarial for class {true_label}")
                    output_spec = OutputSpec(
                        kind=OutKind.MARGIN_ROBUST,
                        y_true=true_label,
                        margin=0.0
                    )
                    return [(input_spec, output_spec)]
            
            # Pattern 2: Simple linear constraint - (assert (<= (+ (* c0 Y_0) (* c1 Y_1) ...) bound))
            linear_pattern = r'\(assert\s+\(\<=\s+\(\+\s+((?:\(\*\s+[-\d.eE+]+\s+Y_\d+\)\s*)+)\)\s+([-\d.eE+]+)\)\)'
            linear_match = re.search(linear_pattern, content)
            
            if linear_match:
                # Extract coefficients
                coeffs = re.findall(r'\(\*\s+([-\d.eE+]+)\s+(Y_(\d+))\)', linear_match.group(1))
                bound = float(linear_match.group(2))
                
                # Build coefficient tensor
                c = torch.zeros(num_outputs, device=self.device, dtype=self.dtype)
                for coeff_val, y_var, y_idx in coeffs:
                    c[int(y_idx)] = float(coeff_val)
                
                print(f"✅ Parsed linear constraint: c^T * Y <= {bound}")
                output_spec = OutputSpec(kind=OutKind.LINEAR_LE, c=c, d=bound)
                return [(input_spec, output_spec)]
            
            # Pattern 3: Range constraint - (assert (and (<= Y_i ub) (>= Y_i lb)))
            range_pattern = r'\(assert\s+\(and\s+\(\<=\s+(Y_\d+)\s+([-\d.eE+]+)\)\s+\(\>=\s+(Y_\d+)\s+([-\d.eE+]+)\)\)\)'
            range_match = re.search(range_pattern, content)
            
            if range_match:
                ub = float(range_match.group(2))
                lb = float(range_match.group(4))
                print(f"✅ Parsed range constraint: {lb} <= Y <= {ub}")
                output_spec = OutputSpec(
                    kind=OutKind.RANGE,
                    lb=torch.tensor([lb], device=self.device, dtype=self.dtype),
                    ub=torch.tensor([ub], device=self.device, dtype=self.dtype)
                )
                return [(input_spec, output_spec)]
            
            # Default: assume margin robustness for class 0
            print("⚠️  No specific output constraint pattern matched, using default margin robustness")
            output_spec = OutputSpec(kind=OutKind.MARGIN_ROBUST, y_true=0, margin=0.0)
            return [(input_spec, output_spec)]
            
        except Exception as e:
            import traceback
            print(f"⚠️  Could not parse VNNLIB {vnnlib_path}: {e}")
            traceback.print_exc()
            return []
    
    def load_all_for_act_backend(self) -> Dict[str, Any]:
        """Load all discovered specifications for ACT backend"""
        discovered = self.discover_all_specs()
        act_ready_specs = {}
        
        # Load VNNLIB specifications
        for vnnlib_path in discovered["vnnlib"]:
            try:
                specs = self.load_vnnlib_specs(vnnlib_path)
                spec_name = Path(vnnlib_path).stem
                act_ready_specs[spec_name] = {
                    "type": "vnnlib",
                    "input_specs": [spec[0] for spec in specs],
                    "output_specs": [spec[1] for spec in specs],
                    "spec_pairs": specs
                }
                print(f"✅ Prepared VNNLIB '{spec_name}' for ACT backend ({len(specs)} pairs)")
            except Exception as e:
                print(f"❌ Failed to prepare VNNLIB {vnnlib_path}: {e}")
        
        # Load JSON specifications with comprehensive tensor conversion
        for json_path in discovered["json"]:
            try:
                from act.front_end.loaders.data_loader import DatasetLoader
                data_loader = DatasetLoader()  # For JSON loading capability
                json_spec = data_loader.load_json_spec(json_path)
                spec_name = Path(json_path).stem
                
                # Convert to InputSpec/OutputSpec format based on available keys
                input_specs = []
                output_specs = []
                
                # Handle box constraints (lb/ub format)
                if all(key in json_spec for key in ["lb", "ub"]):
                    # Ensure lb and ub are torch tensors with proper device/dtype
                    lb_tensor = torch.tensor(json_spec["lb"], device=self.device, dtype=self.dtype)
                    ub_tensor = torch.tensor(json_spec["ub"], device=self.device, dtype=self.dtype)
                    
                    # Create box input spec
                    input_spec = InputSpec(
                        kind=InKind.BOX,
                        lb=lb_tensor,
                        ub=ub_tensor
                    )
                    input_specs.append(input_spec)
                
                # Handle L∞ ball constraints (center + epsilon format)
                elif all(key in json_spec for key in ["center", "epsilon"]):
                    center_tensor = torch.tensor(json_spec["center"], device=self.device, dtype=self.dtype)
                    epsilon = float(json_spec["epsilon"])
                    
                    input_spec = InputSpec(
                        kind=InKind.LINF_BALL,
                        center=center_tensor,
                        eps=epsilon
                    )
                    input_specs.append(input_spec)
                
                # Handle L2 ball constraints (center + radius format) - convert to box
                elif all(key in json_spec for key in ["center", "radius"]):
                    center_tensor = torch.tensor(json_spec["center"], device=self.device, dtype=self.dtype)
                    radius = float(json_spec["radius"])
                    
                    # Convert L2 ball to box approximation
                    lb_tensor = center_tensor - radius
                    ub_tensor = center_tensor + radius
                    
                    input_spec = InputSpec(
                        kind=InKind.BOX,
                        lb=lb_tensor,
                        ub=ub_tensor
                    )
                    input_specs.append(input_spec)
                
                # Handle output specifications
                if "target_label" in json_spec:
                    target_label = int(json_spec["target_label"])
                    margin = float(json_spec.get("margin", 0.0))
                    
                    output_spec = OutputSpec(
                        kind=OutKind.MARGIN_ROBUST,
                        y_true=target_label,
                        margin=margin
                    )
                    output_specs.append(output_spec)
                
                # Handle linear constraints (coeffs + bound format)
                elif all(key in json_spec for key in ["coeffs", "bound"]):
                    coeffs_tensor = torch.tensor(json_spec["coeffs"], device=self.device, dtype=self.dtype)
                    bound = float(json_spec["bound"])
                    
                    output_spec = OutputSpec(
                        kind=OutKind.LINEAR_LE,
                        c=coeffs_tensor,
                        d=bound
                    )
                    output_specs.append(output_spec)
                
                # Convert any remaining numeric arrays to tensors in raw_data
                tensor_converted_data = {}
                for key, value in json_spec.items():
                    if isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
                        # Convert numeric lists to tensors
                        tensor_converted_data[key] = torch.tensor(value, device=self.device, dtype=self.dtype)
                    else:
                        tensor_converted_data[key] = value
                
                # Store the processed specification
                if input_specs or output_specs:
                    act_ready_specs[spec_name] = {
                        "type": "json_structured",
                        "input_specs": input_specs,
                        "output_specs": output_specs,
                        "raw_data": tensor_converted_data
                    }
                    print(f"✅ Prepared JSON '{spec_name}' for ACT backend ({len(input_specs)} input, {len(output_specs)} output specs)")
                else:
                    # Even if no structured specs, convert numeric data to tensors
                    act_ready_specs[spec_name] = {
                        "type": "json_raw", 
                        "raw_data": tensor_converted_data
                    }
                    print(f"✅ Prepared JSON '{spec_name}' with tensor-converted data")
                    
            except Exception as e:
                print(f"❌ Failed to prepare JSON {json_path}: {e}")
                
        return act_ready_specs



if __name__ == "__main__":
    # Simple test for VNNLIB loader
    print("=" * 80)
    print("VNNLIB Specification Loader Test")
    print("=" * 80)

    spec_loader = SpecLoader()
    test_vnnlib = "/home/guanqizh/Data/postdoc/ACT/models/vnnmodels/relusplitter/vnnlib/cifar_biasfield_vnncomp2022_prop_0.vnnlib"

    print(f"Testing file: {test_vnnlib}")
    specs = spec_loader.load_vnnlib_specs(test_vnnlib)

    if not specs:
        print("❌ Failed to load specifications!")
    else:
        input_spec, output_spec = specs[0]
        valid_bounds = (input_spec.lb < input_spec.ub).all()
        print(f"Loaded {len(specs)} spec(s)")
        print(f"Input: kind={input_spec.kind}, shape={input_spec.lb.shape}, bounds valid={valid_bounds}")
        print(f"  lb[:5]={input_spec.lb[:5].tolist()} ... ub[:5]={input_spec.ub[:5].tolist()}")
        print(f"Output: kind={output_spec.kind}")
        if hasattr(output_spec, "y_true"):
            print(f"  True label: {output_spec.y_true}, margin: {output_spec.margin}")
        elif hasattr(output_spec, "c"):
            print(f"  Coefficients shape: {output_spec.c.shape}, bound: {output_spec.d}")
        elif hasattr(output_spec, "lb") and hasattr(output_spec, "ub"):
            print(f"  Lower Bound: {output_spec.lb}, Upper Bound: {output_spec.ub}")
        print("Test finished.")