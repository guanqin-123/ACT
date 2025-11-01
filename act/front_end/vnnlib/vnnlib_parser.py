#===- act/front_end/vnnlib/vnnlib_parser.py - VNNLIB Parser ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Parse VNNLIB SMT-LIB format files to extract input tensors and constraints.
#   Converts VNNLIB specifications to InputSpec and OutputSpec objects.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import torch
import re

from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind

logger = logging.getLogger(__name__)


class VNNLibParseError(Exception):
    """Exception raised when VNNLIB parsing fails."""
    pass


def parse_vnnlib_to_tensors(
    vnnlib_path: Path,
    input_shape: Optional[Tuple[int, ...]] = None
) -> Tuple[torch.Tensor, Dict[str, any]]:
    """
    Parse a VNNLIB file to extract input tensor and metadata.
    
    The input tensor represents the center of the constrained input region.
    For box constraints with bounds [lb, ub], the center is (lb + ub) / 2.
    
    Args:
        vnnlib_path: Path to .vnnlib file
        input_shape: Expected input shape (optional, can be inferred)
        
    Returns:
        Tuple of (input_tensor, metadata_dict) where:
        - input_tensor: torch.Tensor representing input sample
        - metadata_dict: Contains 'input_bounds', 'num_outputs', 'property_type'
        
    Raises:
        VNNLibParseError: If parsing fails
    """
    if not vnnlib_path.exists():
        raise VNNLibParseError(f"VNNLIB file not found: {vnnlib_path}")
    
    try:
        with open(vnnlib_path, 'r') as f:
            content = f.read()
        
        # Extract variable declarations to determine shapes
        num_inputs = _extract_num_inputs(content)
        num_outputs = _extract_num_outputs(content)
        
        # Extract input bounds (X_i constraints)
        input_bounds = _extract_input_bounds(content, num_inputs)
        
        # Create input tensor from bounds center
        input_values = []
        for i in range(num_inputs):
            if i in input_bounds:
                lb, ub = input_bounds[i]
                center = (lb + ub) / 2.0
            else:
                # Default to 0 if no constraint
                center = 0.0
            input_values.append(center)
        
        input_tensor = torch.tensor(input_values, dtype=torch.float32)
        
        # Reshape if shape provided
        if input_shape is not None:
            expected_numel = 1
            for dim in input_shape:
                expected_numel *= dim
            if input_tensor.numel() != expected_numel:
                raise VNNLibParseError(
                    f"Input size mismatch: got {input_tensor.numel()} "
                    f"values but expected {expected_numel} from shape {input_shape}"
                )
            input_tensor = input_tensor.view(*input_shape)
        
        # Infer property type
        property_type = _infer_property_type(content, num_outputs)
        
        metadata = {
            'input_bounds': input_bounds,
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'property_type': property_type,
            'vnnlib_path': str(vnnlib_path)
        }
        
        logger.info(
            f"Parsed VNNLIB: {num_inputs} inputs, {num_outputs} outputs, "
            f"type={property_type}"
        )
        
        return input_tensor, metadata
        
    except Exception as e:
        raise VNNLibParseError(f"Failed to parse {vnnlib_path}: {str(e)}")


def parse_vnnlib_to_specs(
    vnnlib_path: Path,
    input_shape: Optional[Tuple[int, ...]] = None
) -> Tuple[InputSpec, OutputSpec]:
    """
    Parse VNNLIB file to create InputSpec and OutputSpec objects.
    
    Args:
        vnnlib_path: Path to .vnnlib file
        input_shape: Expected input shape (optional)
        
    Returns:
        Tuple of (InputSpec, OutputSpec)
        
    Raises:
        VNNLibParseError: If parsing fails
    """
    try:
        with open(vnnlib_path, 'r') as f:
            content = f.read()
        
        num_inputs = _extract_num_inputs(content)
        num_outputs = _extract_num_outputs(content)
        input_bounds = _extract_input_bounds(content, num_inputs)
        
        # Create InputSpec (BOX constraints)
        lb_values = []
        ub_values = []
        for i in range(num_inputs):
            if i in input_bounds:
                lb, ub = input_bounds[i]
            else:
                lb, ub = float('-inf'), float('inf')
            lb_values.append(lb)
            ub_values.append(ub)
        
        lb_tensor = torch.tensor(lb_values, dtype=torch.float32)
        ub_tensor = torch.tensor(ub_values, dtype=torch.float32)
        
        if input_shape is not None:
            lb_tensor = lb_tensor.view(*input_shape)
            ub_tensor = ub_tensor.view(*input_shape)
        
        input_spec = InputSpec(
            kind=InKind.BOX,
            lb=lb_tensor,
            ub=ub_tensor
        )
        
        # Create OutputSpec (LINEAR_LE constraints)
        output_constraints = _extract_output_constraints(content, num_outputs)
        
        if output_constraints:
            # Use first constraint as representative
            c, d = output_constraints[0]
            output_spec = OutputSpec(
                kind=OutKind.LINEAR_LE,
                c=torch.tensor(c, dtype=torch.float32),
                d=float(d),
                meta={'all_constraints': output_constraints}
            )
        else:
            # Fallback: RANGE with no specific bounds
            output_spec = OutputSpec(
                kind=OutKind.RANGE,
                lb=torch.tensor([float('-inf')] * num_outputs, dtype=torch.float32),
                ub=torch.tensor([float('inf')] * num_outputs, dtype=torch.float32)
            )
        
        logger.info(f"Created specs from VNNLIB: {input_spec.kind}, {output_spec.kind}")
        
        return input_spec, output_spec
        
    except Exception as e:
        raise VNNLibParseError(f"Failed to create specs from {vnnlib_path}: {str(e)}")


def _extract_num_inputs(content: str) -> int:
    """
    Extract number of input variables from VNNLIB content.
    
    Looks for patterns like:
    - (declare-const X_0 Real)
    - (declare-const X_1 Real)
    """
    x_vars = set()
    # Match X_<number>
    pattern = r'X_(\d+)'
    matches = re.findall(pattern, content)
    for match in matches:
        x_vars.add(int(match))
    
    if not x_vars:
        raise VNNLibParseError("No input variables (X_i) found")
    
    # Number of inputs is max index + 1 (assuming 0-indexed)
    return max(x_vars) + 1


def _extract_num_outputs(content: str) -> int:
    """
    Extract number of output variables from VNNLIB content.
    
    Looks for patterns like:
    - (declare-const Y_0 Real)
    - (declare-const Y_1 Real)
    """
    y_vars = set()
    # Match Y_<number>
    pattern = r'Y_(\d+)'
    matches = re.findall(pattern, content)
    for match in matches:
        y_vars.add(int(match))
    
    if not y_vars:
        logger.warning("No output variables (Y_i) found in VNNLIB")
        return 0
    
    return max(y_vars) + 1


def _extract_input_bounds(content: str, num_inputs: int) -> Dict[int, Tuple[float, float]]:
    """
    Extract lower and upper bounds for each input variable.
    
    Parses constraints like:
    - (assert (>= X_0 0.5))  -> lower bound
    - (assert (<= X_0 1.5))  -> upper bound
    
    Returns:
        Dict mapping input index to (lower_bound, upper_bound)
    """
    bounds = {}
    
    # Initialize with infinity bounds
    for i in range(num_inputs):
        bounds[i] = [float('-inf'), float('inf')]
    
    # Match lower bounds: (>= X_i value) or (>= value X_i)
    lb_patterns = [
        r'\(>=\s+X_(\d+)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)',
        r'\(>=\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+X_(\d+)\)'
    ]
    
    for pattern in lb_patterns:
        for match in re.finditer(pattern, content):
            groups = match.groups()
            # Handle both orderings
            if groups[0].replace('.', '').replace('-', '').replace('+', '').replace('e', '').replace('E', '').isdigit():
                # First group is number
                idx = int(groups[1])
                lb = float(groups[0])
            else:
                # Second group is number
                idx = int(groups[0])
                lb = float(groups[1])
            
            if idx < num_inputs:
                bounds[idx][0] = max(bounds[idx][0], lb)
    
    # Match upper bounds: (<= X_i value) or (<= value X_i)
    ub_patterns = [
        r'\(<=\s+X_(\d+)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)',
        r'\(<=\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+X_(\d+)\)'
    ]
    
    for pattern in ub_patterns:
        for match in re.finditer(pattern, content):
            groups = match.groups()
            # Handle both orderings
            if groups[0].replace('.', '').replace('-', '').replace('+', '').replace('e', '').replace('E', '').isdigit():
                # First group is number
                idx = int(groups[1])
                ub = float(groups[0])
            else:
                # Second group is number
                idx = int(groups[0])
                ub = float(groups[1])
            
            if idx < num_inputs:
                bounds[idx][1] = min(bounds[idx][1], ub)
    
    # Convert to tuples and filter infinite bounds
    result = {}
    for i, (lb, ub) in bounds.items():
        if lb != float('-inf') or ub != float('inf'):
            result[i] = (lb, ub)
    
    return result


def _extract_output_constraints(content: str, num_outputs: int) -> List[Tuple[List[float], float]]:
    """
    Extract output constraints (linear combinations of Y_i).
    
    Returns list of (coefficients, bias) tuples representing c^T * y <= d.
    """
    constraints = []
    
    # This is a simplified parser for common patterns
    # Full VNNLIB can have complex nested assertions
    
    # Match patterns like: (<= (+ (* a0 Y_0) (* a1 Y_1) ...) d)
    # This would require more sophisticated parsing for general VNNLIB
    
    # For now, return empty list (would need full SMT-LIB parser for complete support)
    logger.debug("Output constraint extraction not fully implemented (requires full SMT-LIB parser)")
    
    return constraints


def _infer_property_type(content: str, num_outputs: int) -> str:
    """
    Infer the property type from VNNLIB content.
    
    Returns:
        One of: 'classification', 'safety', 'unknown'
    """
    content_lower = content.lower()
    
    # Classification properties often involve comparisons between outputs
    if 'y_' in content_lower and num_outputs > 1:
        # Check for patterns like Y_i - Y_j > 0 (classification margin)
        if re.search(r'y_\d+\s*[-]\s*y_\d+', content_lower):
            return 'classification'
    
    # Safety properties typically have output range constraints
    if num_outputs == 1 or 'range' in content_lower:
        return 'safety'
    
    return 'unknown'


def validate_vnnlib_file(vnnlib_path: Path) -> bool:
    """
    Validate that a VNNLIB file is parseable.
    
    Args:
        vnnlib_path: Path to .vnnlib file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        parse_vnnlib_to_tensors(vnnlib_path)
        return True
    except VNNLibParseError as e:
        logger.error(f"VNNLIB validation failed: {e}")
        return False


def list_vnnlib_variables(vnnlib_path: Path) -> Dict[str, int]:
    """
    List all variables declared in a VNNLIB file.
    
    Args:
        vnnlib_path: Path to .vnnlib file
        
    Returns:
        Dict with 'num_inputs' and 'num_outputs'
    """
    try:
        with open(vnnlib_path, 'r') as f:
            content = f.read()
        
        return {
            'num_inputs': _extract_num_inputs(content),
            'num_outputs': _extract_num_outputs(content)
        }
    except Exception as e:
        logger.error(f"Failed to list variables: {e}")
        return {'num_inputs': 0, 'num_outputs': 0}
