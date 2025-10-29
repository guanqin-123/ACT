#!/usr/bin/env python3
#===- act/pipeline/validate_verifier.py - Verifier Correctness Validation ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Validate formal verifier correctness using concrete counterexamples.
#   If concrete inference finds a counterexample, the formal verifier MUST
#   report SAT/COUNTEREXAMPLE (not CERTIFIED), otherwise verifier is unsound.
#
# Key Insight:
#   Concrete execution provides ground truth - if we find a real counterexample
#   at runtime, the formal verifier cannot claim the property is certified.
#   This is a soundness check for the verification backend.
#
# Validation Strategy:
#   1. For each network, generate strategic test cases:
#      - Center: Input at center of input spec (typically safe)
#      - Boundary: Input near boundary of input spec (risky)
#      - Random: Random input within input spec (varied)
#
#   2. Run concrete execution to find violations
#   3. If counterexample found, run formal verification
#   4. Cross-validate using matrix below
#
# Validation Matrix:
#   ┌─────────────────────────┬────────────────────────────────────┬──────────────┐
#   │ Concrete Counterexample │ Verifier Result                    │ Validation   │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ FOUND                   │ CERTIFIED / UNSAT / INFEASIBLE     │ ❌ FAILED    │
#   │                         │ (Soundness Bug - false negative)   │              │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ FOUND                   │ COUNTEREXAMPLE / SAT / FEASIBLE    │ ✅ PASSED    │
#   │                         │ (Correct - verifier found issue)   │              │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ FOUND                   │ UNKNOWN / TIMEOUT                  │ ⚠️ ACCEPTABLE│
#   │                         │ (Incomplete but sound)             │              │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ NOT FOUND               │ Any Result                         │ ❓ INCONC.   │
#   │                         │ (Cannot validate - no ground truth)│              │
#   └─────────────────────────┴────────────────────────────────────┴──────────────┘
#
#   Legend:
#     FAILED       - Critical soundness bug (false negative)
#     PASSED       - Verifier correct
#     ACCEPTABLE   - Verifier incomplete but sound (conservative)
#     INCONCLUSIVE - No concrete counterexample to validate against
#
# Usage:
#   python act/pipeline/validate_verifier.py
#
# Exit Codes:
#   0 - All validations passed (no soundness bugs)
#   1 - Soundness bugs detected (verifier claimed safety despite counterexample)
#
#===---------------------------------------------------------------------===#

import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from act.pipeline.model_factory import ModelFactory
from act.pipeline.torch2act import TorchToACT
from act.back_end.verifier import verify_once
from act.back_end.solver.solver_gurobi import GurobiSolver
from act.back_end.solver.solver_torch import TorchLPSolver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerifierValidator:
    """Validate formal verifier correctness using concrete counterexamples."""
    
    def __init__(self):
        self.factory = ModelFactory()
        self.validation_results = []
    
    def find_concrete_counterexample(
        self, 
        name: str, 
        model: torch.nn.Module
    ) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Try to find a concrete counterexample through inference testing.
        
        Returns:
            (counterexample_input, results_dict) if found, None otherwise
        """
        test_cases = ['center', 'boundary', 'random']
        
        for test_case in test_cases:
            input_tensor = self.factory.generate_test_input(name, test_case)
            results = model(input_tensor)
            
            # Check if this is a counterexample
            if isinstance(results, dict):
                if results['input_satisfied'] and not results['output_satisfied']:
                    logger.info(f"  🔴 Counterexample found in '{test_case}' test case")
                    logger.info(f"     Input explanation: {results['input_explanation']}")
                    logger.info(f"     Output explanation: {results['output_explanation']}")
                    return (input_tensor, results)
        
        return None
    
    def validate_network(
        self, 
        name: str, 
        solver: str = 'gurobi'
    ) -> Dict[str, Any]:
        """
        Validate verifier correctness for a single network.
        
        Args:
            name: Network name from examples_config.yaml
            solver: 'gurobi' or 'torchlp'
            
        Returns:
            Validation result dictionary with status and details
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Validating: {name} (solver: {solver})")
        logger.info(f"{'='*80}")
        
        # Step 1: Create PyTorch model and find concrete counterexample
        model = self.factory.create_model(name, load_weights=True)
        counterexample = self.find_concrete_counterexample(name, model)
        
        # Step 2: Convert to ACT Net
        converter = TorchToACT(model)
        act_net = converter.run()
        
        # Step 3: Run formal verifier
        logger.info(f"\n  🔍 Running formal verifier ({solver})...")
        
        try:
            if solver == 'gurobi':
                solver_instance = GurobiSolver()
            elif solver == 'torchlp':
                solver_instance = TorchLPSolver()
            else:
                raise ValueError(f"Unknown solver: {solver}")
            
            verify_result = verify_once(act_net, solver=solver_instance)
            logger.info(f"     Verifier result: {verify_result}")
            
        except Exception as e:
            logger.error(f"     Verifier failed: {e}")
            return {
                'network': name,
                'solver': solver,
                'status': 'ERROR',
                'error': str(e)
            }
        
        # Step 4: Cross-validate results
        validation = self._cross_validate(
            network_name=name,
            solver_name=solver,
            concrete_counterexample=counterexample,
            verifier_result=verify_result
        )
        
        self.validation_results.append(validation)
        return validation
    
    def _cross_validate(
        self,
        network_name: str,
        solver_name: str,
        concrete_counterexample: Optional[Tuple],
        verifier_result: str
    ) -> Dict[str, Any]:
        """
        Cross-validate concrete inference vs formal verification.
        
        Validation Rules:
        1. If concrete counterexample found → verifier MUST report SAT/COUNTEREXAMPLE/UNKNOWN
        2. If no concrete counterexample → verifier can report anything (testing incomplete)
        """
        result = {
            'network': network_name,
            'solver': solver_name,
            'concrete_counterexample': counterexample is not None,
            'verifier_result': verifier_result,
            'validation_status': None,
            'explanation': None
        }
        
        if concrete_counterexample is not None:
            # We found a real counterexample - verifier MUST NOT claim CERTIFIED
            input_tensor, inference_results = concrete_counterexample
            
            if verifier_result in ['CERTIFIED', 'UNSAT', 'INFEASIBLE']:
                # CRITICAL BUG: Verifier claims safe, but we have a counterexample!
                result['validation_status'] = 'FAILED'
                result['explanation'] = (
                    f"🚨 SOUNDNESS BUG DETECTED! Verifier claims CERTIFIED but "
                    f"concrete counterexample exists. This is a false negative."
                )
                logger.error(f"\n  {result['explanation']}")
                logger.error(f"     Counterexample input: {input_tensor.shape}, "
                            f"range=[{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
                logger.error(f"     Output violation: {inference_results['output_explanation']}")
                
            elif verifier_result in ['COUNTEREXAMPLE', 'SAT', 'FEASIBLE']:
                # CORRECT: Verifier correctly identified the issue
                result['validation_status'] = 'PASSED'
                result['explanation'] = (
                    f"✅ CORRECT - Verifier correctly reported counterexample "
                    f"(matches concrete execution)"
                )
                logger.info(f"\n  {result['explanation']}")
                
            elif verifier_result in ['UNKNOWN', 'TIMEOUT']:
                # ACCEPTABLE: Verifier couldn't decide (incomplete but sound)
                result['validation_status'] = 'ACCEPTABLE'
                result['explanation'] = (
                    f"⚠️ INCOMPLETE - Verifier returned UNKNOWN, but concrete "
                    f"counterexample exists (verifier is sound but incomplete)"
                )
                logger.warning(f"\n  {result['explanation']}")
                
            else:
                result['validation_status'] = 'UNKNOWN'
                result['explanation'] = f"Unknown verifier result: {verifier_result}"
                logger.warning(f"\n  {result['explanation']}")
        
        else:
            # No concrete counterexample found in testing
            result['validation_status'] = 'INCONCLUSIVE'
            result['explanation'] = (
                f"⚪ INCONCLUSIVE - No counterexample found in concrete testing. "
                f"Verifier result: {verifier_result} (cannot validate with this test)"
            )
            logger.info(f"\n  {result['explanation']}")
        
        return result
    
    def validate_all_networks(self, solvers: list = ['gurobi', 'torchlp']) -> Dict[str, Any]:
        """
        Validate verifier correctness across all networks and solvers.
        
        Returns:
            Summary statistics and detailed results
        """
        networks = self.factory.list_networks()
        
        print("\n" + "="*80)
        print("FORMAL VERIFIER VALIDATION TEST SUITE")
        print("="*80)
        print(f"Networks: {len(networks)}")
        print(f"Solvers: {', '.join(solvers)}")
        print("="*80)
        
        for network in networks:
            for solver in solvers:
                try:
                    self.validate_network(network, solver)
                except Exception as e:
                    logger.error(f"Validation failed for {network}/{solver}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Compute summary statistics
        summary = self._compute_summary()
        self._print_summary(summary)
        
        return summary
    
    def _compute_summary(self) -> Dict[str, Any]:
        """Compute validation summary statistics."""
        total = len(self.validation_results)
        
        if total == 0:
            return {'total': 0, 'error': 'No validation results'}
        
        passed = sum(1 for r in self.validation_results if r['validation_status'] == 'PASSED')
        failed = sum(1 for r in self.validation_results if r['validation_status'] == 'FAILED')
        acceptable = sum(1 for r in self.validation_results if r['validation_status'] == 'ACCEPTABLE')
        inconclusive = sum(1 for r in self.validation_results if r['validation_status'] == 'INCONCLUSIVE')
        
        counterexamples_found = sum(1 for r in self.validation_results if r['concrete_counterexample'])
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'acceptable': acceptable,
            'inconclusive': inconclusive,
            'counterexamples_found': counterexamples_found,
            'critical_bugs': failed,  # FAILED status means soundness bug
            'results': self.validation_results
        }
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print validation summary."""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print(f"Total validation tests: {summary['total']}")
        print(f"Concrete counterexamples found: {summary['counterexamples_found']}")
        print()
        print(f"✅ PASSED (verifier correct):     {summary['passed']}")
        print(f"⚠️  ACCEPTABLE (incomplete):       {summary['acceptable']}")
        print(f"⚪ INCONCLUSIVE (no test data):   {summary['inconclusive']}")
        print(f"🚨 FAILED (soundness bugs):       {summary['failed']}")
        print("="*80)
        
        if summary['failed'] > 0:
            print("\n🚨 CRITICAL: Soundness bugs detected in verifier!")
            print("The following networks have false negatives:")
            for result in summary['results']:
                if result['validation_status'] == 'FAILED':
                    print(f"  - {result['network']} ({result['solver']})")
            print()
        elif summary['counterexamples_found'] > 0:
            print("\n✅ Verifier validation PASSED!")
            print("All concrete counterexamples were correctly identified by verifier.")
        else:
            print("\n⚪ No counterexamples found in concrete testing.")
            print("Cannot fully validate verifier soundness (need more test cases).")
        
        print("="*80)


def main():
    """Run verifier validation test suite."""
    validator = VerifierValidator()
    
    # Validate with both solvers
    summary = validator.validate_all_networks(solvers=['gurobi', 'torchlp'])
    
    # Exit code: 0 if no critical bugs, 1 if soundness bugs detected
    exit_code = 1 if summary['failed'] > 0 else 0
    return exit_code


if __name__ == "__main__":
    import sys
    sys.exit(main())
