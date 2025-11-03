"""
ACTFuzzer: Inference-based whitebox fuzzing for neural network verification.

Main fuzzer engine that orchestrates mutation, coverage tracking, and
property checking to find counterexamples.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time
import json
import torch
import torch.nn as nn
from pathlib import Path

from act.front_end.specs import InputSpec, OutputSpec
from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.verifiable_model import InputSpecLayer, OutputSpecLayer
from act.pipeline.fuzzing.mutations import MutationEngine
from act.pipeline.fuzzing.coverage import CoverageTracker
from act.pipeline.fuzzing.corpus import SeedCorpus, FuzzingSeed
from act.pipeline.fuzzing.checker import PropertyChecker, Counterexample


@dataclass
class FuzzingConfig:
    """
    Fuzzing configuration (immutable).
    
    Attributes:
        max_iterations: Maximum fuzzing iterations
        timeout_seconds: Total time budget
        seed_selection_strategy: "energy" or "random"
        mutation_weights: Dict of strategy weights
        device: Torch device ("cuda" or "cpu")
        save_counterexamples: Whether to save counterexamples incrementally
        output_dir: Output directory for results
        report_interval: Print progress every N iterations
    """
    max_iterations: int = 10000
    timeout_seconds: float = 3600.0
    seed_selection_strategy: str = "energy"
    mutation_weights: Dict[str, float] = field(default_factory=lambda: {
        "gradient": 0.4,
        "activation": 0.3,
        "boundary": 0.2,
        "random": 0.1
    })
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_counterexamples: bool = True
    output_dir: Path = field(default_factory=lambda: Path("fuzzing_results"))
    report_interval: int = 100


@dataclass
class FuzzingReport:
    """
    Fuzzing results summary.
    
    Attributes:
        total_iterations: Number of iterations completed
        total_time: Time elapsed in seconds
        counterexamples: List of found counterexamples
        neuron_coverage: Final neuron coverage (0.0 to 1.0)
        total_mutations: Total mutations applied
        seeds_explored: Number of unique seeds explored
    """
    total_iterations: int
    total_time: float
    counterexamples: List[Counterexample]
    neuron_coverage: float
    total_mutations: int
    seeds_explored: int
    
    def save(self, output_dir: Path):
        """Save report and counterexamples to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary as JSON
        summary = {
            "iterations": self.total_iterations,
            "time_seconds": self.total_time,
            "counterexamples_found": len(self.counterexamples),
            "neuron_coverage": self.neuron_coverage,
            "mutations": self.total_mutations,
            "seeds_explored": self.seeds_explored
        }
        
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save counterexamples
        for i, ce in enumerate(self.counterexamples):
            ce.save(output_dir / f"counterexample_{i}.pt")
        
        print(f"✅ Report saved to {output_dir}")


class ACTFuzzer:
    """
    Inference-based whitebox fuzzer for neural network verification.
    
    Features:
    - Gradient-guided mutations (FGSM-style)
    - Neuron coverage tracking (DeepXplore)
    - Energy-based seed scheduling (AFL)
    - OutputSpec violation detection
    - InputSpec constraint projection
    
    Workflow:
    1. Initialize with wrapped model and seeds
    2. Loop: Select seed → Mutate → Inference → Check violation → Update coverage
    3. Return report with counterexamples
    
    Example:
        >>> fuzzer = ACTFuzzer(
        ...     wrapped_model=model,
        ...     initial_seeds=labeled_tensors,
        ...     config=FuzzingConfig(max_iterations=5000)
        ... )
        >>> report = fuzzer.fuzz()
        >>> print(f"Found {len(report.counterexamples)} violations")
    """
    
    def __init__(self,
                 wrapped_model: nn.Sequential,
                 initial_seeds: List[LabeledInputTensor],
                 config: Optional[FuzzingConfig] = None):
        """
        Initialize ACTFuzzer.
        
        Args:
            wrapped_model: VerifiableModel from model_synthesis
                          (contains InputSpecLayer and OutputSpecLayer)
            initial_seeds: List of LabeledInputTensor from spec creators
            config: Fuzzing configuration (uses defaults if None)
        """
        self.config = config or FuzzingConfig()
        self.model = wrapped_model.to(self.config.device)
        self.device = torch.device(self.config.device)
        
        # Extract specs from model
        self.input_spec = self._extract_spec(InputSpecLayer)
        self.output_spec = self._extract_spec(OutputSpecLayer)
        
        # Initialize components
        self.mutation_engine = MutationEngine(
            model=self.model,
            input_spec=self.input_spec,
            weights=self.config.mutation_weights,
            device=self.device
        )
        self.coverage_tracker = CoverageTracker(self.model)
        self.property_checker = PropertyChecker(self.output_spec)
        self.seed_corpus = SeedCorpus(
            initial_seeds=initial_seeds,
            strategy=self.config.seed_selection_strategy
        )
        
        # Statistics
        self.counterexamples: List[Counterexample] = []
        self.iterations = 0
        self.start_time = 0.0
    
    def _extract_spec(self, layer_type) -> Optional[InputSpec | OutputSpec]:
        """Extract spec from wrapper layer."""
        for layer in self.model:
            if isinstance(layer, layer_type):
                return layer.spec
        return None
    
    def fuzz(self) -> FuzzingReport:
        """
        Main fuzzing loop.
        
        Returns:
            FuzzingReport with counterexamples and statistics
        """
        print(f"{'='*80}")
        print(f"ACT: Abstract Constraint Transformer")
        print(f"Inference-based whitebox fuzzing for neural network verification")
        print(f"{'='*80}\n")
        
        print(f"🚀 Starting ACTFuzzer with {len(self.seed_corpus)} seeds")
        print(f"   Device: {self.device}")
        print(f"   Max iterations: {self.config.max_iterations}")
        print(f"   Timeout: {self.config.timeout_seconds}s\n")
        
        self.start_time = time.time()
        
        for iteration in range(self.config.max_iterations):
            # Check timeout
            if time.time() - self.start_time > self.config.timeout_seconds:
                print(f"⏱️  Timeout reached after {iteration} iterations")
                break
            
            # Fuzzing iteration
            self._fuzz_iteration(iteration)
            
            # Periodic reporting
            if iteration > 0 and iteration % self.config.report_interval == 0:
                self._print_progress(iteration)
        
        return self._generate_report()
    
    def _fuzz_iteration(self, iteration: int):
        """Single fuzzing iteration."""
        # 1. Select seed
        seed = self.seed_corpus.select()
        
        # 2. Ensure batch dimension (required for models)
        seed_tensor = seed.tensor
        if seed_tensor.dim() == 3:  # CHW without batch
            seed_tensor = seed_tensor.unsqueeze(0)  # Add batch dimension
        
        # 3. Mutate with feedback
        candidate = self.mutation_engine.mutate(seed_tensor)
        
        # 4. Run inference
        with torch.no_grad():
            output_dict = self.model(candidate)
        
        # Handle VerifiableModel output (dict) or plain tensor
        if isinstance(output_dict, dict):
            output = output_dict['output']
        else:
            output = output_dict
        
        # 4. Check violation
        violation = self.property_checker.check(
            input_tensor=candidate,
            output=output,
            label=seed.label,
            seed_tensor=seed.tensor
        )
        
        # 5. Update coverage
        activations = self.mutation_engine.get_last_activations()
        coverage_delta = self.coverage_tracker.update(candidate, activations)
        
        # 6. Handle results
        if violation:
            self.counterexamples.append(violation)
            print(f"🚨 Counterexample #{len(self.counterexamples)}: {violation.summary()}")
            
            if self.config.save_counterexamples:
                self.config.output_dir.mkdir(parents=True, exist_ok=True)
                violation.save(self.config.output_dir / f"ce_{len(self.counterexamples)}.pt")
        
        # 7. Add to corpus if interesting
        if violation or coverage_delta > 0:
            energy = self._compute_energy(coverage_delta, violation is not None)
            new_seed = FuzzingSeed(
                tensor=candidate.cpu(),
                label=seed.label,
                energy=energy,
                depth=seed.depth + 1
            )
            self.seed_corpus.add(new_seed)
        
        self.iterations = iteration + 1
    
    def _compute_energy(self, coverage_delta: float, found_violation: bool) -> float:
        """Compute seed energy (higher = more interesting)."""
        energy = coverage_delta * 10.0
        if found_violation:
            energy += 100.0  # Violations are very interesting
        return max(energy, 0.1)  # Minimum energy
    
    def _print_progress(self, iteration: int):
        """Print fuzzing progress."""
        elapsed = time.time() - self.start_time
        iter_per_sec = iteration / elapsed if elapsed > 0 else 0
        coverage = self.coverage_tracker.get_coverage()
        
        print(f"📊 Iteration {iteration:6d} | "
              f"Coverage: {coverage:6.2%} | "
              f"Seeds: {len(self.seed_corpus):4d} | "
              f"Violations: {len(self.counterexamples):3d} | "
              f"Speed: {iter_per_sec:5.1f} it/s")
    
    def _generate_report(self) -> FuzzingReport:
        """Generate final report."""
        total_time = time.time() - self.start_time
        
        report = FuzzingReport(
            total_iterations=self.iterations,
            total_time=total_time,
            counterexamples=self.counterexamples,
            neuron_coverage=self.coverage_tracker.get_coverage(),
            total_mutations=self.mutation_engine.total_mutations,
            seeds_explored=len(self.seed_corpus)
        )
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"🎉 ACTFuzzer completed in {total_time:.1f}s")
        print(f"   Iterations: {report.total_iterations}")
        print(f"   Counterexamples: {len(report.counterexamples)}")
        print(f"   Coverage: {report.neuron_coverage:.2%}")
        print(f"   Seeds explored: {report.seeds_explored}")
        print(f"{'='*80}\n")
        
        if self.config.save_counterexamples and report.counterexamples:
            report.save(self.config.output_dir)
        
        return report
