"""
ACTFuzzer: Inference-based whitebox fuzzing for neural network verification.

Main fuzzer engine that orchestrates mutation, coverage tracking, and
property checking to find counterexamples.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import time
import json
import torch
import torch.nn as nn
from pathlib import Path

from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind, BatchedInputSpec, BatchedOutputSpec
from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.verifiable_model import (
    VerifiableModel, InputSpecLayer, OutputSpecLayer,
)
from act.pipeline.fuzzing.mutations import MutationEngine
from act.pipeline.fuzzing.coverage import CoverageTracker
from act.pipeline.fuzzing.corpus import SeedCorpus, FuzzingSeed
from act.pipeline.fuzzing.checker import Counterexample, PropertyChecker
from act.util.path_config import get_pipeline_log_dir


@dataclass
class FuzzingConfig:
    """
    Fuzzing configuration (immutable).
    
    Attributes:
        max_iterations: Maximum fuzzing iterations
        timeout_seconds: Total time budget
        seed_selection_strategy: "energy" or "random"
        mutation_weights: Dict of strategy weights
        perturb_mode: Perturbation size computation mode ("adaptive_scalar", "adaptive_perdim", "fixed")
        perturb_scale: Fraction of range per mutation perturbation (e.g., 0.1 = 10% = ~10 steps to traverse)
        device: Torch device ("cuda" or "cpu")
        save_counterexamples: Whether to save counterexamples incrementally
        output_dir: Output directory for results
        report_interval: Print progress every N iterations
        verbose: Logging verbosity (0=silent, 1=report violations in progress only, 2=print each violation immediately)
        trace_level: Execution tracing level (0=disabled, 1=default, 2=full, 3=debug)
        trace_sample_rate: Capture every Nth iteration (1=all iterations)
        trace_storage: Storage backend ("hdf5" or "json")
        trace_output: Trace output path (None=auto-generate)
    
    Perturbation Size Configuration:
        NOTE: We use "perturb_size" (not "epsilon") to avoid confusion with InputSpec.eps (L∞ radius).
        - InputSpec.eps: Defines constraint boundaries (e.g., center ± eps for LINF_BALL)
        - Mutation perturb_size: Controls mutation perturbation magnitude (exploration granularity)
        
        perturb_mode determines how mutation perturbation sizes are computed:
        - "adaptive_scalar": Single perturb_size from mean(ub-lb) * perturb_scale (default, best for uniform ranges)
        - "adaptive_perdim": Per-dimension perturb_size from (ub-lb) * perturb_scale (best for non-uniform ranges)
        - "fixed": Legacy hardcoded values (0.01 for gradient/activation, 0.005 for boundary/random)
        
        coverage_strategy determines the coverage strategy to use:
        - "BestInputCov": Per-input coverage (best per-input coverage over time)
        - "GlobalCov": Global union coverage (monotonic union over all inputs)
        
        perturb_scale interpretation:
        - Fraction of feasible range each mutation perturbation covers
        - steps_to_traverse = 1 / perturb_scale
        - Example: perturb_scale=0.1 → 10% per perturbation → ~10 steps to traverse from lb to ub
    """
    max_iterations: int = 10000
    timeout_seconds: float = 3600.0
    seed_selection_strategy: str = "energy"
    mutation_weights: Dict[str, float] = field(default_factory=lambda: {
        "gradient": 0.2,
        "pgd": 0.2,
        "activation": 0.3,
        "boundary": 0.2,
        "random": 0.1
    })
    coverage_strategy: str = "BestInputCov"  # "BestInputCov"/"GlobalCov"
    activation_threshold: float = 0.1  # Neuron activation threshold
    perturb_mode: str = "adaptive_scalar"  # Perturbation size computation mode
    perturb_scale: float = 0.1  # Fraction of range per perturbation (10% → ~10 steps)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_counterexamples: bool = True
    output_dir: Path = field(default_factory=lambda: Path(get_pipeline_log_dir()) / "fuzzing_results")
    report_interval: int = 100
    verbose: int = 1  # 0=silent, 1=report in progress only, 2=print each violation immediately
    
    # Tracing configuration
    trace_level: int = 0  # 0=disabled, 1=default, 2=full, 3=debug
    trace_sample_rate: int = 1  # Capture every Nth iteration
    trace_storage: str = "json"  # "json" or "hdf5"
    trace_output: Optional[Path] = None  # Auto-generated if None
    
    # Batched fuzzing configuration
    batch_size: int = 1  # Number of samples to process per iteration (1 = sequential, >1 = batched)


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
        num_of_never_activated_neurons: Number of neurons that were never activated across all iterations
        never_activated_neurons: Sample of never-activated neuron ids (layer_name, neuron_idx)
    """
    total_iterations: int
    total_time: float
    counterexamples: List[Counterexample]
    neuron_coverage: float
    total_mutations: int
    seeds_explored: int
    num_of_never_activated_neurons: int = 0
    never_activated_neurons: List[Dict[str, Any]] = field(default_factory=list)
    
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
            "seeds_explored": self.seeds_explored,
            "num_of_never_activated_neurons": self.num_of_never_activated_neurons,
            # JSON-friendly: list of [layer_name, neuron_idx]
            "never_activated_neurons": [[ln, int(i)] for (ln, i) in self.never_activated_neurons]
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
                 wrapped_model: nn.Module,
                 initial_seeds: List[LabeledInputTensor],
                 config: Optional[FuzzingConfig] = None):
        """
        Initialize ACTFuzzer.
        
        Args:
            wrapped_model: VerifiableModel from model_synthesis
                          (contains InputSpecLayer and OutputSpecLayer)
            initial_seeds: List of LabeledInputTensor from spec creators
            config: Fuzzing configuration (uses defaults if None)
        
        Note:
            VerifiableModel supports batching natively. Specs are extracted
            for the MutationEngine. The core model (without spec layers) is
            extracted for inference and avoid mismatch between spec and model.
        """
        self.config = config or FuzzingConfig()
        self.device = torch.device(self.config.device)
        
        # Extract specs and core model (supports both single and batched models)
        self.input_spec, self.output_spec, core_model = self._extract_specs_and_model(wrapped_model)
        self.model = core_model.to(self.config.device)
        
        # Initialize components
        self.mutation_engine = MutationEngine(
            model=self.model,
            input_spec=self.input_spec,
            weights=self.config.mutation_weights,
            device=self.device,
            perturb_mode=self.config.perturb_mode,
            perturb_scale=self.config.perturb_scale
        )
        self.coverage_tracker = CoverageTracker(model=self.model,threshold=self.config.activation_threshold, strategy=self.config.coverage_strategy)
        self.property_checker = PropertyChecker(self.output_spec)
        self.seed_corpus = SeedCorpus(
            initial_seeds=initial_seeds,
            strategy=self.config.seed_selection_strategy
        )
        
        # Initialize tracer (only if trace_level > 0)
        if self.config.trace_level > 0:
            from act.pipeline.fuzzing.tracer import ExecutionTracer
            
            # Auto-generate trace output path if not specified
            trace_output = self.config.trace_output or (
                self.config.output_dir / f"traces.{self._get_trace_ext()}"
            )
            
            self.tracer = ExecutionTracer(
                level=self.config.trace_level,
                sample_rate=self.config.trace_sample_rate,
                storage_backend=self.config.trace_storage,
                output_path=trace_output
            )
            
            print(f"📊 Tracing enabled: Level {self.config.trace_level}, "
                  f"sampling every {self.config.trace_sample_rate} iteration(s)")
            print(f"   Output: {trace_output}")
        else:
            self.tracer = None  # No overhead when disabled
        
        # Statistics
        self.counterexamples: List[Counterexample] = []
        self.iterations = 0
        self.start_time = 0.0
        self.never_activated_neurons: List[Dict[str, Any]] = []
        self.last_report_ce_count = 0  # Track counterexamples count at last report
    
    def _get_trace_ext(self) -> str:
        """Get file extension for trace storage."""
        return {"hdf5": "h5", "json": "json"}[self.config.trace_storage]
    
    @staticmethod
    def _batched_input_to_single(batched_in: BatchedInputSpec) -> InputSpec:
        """Convert BatchedInputSpec to single-sample InputSpec (using first sample)."""
        if batched_in.kind == InKind.LINF_BALL:
            eps_val = batched_in.eps
            if not isinstance(eps_val, (int, float)):
                eps_val = float(eps_val[0].item())  # type: ignore
            return InputSpec(
                kind=InKind.LINF_BALL,
                center=batched_in.center[0:1] if batched_in.center is not None else None,
                eps=eps_val,
            )
        elif batched_in.kind == InKind.BOX:
            return InputSpec(
                kind=InKind.BOX,
                lb=batched_in.lb[0:1] if batched_in.lb is not None else None,
                ub=batched_in.ub[0:1] if batched_in.ub is not None else None,
            )
        return InputSpec(kind=batched_in.kind)  # Fallback

    @staticmethod
    def _batched_output_to_single(batched_out: BatchedOutputSpec) -> OutputSpec:
        """Convert BatchedOutputSpec to single-sample OutputSpec (using first sample)."""
        margin_val = batched_out.margin
        if margin_val is not None and not isinstance(margin_val, (int, float)):
            margin_val = float(margin_val[0].item())  # type: ignore
        return OutputSpec(
            kind=batched_out.kind,
            y_true=int(batched_out.y_true[0].item()) if batched_out.y_true is not None else None,
            margin=margin_val if margin_val is not None else 0.0,
            lb=batched_out.lb[0:1] if hasattr(batched_out, 'lb') and batched_out.lb is not None else None,
            ub=batched_out.ub[0:1] if hasattr(batched_out, 'ub') and batched_out.ub is not None else None,
        )

    def _extract_specs_and_model(
        self, wrapped_model: nn.Module
    ) -> Tuple[Optional[InputSpec], Optional[OutputSpec], nn.Module]:
        """
        Extract InputSpec, OutputSpec, and core model from wrapped model.
        
        Supports VerifiableModel (which now handles both single and batched inputs).
        If the model's spec layers have a larger batch size than config.batch_size,
        the spec layer tensors are capped to match the fuzzer's batch size.
        """
        input_spec: Optional[InputSpec] = None
        output_spec: Optional[OutputSpec] = None
        max_batch = self.config.batch_size
        
        for layer in wrapped_model.children():
            if isinstance(layer, InputSpecLayer):
                input_spec = layer.spec
                self._cap_spec_layer_tensors(layer, layer.spec, ("lb", "ub", "center", "A", "b"), max_batch)
            elif isinstance(layer, OutputSpecLayer):
                output_spec = layer.spec
                self._cap_spec_layer_tensors(layer, layer.spec, ("c", "lb", "ub"), max_batch)
                # Handle y_true separately (not a buffer, stored as attribute)
                y_true = layer.y_true
                if isinstance(y_true, torch.Tensor) and y_true.dim() > 0 and y_true.shape[0] > max_batch:
                    capped_y_true = y_true[:max_batch]
                    layer.y_true = capped_y_true
                    if layer.spec is not None:
                        layer.spec.y_true = capped_y_true
        
        return input_spec, output_spec, wrapped_model
    
    def _cap_spec_layer_tensors(
        self, 
        layer: nn.Module, 
        spec: Optional[object], 
        field_names: Tuple[str, ...], 
        max_batch: int
    ) -> None:
        """
        Cap tensor fields in a spec layer to max_batch if they exceed it.
        
        Modifies the layer buffers and spec attributes in-place.
        """
        for name in field_names:
            tensor = getattr(layer, name, None)
            if isinstance(tensor, torch.Tensor) and tensor.dim() > 0 and tensor.shape[0] > max_batch:
                capped = tensor[:max_batch]
                layer.register_buffer(name, capped)
                if spec is not None:
                    setattr(spec, name, capped)
    
    def fuzz(self) -> FuzzingReport:
        """
        Main fuzzing loop.
        
        Automatically uses batched mode if config.batch_size > 1.
        
        Returns:
            FuzzingReport with counterexamples and statistics
        """
        print(f"{'='*80}")
        print(f"ACT: Abstract Constraint Transformer")
        print(f"Inference-based whitebox fuzzing for neural network verification")
        print(f"{'='*80}\n")
        
        batch_size = self.config.batch_size
        mode_str = f"batched (batch_size={batch_size})" if batch_size > 1 else "sequential"
        
        print(f"🚀 Starting ACTFuzzer with {len(self.seed_corpus)} seeds")
        print(f"   Device: {self.device}")
        print(f"   Mode: {mode_str}")
        print(f"   Max iterations: {self.config.max_iterations}")
        print(f"   Timeout: {self.config.timeout_seconds}s\n")
        
        self.start_time = time.time()
        
        if batch_size > 1:
            # Batched fuzzing loop
            self._fuzz_batched_loop()
        else:
            # Sequential fuzzing loop (original)
            self._fuzz_sequential_loop()
        
        return self._generate_report()
    
    def _fuzz_sequential_loop(self):
        """Original sequential fuzzing loop (batch_size=1)."""
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
    
    def _fuzz_batched_loop(self):
        """Batched fuzzing loop for GPU efficiency."""
        batch_size = self.config.batch_size
        iteration = 0
        
        while iteration < self.config.max_iterations:
            # Check timeout
            if time.time() - self.start_time > self.config.timeout_seconds:
                print(f"⏱️  Timeout reached after {iteration} iterations")
                break
            
            # Process a batch
            actual_batch_size = min(batch_size, self.config.max_iterations - iteration)
            self._fuzz_batch_iteration(iteration, actual_batch_size)
            iteration += actual_batch_size
            
            # Periodic reporting
            if iteration > 0 and iteration % self.config.report_interval < batch_size:
                self._print_progress(iteration)
    
    def _fuzz_batch_iteration(self, start_iteration: int, batch_size: int):
        """
        Process a batch of B samples with single inference call AND batched mutation.
        
        Key optimizations:
        1. Batched mutation: Single forward pass for gradient-based mutations (PGD/FGSM)
        2. Batched inference: Single forward pass for B samples
        3. Batched property checking: Vectorized violation detection
        
        Args:
            start_iteration: Starting iteration number
            batch_size: Number of samples to process in this batch
        """
        # 1. Select B seeds
        seeds = [self.seed_corpus.select() for _ in range(batch_size)]
        
        # 2. BATCHED mutation (key optimization!)
        # Instead of B sequential mutations, do 1 batched mutation
        batch_input = self.mutation_engine.mutate_batch(seeds)  # [B, C, H, W]
        
        with torch.no_grad():
            output_dict = self.model(batch_input)
        
        # Handle VerifiableModel output (dict) or plain tensor
        if isinstance(output_dict, dict):
            batch_output = output_dict['output']
        else:
            batch_output = output_dict
        
        # 3. BATCHED property checking
        labels = [s.label for s in seeds]
        seed_tensors = [s.tensor for s in seeds]
        violations = self.property_checker.check_batch(
            inputs=batch_input,
            outputs=batch_output,
            labels=labels,
            seed_tensors=seed_tensors
        )
        
        # 4. Process results (sequential but cheap)
        # Get activations from batched forward (shared across samples)
        activations = self.mutation_engine.get_activation_map()
        
        for i, (seed, violation) in enumerate(zip(seeds, violations)):
            iteration = start_iteration + i
            
            # Extract single sample from batch for corpus: [B, C, H, W] -> [1, C, H, W]
            candidate_single = batch_input[i:i+1]
            
            # Update coverage (approximate: use shared batch activations)
            # This is a trade-off for speed - coverage is approximate in batched mode
            coverage_delta = self.coverage_tracker.update(candidate_single, activations)
            
            # Compute energy
            if violation or coverage_delta > 0:
                energy = self._compute_energy(coverage_delta, violation is not None)
            else:
                energy = 0.0
            
            # Handle violations
            if violation:
                self.counterexamples.append(violation)
                if self.config.verbose >= 2:
                    print(f"🚨 Counterexample #{len(self.counterexamples)}: {violation.summary()}")
                
                if self.config.save_counterexamples:
                    self.config.output_dir.mkdir(parents=True, exist_ok=True)
                    violation.save(self.config.output_dir / f"ce_{len(self.counterexamples)}.pt")
            
            # Add to corpus if interesting
            if violation or coverage_delta > 0:
                new_seed = FuzzingSeed(
                    tensor=candidate_single.cpu(),
                    label=seed.label,
                    energy=energy,
                    depth=seed.depth + 1,
                    parent_id=seed.id
                )
                self.seed_corpus.add(new_seed)
        
        self.iterations = start_iteration + batch_size
    
    def _fuzz_iteration(self, iteration: int):
        """Single fuzzing iteration with optional tracing."""
        # 1. Select seed
        seed = self.seed_corpus.select()
        
        # 2. Get seed tensor (already has batch dimension)
        seed_tensor = seed.tensor  # Already (1, C, H, W)
        
        # 3. Mutate with feedback (pass labeled_tensor)
        candidate = self.mutation_engine.mutate(seed)
        mutation_strategy = self.mutation_engine.last_strategy
        
        # 4. Run inference
        with torch.no_grad():
            output_dict = self.model(candidate)
        
        # Handle VerifiableModel output (dict) or plain tensor
        if isinstance(output_dict, dict):
            output = output_dict['output']
        else:
            output = output_dict
        
        # 5. Check violation
        violation = self.property_checker.check(
            input_tensor=candidate,
            output=output,
            label=seed.label,
            seed_tensor=seed.tensor
        )
        
        # 6. Update coverage
        activations = self.mutation_engine.get_activation_map()

        
        coverage_delta = self.coverage_tracker.update(candidate, activations)
        coverage = self.coverage_tracker.get_coverage()
        
        # 7. Compute energy
        if violation or coverage_delta > 0:
            energy = self._compute_energy(coverage_delta, violation is not None)
        else:
            energy = 0.0
        
        # 8. UNIFIED TRACING HOOK (all levels)
        if self.tracer and self.tracer.should_trace(iteration):
            # Collect gradients only if Level 3
            gradients = None
            loss_value = None
            if self.config.trace_level >= 3:
                gradients = self.mutation_engine.get_last_gradients()
                loss_value = self.mutation_engine.get_last_loss()
            
            # Single tracing call - tracer handles level-specific storage
            self.tracer.record_iteration(
                iteration=iteration,
                timestamp=time.time(),
                mutation_strategy=mutation_strategy,
                violation=violation,
                coverage=coverage,
                coverage_delta=coverage_delta,
                energy=energy,
                seed_id=seed.id,
                # Level 1+ data
                input_before=seed_tensor,
                input_after=candidate,
                parent_id=seed.parent_id,
                depth=seed.depth,
                # Level 2+ data
                activations=activations,
                # Level 3+ data
                gradients=gradients,
                loss_value=loss_value
            )
        
        # 9. Handle results
        if violation:
            self.counterexamples.append(violation)
            # Only print individual counterexamples in debug mode (verbose >= 2)
            # Regular reporting happens every report_interval in _print_progress
            if self.config.verbose >= 2:
                print(f"🚨 Counterexample #{len(self.counterexamples)}: {violation.summary()}")
            
            if self.config.save_counterexamples:
                self.config.output_dir.mkdir(parents=True, exist_ok=True)
                violation.save(self.config.output_dir / f"ce_{len(self.counterexamples)}.pt")
        
        # 10. Add to corpus if interesting
        if violation or coverage_delta > 0:
            new_seed = FuzzingSeed(
                tensor=candidate.cpu(),
                label=seed.label,
                energy=energy,
                depth=seed.depth + 1,
                parent_id=seed.id
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
        """Print fuzzing progress with incremental counterexample count."""
        elapsed = time.time() - self.start_time
        iter_per_sec = iteration / elapsed if elapsed > 0 else 0
        coverage = self.coverage_tracker.get_coverage()
        
        # Calculate new counterexamples since last report
        ce_total = len(self.counterexamples)
        ce_new = ce_total - self.last_report_ce_count
        self.last_report_ce_count = ce_total
        
        print(f"📊 Iteration {iteration:6d} | "
              f"Coverage: {coverage:6.2%} | "
              f"Seeds: {len(self.seed_corpus):4d} | "
              f"Violations: {ce_total:3d} (+{ce_new}) | "
              f"Speed: {iter_per_sec:5.1f} it/s")
    
    def _generate_report(self) -> FuzzingReport:
        """Generate final report."""
        total_time = time.time() - self.start_time

        # Neurons that were never activated across all iterations (per coverage tracker definition)
        never_activated_neurons: List[Tuple[str, int]] = []
        never_activated_count = 0
        try:
            uncovered = self.coverage_tracker.get_uncovered_neurons()
            never_activated_count = len(uncovered)
            # Deterministic small sample for logs/report
            never_activated_neurons = sorted(list(uncovered))[:20]
        except Exception:
            # Fallback if tracker doesn't support uncovered queries
            never_activated_count = 0
            never_activated_neurons = []
        
        report = FuzzingReport(
            total_iterations=self.iterations,
            total_time=total_time,
            counterexamples=self.counterexamples,
            neuron_coverage=self.coverage_tracker.get_coverage(),
            total_mutations=self.mutation_engine.total_mutations,
            seeds_explored=len(self.seed_corpus),
            num_of_never_activated_neurons=never_activated_count,
            never_activated_neurons=never_activated_neurons,
        )
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"🎉 ACTFuzzer completed in {total_time:.1f}s")
        print(f"   Iterations: {report.total_iterations}")
        print(f"   Counterexamples: {len(report.counterexamples)}")
        print(f"   Coverage: {report.neuron_coverage:.2%}")
        print(f"   Seeds explored: {report.seeds_explored}")
        print(f"   Never-activated neurons: {report.num_of_never_activated_neurons}")
        if report.never_activated_neurons:
            sample_str = ", ".join([f"{ln}[{i}]" for (ln, i) in report.never_activated_neurons[:10]])
            print(f"   Never-activated sample: {sample_str}")
        print(f"{'='*80}\n")
        
        if self.config.save_counterexamples and report.counterexamples:
            report.save(self.config.output_dir)
        
        # Close tracer if enabled
        if self.tracer:
            self.tracer.close()
        
        return report