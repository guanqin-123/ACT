
# act.back_end — Torch‑Native DNN Verification

A concise, modular verification toolkit that does **all analysis in PyTorch** (for
ergonomics and optional GPU). Two verification pipelines share the same `Net` IR:

- **MILP/LP** (`verify_once`): IntervalTF / HybridzTF → constraint templates →
  NumPy boundary → Gurobi or TorchLP. Complete + exact CE.
- **Dual + BaB** (`verify_bab`): DualTF (forward bounds) + DualSolver (certified
  dual backward + joint α/η KKT Adam) + BaBSR branching. Incomplete but
  differentiable and cheap enough for thousands of node evaluations.

Supports MLPs, CNNs, RNNs, and key Transformer components via per-kind transfer
function registries.

---

## Why this design?

- **Torch inside**: easy interop with models, layers, and CE validation; one tensor world.
- **Single export boundary**: solver backends remain NumPy‑facing; plug in Gurobi/HiGHS/CBC, etc.
- **DAG‑aware worklist**: efficient, incremental propagation with caching and change detection.
- **Layer coverage**: MLP + less‑common activations + Transformer blocks with sound relaxations.
- **Refinement loop**: BaB + CE validation to distinguish true vs. false counterexamples.

---

## Core Data Structures

```python
@dataclass(eq=True, frozen=True)
class Bounds:      # Box bounds for a contiguous vector of variables
    lb: torch.Tensor
    ub: torch.Tensor

@dataclass(eq=False)
class Con:         # Canonical constraint template (solver-agnostic)
    kind: str                  # 'EQ' | 'INEQ' | 'BIN'
    var_ids: Tuple[int, ...]   # variable ids referenced by this constraint
    meta: Dict[str, Any]       # parameters (Torch tensors allowed) + {'tag': str}
    # Optional numeric payloads (unused in-core; only for compatibility)
    A: Any=None; b: Any=None; C: Any=None; d: Any=None
    def signature(self) -> Tuple[str, Tuple[int, ...], str]: ...

@dataclass
class ConSet:       # Replace-by-signature semantics
    S: Dict[Tuple[str, Tuple[int, ...], str], Con]
    def replace(self, c: Con): ...
    def add_box(self, layer_id: int, var_ids: List[int], B: Bounds): ...

@dataclass
class Fact:         # Dataflow fact per layer: (bounds, constraints)
    bounds: Bounds
    cons: ConSet

@dataclass
class Layer:
    id: int
    kind: str       # e.g. 'DENSE', 'RELU', 'SOFTMAX', ...
    params: Dict[str, Any]
    in_vars: List[int]
    out_vars: List[int]
    cache: Dict[str, Any]      # prev_lb/prev_ub/masks (for change detection)

@dataclass
class Net:          # Topo-ordered DAG
    layers: List[Layer]
    preds: Dict[int, List[int]]
    succs: Dict[int, List[int]]
    by_id: Dict[int, Layer] = field(init=False)
```

- **`meta['tag']`** drives export logic (e.g. `relu:{id}`, `dense:{id}`, `softmax:simplex:{id}`, …).
- Constraints are **templates** kept compact during analysis, then materialized at export time.

---

## Device & Dtype Policy

- Analysis uses **Torch** everywhere with `DEFAULT_DEVICE` (CPU/GPU) and `DEFAULT_DTYPE` (e.g. `float32`).
- Helpers:
  - `as_t(x, device, dtype)` converts inputs and parameters consistently.
  - `@torch.no_grad()` guards analysis & CE validation paths.
- Export converts tensors to **NumPy float64** in one place: `exporter.to_numpy(x)`.

---

## Supported Layers

Canonical layer kinds are defined in `layer_schema.py` (enum `LayerKind`). The three
active transfer functions (IntervalTF, HybridzTF, DualTF) share this schema; coverage
per-mode is advertised via `supports_layer(kind)`.

### Structural / shape
`INPUT, INPUT_SPEC, ASSERT, FLATTEN, RESHAPE, TRANSPOSE, SQUEEZE, UNSQUEEZE, ADD, CONCAT`

### MLP basics
`DENSE, BIAS, SCALE, BN, RELU, LRELU, ABS, CLIP, MUL`

### Smooth activations
`SIGMOID, TANH, SOFTPLUS, SILU, GELU, SQUARE, POWER, MAX, MIN`

### CNN
`CONV2D, MAXPOOL2D, AVGPOOL2D`

### RNN
`LSTM, GRU`

### Transformer
`EMBEDDING, POSENC, LAYERNORM, ATT_SCORES, SOFTMAX, ATT_MIX, MHA_SPLIT, MHA_JOIN, MASK_ADD`

Each TF implementation registers per-kind handlers. Handler contracts differ by mode:

- **IntervalTF / HybridzTF**: handler returns `Fact(bounds, cons)` where `cons` are
  solver-agnostic **template constraints** (`meta['tag']` drives downstream materialization).
- **DualTF**: handler returns `Fact(bounds, ConSet())` — no constraints. Bounds propagation
  only. The dual certified bound is computed separately by `DualSolver` using the per-kind
  **backward kernels** registered in `DualTF._BACKWARD_REGISTRY`.

---

## Bounds Propagation (Worklist, DAG‑aware)

- Start from the **entry layer**: seed bounds with the input spec (box / ℓ∞ ball).
- Use a **worklist** (queue) of layer ids.
- For a layer `L`:
  1. **Join** predecessors’ `after` bounds into `before[L]`.
  2. Run `dispatch_tf(L, before, after, net)` ⇒ `out_fact`.
  3. If changed (bounds or masks), update caches, merge constraints, push successors.
- Termination when the worklist empties (monotone joins + finite precision).

**Performance**: vectorized Torch ops, caching, and dedup constraints (replace-by-signature).

---

## Constraint Export (Solver‑Agnostic)

- `export_to_solver(globalC, solver)` performs the **only** Torch→NumPy conversion.
- Merges all `box:*` constraints into global variable bounds.
- Materializes per‑layer templates:
  - `dense:` → linear equalities (y = Wx + b)
  - `relu:` → phase splits (on/off) and convex relaxation for ambiguous
  - `mcc:`  → McCormick envelopes for bilinear terms (e.g., `MUL`, attention mixes)
  - `softmax:simplex:` → probability simplex per row (≥0, sum=1)
  - etc.
- Backends implement `Solver` (see next section).

---

## Solvers

Two families of solvers coexist; they serve different verification modes.

### (1) Constraint-based MILP/LP family — used by `verify_once`

NumPy-facing; consumes the constraint templates built up during `analyze()`.

```python
class Solver:  # solver/solver_base.py
    def begin(...); def add_vars(n); def set_bounds(idxs, lb, ub)
    def add_lin_eq(...); def add_lin_le(...); def add_lin_ge(...)
    def add_sum_eq(...); def add_ge_zero(...); def add_sos2(...)
    def set_objective_linear(..., sense="min"); def optimize(tlim=None)
    def status() -> str; def has_solution() -> bool; def get_values(ids) -> np.ndarray
    @property def n(self) -> int
```

Implementations:
- **`GurobiSolver`** (`solver_gurobi.py`) — MILP, complete + exact
- **`TorchLPSolver`** (`solver_interval.py`) — PyTorch-native LP, no external license
- **`HZSolver`** (`solver_hz.py`) — zonotope operations (used by HybridzTF, not a standalone verifier)

### (2) Certified-bound family — used by `verify_bab`

Torch-native; consumes `DualTF` forward bounds + backward kernels. No constraint
materialization; no NumPy boundary.

- **`DualSolver`** (`solver_dual.py`) — Wong-Kolter dual certified bounds with optional
  joint α/η KKT Adam refinement and warm-start. Key API:
  - `compute_bound(net, bounds_dict, c, n_iters, warm_alphas, warm_etas) -> Tensor[B]`
  - `compute_robust_bound(net, bounds_dict, y_true, num_classes, ...) -> (min_slack, certified, margins, out_alphas, out_etas)`
  - `compute_linear_bound(net, bounds_dict, C, d, ...) -> (...)` for polytope specs
  - `evaluate_spec(net, bounds_dict, out_spec, num_classes, ...) -> SpecBatchResult`

### Why two pipelines

| | MILP (Gurobi/TorchLP) | Dual (DualSolver) |
|---|---|---|
| Completeness | Complete (exact CE) | Incomplete (sound LB only) |
| Differentiable | No | Yes — α/η optimized via Adam |
| Cost per call | High | Low (single backward pass) |
| Used by | `verify_once` | `verify_bab` (per-node) |
| CE recovery | From LP solution | From concrete forward with violating x\* |

Collapsing them would cost either completeness (if BaB routed through Gurobi, too slow)
or exactness of falsification (if `verify_once` routed through DualSolver, no CE).

---

## Input / Output Specs

**InputSpec**: `BOX`, `LINF_BALL`, or extra **linear polytope** constraints (`A x ≤ b`).  
**OutputSpec** (negated for searching counterexamples):
- `LINEAR_LE`: find `c^T y ≥ d + ε`
- `TOP1_ROBUST`: find a class with `y_j ≥ y_true`
- `MARGIN_ROBUST`: find `y_j − y_true ≥ δ`

`verify_once(...)`:
1. Run `analyse()` to collect bounds + constraint templates.
2. Export to solver, add input spec and **negated** output spec.
3. Optimize (optionally maximizing violation) →
   - **INFEASIBLE** ⇒ `CERTIFIED`
   - **FEASIBLE** ⇒ return **counterexample** (input/output witness).

---

## Branch‑and‑Bound (BaB) + CE Validation

`verify_bab` (`bab/bab.py`) routes every node through **`DualSolver`** — never through
Gurobi/TorchLP. BaB consumes forward bounds from `DualTF`, computes a certified lower
bound via the dual backward pass, and optionally refines that bound with joint α/η Adam
iterations (`n_iters > 0`, `force_kkt`).

### Node processing (batched)

1. **Forward**: `DualTF.compute_forward_bounds(net, lb, ub, eta_state)` with the per-node
   η-clamp active on any split neurons (sign convention: `+1 = INACTIVE`, `-1 = ACTIVE`).
2. **Backward + certify**: `DualSolver.evaluate_spec(net, bounds_dict, out_spec, ...)`
   returns `SpecBatchResult` with `(min_slack, certified, margins, out_alphas, out_etas)`.
   Warm-starts α/η from the parent node.
3. **Parent-margin safety clamp**: after α changes during optimization, child bound ≥
   parent bound is not guaranteed (α is not monotone across sub-problems). BaB falls back
   to the parent bound when a child regresses — the child's feasible region is a subset
   of the parent's, so the parent bound is always a valid lower bound.
4. **CE candidate extraction**: on non-certified rows, extract violating inputs via
   `_extract_ces()` and validate with a concrete model forward
   (`check_violation_at_point_batched`). True violation → `FALSIFIED`; otherwise treat as
   a dual-gap artifact and proceed to branch.
5. **Branch**: BaBSR scoring (`bab/branching/babsr.py`) picks the splitting neuron;
   children enqueued via `BoundingStrategy` (BFS default).
6. **Optional trace**: `BaBConfig.record_bound_trace=True` records per-(subproblem, bab_iter)
   Adam trajectory for diagnostics; see `bab/trace.py`.

Termination: `CERTIFIED` if no true CE surfaces within `time_budget_s` / `max_nodes`.

### Why no Gurobi in BaB

BaB runs the solver thousands of times per problem. DualSolver is single-backward-pass
cheap, differentiable (for α/η optimization), and sound. MILP completeness would collapse
throughput — and is unnecessary because BaB's branching itself closes the dual gap.

---

## Extending the System

- **New layer kind**: (1) add to `LayerKind` in `layer_schema.py`; (2) add a per-kind
  handler in every active TF module (`interval_tf/tf_*.py`, `hybridz_tf/tf_*.py`, and —
  for dual-path coverage — a `forward_*` + `backward_*` pair in `dual_tf/tf_*.py`
  registered in both `_FORWARD_REGISTRY` and `_BACKWARD_REGISTRY`); (3) add the
  materialization case in `cons_exportor.py` for MILP/LP coverage.
- **New constraint-based solver**: subclass `Solver` in `solver/solver_base.py`.
- **Advanced relaxations**: store extra parameters in `Con.meta` (Torch tensors)
  and materialize in `cons_exportor.py`. For dual-path relaxations, extend the
  backward kernel and register in `DualTF._BACKWARD_REGISTRY`.
- **Registry invariant**: `DualTF._FORWARD_REGISTRY` and `_BACKWARD_REGISTRY` MUST
  share identical keysets. An assertion at module import enforces this.

---

## Performance Tips

- Set a **global device** (CPU/GPU) and dtype up front via
  `act.util.device_manager.initialize_device("cuda"|"cpu", "float32"|"float64")`.
- Use `@torch.no_grad()` (or `torch.no_grad()` blocks) in **MILP** analysis & CE
  validation. For the **dual path**, grad flow is deliberate — `DualSolver` uses a
  conditional context (`nullcontext` when α/η is trainable, `torch.no_grad` otherwise).
  Do not re-add an unconditional `@torch.no_grad()` decorator to
  `compute_forward_bounds` or `DualSolver._backward_pass`; α optimization silently
  becomes a no-op.
- BaB is batched by default via `SubproblemBatch`; tune `subproblem_batch_size` to GPU
  memory.
- Keep MILP constraints minimal; rely on `box:*` and a few tight relaxations.
- Adam is secondary; BaB branching is the primary bound-tightening tool. Keep
  `n_iters`/`joint_iters` modest (default 10) and rely on plateau early-stop
  (`threshold=5e-4`, `patience=2`).
- Watch BLAS threading: Torch vs. Gurobi (set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`
  if needed).

---

## Minimal Usage Sketch

The back-end is spec-free: `INPUT_SPEC` and `ASSERT` layers are embedded in the `Net`
and `verify_once` / `verify_bab` read them directly. The CLI (`python -m act.back_end
--verify --network ... [--bab]`) is the usual entry point; for a programmatic example
see `act/back_end/examples/` or `ipynb/bab_neuron_split_demo.ipynb`.

### MILP pipeline — `verify_once`

```python
from act.back_end import GurobiSolver, verify_once
from act.back_end.serialization.serialization import load_net_from_file

net = load_net_from_file("act/back_end/examples/nets/mnist_robust_easy.json")
result = verify_once(net, solver=GurobiSolver(), timelimit=60.0)
print(result.status)            # CERTIFIED / FALSIFIED / UNKNOWN / TIMEOUT
```

Swap `GurobiSolver()` for `TorchLPSolver()` (`from act.back_end.solver import
TorchLPSolver`) if you don't have a Gurobi license — weaker relaxation but no external
dependency.

### Dual + BaB pipeline — `verify_bab`

```python
from act.back_end import verify_bab
from act.back_end.solver import DualSolver
from act.back_end.dual_tf import DualTF
from act.back_end.config import BaBConfig

dual_solver = DualSolver(DualTF(), n_iters=10)  # α/η Adam iterations per node
config = BaBConfig(
    branching_method="babsr",
    bounding_method="bfs",
    subproblem_batch_size=16,
    eta_iters=10, lr_eta=0.05,
    time_budget_s=100.0,
    record_bound_trace=False,
)
result = verify_bab(net, solver=GurobiSolver(), dual_solver=dual_solver, config=config)
```

The `solver=` argument is retained for CE-extraction fallback but BaB's hot path is
`dual_solver`. Set `record_bound_trace=True` to collect per-subproblem Adam trajectories
in a `BoundTrace` (see `bab/trace.py`).

---

## File Layout (actual)

```
act/back_end/
  core.py                    # Bounds, Con, ConSet, Fact, Layer, Net
  utils.py                   # affine_bounds, validate_constraints, box_join
  analyze.py                 # worklist-based bounds propagation
  verifier.py                # verify_once() → analyze + export + MILP/LP solve
  transfer_functions.py      # TF base, dispatch registry, mode selection
  cons_exportor.py           # template → MILP/LP materialization (NumPy boundary)
  layer_schema.py            # LayerKind enum + per-kind param/meta schemas
  layer_util.py              # layer construction + validation
  net_factory.py             # YAML-driven network factory
  cli.py  __main__.py        # `python -m act.back_end` entry point

  interval_tf/               # Fast interval TF + template constraints
    interval_tf.py, tf_mlp.py, tf_cnn.py, tf_rnn.py, tf_transformer.py

  hybridz_tf/                # Zonotope-enhanced TF (falls back to interval)
    hybridz_tf.py, tf_mlp.py, tf_cnn.py, tf_rnn.py, tf_transformer.py

  dual_tf/                   # CROWN-style forward bounds + dual backward kernels
    dual_tf.py               # DualTF class + _FORWARD_REGISTRY + _BACKWARD_REGISTRY
    tf_forward.py            # LinearBound, dual-track forward, concretization
    tf_mlp.py  tf_cnn.py  tf_smooth.py  tf_rnn.py  tf_transformer.py

  solver/
    solver_base.py           # Solver, SolverCaps, SolveStatus
    solver_gurobi.py         # GurobiSolver (MILP, license-gated)
    solver_interval.py       # TorchLPSolver (PyTorch-native LP)
    solver_hz.py             # HZono operations (used by HybridzTF)
    solver_dual.py           # DualSolver — certified bounds + α/η Adam KKT
    spec_batching.py         # SpecBatch helper for batched spec evaluation

  bab/
    bab.py                   # verify_bab, _verify_bab_batched, CE validation
    node.py                  # BabNode, SubproblemBatch, split_subproblems
    eta.py                   # EtaState, η-clamp helpers
    trace.py                 # BoundTrace (per-subproblem Adam trajectory)
    branching/
      branching.py           # BranchingStrategy base
      bounding.py            # BoundingStrategy, BFSBounding
      babsr.py               # BaBSR branching heuristic

  serialization/             # NetSerializer (JSON round-trip)
  examples/                  # YAML-defined example networks + generated nets/
```

---

## License & Notes

- This blueprint focuses on clarity, extensibility, and solver portability.
- Replace/extend relaxations as needed for tighter bounds (CROWN, triangle relaxations, etc.).
- Backends other than Gurobi are straightforward once the `Solver` interface is implemented.
