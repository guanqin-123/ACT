# Patches Porting Guide

## Background

ACT's dual verifier originally represented every Conv2D linear form as a dense
`LinearBound` matrix. That is simple and robust, but it is memory-heavy on
image models because a conv operator is locally sparse while the matrix view is
globally dense. α-β-CROWN avoids this blow-up with a `Patches` representation:

- instead of materializing a full `[rows, input_features]` matrix,
- it stores local receptive-field coefficients as
  `(out_c, B, out_h, out_w, in_c, k_h, k_w)`,
- and only materializes when a downstream consumer genuinely needs a dense
  matrix view.

This port brings the same idea into ACT's dual stack so that Conv→Conv chains,
ResNet skips, and mixed-mode dispatcher paths can keep sparse structure longer.

In practice, the win is memory realism rather than a semantic change:

- Conv hot paths preserve soundness.
- Dense baselines still exist and remain the fallback.
- The dispatcher centralizes mixed-mode behavior so materialization is visible
  instead of silent.

## What shipped in Waves 1-5

Waves 1-5 introduced:

- a `Patches` dataclass and helper API in `act/back_end/patches.py`
- a central dispatcher in `act/back_end/bounds_dispatch.py`
- zero-copy rank-3 spec expansion for bounds-only batching
- patches-aware Conv2D forward/backward kernels
- per-op support for ReLU, BN, Add, Concat, pool materialization, and dense
  boundary fallback
- ResNet parity/soundness tests and a realistic CIFAR-100 benchmark harness

Wave 6 does not change hot-path math. It finalizes documentation, deterministic
mode hardening, and soundness coverage.

## Locked scope decisions (S1-S4)

These are copied verbatim from the Wave 6 plan.

| ID | Decision | Rationale |
|---|---|---|
| **S1** | v1 only supports CIFAR-100 ResNet-medium Conv regime: **stride ∈ {1,2}, padding ∈ {0,1}, no dilation, groups=1** | Extending to generic Conv doubles work; current benchmark only needs this |
| **S2** | `identity` fast-path goes to **v1** (NOT deferred) | Oracle flagged ResNet skip-connection as first-order perf issue, not polish |
| **S3** | α-sensitive BaB runs **force `spec_chunk_size=None`**; chunked path only emits `log.warning` | CIFAR-100 M=99 @ batch=256 with patches fits 16GB (Oracle estimate) |
| **S4** | Deterministic mode uses **explicit opt-out ctx manager** for `as_strided` in `patches_to_matrix` | Only in test/debug code; hot path uses `F.conv_transpose2d` (natively deterministic) |

## Architecture overview

### 1. `bounds_dispatch.py` is the central switchboard

All consumers are expected to route Conv-sensitive logic through dispatcher
entrypoints such as:

- `dispatch_conv_forward`
- `dispatch_bn_forward`
- `dispatch_add_forward`
- `dispatch_concat_forward`
- `dispatch_pool_forward`

This is intentional. Direct `isinstance(bounds, Patches)` checks scattered
across unrelated files would make mixed-mode behavior impossible to audit.

### 2. Matrix vs patches is a representation choice, not a semantic fork

- **matrix mode** keeps classic `LinearBound`
- **patches mode** preserves `Patches` wherever the current v1 implementation
  can do so soundly
- **mixed mode** is explicit: warnings, counters, and summaries show when ACT
  had to materialize and why

The dispatcher keeps these cases centralized so tests can assert parity and
warning behavior.

### 3. Dense is the materialization boundary

The current rule is simple and deliberate:

- Conv-like structure stays sparse when possible.
- Once the flow crosses the Conv→Dense boundary, `Patches` materializes back to
  dense matrix form.

This mirrors α-β-CROWN's practical convention. It avoids infecting fully dense
MLP logic with sparse-conv bookkeeping that buys little in late layers.

### 4. Pooling is v1 materialize-on-entry

`MAXPOOL2D` and `AVGPOOL2D` currently materialize patch inputs before applying
interval pool bounds. This is an explicit v1 choice, not a hidden regression.
For the target ResNet-medium benchmark this is acceptable because the only hot
pool is near the dense boundary.

## Tuning knobs

### `backend.conv_mode`

Configured in backend config and read by `bounds_dispatch`.

- `matrix`: baseline dense behavior
- `patches`: preserve sparse conv structure where supported

The default on branch `bab` is now `patches`.

Important nuance:

- this default is additive, not a matrix-path rewrite
- if the incoming representation is already `LinearBound`, dispatcher fallback
  materializes explicitly and continues soundly
- matrix-only code paths do not regress just because the default changed

### `strict_patches`

`strict_patches=True` turns materialization into a hard error. Use it when you
want to prove a path is still fully sparse end-to-end.

Good uses:

- debugging first-conv densification
- asserting benchmark harness invariants
- preventing accidental mixed-mode regressions during development

### `deterministic_patches_ctx`

`deterministic_patches_ctx()` is the explicit guard for deterministic testing.

What it now does:

- toggles the thread-local patches determinism flag
- snapshots and restores `torch.use_deterministic_algorithms(...)`
- preserves deterministic environment overrides such as
  `PYTHONHASHSEED=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- emits a `DeprecationWarning` when deterministic mode forces `F.unfold`
  instead of the `as_strided` fallback path

Use it in tests and debugging, not as a blanket performance toggle.

Example:

```python
from act.back_end.patches import deterministic_patches_ctx, inplace_unfold

with deterministic_patches_ctx(True):
    cols = inplace_unfold(x, kernel_size=3, stride=1, padding=1)
```

## Warning and materialization summaries

Tier 1 adds first-touch-only throttling for the two noisiest sites:

- Conv dispatcher fallback warnings
- BaBSR lA patch materialization warnings

Behavior:

- first `(site, layer_id)` hit logs at `warning`
- subsequent hits downgrade to `debug`
- a summary table is emitted on reset and optionally at process exit

Example summaries:

```text
[bounds_dispatch] conv materializations during run:
  layer_id=2  site=dispatch_conv_forward  count=18
[bab.branching.babsr] lA materializations during run:
  layer_id=2 count=9
```

This keeps `[BaB]` output readable while preserving aggregate visibility.

## Known limitations in v1

These items are intentionally out of scope for the current merge:

1. Dilation > 1 Conv
2. Grouped Conv (`groups > 1`)
3. Conv transpose, 3D Conv, and 1D Conv patches coverage
4. `_chunked_eval` full α warm-start fix
5. Smooth-split `-η·sign·p` constant-term follow-up
6. Independent α-intermediate / α-final optimization
7. Instance-level batching `verify_bab(nets=[...])`
8. `EtaState` multi-slot support for smooth-activation resplit
9. `lr_eta` → `lr_joint` rename

Also explicit non-goals for v1 patches propagation:

- no end-to-end sparse path through a matrix-seeded `InputSpec` yet
- no pool-native sparse kernel yet
- no Tier 2 kernel rewrites in this merge-prep tier

## Troubleshooting

### `dispatch_conv_forward: conv_mode=patches received LinearBound input; materializing and falling back to matrix path`

Meaning:

- the graph entered the conv dispatcher with a dense `LinearBound`
- patches mode was requested
- ACT stayed sound by materializing and taking the matrix path

Most common cause today:

- the input spec is still matrix-seeded, so the first Conv cannot start from a
  sparse `Patches` seed yet

Action:

- this is expected in Tier 1
- see Tier 2 work for input-spec seeding if you need end-to-end sparse Conv

### `BaBSR select_neurons: materializing patches lA for layer X`

Meaning:

- BaBSR scoring needed a dense tensor view for per-neuron ranking
- the sparse lA structure stayed sound, but the scoring pass materialized it

Tier 1 throttles this warning so it fires once per layer and then moves to
debug-level logging.

### Pool warnings in patches mode

Warnings from `forward_maxpool2d` or `forward_avgpool2d` mean ACT took the
documented v1 materialize-on-entry path. This is expected.

### `strict_patches=True` raises immediately

That means ACT found a representation boundary that still requires dense
materialization. The failure is informative: it identifies exactly where sparse
propagation stops.

## Contributor notes

- prefer dispatcher entrypoints over direct mixed-mode branching
- keep soundness separate from parity in tests
- when adding a new patches-capable operator, preserve sparse structure only if
  the implementation stays auditable and the fallback remains explicit

If you need a shorter contributor-oriented checklist, see
`docs/conv_mode_migration.md`.
