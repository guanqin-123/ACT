# Conv Mode Migration Guide

## Who this is for

This note is for contributors touching ACT's dual forward registry or adding a
new operator that may encounter `Patches` inputs.

## Rule 1: use the dispatcher for Conv-sensitive forward paths

Use `dispatch_conv_forward(...)` when:

- the operator is `CONV2D`
- the caller may run under `backend.conv_mode=patches`
- you want fallback, throttling, strict mode, and materialization accounting to
  stay centralized

Do **not** open-code ad hoc logic like:

```python
if isinstance(bounds, Patches):
    ...
else:
    ...
```

unless the file is itself the dispatcher or the operator-specific patches
implementation.

Direct `tf_cnn.forward_conv2d(...)` calls are only appropriate when the caller
already knows it is intentionally staying on the matrix path.

## Rule 2: preserve `Patches` where it is natural

When adding a new forward op to `DualTF._FORWARD_REGISTRY`:

1. register the handler in `_FORWARD_REGISTRY`
2. decide whether the op can preserve `Patches` soundly
3. add an explicit `isinstance(bounds, Patches)` branch if yes
4. materialize only where representation boundaries are intentional

Current intentional materialization points:

- Conv dispatcher fallback from matrix-seeded input
- pool ops in v1
- dense boundary
- BaBSR scoring when a dense lA tensor is required

## Rule 3: dense boundary only

For Conv-related sparse propagation, keep the mental model simple:

- Conv / skip / ReLU / BN can keep sparse structure where implemented
- Dense is the boundary where sparse conv structure becomes ordinary matrix form

If a change materializes earlier than Dense, it should normally:

- log clearly
- be covered by a test
- justify why the earlier boundary is required

## Rule 4: parity and soundness are separate tests

Every new patches-aware operator should get both:

### Parity

Compare patches vs matrix outputs on the same fixture.

- float64 preferred
- target tolerance: `rtol=1e-5`, `atol=1e-7`

### Soundness

Sample 100 random `x` values inside the input box and assert:

- `LB <= f(x)`
- `f(x) <= UB`

Parity catches regressions. Soundness catches bugs.

## Rule 5: deterministic mode is opt-in

If a helper uses `as_strided` or another non-deterministic shortcut:

- keep the fast path explicit
- provide a deterministic guard where appropriate
- do not silently change hot-path behavior for all users

Use `deterministic_patches_ctx(...)` in tests when you need deterministic
behavior or warning coverage.

## Checklist for adding a new op

- add the forward handler to `DualTF._FORWARD_REGISTRY`
- keep `_BACKWARD_REGISTRY` in sync if backward semantics exist
- preserve `Patches` if feasible and auditable
- otherwise materialize explicitly and log why
- add parity coverage
- add 100-sample soundness coverage
- run matrix and patches-mode tests before merging

## Common mistake

If you see:

`dispatch_conv_forward received LinearBound input`

that usually means the path was not Patches-seeded upstream yet. In Tier 1 that
is expected for the first Conv. Do not hide the fallback; test and document it.
