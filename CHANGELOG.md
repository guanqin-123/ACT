# Changelog

## [Unreleased] — branch: bab

### Added

- Patches infrastructure for sparse Conv2D linear-form propagation in the dual
  verifier
- central bounds dispatcher for matrix vs patches routing and mixed-mode safety
- `backend.conv_mode` configuration with explicit dispatcher fallback behavior
- deterministic patches context hardening and operator-wide soundness sweep
- realistic CIFAR-100 ResNet-medium benchmark and verification harness
- 205+ new tests across Waves 1-6, including parity, soundness, and end-to-end
  ResNet coverage

### Changed

- default `conv_mode` changed from `matrix` to `patches`
- this is intended to be performance-neutral for matrix-seeded paths: when a
  Conv receives `LinearBound`, dispatcher fallback materializes explicitly and
  matrix-only code paths keep their previous semantics
- warning spam from repeated dispatcher/BaBSR materializations is now throttled
  and summarized

### User-facing summary of Waves 1-5

- Wave 1: shared state API, Patches scaffold, dispatcher skeleton
- Wave 2: full Patches helpers, rank-3 bounds fan-out, config toggle
- Wave 3: Conv2D patches kernels, parity/soundness coverage, patches default
- Wave 4: per-op patches support for ReLU, BN, Add, Concat, pool, dense boundary
- Wave 5: ResNet parity/soundness tests and CIFAR-100 A/B benchmark harness

### Breaking changes

- none; all changes are additive from a verifier-API perspective

### Known limitations

- see `docs/patches_porting.md` for explicit v1 limitations and deferred items
