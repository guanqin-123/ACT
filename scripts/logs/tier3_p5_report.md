# Tier 3 Phase 5 Empirical Report

**Branch HEAD**: 6aec8cc
**Date**: 2026-04-25T09:01:17Z
**Instance**: 3995 (CIFAR-100 ResNet-medium prop_idx_3995_sidx_7978_eps_0.0039)
**Configuration**: matrix mode, batch 16, budget 60s

## Pre-Tier-3 baseline
- verdict: UNKNOWN
- wall-clock: 60.73s
- nodes: 495
- peak VRAM: 83.3 GB

## --alpha-split ON
- verdict: UNKNOWN
- wall-clock: 64.64s
- nodes: 495
- peak VRAM: 83.3 GB

## α-β-CROWN reference
- verdict: CERTIFIED
- wall-clock: 43.5s
- nodes: 516

## Per-layer width tightening ratio
```
baseline keys: 43 on layers
alpha-split keys: 43 on layers
baseline verdict: UNKNOWN wall: 60.73s nodes: 495
alpha-split verdict: UNKNOWN wall: 64.64s nodes: 495

per-layer tightening ratio (median, max, min, n):
  L0: median=+0.0000 max=+0.0000 min=+0.0000 n=1000
  L1: median=+0.0000 max=+0.0000 min=+0.0000 n=1000
  L2: median=+0.0000 max=+0.0000 min=+0.0000 n=1000
  L3: median=+0.0000 max=+0.0000 min=+0.0000 n=1000
  L4: median=+0.0010 max=+0.0473 min=-0.0395 n=1000
  L5: median=+0.0000 max=+0.0311 min=-0.0412 n=1000
  L6: median=+0.0000 max=+0.0091 min=-0.0085 n=1000
  L7: median=+0.0000 max=+0.0000 min=+0.0000 n=1000
  L8: median=+0.0000 max=+0.0042 min=-0.0039 n=1000
  L9: median=+0.0789 max=+0.2027 min=+0.0202 n=1000
  L10: median=+0.0000 max=+0.0020 min=-0.0023 n=1000
  L11: median=+0.0000 max=+0.3333 min=-1.0000 n=809
  L12: median=+0.0000 max=+0.0042 min=-0.0039 n=1000
  L13: median=+0.1334 max=+0.5191 min=+0.0000 n=1000
  L14: median=+0.0000 max=+0.0047 min=-0.0030 n=1000
  L15: median=+0.0000 max=+0.3333 min=-0.3333 n=543
  L16: median=+0.0000 max=+0.0042 min=-0.0039 n=1000
  L17: median=+0.3894 max=+0.5979 min=+0.0001 n=1000
  L18: median=+0.0000 max=+0.0054 min=-0.0051 n=1000
  L20: median=+0.0000 max=+0.0042 min=-0.0039 n=1000
  L21: median=+0.0795 max=+0.8420 min=+0.0018 n=1000
  L22: median=+0.0000 max=+0.7938 min=-0.0031 n=1000
  L23: median=+0.0000 max=+0.0392 min=-11.4951 n=1000
  L24: median=+0.0000 max=+0.0227 min=-0.0179 n=1000
  L25: median=+0.0000 max=+0.0065 min=-0.9340 n=1000
  L26: median=+0.5627 max=+3.0385 min=+0.0289 n=1000
  L27: median=-0.0003 max=+0.0182 min=-0.0357 n=1000
  L29: median=+0.0000 max=+0.0065 min=-0.9339 n=1000
  L30: median=+0.6224 max=+1.1273 min=+0.2173 n=1000
  L31: median=-0.0008 max=+0.0123 min=-0.0400 n=1000
  L33: median=+0.0000 max=+0.0065 min=-0.9340 n=1000
  L34: median=+0.5670 max=+1.5750 min=+0.1641 n=1000
  L35: median=-0.0001 max=+0.0800 min=-0.0667 n=1000
  L37: median=+0.0000 max=+0.0065 min=-0.9340 n=1000
  L38: median=+0.0000 max=+0.0065 min=-0.9340 n=1000
  L39: median=+0.5565 max=+0.6322 min=+0.4275 n=1000
  L40: median=-0.0016 max=+0.0003 min=-0.0054 n=1000
  L41: median=-0.0025 max=-0.0018 min=-0.0030 n=1000
  L42: median=-0.0025 max=-0.0018 min=-0.0030 n=1000

overall median tightening (median of per-layer medians): 0.0000

MUST-have (>= 10% median tightening OR CERTIFIED): FAIL
stretch (CERTIFIED in <60s): FAIL
P5_PARTIAL
```

## Conclusion
**MUST-have NOT PASSED** — empirical median tightening below the 10% threshold or verdict still UNKNOWN. See per-layer table for diagnosis.
