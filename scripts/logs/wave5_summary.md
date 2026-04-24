# Wave 5 A/B Benchmark Summary

## Instance 0 (CIFAR100_resnet_medium_prop_idx_3965_sidx_688_eps_0.0039.vnnlib)
| Config | Verdict | Wall-clock | Nodes | Peak VRAM | vs α-β-CROWN |
|---|---|---|---|---|---|
| ACT baseline (matrix) | UNKNOWN | 44.6s | 303 | 83.13 GB | — |
| ACT patches (strict) | ERROR | 0.1s | — | 0.19 GB | — |
| ACT patches (mixed) | UNKNOWN | 126.1s | 239 | 83.09 GB | — |

Notes:
- ACT patches (strict): dispatch_conv_forward: conv_mode=patches received LinearBound input; materializing and falling back to matrix path
- ACT patches (strict): conv materializations=1
- ACT patches (mixed): conv materializations=342

## Instance 1 (CIFAR100_resnet_medium_prop_idx_8028_sidx_5238_eps_0.0039.vnnlib)
| Config | Verdict | Wall-clock | Nodes | Peak VRAM | vs α-β-CROWN |
|---|---|---|---|---|---|
| ACT baseline (matrix) | FALSIFIED | 0.3s | 1 | 5.28 GB | — |
| ACT patches (strict) | ERROR | 0.1s | — | 0.23 GB | — |
| ACT patches (mixed) | FALSIFIED | 0.4s | 1 | 5.30 GB | — |

Notes:
- ACT patches (strict): dispatch_conv_forward: conv_mode=patches received LinearBound input; materializing and falling back to matrix path
- ACT patches (strict): conv materializations=1
- ACT patches (mixed): conv materializations=19

## Instance 2 (CIFAR100_resnet_medium_prop_idx_3995_sidx_7978_eps_0.0039.vnnlib)
| Config | Verdict | Wall-clock | Nodes | Peak VRAM | vs α-β-CROWN |
|---|---|---|---|---|---|
| α-β-CROWN baseline | CERTIFIED | 43.5s | 516 | — | 1.00× |
| ACT baseline (matrix) | UNKNOWN | 44.8s | 303 | 83.17 GB | 1.03× |
| ACT patches (strict) | ERROR | 0.1s | — | 0.23 GB | 0.00× |
| ACT patches (mixed) | UNKNOWN | 126.2s | 239 | 83.13 GB | 2.90× |

Notes:
- ACT patches (strict): dispatch_conv_forward: conv_mode=patches received LinearBound input; materializing and falling back to matrix path
- ACT patches (strict): conv materializations=1
- ACT patches (mixed): conv materializations=342

## Instance 3995 (CIFAR100_resnet_medium_prop_idx_3995_sidx_7978_eps_0.0039.vnnlib)
| Config | Verdict | Wall-clock | Nodes | Peak VRAM | vs α-β-CROWN |
|---|---|---|---|---|---|
| α-β-CROWN baseline | CERTIFIED | 43.5s | 516 | — | 1.00× |
| ACT baseline (matrix) | UNKNOWN | 44.7s | 303 | 83.13 GB | 1.03× |
| ACT patches (strict) | ERROR | 0.1s | — | 0.19 GB | 0.00× |
| ACT patches (mixed) | UNKNOWN | 126.3s | 239 | 83.09 GB | 2.90× |

Notes:
- ACT patches (strict): dispatch_conv_forward: conv_mode=patches received LinearBound input; materializing and falling back to matrix path
- ACT patches (strict): conv materializations=1
- ACT patches (mixed): conv materializations=342
