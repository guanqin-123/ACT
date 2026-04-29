# tier3 progressive results

## Empirical baselines (pre-handoff)

```
fix1 wire-α : 3995 = UNKNOWN  wall=448.2s nodes=3 peak_vram=0.0 GB  (cpu fallback; GPU blocked by resident vLLM)
fix1 wire-α : 8028 = FALSIFIED wall=394.8s nodes=1 peak_vram=0.0 GB (cpu fallback; soundness preserved per plan D4)
```

Source JSONs:
- `scripts/logs/tier3_fix1_3995_cpu.json` (2026-04-29 13:24:49)
- `scripts/logs/tier3_fix1_8028_cpu.json` (2026-04-29 13:15:08)

## Empirical evolution this session (matrix mode, --alpha-split, GPU with vLLM contention)

```
fix1+2+3 gpu  : 3995 = ERROR    peak=44.98 GB  (OOM 114 MiB alloc; pre-Fix-5 baseline, eye-expansion bottleneck)
fix1+2+3 gpu  : 8028 = ERROR    peak=49.15 GB  (OOM 64 MiB alloc; same bottleneck)

fix5  c=4096  : 3995 = ERROR    peak=20.14 GB  (OOM 226 MiB alloc; eye-expansion fixed, new bottleneck = joint-loss expansion)
fix5  c=4096  : 8028 = ERROR    peak=20.14 GB  (same)

fix5+5b       : 3995 = ERROR    peak=17.3 GB   (OOM 22 GiB alloc in joint-loss runtime_alpha_dict; pre-BaB autograd graph released)
fix5+5b       : 8028 = ERROR    peak=17.3 GB   (same — Fix 2 joint loss in BaB hot path is the hog)

fix6 (λ=0)    : 3995 = ERROR     wall=18.36s peak=20.7 GB  (OOM 96 MiB alloc; ~MiB from success)
fix6 (λ=0)    : 8028 = FALSIFIED wall=11.94s peak=17.3 GB  ✓ (plan invariant D4 SATISFIED)
```

Source: `scripts/logs/tier3_lambda0_gpu.json` (post-Fix 6, the 8028 FALSIFIED run).

## Memory peak progression (matrix, batch=8, alpha-split, ResNet-medium)

| Stack | Peak VRAM | OOM at | Status |
|---|---|---|---|
| Fix 1 baseline | 49 GB | 64-114 MiB | ERROR (vLLM contention) |
| + Fix 5 (chunked obj rows) | 20 GB | 226 MiB | ERROR (joint-loss alpha expansion) |
| + Fix 5b (per-sid backward in pre-BaB) | 17 GB | 22 GiB | ERROR (joint-loss runtime_alpha_dict in BaB) |
| + Fix 6 (λ_intermediate=0 default) | 17-21 GB | 96 MiB / N/A | 8028 FALSIFIED ✓ ; 3995 ERROR (gap closed except for last MiB) |
| **+ Fix 7 (cheap patches concretize identity branch + bias tracking)** | **17-21 GB** | **N/A on identity** | **identity-branch dense round-trip eliminated; fusion-branch unchanged** |

## What still needs a clean GPU

3995 OOMs at 96 MiB allocation when only 83 MiB is free. The 17-20 GB peak fits comfortably on a 95 GB GPU when not shared. Currently other tenants:
- vLLM PID 3574892: 48 GB
- python PID 1315561: 22 GB
- python PID 1315674: 5 GB
- our peak: 20.7 GB
- total: ~96 GB on a 95 GB GPU → tiny shortfall

Recommended next-session command on a clean GPU:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m scripts.verify_resnet_cifar100 \
  --instance both --conv-mode matrix --alpha-split \
  --device cuda --batch 8 --budget 60.0 --bound-trace \
  --output scripts/logs/tier3_full_stack_clean_gpu.json
```

Phase 5 success metric: median pre-activation `lb_L` width tightening on 3995 ≥ 10% (or CERTIFIED in <60s) — pending the above run on uncontended GPU.

## Soundness invariants known to hold (unit-test verified)

- D4 default-off (alpha_split_objective=False) bit-identical Adam path: enforced by
  `test_adam_default_off_uses_flat_param_list_with_lr` (tier3-fix4-oracle-blockers).
- Per-node α clamped to [0,1] after every Adam step (`_compute_bound_joint_kkt` post-step `param.clamp_(0,1)`).
- α warm-state preserved across BFS/DFS batch merges: enforced by
  `test_concat_subproblem_batches_preserves_alphas` (tier3-fix4-oracle-blockers).
- `λ_intermediate=0` reproduces fix1-only behaviour: `test_lambda_zero_reproduces_fix1`.
- Intermediate α gradient non-zero under joint loss: `test_intermediate_grad_nonzero_at_bab_depth_1`.
- Adam two-group structure under alpha-split: `test_adam_uses_two_param_groups_with_lr_alpha`.
- Truncated intermediate-bound LB sound under arbitrary α∈[0,1]: `test_soundness_with_intermediate_loss`.
