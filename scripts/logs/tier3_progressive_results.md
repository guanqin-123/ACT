# tier3 progressive results

## Empirical baselines (pre-handoff)

```
fix1 wire-α : 3995 = UNKNOWN  wall=448.2s nodes=3 peak_vram=0.0 GB  (cpu fallback; GPU blocked by resident vLLM)
fix1 wire-α : 8028 = FALSIFIED wall=394.8s nodes=1 peak_vram=0.0 GB (cpu fallback; soundness preserved per plan D4)
```

Source JSONs:
- `scripts/logs/tier3_fix1_3995_cpu.json` (2026-04-29 13:24:49)
- `scripts/logs/tier3_fix1_8028_cpu.json` (2026-04-29 13:15:08)

## Empirical attempts (this session, after Fix 2/3/4)

```
fix1+2+3 cpu : 3995 = SIGKILL  wall=>15min  (OOM during pre-BaB α-CROWN; tier3_fix123_cpu.log)
```

GPU rerun blocked: resident vLLM (PID 3574892, 48.29 GB) + ResNet-medium pre-BaB α-CROWN (~50 GB peak)
together exceed the 95 GB GPU. Tried `expandable_segments:True` + `--batch={4,8}`; same OOM at ~49 GB.
CPU run with `--instance both --budget 60` was silently OOM-killed by the host.

Recommended next-session empirical:
- Wait for resident vLLM to clear OR use a host with ≥80 GB GPU headroom OR ≥80 GB RAM for CPU.
- Run `python -m scripts.verify_resnet_cifar100 --instance both --conv-mode matrix --alpha-split
  --device cuda --batch 16 --budget 60.0 --bound-trace --output scripts/logs/tier3_full_stack.json`
- Compare against `wave5_instance3995_baseline_matrix_b16.json` for Phase 5 success metric
  (median pre-activation lb_L width tightening ≥ 10%).
- Re-confirm 8028 stays FALSIFIED with `--alpha-split` ON (plan invariant D4).

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
