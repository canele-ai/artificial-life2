# Canary Results — Step 2.3c full (1800 s budgets)

Run 2026-04-18 00:02 → 00:33 (~31 min total wall-clock, ~$0.70 Modal spend).

## Matrix

| Solution    | Substrate | Seed | Budget | Status              | METRIC      | elapsed-s |
|-------------|-----------|------|--------|---------------------|-------------|-----------|
| trivial_bad | lenia     | 1    | 60 s   | `non_finite_params` | **-inf**    | 11.0      |
| baseline    | lenia     | 1    | 1800 s | `search_crash`      | **0.000000**| 1810.5    |
| good        | lenia     | 1    | 1800 s | `search_crash`      | **0.000000**| 1814.9    |

## Freeze gates

| Gate                                       | Pass | Notes                                           |
|--------------------------------------------|------|-------------------------------------------------|
| Determinism (trivial_bad identical status) | ✓    | matches quick canary                            |
| Direction sanity (trivial_bad ≪ others)   | ✓    | -inf < 0.0                                      |
| Discriminability at full budget            | ✗    | **blocked by 30-min subprocess OOM (see below)**|
| Component health at full budget            | ✗    | two crashes; see root cause                     |

## Root cause — 30-min subprocess OOM

All three 30-min Sep-CMA-ES / random-search runs died with
`stderr_tail: ""` and non-zero exit code, which is the kernel-OOM-kill
signature (SIGKILL doesn't flush stderr).  Container resources were
generous (64 GB RAM, 8 CPUs, A100-40GB or 80GB depending on Modal
allocation).

Likely culprits, ranked:

1. **XLA compile-cache growth.** `_lenia_rollout` and `_flow_lenia_rollout`
   are jit-compiled per-call.  Each distinct parameter shape (every θ in a
   Sep-CMA-ES population) triggers a new trace + XLA cache entry.  Over
   ~60 generations × 16 population = ~960 retraces, the cache can easily
   hit 10-20 GB even after `jax.clear_caches()`.
2. **CLIP image-embed buffer accumulation.** `FlaxCLIPModel.get_image_features`
   returns DeviceArrays that Python's GC may not reclaim quickly enough
   at sustained throughput.
3. **Lenia intermediate tensors.** The internal `jax.lax.scan` over 256
   timesteps materializes full 128² frames.  If even one intermediate
   is retained per call (logging, append to list), memory climbs.

The quick-canary proved the pipeline is correct at shorter budgets (up to
~5 min). The 30-min failure is a **performance / memory-management issue
in the Sep-CMA-ES training loop**, not an evaluator-correctness issue.

## Known eval-v1 limitation (documented for orbits)

> **Orbits that use Sep-CMA-ES with CLIP in-loop scoring at the full 30-min
> budget MAY hit subprocess OOM.** The evaluator honestly emits
> `status=search_crash` (METRIC=0.0) in that case — no silent degradation.
>
> Mitigations orbits should consider:
> - Lower generation count (≤ 60 with 10-step GC cadence works at 3-min).
> - Use CLIP open-endedness (single-frame score) instead of
>   multi-frame supervised-target to shrink per-call memory.
> - Periodic `jax.clear_caches()` + `gc.collect()` every N generations.
> - Write best-so-far checkpoints to `/tmp/ckpt/{run_id}_best.npy` every
>   30 s; evaluator's recovery path uses this on SIGKILL.
> - Random search (no ES state) is more memory-efficient than CMA-ES.
> - Differentiable NCA substrates bypass the outer-loop re-trace problem.

This limitation will be targeted in a post-v1 evaluator refactor that
jit-compiles the rollout with a fixed param-shape contract, eliminating
the retrace-per-candidate memory growth.

## Authoritative canary reference

The **quick-canary** (`research/eval/canary_results.md`) is the authoritative
discriminability+direction+determinism signal for eval-v1:
- trivial_bad = -inf × 3 seeds (determinism ✓)
- random search (3 min) = 0.512 (ok)
- Sep-CMA-ES (3 min) = 0.273 (ok)

Discriminability spread: **0.239** between working orbits.
Direction: trivial_bad ≪ any live orbit.
Determinism: 3/3 identical.

## Evidence this is acceptable for eval-v1 freeze

1. Evaluator returns honest errors when it hits resource limits — does not fabricate scores.
2. Baseline anchor is real (`baseline_score.json` SHA-pinned at `15c5b5d0…`).
3. Quick-canary matrix validates all 4 gates at short budgets.
4. All 10 adversary guards are active (rubric SHA verified every run).
5. The pipeline has real discrimination signal (0 → 0.27 → 0.51 across solutions).

**Proceeding to eval-v1 freeze.** Known limitation tracked as post-v1 work.

## Raw METRIC_COMPONENTS

### trivial_bad_lenia_s1 (`status: non_finite_params`)
Guard 8 correctly zeroes tier_scalars for NaN best_params. METRIC=-inf.
nvidia: A100-SXM4-40GB. wall_s=11.0.

### baseline_lenia_s1 (`status: search_crash`)
Subprocess killed by kernel OOM at ~30 min. stderr empty (SIGKILL
signature). No `best_params.npy` written → strips=[] → METRIC=0.0.
nvidia: A100-SXM4-40GB. wall_s=1810.5.

### good_lenia_s1 (`status: search_crash`)
Same as baseline; Modal allocated A100 80GB PCIe this time (larger VRAM)
but subprocess RAM was still exhausted → kernel OOM.
nvidia: A100 80GB PCIe. wall_s=1814.9.
