---
issue: 3
parents: []
eval_version: eval-v1
metric: 0.279250
---

# Orbit 02 — ASAL-mature-organism

## Research Notes

### Hypothesis

Sep-CMA-ES + ASAL **supervised-target** on a **single strong prompt** (with
two variant phrasings, take max) that emphasizes the tiers the pinned ASAL
baseline scores weakest on — **Coherence** and **Existence**. We score at
the median rollout frame `t=127` only (the "mature" stage), not all 5
frames. Rationale: narrower target direction + 5× cheaper per-candidate
cost → pop_size=32 instead of 16, same wall-clock budget.

This orbit is the foil to orbit 01 (life-cycle prompt bank): we test
whether a single crisp target beats a rich multi-stage bank inside
Sep-CMA-ES's low-population regime.

### Prompt bank (3 variants, max-prompt soft-CE)

1. `"a mature self-sustaining organism with crisp geometric structure on a dark background"`
2. `"a bilaterally symmetric living cell with smooth clean boundaries"`
3. `"a soft glowing biological organism clearly distinct from empty space"`

ASAL's symmetric cross-entropy on `softmax(z_img @ z_txt.T / τ)` reduces,
for a single image and multiple prompts, to `τ · (logsumexp(cos / τ) −
log n_prompts)` — a smooth upper bound on `max_prompt cosine`, which is the
"take-max-across-variants" hedge the campaign context suggested. τ = 0.07
(CLIP default).

### Hyperparameters

| field           | value                               |
|-----------------|-------------------------------------|
| algorithm       | Sep-CMA-ES (evosax)                 |
| pop_size        | 32                                  |
| sigma_init      | 0.12                                |
| n_generations   | ≤ 50 (wall-clock gated)             |
| CLIP model      | openai/clip-vit-base-patch32        |
| score frame     | t=127 only (median of T=256)        |
| n_prompts       | 3 variants (take soft-max)          |
| clear_cache_gen | every 10 gens (`gc.collect + jax.clear_caches`) |
| checkpoint      | `/tmp/ckpt/{run_id}_best.npy` every 30 s |
| wall_clock_margin | 15 s before hard cap              |

### OOM mitigations (per `canary_results_full.md` 30-min finding)

- Module-level `_CLIP_CACHE` — prevents re-loading CLIP per generation.
- `gc.collect() + jax.clear_caches()` every 10 gens.
- Self-limit `wall_clock_s − 15`.
- Single-frame scoring (≤ 1 CLIP image forward per candidate, vs baseline's 5).
- Checkpoint `best_params` to `/tmp/ckpt/` every 30 s for SIGKILL recovery.

### Expected behavior

- If the "single-strong-target" theory is right: higher Coherence +
  Existence tier scores under the Sonnet judge than the pinned ASAL
  baseline (0.1185), with slightly lower Agency (motion is not directly
  rewarded at t=127 alone).
- Risk: t=127 may be dominated by a collapsed / blown-up state on many θ
  (Lenia's chaotic regime) — the prompt-bank max should still give the
  CMA-ES enough gradient to escape into the stable-pattern manifold.

## Results

(Populated after Modal runs complete.)

## Glossary

- **ASAL** — Automating Artificial Life Search (Kumar et al. 2024).
- **CLIP** — Contrastive Language-Image Pre-training (Radford et al. 2021).
- **Sep-CMA-ES** — Separable CMA Evolution Strategy (Ros & Hansen 2008).
- **LCF** — Life-Cycle Fidelity score (campaign-defined metric).
- **Soft-CE** — softmax-cross-entropy, ASAL's collapse-resistant variant of supervised-target alignment.
- **OOM** — out-of-memory (subprocess SIGKILL signature in eval-v1).
- **VLM** — Vision-Language Model (here: Claude Sonnet 4.6 as judge).
