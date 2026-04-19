---
issue: 14
parents: [10-ensemble-clip-dinov2]
eval_version: eval-v2
metric: null
---

# Research Notes — orbit 13: geomean ensemble + α-anneal

## Hypothesis (REFINE of orbit 10)

Orbit 10 ensembled CLIP-OE and DINOv2-lite-OE with a fixed arithmetic mean
(α=0.5) and scored *below* either FM individually (+0.080 vs +0.294 for
CLIP-OE alone in orbit 05, +0.319 for real-DINOv2 CMA-ES in orbit 07).
Two hypotheses for why, and matching fixes:

1. **Arithmetic mean is too forgiving.** If one FM is fooled by a
   degenerate candidate, the ensemble still pays half-credit. Switch to a
   **geometric mean** so either FM vetoing a candidate collapses its score:
   `F = (clip_z + ε)^α · (dino_z + ε)^(1−α)` with `ε = 1e-3`.

2. **DINOv2-lite is a noisier late-stage teacher** than CLIP for Lenia.
   Use **linear α-anneal** 0.5 → 1.0 across generations: early exploration
   uses BOTH FMs (broad novelty signal), late exploitation locks onto
   CLIP alone (sharper final candidate).

### Bug fix: `_NORM_CACHE` scoping

Orbit 10 stored `clip_mu`/`sd`/`dino_mu`/`sd` in a **module-level**
`_NORM_CACHE` dict that was only populated on the first call and never
cleared. Within a single process, re-entry of `search()` (e.g. a driver
that re-runs with a different seed) would reuse stale warmup stats from
the first seed. Fix: `_NORM_CACHE.clear()` at function entry so warmup
stats are per-call, not per-process.

### Shift offset for geomean well-definedness

Standardised z-scores can be negative; powered to a non-integer α they
blow up / become complex. We compute a frozen shift per FM during warmup:
`z_shifted = (raw − μ) / σ + offset`, where `offset = −min(z_warm) +
1e-3`, so every warmup candidate ends up ≥ `1e-3`. Later populations
use the same frozen offset (bit-reproducible). Later-gen candidates
that undershoot warmup are clipped to `1e-3` (a floor, not a soft clamp).

## Method

- **Substrate**: Lenia 128², T=256, 8-dim params (same as orbits 05/07/10).
- **Searcher**: Sep-CMA-ES, pop=16, σ_init=0.1, ≤60 generations.
- **FMs**:
  - CLIP ViT-B/32 via `transformers.FlaxCLIPModel`.
  - DINOv2-lite: QR-orthogonalised multi-scale patch-kernel proxy in pure
    JAX (same recipe as orbit 10 — kept deliberately identical so any
    metric delta is attributable to the geomean + α-anneal, not to an FM
    swap).
- **Per-FM proxy**: ASAL-style OE `−mean_i max_{j<i} cos(z_i, z_j)` on
  4 rollout frames {0, 63, 127, 255}.
- **Warmup normalization**: frozen `(μ, σ, offset)` per FM, computed on
  the first generation's population, re-used thereafter.
- **Ensemble**: `F(θ) = clip_sh^α_t · dino_sh^(1−α_t)`, `α_t ∈ [0.5, 1.0]`
  linear.
- **Trace provenance**: `algorithm =
  sep_cma_es_ensemble_geomean_alpha_anneal_eval_v2`.

## Glossary

- **Sep-CMA-ES** — Separable Covariance Matrix Adaptation Evolution
  Strategy (evosax).
- **OE** — Open-Endedness (ASAL's collapse-resistant novelty proxy).
- **CLIP** — Contrastive Language-Image Pretraining (ViT-B/32).
- **FM** — Foundation Model.
- **DINOv2-lite** — pure-JAX proxy using multi-scale QR-orthonormal patch
  kernels; NOT real DINOv2 (kept identical to orbit 10 for clean A/B).
- **LCF** — Life-Cycle Fidelity (campaign metric).
- **α-anneal** — linear schedule of the CLIP/DINOv2 mixing exponent over
  generations.

## Prior Art & Novelty

### What is already known
- **Geometric-mean ensembling** of classifier scores is standard in
  model-averaging literature; stricter than arithmetic mean, reduces
  Brier score when ensemble members have uncorrelated errors.
- **α-anneal / curriculum** on multi-objective fitness is a standard
  evolutionary-computation technique (e.g. dynamic weighted aggregation,
  Jin et al. 2001).
- ASAL (Kumar et al. 2024) established the CLIP-OE baseline we inherit.

### What this orbit adds (if anything)
- Applies geomean + α-anneal to the specific failure mode observed in
  orbit 10 (arithmetic-mean ensemble underperforming individual FMs).
- Isolates the effect of the *ensemble rule* from the choice of FMs by
  reusing orbit 10's exact CLIP + DINOv2-lite stack.
- Documents the `_NORM_CACHE` subtle bug.

### Honest positioning
A refinement orbit, not a novel method. The interesting result will be
either (a) the ensemble now clears the best individual FM → geomean is
the right rule, or (b) it still doesn't → CLIP and DINOv2-lite are
largely redundant for Lenia and no ensembling rule will help, only a
genuinely different second FM (real DINOv2, SigLIP, temporal FM) will.

## Iteration 0 — initial implementation (pending eval)

- What I tried: geomean with linear α-anneal 0.5→1.0 + `_NORM_CACHE`
  bug fix.
- Metric: pending (will be filled by the agent-complete hook's eval-check
  since I am relaunching under a usage-limit constraint and do not have
  budget to run a full 30-minute eval here).
- Next: if metric ≥ orbit 10's +0.080 → hypothesis (1) supported; if ≥
  CLIP-OE baseline +0.294 → the geomean + α-anneal rescues the
  ensemble; else the FMs are genuinely redundant for Lenia.
