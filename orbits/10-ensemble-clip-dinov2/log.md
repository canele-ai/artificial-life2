---
issue: 11
parents: []
eval_version: eval-v2
metric: null
---

# Orbit 10-ensemble-clip-dinov2 — Two-FM ensemble open-endedness (Axis 2, novel combination)

## Hypothesis

Batch-1 results suggest CLIP ViT-B/32 and DINOv2 see genuinely different
aspects of life-likeness on Lenia rollouts. Orbit 05 measured Pearson
r≈0.66 and top-5 Jaccard≈0.54 between CLIP-like and patch-structural
proxies — half of each proxy's top candidates is different. If CLIP
(text-aligned semantic global features) and DINOv2 (patch-structural
object-centric features) are genuinely complementary, ensembling them
should dominate either alone on the judge's 5-tier rubric. If not, this
orbit's null result tells us the two FMs are largely redundant for Lenia
open-endedness under Sep-CMA-ES.

## Approach

Sep-CMA-ES inner loop (pop_size=16, sigma_init=0.1, ≤60 gens, OOM-safe
profile matching orbits 05 & 07). Per candidate:

1. Roll out Lenia for T=256 steps, pick 4 frames (t = 0, 63, 127, 255).
2. Embed frames with CLIP ViT-B/32 (FlaxCLIPModel, same as good.py).
3. Embed the same frames with DINOv2-small — three-tier backend load:
   `FlaxDinov2Model` → `Dinov2Model(torch)` → DINOv2-lite pure-JAX fallback.
   On eval-v2's image, torch is absent → we run the fallback (same recipe
   as orbit 05, with the orthogonalisation bug fixed via QR).
4. Compute ASAL-style OE score per FM: `-mean_i max_{j<i} cos(z_i, z_j)`.
5. Combined fitness: `0.5 · clip_oe_z + 0.5 · dino_oe_z` where each
   proxy is standardised to mean-0, std-1 on warmup stats frozen at gen 0.
   Normalisation is essential — CLIP cos-sim magnitudes are ≈3× DINOv2's,
   so raw summation would let CLIP dominate.

evosax is required (eval-v2 fail-loud contract). Module-level caches for
both FMs prevent per-generation OOM. `gc.collect() + jax.clear_caches()`
every 10 generations.

## Expected interpretation of results

- **Metric > CLIP-alone baseline (orbit C6) AND > DINOv2-alone (orbit 07):**
  ensembling helps → the two FMs encode orthogonal evidence.
- **Metric ≈ max(CLIP, DINOv2):** ensembling is no-op → redundancy.
  Value here is the negative result.
- **Metric < both single-FM:** ensembling averages away signal → the FMs
  disagree on *what's good* (not merely on *which is best*), and a
  min-style ensemble or learned weighting would be needed.

Baseline anchor (LCF_judge at eval-v2): 0.1185. Absolute metric is the
mean judge score minus this anchor.

## Prior Art & Novelty

### What is already known
- [Kumar et al. (2024), ASAL](https://arxiv.org/abs/2412.17799) — single-FM
  (CLIP) foundation-model-guided Lenia search. This orbit extends it.
- [Oquab et al. (2023), DINOv2](https://arxiv.org/abs/2304.07193) —
  self-supervised patch-structural ViT; the second FM in this ensemble.
- Ensemble foundation models for search/reward modelling is common in
  reward-model-RL literature; the novelty here is applying it to
  foundation-model-guided artificial-life search.

### What this orbit adds
- First measurement on the eval-v2 Lenia substrate of whether CLIP + DINOv2
  ensemble beats single-FM OE proxies for Sep-CMA-ES inner-loop search.
- Explicit normalisation protocol for mixing two FMs with different
  cos-sim magnitude scales.

### Honest positioning
This orbit tests the **FM-complementarity hypothesis** under axis 2 of
problem.md. It does NOT claim novelty over CLIP or DINOv2 themselves.
If the ensemble ties or loses, the finding — *the two FMs are redundant
under ASAL-OE on Lenia* — is itself useful negative evidence.

## Glossary
- **ASAL** — Automating the Search for Artificial Life (Kumar et al. 2024).
- **CLIP** — Contrastive Language-Image Pre-training (Radford et al. 2021).
- **CMA-ES** — Covariance Matrix Adaptation Evolution Strategy.
- **DINOv2** — Self-supervised ViT (Oquab et al. 2023).
- **FM** — Foundation Model.
- **LCF** — Life-Cycle Fidelity (the campaign's judged metric).
- **OE** — Open-Endedness (ASAL's "trajectory keeps moving" score).
- **Sep-CMA-ES** — Separable (diagonal-covariance) CMA-ES.

## References
- Kumar, A. et al. (2024). *Automating the Search for Artificial Life
  with Foundation Models.* arXiv:2412.17799.
- Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features
  Without Supervision.* arXiv:2304.07193.
- Radford, A. et al. (2021). *Learning Transferable Visual Models from
  Natural Language Supervision.* arXiv:2103.00020.
