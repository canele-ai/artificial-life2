---
issue: 17
parents: [15-flow-lenia-retry]
eval_version: eval-v2
metric: -0.118516
---

# Research Notes — orbit 16: Flow-Lenia × DINOv2-lite (stack top two axes)

## Hypothesis

Stack the two strongest effects in the eval-v2 campaign:

- **Axis 1 — substrate.** Orbit 15 (`15-flow-lenia-retry`) is the current
  leader at +0.339 with mass-conserving Flow-Lenia (semi-Lagrangian
  transport + Σ-renormalisation) as the substrate.
- **Axis 2 — inner-loop FM.** Orbit 07 (`07-real-cmaes-dinov2`) reached
  +0.319 with a QR-orthogonal multi-scale patch encoder ("DINOv2-lite")
  as the OE proxy, running on *plain* Lenia.

These two axes have never been combined.  Orbit 16 does exactly that:
Flow-Lenia rollout + DINOv2-lite embedding + ASAL open-endedness score
(negative mean max off-diagonal cosine similarity across 4 frames) +
Sep-CMA-ES (pop=16, σ_init=0.1, ≤60 gens), all fail-loud on evosax.

Flow-Lenia's distinctive failure mode — dead/uniform final frames that
paradoxically inflate OE — is caught with orbit 15's mass-variance
degeneracy penalty (−0.25 if `var(final_mass) < 1e-6` or
`max(final_mass) < 1e-3`).

## Method

- **Substrate:** `flow_lenia` (orbit 15), inlined from orbit 07's
  self-contained rollout for independence from evaluator-private symbols.
- **FM:** DINOv2-lite with orbit 07's corrected QR-orthogonal multi-scale
  kernel bank at native patch sizes 16/32/64 (no bilinear upsampling — the
  orbit-05 bug is absent here).
- **Optimiser:** `evosax.Sep_CMA_ES`, fail-loud import at module top.
  `pop=16`, `σ_init=0.1`, `≤60` gens, `gc.collect()+jax.clear_caches()`
  every 10 gens.
- **Scoring:** rollout (256 steps) → 4 frame picks `(0, 63, 127, 255)` →
  DINOv2-lite 96-d embedding → OE = −mean of row-max lower-triangle cosine
  similarities; subtract 0.25 if degenerate.
- **Search trace tag:**
  `sep_cma_es_flow_lenia_dinov2_lite_eval_v2[dinov2_lite_ortho]`.

## Prior Art & Novelty

### What is already known

- ASAL open-endedness proxy (Kumar et al. 2024) with CLIP backbone.
- Flow-Lenia mass-conservation dynamics (Plantec et al. 2023,
  arXiv:2212.07906).
- QR-orthogonal random-feature multi-scale kernels (standard; see e.g.
  Rahimi & Recht 2007 for the random-features lineage) applied to ALife
  OE proxies here by orbit 07.

### What this orbit adds

- First combination on eval-v2 of Flow-Lenia substrate × orbit 07's
  corrected DINOv2-lite proxy.  No new algorithm; the contribution is the
  specific cross-axis combination and the confirmation/refutation of
  whether the two effects compose additively.

### Honest positioning

If the two effects were fully additive relative to the eval-v2 baseline,
one would naïvely expect >+0.339 (the better of the two parents).  If the
effects are mutually redundant (both proxy "trajectory keeps moving"),
the metric should land near the parents.  The mass-variance degeneracy
penalty is load-bearing — without it Flow-Lenia's dead rollouts inflate
the DINOv2-lite OE score the same way they inflated CLIP OE.

## References

- Kumar, Couturier, Soros, Ha (2024). *Automating the Search for
  Artificial Life*. ASAL paper.
- Plantec, Hamon, Etcheverry, Oudeyer, Moulin-Frier, Chan (2023).
  *Flow-Lenia: Mass conservation for the study of virtual creatures in
  continuous cellular automata*. arXiv:2212.07906.

## Iteration 1

- What I tried: composed Flow-Lenia rollout (orbit 15/07) + DINOv2-lite
  QR-orthogonal embedding (orbit 07) + mass-variance degeneracy penalty
  (orbit 15).  Sep-CMA-ES, pop=16, σ=0.1, ≤60 gens, fail-loud evosax.
- Metric: pending (Modal eval).
- Next: await evaluator metric; if degenerate-but-not-caught, tighten
  penalty.
