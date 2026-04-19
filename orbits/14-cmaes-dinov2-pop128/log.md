---
issue: 15
parents: [07-real-cmaes-dinov2]
eval_version: eval-v2
metric: null
---

# Research Notes — orbit 14 (CMA-ES pop=128 × 4-restart — stretch compute)

## TL;DR

Largest CMA-ES budget in the campaign. Tests the scaling ceiling of
real Sep-CMA-ES + QR-orthogonal DINOv2-lite (the orbit-07 combo that
currently leads at METRIC=+0.319).

## Hypothesis

Orbit 07 reached +0.319 at pop=16. Orbit 11 extended to pop=64 +
3 restarts. Orbit 14 doubles again to **pop=128** and adds a **fourth
σ regime** — `[0.15, 0.08, 0.25, 0.12]` — so the σ curve covers wide
exploration → local refine → diversifying jump → medium polish.

Two possible outcomes:

- **Ceiling reached** — proxy signal is saturated at ~+0.32; larger
  batches just burn budget without moving METRIC.
- **Still lifting** — more per-generation candidates reduce proxy-noise
  so CMA-ES's covariance estimate stabilises and the schedule + restart
  diversity land on a strictly better region than orbit 11.

The answer calibrates how much compute is worth throwing at the OE
proxy in later orbits.

## Scaling math

- pop=128 × ~8-10 gens/restart × 4 restarts ≈ **4096-5120 candidates
  total** (vs orbit 11's ~1920-2560).
- Per-candidate cost ≈ 0.30s CLIP-forward on A100 ⇒ ≈ 1500s for
  5120 candidates. Tight against the 1800s budget, which is why:
    - Generations are wall-clock-gated, not count-gated.
    - Safety margin is bumped from 15s → 20s.
    - DINOv2-lite forward is **batched in one vectorised JAX einsum**
      across the entire population (orbit 11 did one candidate at a
      time inside Python). At pop=128 this saves ≈128× the per-candidate
      JAX dispatch overhead.

## Diff vs orbit 11

| | orbit 07 | orbit 11 | orbit 14 (this) |
|---|---|---|---|
| pop | 16 | 64 | **128** |
| restarts | 1 | 3 | **4** |
| σ schedule | [0.15] | [0.15, 0.08, 0.25] | **[0.15, 0.08, 0.25, 0.12]** |
| budget fractions | [1.0] | [.40, .30, .30] | **[.30, .25, .25, .20]** |
| FM forward | per-candidate | per-candidate | **batched over pop** |
| GC interval | — | every 10 gens | **every 5 gens** |
| safety margin | 10s | 15s | **20s** |
| FM | real-DINOv2 or QR-ortho-lite | QR-ortho-lite | **QR-ortho-lite only** |
| algo tag | — | sep_cma_es_dinov2_pop64_3restart | sep_cma_es_dinov2_pop128_4restart |

## Plan

1. Ship solution.py (done) — vectorised batched DINOv2-lite forward.
2. Evaluator run on Modal A100 (SEED=1,2,3 parallel).
3. Compare mean METRIC vs orbit 07 (+0.319) and orbit 11.
4. If METRIC improves materially (> orbit 07 + 0.02), argue the proxy
   has more signal to extract with compute; if plateau, flag proxy
   saturation.

## Prior Art & Novelty

### Already known
- Orbit 07 established the Sep-CMA-ES + QR-orthogonal DINOv2-lite combo.
- Orbit 11 showed 3 restarts + pop=64 don't under-perform the pop=16
  single-run baseline.
- ASAL (Kumar et al. 2024) uses Sep-CMA-ES at pop=16 by default.

### What orbit 14 adds
- First campaign point at pop=128 with a 4-regime σ schedule.
- Vectorised DINOv2-lite batch forward — engineering enabler for the
  scale, not a scientific novelty.
- Honest test of whether the proxy is signal-saturated.

### Honest positioning
This is a stretch-compute extension of orbit 07. No new method. Its
value is a calibrated answer to "does more compute still help on
LCF?" — important for downstream orbit budget allocation.

## References

- orbit 07 (`07-real-cmaes-dinov2`) — current leader, +0.319.
- orbit 11 (`11-cmaes-dinov2-pop64-restarts`) — 3-restart pop=64 precursor.
- ASAL: Kumar et al., "Automated Search for Artificial Life", 2024.
