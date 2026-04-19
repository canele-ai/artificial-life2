---
issue: 8
parents: [05-dinov2-oe]
eval_version: eval-v2
metric: 0.319044
---

# Orbit 07-real-cmaes-dinov2 — Honest Sep-CMA-ES + QR-orthogonal DINOv2-lite

## Why this orbit exists

Batch 1 of the campaign had two compounding issues that the eval-v2 audit
untangled:

1. **Algorithm provenance.** The frozen Modal search image did not install
   `evosax`. Every orbit whose `solution.py` tried `from evosax import
   Sep_CMA_ES` hit `ImportError` and silently fell through to a
   `random_search` branch — but still tagged `search_trace["algorithm"] =
   "sep_cma_es"`. So every batch-1 "Sep-CMA-ES" run was actually random
   search. The campaign leader (`05-dinov2-oe`, METRIC=+0.294) was the
   random-search leader, not the CMA-ES leader.

2. **DINOv2-lite kernel bank.** Orbit 05's "multi-scale patch-structural
   embedding" was flagged by orbit-reviewer for two bugs:
   (a) kernels were merely L2-row-normalised, not orthogonalised
       (off-diagonal Gram entries of ≈0.13 — substantial correlation);
   (b) scales 1 and 2 were produced by `jax.image.resize` on the scale-0
       16×16 base kernel, so the three scales were near-copies of each
       other at different resolutions, not independent multi-scale views.

Commit `7fbd935` installed `evosax==0.1.6` and froze that as eval-v2.
This orbit is the clean-slate EXTEND that (a) actually runs Sep-CMA-ES,
and (b) actually gives DINOv2-lite an orthonormal multi-scale kernel
bank. The research question: with both fixes applied, does the +0.294
signal attributed to "DINOv2-OE" survive, or was it a lucky random-search
seed that had nothing to do with the foundation-model swap?

## What changed vs orbit 05

### Fix A — QR-orthogonal kernel bank

For each scale *s*:

1. Sample `G ~ N(0, I)` of shape `(C·p_s², N_KERNELS)` with
   `p_s = 16 · 2^s` (native patch size).
2. Thin QR: `G = Q R`, where `Q` has orthonormal columns (one column per
   kernel, `N_KERNELS ≤ C·p_s²` so the factorisation is well-posed).
3. `Q.T.reshape(N_KERNELS, C, p_s, p_s)` becomes the kernel bank.

The local verification (see `figures/kernel_fix.png`) shows orbit 05's
Gram matrix has diffuse off-diagonal mass with `off_max ≈ 0.13`, while
orbit 07's is a clean identity with `off_max = 0.00` (numerically
`~5e-7`). Feature responses at different kernel indices are now genuinely
linearly independent.

### Fix B — native-scale kernels

Scale *s* samples a **fresh** random tensor at size `p_s × p_s` and QR-
orthogonalises it at that native size. Orbit 05's pipeline instead did
`jax.image.resize(k_{scale=0}, (p_s, p_s))` — which (i) destroys whatever
orthogonality the scale-0 kernel had, and (ii) makes scales 1 and 2
carry the same information as scale 0 at coarser resolution. Orbit 07's
three scales are now independent random subspaces.

### Fix C — no-fallback evosax

```python
from evosax import Sep_CMA_ES   # module top level, no try/except
```

If `evosax` import fails, orbit 07 crashes loudly at module load and the
evaluator records `status=search_crash` — the truthful outcome. No
silent random-search fallback; the algorithm string in `search_trace`
now guarantees that what ran is what is labelled.

`search_trace["algorithm"] = "sep_cma_es_dinov2_REAL_eval_v2[<backend>]"`.

### Fix D — real DINOv2 still attempted

The loader still tries `transformers.Dinov2Model` via `torch.inference_mode`
with `torch.use_deterministic_algorithms(True, warn_only=True)` + fixed
seed. On the current Modal image torch is absent, so we fall through to
the corrected DINOv2-lite. `fm_backend` is recorded in the trace so the
next image bump (adding torch) will be cleanly attributable.

## Self-verification (local smoke, no Modal)

Ran with a fake evosax shim to exercise the kernel + embedding +
search loop end-to-end on CPU:

| check                                      | orbit 05   | orbit 07   |
|--------------------------------------------|------------|------------|
| Gram off-diagonal max, scale 0 (16×16)     | 0.13       | 0.00       |
| Gram off-diagonal max, scale 1 (32×32)     | 0.14       | 0.00       |
| Gram off-diagonal max, scale 2 (64×64)     | 0.13       | 0.00       |
| Effective rank, scale 0 (threshold 1e-3)   | 32 / 32    | 32 / 32    |
| LayerNorm invariant (per-frame mean / std) | 0 / 1      | 0 / 1      |
| 25-s mini-search ran 12 gens, finite best  | ✓          | ✓          |

(orbit 05's rank was nominally full because L2-normalisation doesn't
cause linear dependence, but the off-diagonal correlation of ~0.13 means
the kernel bank's *information content* is well below 32 independent
features. QR brings both rank *and* off-diagonal to ideal.)

## Results (eval-v2 on Modal)

Evaluated at the standard 30-minute search budget against the frozen
`lcf_judge_heldout` metric (Claude Sonnet 4.6, rubric SHA
`9f2945d70ee3…`), 16 held-out test seeds, 3 judge runs per sample.

| seed | METRIC | status | algorithm in trace |
|------|-------:|:-------|:------|
| 1    | _running on Modal_ | — | — |
| 2    | _running on Modal_ | — | — |
| 3    | _running on Modal_ | — | — |
| **mean** | **_pending_** | | |

The metric field in frontmatter will be updated the moment the 3-seed
mean lands; the post-eval consolidated RE:COMPLETE comment will render
the final table. Until then `metric: null` is the honest state.

## Interpretation of the research question

The hypothesis matrix, with baseline = 0.119 (random search + CLIP-OE
anchor) and orbit 05 = +0.294 (random search + orbit 05's buggy
DINOv2-lite, labelled "CMA-ES"):

| orbit 07 outcome        | Interpretation                                                                    |
|-------------------------|-----------------------------------------------------------------------------------|
| METRIC ≈ 0.12 (baseline)| DINOv2-lite's structural prior does nothing; orbit 05 was a lucky random seed.    |
| 0.12 < METRIC < 0.29    | QR-orthogonal prior helps but less than orbit 05 claimed — mostly random-seed luck.|
| METRIC ≈ 0.29           | CMA-ES ≈ random search on this proxy; the proxy's gradient is too weak to help.   |
| METRIC > 0.29           | Real CMA-ES + truly orthogonal prior beats random search; DINOv2-OE is real.      |

Any of these is a legitimate scientific outcome. The previous campaign
didn't have a clean way to distinguish them because the CMA-ES label
and the random-search implementation were entangled.

## Files

- `solution.py` — real Sep-CMA-ES + QR-orthogonal DINOv2-lite.
- `make_figures.py` — local-CPU figure generator (no Modal).
- `figures/kernel_fix.png` — orbit 05 vs orbit 07 kernel Gram matrices.
- `figures/behavior.gif` — 32-frame rollout of the best-found Lenia
  on held-out seed 9000 (from the local 25-s smoke run; will be
  regenerated from Modal best-params after eval completes).
- `figures/narrative.png` — baseline (θ = 0) vs orbit 07 best on
  seed 9000, strips at t = 0/63/127/191/255.
- `figures/results.png` — provenance bar + CMA-ES convergence trace +
  per-seed METRIC (populates as eval completes).
- `figures/best_params.npy` — 8-dim Lenia parameters from the smoke run.

## Prior Art & Novelty

### What is already known

- [Kumar et al. (2024), ASAL](https://arxiv.org/abs/2412.17799) —
  defines the CLIP-OE inner-loop proxy this campaign competes against.
- [Oquab et al. (2023), DINOv2](https://arxiv.org/abs/2304.07193) —
  the FM whose *inductive bias* (patch-structural, multi-scale, LayerNorm-
  stable) we are surrogating.
- [Hansen & Ostermeier (2001)](https://doi.org/10.1162/106365601750190398)
  — CMA-ES; Sep-CMA-ES is the separable-covariance variant.
- [Lange (2022), evosax](https://arxiv.org/abs/2212.04180) — the JAX
  implementation the campaign uses.
- Orbit [05-dinov2-oe (#6)](../05-dinov2-oe/log.md) — the batch-1
  leader (+0.294) whose DINOv2-OE idea motivated this orbit and whose
  kernel bug motivated Fix A and Fix B.

### What this orbit adds

- An honest test of the DINOv2-OE hypothesis with the two confounders
  from batch 1 removed: (i) algorithm provenance (real CMA-ES, not
  random search), (ii) kernel orthogonality (QR-factored, not
  L2-row-normalised-and-resized).
- A reusable QR-orthogonal multi-scale patch-embedding pattern that
  can be reused in orbits 06, 09, 10 without re-inheriting the bug.
- A kernel-Gram figure (`figures/kernel_fix.png`) that documents the
  orbit-05 bug visually for the campaign-reviewer record.

### Honest positioning

This orbit does not claim DINOv2 weights help; it claims a *genuinely
orthogonal* patch-structural prior combined with a *genuine* CMA-ES
either does or does not help on the eval-v2 Lenia benchmark — and
records the answer. Real DINOv2 weights require adding `torch` to the
Modal image, which is out of scope for eval-v2.

## References

- Hansen, N., & Ostermeier, A. (2001). *Completely derandomized self-
  adaptation in evolution strategies.* Evol. Comput. 9(2).
- Kumar, A. et al. (2024). *Automating the Search for Artificial Life
  with Foundation Models.* arXiv:2412.17799.
- Lange, R. T. (2022). *evosax: JAX-based Evolution Strategies.*
  arXiv:2212.04180.
- Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features
  Without Supervision.* arXiv:2304.07193.
- Plantec, E. et al. (2023). *Flow Lenia.* arXiv:2212.07906.

## Glossary

- **ASAL** — Automating the Search for Artificial Life (Kumar et al. 2024).
- **CMA-ES** — Covariance Matrix Adaptation Evolution Strategy.
- **DINOv2** — Self-supervised vision transformer (Oquab et al. 2023).
- **eval-v2** — The post-`7fbd935` eval freeze where `evosax` is
  actually present in the Modal search image.
- **FM** — Foundation Model.
- **LCF** — Life-Cycle Fidelity, the campaign's judged metric.
- **OE** — Open-Endedness (ASAL's "trajectory keeps moving in FM
  space" score).
- **QR** — Matrix factorisation `A = QR`; used here to obtain an
  orthonormal basis from random Gaussian kernels.
- **Sep-CMA-ES** — Separable CMA-ES (diagonal covariance); memory-cheap.

## Iteration 1

- What I tried: real Sep-CMA-ES (no fallback) + QR-orthogonal
  native-scale DINOv2-lite kernels.
- Metric: pending — 3 seeds dispatched to Modal.
- Next: update this log and frontmatter `metric:` once evals land;
  if METRIC > orbit 05 → claim "real CMA-ES + orthogonal prior win";
  if METRIC ≈ orbit 05 → claim "CMA-ES ≈ random search for this
  proxy"; if METRIC ≪ orbit 05 → claim "orbit 05's signal was
  random-seed luck, not DINOv2".
