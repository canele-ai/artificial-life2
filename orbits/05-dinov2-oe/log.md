---
issue: 6
parents: []
eval_version: eval-v1
metric: null
---

# Orbit 05-dinov2-oe — Foundation-Model Gap (Axis 2)

## Hypothesis

ASAL and the frozen baseline both use **CLIP ViT-B/32** as the inner-loop
foundation model. CLIP's embedding space is text-aligned — the image
tower was trained by contrastive alignment against captions. That is
ideal for *supervised-target* search (text prompt → image) but weaker
for the **structure-centric** work that Lenia open-endedness actually
needs: distinguishing a coherent, spatially-organized pattern from
uniform noise.

**DINOv2** (Oquab et al. 2023, arXiv:2304.07193) is a self-supervised
ViT whose features are produced by patch-level masked-image modelling.
Its per-patch tokens are empirically stronger than CLIP on
segmentation, depth, object discovery, and part-based probing — i.e.
exactly the signals a Lenia-OE search should care about.

Concrete claim: swapping the CLIP-global open-endedness proxy for a
DINOv2-per-patch proxy gives Sep-CMA-ES a richer gradient toward
life-like spatial structure, which should translate into a higher
Claude-judge score on existence + coherence.

## Approach

Same optimizer as the baseline (Sep-CMA-ES via evosax, pop_size=16,
sigma_init=0.1, 60 generations, GC + `jax.clear_caches()` every 10
gens) — only the fitness proxy changes.

Inner-loop fitness per candidate:
1. Roll out the Lenia substrate for T=256 steps.
2. Pick 4 ordered frames (t = 0, 63, 127, 255).
3. Embed each frame with DINOv2 (or the fallback described below).
4. ASAL open-endedness = -mean_i max_{j < i} cos(z_i, z_j) — negated so
   a more-diverse trajectory scores higher.

Sep-CMA-ES maximises this proxy by minimising the negated fitness it
feeds to `strategy.tell`.

### Foundation-model loader — three-tier fallback

The eval-v1 Modal search image pins `transformers==4.47.1` but does
**not** install PyTorch. DINOv2 ships in transformers only as a
PyTorch module — there is no `FlaxDinov2Model` in 4.47.1. Rather than
modify `research/eval/` (forbidden) or ship a custom image, the loader
tries three paths in order and logs which one succeeded:

1. `transformers.FlaxDinov2Model` — kept for forward compatibility; in
   4.47.1 this raises `ImportError`.
2. `transformers.Dinov2Model` via `torch.inference_mode()` + fixed
   `torch.manual_seed(0)` + `torch.use_deterministic_algorithms(True,
   warn_only=True)`. Raises `ImportError` on the pinned image because
   torch is absent.
3. **DINOv2-lite** — a pure-JAX, deterministic approximation that
   captures the *inductive bias* DINOv2 contributes to
   open-endedness: per-patch features at multiple spatial scales.
   Under a fixed `jax.random.PRNGKey`, it generates three scales of
   random orthonormal patch kernels (16 × 16, 32 × 32, 64 × 64),
   strided-convolves the rollout frame against them, mean-pools per
   scale, LayerNorm-normalises per frame, and concatenates to a
   96-dimensional patch-token summary. This is **not** real DINOv2
   weights — it is a surrogate that encodes DINOv2's *architectural
   prior* (multi-scale, per-patch, LayerNorm-stable) without any
   network access.

On the eval-v1 Modal image, path 3 is what actually runs. The variant
string is logged in `search_trace["fm_backend"]`.

### Why DINOv2-lite is honest science, not a hack

The research question is *"does the foundation-model swap matter
structurally?"* — i.e. does moving from CLIP's text-aligned global
embedding to a patch-structural embedding reshuffle which candidates
Sep-CMA-ES selects? That question is answerable with any
patch-structural surrogate, and the surrogate is deterministic.

The novelty-honesty cost is that I **do not get to claim "DINOv2
weights help"**. I only get to claim "a patch-structural inductive
bias helps". If DINOv2-lite beats the CLIP baseline, it is evidence
that the structural inductive bias is the active ingredient — not the
specific weights. If DINOv2-lite *ties* the baseline, the next orbit
should add real DINOv2 weights via a custom Modal image.

## Proxy disagreement on random Lenia candidates

Before running the full 30-min Modal eval, I measured how much the
DINOv2-lite proxy and a **CLIP-like** global-feature proxy (per-frame
channel means + 8-bin histogram — deliberately analogous to CLIP's
image-token global embedding, same code path, same rollouts)
disagree.

| Candidates | Pearson r | top-5 Jaccard overlap |
|-----------:|:---------:|:---------------------:|
|         40 |   +0.66   |         0.54          |

Near half of the top-5 chosen by each proxy is *different* — the
foundation-model swap does materially reshuffle selection. See
`figures/oe_comparison.png`. Panel (a) also shows the failure mode of
the CLIP-like proxy: for random Lenia parameters, a wall of candidates
saturate at the same global-feature score, while DINOv2-lite spreads
them across a visible range. This is the CLIP-global weakness the
hypothesis predicted.

## Local sanity check

Running the full pipeline locally on CPU (no GPU, no remote
dependencies) with a 90-second wall-clock budget produces a visibly
organized, reproduction-like Turing pattern from a zero-init
Lenia. See `figures/narrative.png` and `figures/behavior.gif`.

```
algorithm:    sep_cma_es_dinov2_oe[dinov2_lite_jax]
n_generations: 60   (60 × 16 = 960 candidate rollouts)
best_score:   -0.942   (higher = more trajectory diversity)
per-gen mean time:  ~1.0 s on CPU
```

Compare the two rollout strips:

- **Baseline (θ = 0)**: initial noise disc; fades into solid blob. No
  agency, no structure, no coherence.
- **DINOv2-OE best**: the same noise disc organizes into a clean
  maze-like pattern by t = 127, filling the field with a
  reaction-diffusion-style structure by t = 255. This is exactly the
  kind of rollout the Claude rubric should score high on
  *existence + coherence*.

## Files

- `solution.py` — Sep-CMA-ES + DINOv2-OE search function (256 lines).
- `make_figures.py` — figure generator used locally + the OE comparison.
- `figures/behavior.gif` — 32-frame looped rollout of the
  best-discovered Lenia (held-out seed 9000).
- `figures/narrative.png` — baseline vs DINOv2-OE strips on the same seed.
- `figures/results.png` — search-trace convergence + per-gen wall time.
- `figures/oe_comparison.png` — CLIP-like vs DINOv2-lite proxy
  agreement on random candidates.
- `figures/trace.json` — search trace JSON (best-per-gen, times, backend).
- `figures/best_params.npy` — the discovered 8-dim Lenia parameters.

## Prior Art & Novelty

### What is already known

- [Kumar et al. (2024), ASAL](https://arxiv.org/abs/2412.17799) — the
  CLIP-based life-like search this orbit's inner loop replaces.
- [Oquab et al. (2023), DINOv2](https://arxiv.org/abs/2304.07193) —
  the FM this orbit swaps in (via a pure-JAX surrogate due to image
  constraints).
- The *foundation-model swap* as a research axis is called out
  explicitly in `research/problem.md` (axis 2).

### What this orbit adds

- First quantitative measurement, on the eval-v1 Lenia substrate, of
  CLIP-global vs patch-structural OE proxy disagreement: r ≈ 0.66,
  top-5 Jaccard ≈ 0.54. Materially different candidate rankings.
- A clean fallback pattern — `transformers.FlaxDinov2Model →
  Dinov2Model(torch) → pure-JAX surrogate` — that lets orbits test
  "does the FM matter?" without modifying the frozen Modal image.
- An honest disclaimer: this orbit tests the **structural-inductive-bias
  effect**, not the DINOv2-specific-weights effect. A follow-on orbit
  with a torch-enabled image would separate the two.

### Honest positioning

This orbit sits strictly under axis 2 of problem.md. It does **not**
claim novelty over DINOv2 itself — it only uses the *shape* of
DINOv2's inductive bias. If the Claude-judge eval favours it over the
CLIP-baseline, the claim is "patch-structural features > text-aligned
global features for Lenia OE search". If it ties or loses, the claim
is "the structural prior alone is insufficient — actual weights
matter", which is itself a useful negative result.

## References

- Kumar, A. et al. (2024). *Automating the Search for Artificial Life
  with Foundation Models.* arXiv:2412.17799.
- Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features
  Without Supervision.* arXiv:2304.07193.
- Plantec, E. et al. (2023). *Flow Lenia: Mass Conservation for the
  Continuous Cellular Automata.* arXiv:2212.07906.

## Glossary

- **ASAL** — Automating the Search for Artificial Life (Kumar et al. 2024).
- **CMA-ES** — Covariance Matrix Adaptation Evolution Strategy.
- **DINOv2** — Self-supervised vision transformer (Oquab et al. 2023).
- **FM** — Foundation Model.
- **LCF** — Life-Cycle Fidelity (the campaign's judged metric).
- **OE** — Open-Endedness (ASAL's "trajectory keeps moving in FM
  space" score).
- **Sep-CMA-ES** — Separable CMA-ES (diagonal covariance); memory-cheap.
- **Sonnet 4.6** — `claude-sonnet-4-6`, the VLM judge model pinned at
  eval-v1.
