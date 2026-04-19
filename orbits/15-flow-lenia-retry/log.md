---
issue: 16
parents: [09-flow-lenia-real-cmaes]
eval_version: eval-v2
metric: null
---

# Research Notes — orbit 15-flow-lenia-retry

## Hypothesis (campaign axis 1 — substrate gap)

Flow-Lenia (Plantec et al. 2023, arXiv:2212.07906) enforces mass conservation
via semi-Lagrangian transport + Σ-renormalisation. The conjecture: that
invariant translates into higher judge scores on tiers that reward persistent
identity (existence, robustness, reproduction) relative to plain Lenia.

ASAL did not test Flow-Lenia; no orbit in this campaign has yet produced a
non-crashed Flow-Lenia result under eval-v2.

## Why this is a retry (not a new idea)

Orbit 09 (`09-flow-lenia-real-cmaes`, issue #10) was the first attempt to
test this hypothesis under eval-v2. It crashed inside the Modal search
container with `preprocessor_config.json` missing from `/cache/hf`.

Root cause: `search_container` had `block_network=True`, so when the mounted
HuggingFace cache Volume was missing a file from the CLIP snapshot, the
container could not refetch it — and `CLIPProcessor.from_pretrained(...)`
errored out before the first generation.

Infra fix (commit `08e6131`, "eval-v2 infra: drop block_network=True on
search_container") removed that guard. The cache can now repopulate
transparently. This orbit re-runs orbit 09's solution verbatim on the fixed
infra.

Any delta here is attributable to the infra fix alone — the search code,
substrate, and scoring proxy are identical to orbit 09.

## Approach

- Substrate: `flow_lenia` (NOT `lenia`; this is the critical axis-1 difference)
- Inner-loop proxy: ASAL open-endedness (mean max off-diagonal CLIP cos-sim,
  sign flipped for maximisation) over 4 frames at t ∈ {0, 63, 127, 255}
- Outer-loop: Sep-CMA-ES (evosax 0.1.6), pop=16, up to 4 restarts with
  σ_init schedule [0.15, 0.08, 0.25, 0.12], ≤50 gens per restart
- Degenerate-rollout penalty: −0.25 if final-frame mass variance < 1e-6 or
  max < 1e-3 (prevents dead/uniform rollouts from inflating CLIP OE)
- Memory discipline: module-level `_CLIP_CACHE`, `jax.clear_caches()` +
  `gc.collect()` every 10 generations
- Best-so-far checkpointed to `/tmp/ckpt/{run_id}_best.npy` after every
  improvement, so the evaluator's `search_timeout` SIGKILL path can recover
- Fail-loud: `from evosax import Sep_CMA_ES` and
  `from transformers import CLIPProcessor, FlaxCLIPModel` are NOT wrapped in
  try/except — if the container is mis-provisioned, the run crashes with
  legible stderr rather than silently falling back to random search

`search_trace["algorithm"] = "sep_cma_es_flow_lenia_eval_v2_infra_fixed"` is
the provenance tag; it also carries `infra_fix_commit: "08e6131"` and
`parent_orbit: "09-flow-lenia-real-cmaes"`.

## Prior Art

- Plantec et al. 2023 (arXiv:2212.07906) — Flow-Lenia formulation
- Kumar et al. 2024 (arXiv:2412.17799) — ASAL open-endedness proxy + baselines
- Hansen & Ostermeier 2001 — CMA-ES; Sep-CMA-ES is Hansen's separable variant
- Orbit 07 (`07-real-cmaes-dinov2`) is the current best eval-v2 result
  (+0.319) on the *lenia* substrate, so this orbit tests whether the
  substrate switch holds up when the outer loop is weaker (CLIP OE vs DINOv2).

## Iteration 1

- What I tried: inherit orbit 09's solution verbatim; bump provenance tags to
  reflect the infra fix and new parent
- Metric: pending eval
- Next: launch 3-seed eval on Modal; compare against orbit 09 (crashed) and
  the lenia-substrate leaders (07 = +0.319)
