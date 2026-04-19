---
issue: 7
parents: [02-asal-mature-organism]
eval_version: eval-v2
metric: null
---

# Research Notes — orbit 06-real-cmaes-mature

## TL;DR

Canonical EXTEND of orbit 02 (`02-asal-mature-organism`). Same mature-organism
3-prompt bank, same soft-CE scoring at t=127, same pop=32 / sigma_init=0.12,
but with **real Sep-CMA-ES** now that `evosax==0.1.6` is in the Modal image
(eval-v2). **No ImportError fallback** — orbit must crash honestly if evosax
is unavailable, so the batch-1 silent-random-search regression cannot recur.

**Research question.** Does real CMA-ES beat random-search (orbit 02's
accidental-RS baseline at LCF_judge Δ=+0.279) at this hypothesis? Predictions:
+5–15 % if the Lenia 8-D parameter landscape is smooth; ≈ 0 if it is flat
enough that random pop=32 already saturates.

## Why orbit 02's result is "accidental random search"

Orbit 02's solution.py wrapped `from evosax import Sep_CMA_ES` in
`try/except ImportError:` and under eval-v1 the Modal search container did
not ship evosax, so every orbit claiming Sep-CMA-ES silently ran the
fallback random-search loop. The two-part fix in eval-v2 is:

1. `evosax==0.1.6` added to `research/eval/modal_app.py::search_image`.
2. Fail-loud requirement baked into this orbit's solution.py (see below).

This orbit is the canonical "what if we actually ran CMA-ES" test, holding
every other knob (prompt bank, scoring geometry, population, budget) fixed
to orbit 02's exact values.

## Design decisions

### Fail-loud evosax import
`from evosax import Sep_CMA_ES` at the top of `search()` (not module-top, so
that `import solution` during search_worker startup still succeeds for the
non-solver branches — but the first search call crashes loud if evosax is
unavailable). No `try/except ImportError` anywhere in the file. If evosax
is missing, `search_worker.py` exits non-zero, `search_container` reports
`status="search_crash"` with a `stderr_tail` mentioning `ModuleNotFoundError:
No module named 'evosax'`, and the evaluator emits `METRIC=0.0` with a
`search_crash` diagnostic — exactly the honest signal we want.

### Inherited from orbit 02 verbatim
- Prompt bank (3 hand-written "mature organism" variants).
- Scoring: `TAU · (logsumexp(cos / TAU) − log n_prompts)` at frame t=127.
  This is a smooth approximation of `max_prompt cos(z_img, z_txt[i])` —
  the "take max across variants" hedge against CLIP text-encoder quirks.
- pop_size=32, sigma_init=0.12, n_gens ≤ 50.
- Module-level `_CLIP_CACHE` + `gc.collect() + jax.clear_caches()` every
  10 generations.
- `/tmp/ckpt/<run_id>_best.npy` checkpoint every 30 s (Guard 6 recovery).
- Wall-clock self-limit at `wall_clock_s − 15 s`.

### Provenance tag
`search_trace["algorithm"] = "sep_cma_es_mature_target_REAL_eval_v2"` so any
downstream audit distinguishes this run from orbit 02's accidental-RS.

## Prompt bank (frozen)

| ID | Prompt |
|----|--------|
| P1 | a mature self-sustaining organism with crisp geometric structure on a dark background |
| P2 | a bilaterally symmetric living cell with smooth clean boundaries |
| P3 | a soft glowing biological organism clearly distinct from empty space |

All three emphasise **Existence** and **Coherence** — the two tiers the
pinned ASAL baseline scores weakest on (per adversarial_findings_1.md).

## Methodology

1. Search container (Modal A100, `block_network=True`):
   - Subprocess `search_worker.py` loads solution.py.
   - Sep-CMA-ES on 8-D Lenia params, pop=32, 1800 s budget.
   - Every candidate: 1 rollout (T=256, 128×128) → pick frame t=127 → CLIP
     ViT-B/32 image embed → softmax-CE against 3 prompt embeddings.
   - Best params checkpointed periodically; final saved to `out.npy`.
2. Rollout + strip render → 16 sanitised PNG strips (1 per held-out seed).
3. Judge container (Modal CPU, Claude Sonnet 4.6): 16×3 = 48 calls, OCR
   canary check, median-of-3 per tier, geomean across 5 tiers → LCF_θ.
4. Final metric = LCF_θ − baseline_lcf (pinned 0.1185).

## Expected behavior

If CMA-ES *helps*:
- `best_proxy_per_gen` shows monotone improvement beyond generation 5–10.
- Final LCF_judge in `[0.30, 0.35]` range (orbit 02's 0.279 + 5–20 %).

If CMA-ES is *no better than random*:
- `best_proxy_per_gen` plateaus inside the first 3 generations.
- Final LCF_judge statistically indistinguishable from orbit 02's 0.279
  (Wilcoxon p ≫ 0.05 against orbit 02's per-seed scalars).

Either outcome is scientifically informative: it tells us whether the Lenia
8-D parameter landscape has enough structure for a well-known evolutionary
strategy to exploit it, or whether random search is on the efficient
frontier at this budget.

## Results

(To be filled after eval run.)

## Prior Art & Novelty

### What is already known
- Sep-CMA-ES (Ros & Hansen, 2008) is the standard separable-covariance
  CMA-ES; used by ASAL (Kumar et al., 2024, arXiv:2412.17799) as the
  default optimiser for all three objectives.
- CLIP ViT-B/32 (Radford et al., 2021) as an ASAL vision FM; the ASAL
  paper's supervised-target + OE + illumination objectives all build on
  frozen CLIP embeddings of rollout frames.

### What this orbit adds
- A provenance-clean test of the "real CMA-ES beats random search on the
  mature-organism hypothesis" question, holding every other knob fixed to
  orbit 02's values. Rules out the silent-fallback confound that corrupted
  batch 1.

### Honest positioning
No algorithmic novelty — this is a controlled replication study in the
context of a provenance bug. The scientific value is in isolating whether
the 0.279 lift reported by orbit 02 came from CMA-ES's adaptation or just
from random sampling at pop=32.

## References

- Kumar et al. (2024). "Automating the Search for Artificial Life with
  Foundation Models." arXiv:2412.17799.
- Ros, R. & Hansen, N. (2008). "A Simple Modification in CMA-ES Achieving
  Linear Time and Space Complexity." PPSN X.
- Radford et al. (2021). "Learning Transferable Visual Models From Natural
  Language Supervision." ICML.
