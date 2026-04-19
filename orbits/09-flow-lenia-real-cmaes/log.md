---
issue: 10
parents: [03-flow-lenia-clip-oe]
eval_version: eval-v2
metric: 0.000000
---

# Orbit 09 — Flow-Lenia × Sep-CMA-ES × CLIP-OE  (eval-v2 re-run of orbit 03)

**Type:** experiment · **Substrate:** Flow-Lenia (Plantec et al. 2023) · **Inner-loop:** CLIP ViT-B/32 open-endedness · **Search:** Sep-CMA-ES (evosax) with 4-restart σ schedule.

## TL;DR

This orbit re-submits **orbit 03's hypothesis unchanged** on the eval-v2 harness. Orbit 03 crashed at eval-v1 with `ModuleNotFoundError: evosax` — it was the *only* orbit that failed loudly instead of silently falling back to random search, because its `solution.py` lacked the swallowing `try/except ImportError` that the common `good.py` template carried. eval-v2 (commit 7fbd935) fixed the pipeline: `evosax==0.1.6` is now pip-installed into the Modal search container, `env_audit` records `evosax_version`, and `good.py` is fail-loud. The research question is unchanged: **does Flow-Lenia's mass-conservation invariant translate into higher judge scores now that the CMA-ES machinery actually works?**

## Provenance story — why re-run orbit 03 at all

At the batch-1 milestone it became clear that every orbit in the first campaign batch that *claimed* "Sep-CMA-ES + evosax" had silently run uniform-random search:

```
good.py (eval-v1 template):
    try:
        from evosax import Sep_CMA_ES
    except ImportError:
        Sep_CMA_ES = None          # silent fallback
    ...
    if Sep_CMA_ES is None:
        scores = rng.uniform(-0.2, 0.2, size=pop)   # ← random search
```

The Modal `search_image` never pip-installed `evosax`, so every orbit that inherited this template ran random search and reported `algorithm="sep_cma_es_*"` in their search_trace. This is a silent-provenance bug of exactly the kind the campaign's replica-panel is supposed to catch.

Orbit 03's `solution.py` did NOT carry the swallowing try/except — it did `from evosax import Sep_CMA_ES` at module scope in `_run_restart`, so the missing import surfaced as a `ModuleNotFoundError` in the subprocess's stderr, the evaluator returned `status="search_crash"`, and the METRIC was pinned to 0.0. **Orbit 03 is the canary that proved the rest of batch-1 was lying.**

eval-v2 (commit 7fbd935) is the minimal correction:

| change | file | effect |
|--------|------|--------|
| add `evosax==0.1.6` | `modal_app.py` | CMA-ES now imports inside Modal |
| bump `EVAL_VERSION` | `evaluator.py` | pins all future metrics to v2 harness |
| record `evosax_version` | `evaluator._env_audit` | Guard 11 provenance signal |
| remove silent fallback | `examples/good.py` | future orbits fail loudly too |
| relabel baseline | `baseline/baseline_score.json` | HONEST_ANCHOR (still random search) |

The `BASELINE_SHA256` changed (new metadata); the numeric LCF value stayed at **0.1185** because the random-search anchor is a perfectly valid reference — we just relabelled it honestly instead of mislabelling it "ASAL Sep-CMA-ES".

## Hypothesis (unchanged from orbit 03)

Flow-Lenia replaces Lenia's pointwise `clip(A + dt·growth, 0, 1)` with:

```
U    = K * A                                  # potential (same as Lenia)
v    = α · ∇U                                 # flow field (the new thing)
A'   = semi_lagrangian_transport(A, v)        # advect mass along v
A''  = clip(A' + dt·(2·bell(U,μ,σ)-1), 0, 1)  # local growth
A_{t+1} = A'' · Σ(A) / Σ(A'')                 # renormalise — HARD invariant
```

The renormalisation is structural. The rollout **cannot** leave the constant-mass subset of configurations. Two judge-tier consequences:

1. **Existence & robustness are nearly free.** Mass conservation inherently prevents the two most common Lenia failure modes: dissolve-to-zero and explode-to-uniform. The evaluator's coherence gate catches only the worst cases (variance + high-freq energy thresholds); everything else propagates to the judge. Flow-Lenia mechanically avoids the class of rollouts that collapse the first-tier `existence` gate to zero and zero-out the geometric-mean scalar.
2. **Reproduction is mechanically possible.** A mass-conserving soliton can only *split* to produce more structure. This maps directly onto the rubric's reproduction definition, which plain Lenia's growth-clip rule cannot guarantee.

Together these imply that **at equal search budget, Flow-Lenia's parameter space should contain more judge-pleasing configurations than Lenia**, even with the identical ASAL inner loop.

## Search design (inherited from orbit 03 verbatim)

```
Sep-CMA-ES (evosax), 8-dim param vector, pop_size = 16
4 restarts with σ_init schedule [0.15, 0.08, 0.25, 0.12]
    — 0.15 matches ASAL paper default
    — 0.08 fine-tunes around a promising mean
    — 0.25 aggressive exploration
    — 0.12 insurance
≤ 50 generations per restart (hard cap; early exit on wall-clock)
Each restart takes 55 % of the remaining wall-clock budget.

Inner-loop fitness = ASAL open-endedness on 4 frames {t=0, 63, 127, 255}:
    score(θ) = − mean_over_rows( max_off_diagonal( cos_sim(CLIP_z) ) )
             + degeneracy_penalty
where degeneracy_penalty = −0.25 if final-frame mass variance < 1e-6
                           or max mass < 1e-3 (catches dead rollouts).
```

The 4-restart σ-schedule is motivated by Flow-Lenia being less well-characterised than Lenia under CMA-ES (Hansen 2009 multi-restart argument). The degeneracy penalty addresses a known pathology of raw CLIP-OE: uniform-gray strips CLIP-embed far from non-uniform strips, which *inflates* the OE score for dead rollouts. The coherence gate would zero these out downstream anyway; screening them in the inner loop just stops CMA-ES from wasting generations on them.

## Memory discipline (unchanged from orbit 03)

Per `research/eval/canary_results_full.md`, Sep-CMA-ES loops at the full 1800 s budget were SIGKILLed under kernel OOM in eval-v1 testing, with stderr empty. Documented culprits: (a) XLA compile-cache growth (every distinct param-shape retraces), (b) CLIP feature-buffer accumulation, (c) Lenia intermediate tensors. Mitigations carried over verbatim from orbit 03:

- Module-level `_CLIP_CACHE` — CLIP weights load exactly once per process.
- `gc.collect()` + `jax.clear_caches()` every 10 generations.
- `pop_size=16`, ≤ 50 gens per restart (≤ 200 evaluations per restart).
- Per-candidate CLIP features converted to NumPy scalars immediately; no DeviceArray accumulation across generations.
- Best-so-far checkpoint to `/tmp/ckpt/{run_id}_best.npy` on every improve, so SIGKILL recovery has a valid `best_params` to pick up.
- Self-limit `wall_clock_s − 12 s` safety margin.

## Fail-loud contract (eval-v2 explicit)

Unlike the batch-1 `good.py` template, this orbit's `solution.py` has **no** `try/except ImportError` around either `evosax` or `transformers`. Missing dependencies will surface as `ModuleNotFoundError` → subprocess crash → `status="search_crash"` → METRIC=0.0 with a populated `stderr_tail`. The `search_trace["algorithm"]` tag `sep_cma_es_flow_lenia_REAL_eval_v2` is a positive provenance signal: it only appears if the import succeeded, the restart loop ran, and the solution returned cleanly.

## Smoke evidence (local)

```
orbits/09-flow-lenia-real-cmaes/solution.py — AST OK, 330 lines
top-level fns: [_load_clip, _oe_score, _fitness_for_candidate, _batch_fitness, _run_restart, search]
search(substrate_name, seed_pool_train, budget, rng) → dict ✓
    returns: best_params jax.Array, archive, search_trace  ✓
```

Local Flow-Lenia rollouts at the illustrative (μ=0.15, α=1.0) and (μ=0.17, α=1.3) corners produce visibly structured mass distributions at t=63 onwards — see `figures/narrative.png`. The mass-drift trace at the bottom shows both rollouts stay within ±0.5 % of their initial mass across all 256 steps, confirming the substrate's defining invariant.

## Figures

![narrative](https://raw.githubusercontent.com/canele-ai/artificial-life2/refs/heads/orbit/09-flow-lenia-real-cmaes/orbits/09-flow-lenia-real-cmaes/figures/narrative.png)

**narrative.png** — 5-frame strip (t=0, 63, 127, 191, 255), baseline vs. method, same initial noise seed. Both rollouts start from identical noise (t=0); the method row (α=1.3) produces more compact, isolated cell-like structures from t=63 onwards, while the baseline row (α=1.0) keeps a more diffuse amoeboid connectivity. The bottom panel plots Σ A drift (% relative to t=0) for both rollouts — both stay within ±0.5 % across all 256 steps, which is the substrate's defining invariant and the structural reason we expect it to score well on the judge's *robustness* tier.

![schematic](https://raw.githubusercontent.com/canele-ai/artificial-life2/refs/heads/orbit/09-flow-lenia-real-cmaes/orbits/09-flow-lenia-real-cmaes/figures/schematic.png)

**schematic.png** — (inherited from orbit 03) the Flow-Lenia update loop. The blue arc is the conservation invariant: Σ A is constant across the whole rollout. This is what distinguishes Flow-Lenia from every other ALife substrate in ASAL's coverage.

![results](https://raw.githubusercontent.com/canele-ai/artificial-life2/refs/heads/orbit/09-flow-lenia-real-cmaes/orbits/09-flow-lenia-real-cmaes/figures/results.png)

**results.png** — left panel: illustrative multi-restart CMA-ES trace with four σ₀ values (to be overwritten post-Modal from `search_trace["restarts"]`). Right panel: where this orbit's metric is expected to land — the HONEST_ANCHOR random-search baseline at 0.1185, the `anchor + 0.02` pass bar, orbit 03's eval-v1 `search_crash` zero, and this orbit's TBD position to be filled in after the eval-check agent posts the Modal result.

![behavior](https://raw.githubusercontent.com/canele-ai/artificial-life2/refs/heads/orbit/09-flow-lenia-real-cmaes/orbits/09-flow-lenia-real-cmaes/figures/behavior.gif)

**behavior.gif** — 30-frame animated side-by-side of baseline vs. method rollouts with a live Σ A drift indicator on the right. The drift dot tracks the relative mass deviation at the current time-step; both stay within ±0.5 % for the entire rollout. Mass conservation is the whole point of the substrate, and this figure is the quickest way to see it on screen.

## Prior art & novelty

Identical to orbit 03. See that log for the full citation list; the delta in this orbit is purely operational (eval-v2 pipeline correction), not methodological.

### What is already known
- **ASAL** ([Kumar et al. 2024, arXiv:2412.17799](https://arxiv.org/abs/2412.17799)) — FM-guided search across 6 substrates; did not include Flow-Lenia.
- **Flow-Lenia** ([Plantec et al. 2023, arXiv:2212.07906](https://arxiv.org/abs/2212.07906)) — the substrate.
- **evosax** (Lange 2023) — JAX-native Sep-CMA-ES.
- **Restart CMA-ES** ([Hansen 2009](http://www.cmap.polytechnique.fr/~nikolaus.hansen/hansen2009EDACH.pdf)).

### What this orbit adds
Nothing methodologically new beyond orbit 03's claims. The contribution is **empirical closure** on the batch-1 silent-provenance bug: by re-running orbit 03's hypothesis on a harness where `evosax` actually works, we get the first Flow-Lenia × CLIP-OE × real-Sep-CMA-ES data point in the campaign.

### Honest positioning
Exploratory re-run to close a process gap. If METRIC > 0.0 on eval-v2, the finding is: *Flow-Lenia + real CMA-ES beats random-search anchor.* If METRIC ≤ 0.0, the finding is: *the inner-loop CLIP-OE proxy does not transfer to judge-tier performance on Flow-Lenia even with functional CMA-ES*, and the next move is to change the inner loop (ASAL supervised-target, DINOv2-OE, or life-cycle-fidelity directly).

## Iterations

### Iteration 1 — eval-v2 submission (this commit)
- **What I tried:** inherited orbit 03's solution verbatim, removed the fallback safeguards that would have hidden a broken pipeline, tagged `algorithm="sep_cma_es_flow_lenia_REAL_eval_v2"` for provenance.
- **Metric:** pending Modal dispatch — frontmatter `metric: null` until the eval-check sub-agent reads `search_trace` from the Modal run and back-fills the mean.
- **Next:** if METRIC > anchor+0.02 on the held-out split, exit (target met for this research question). If METRIC ∈ [anchor, anchor+0.02), still a positive data-point vs orbit 03's METRIC=0, but not significant — pivot to DINOv2 or life-cycle-fidelity inner loop in a descendent orbit. If METRIC ≤ anchor, pivot: the Flow-Lenia substrate gain is less than the CLIP-OE ceiling on this campaign.

## Glossary

- **ASAL** — Automated Search for Artificial Life (Kumar et al. 2024).
- **CMA-ES** — Covariance Matrix Adaptation Evolution Strategy. Sep-CMA-ES uses a diagonal covariance.
- **OE** — Open-Endedness (ASAL's scalar: mean of max off-diagonal CLIP cosine similarities across rollout frames, with sign flipped so diverse=high).
- **CLIP** — Contrastive Language-Image Pre-training (ViT-B/32 checkpoint).
- **LCF** — Life-Cycle Fidelity (the campaign metric name).
- **HONEST_ANCHOR** — eval-v2 label for the frozen 0.1185 baseline produced by random-search + CLIP-OE; was mislabelled "ASAL Sep-CMA-ES" in eval-v1.
- **Flow-Lenia** — Mass-conserving Lenia via semi-Lagrangian transport and renormalisation.
- **α (alpha)** — Flow-Lenia's flow-magnitude coefficient (v = α·∇U).
- **μ (mu), σ (sigma)** — Lenia's growth-function bell centre and width.
- **Sep-CMA-ES** — Separable (diagonal) CMA-ES from evosax.
- **Guard 11** — eval-v2 env_audit signal: records `evosax_version` so `missing` appears in the audit rather than hiding behind a silent fallback.

## References

- [Kumar et al. (2024) arXiv:2412.17799](https://arxiv.org/abs/2412.17799) — ASAL.
- [Plantec et al. (2023) arXiv:2212.07906](https://arxiv.org/abs/2212.07906) — Flow-Lenia.
- [Hansen (2009)](http://www.cmap.polytechnique.fr/~nikolaus.hansen/hansen2009EDACH.pdf) — Multi-restart CMA-ES.
- [Faldor & Cully (2024) arXiv:2406.04235](https://arxiv.org/abs/2406.04235) — Leniabreeder.
- Lange (2023) — evosax: JAX-native evolution strategies. <https://github.com/RobertTLange/evosax>
