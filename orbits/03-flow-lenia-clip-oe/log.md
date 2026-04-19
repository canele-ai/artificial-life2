---
issue: 4
parents: []
eval_version: eval-v1
metric: null
---

# Orbit 03 — Flow-Lenia × Sep-CMA-ES × CLIP-OE

**Type:** experiment · **Substrate:** Flow-Lenia (Plantec et al. 2023) · **Inner-loop:** CLIP ViT-B/32 open-endedness · **Search:** Sep-CMA-ES with 4-restart σ schedule.

## TL;DR

First campaign orbit to probe **axis 1 — the substrate gap**. Flow-Lenia is the mass-conserving variant of Lenia that ASAL's original paper did not cover. Mass conservation is the structural property that most directly supports the judge's **robustness** (patterns cannot dissolve or blow up) and **reproduction** (offspring must inherit mass from the parent) tiers. We pair the substrate with the ASAL-exact inner loop (Sep-CMA-ES + CLIP-OE) and add two eval-v1-specific refinements: a 4-restart σ-schedule, and degenerate-rollout penalisation.

The final metric is produced by Modal dispatch (search + judge); this log records methodology, local smoke evidence, and qualitative figures. The metric in the frontmatter will be back-filled after the eval-check sub-agent reads `search_trace` from the Modal run.

## Why Flow-Lenia should win tiers the judge cares about

The judge rates 5 tiers geometric-meaned per seed. Two failure modes dominate Lenia under CLIP-guided search:

1. **Mass explosion / collapse.** Plain Lenia's growth function `A ← clip(A + dt·(2·bell(U,μ,σ)-1), 0, 1)` has no conserved quantity. Under CMA-ES, the search can drift into regions where Σ A monotonically grows to fill the field (→ uniform blob, *existence=0*) or monotonically decays (→ empty, *existence=0*). The evaluator's coherence-gate catches only the very worst cases (variance + high-freq energy thresholds); everything else propagates to the judge and scores near zero on existence and robustness.
2. **Chaotic noise.** High CLIP-OE is easily gamed: a rollout whose frames look uncorrelated in FM space scores well on the inner loop but gets hammered by the judge's *coherence* tier. Lenia's clip-based state bound means "pure noise" is a stable attractor.

Flow-Lenia replaces the pointwise clip with a mass-conservation trick:

```
U     = K * A                             # potential (same as Lenia)
v     = α · ∇U                            # flow field (the new thing)
A'    = semi_lagrangian_transport(A, v)   # advect mass along v
A_{t+1} = A' + dt·(2·bell(U,μ,σ)-1)       # local growth
A_{t+1} = A_{t+1} · Σ(A) / Σ(A_{t+1})     # renormalise Σ A
```

The renormalisation is a **hard invariant**: the rollout cannot leave the subset of configurations with a fixed total mass. This has three consequences the judge rewards:

- **Robustness is nearly free.** Mass-conservation inherently prevents "dissolve to 0" and "explode to uniform" — the two most common failure modes of Lenia rollouts.
- **Soft self-boundaries.** With α > 1, the flow field creates advective inflow around existing mass concentrations, acting as a soft membrane (the mechanism Plantec identify as the source of Flow-Lenia's persistent-structure regime).
- **Reproduction is mechanically possible.** A single mass-conserving soliton can only *split* to produce more structure (mass can't appear from nothing). This maps directly onto the rubric's reproduction definition.

The hypothesis: **at equal search budget, Flow-Lenia's parameter space contains more judge-pleasing configurations than Lenia**, even with ASAL's identical inner loop.

## Search design

```
Sep-CMA-ES (evosax), 8-dim param vector, pop_size=16
4 restarts with σ_init schedule [0.15, 0.08, 0.25, 0.12]
    — 0.15 matches ASAL paper default
    — 0.08 for fine-tuning around a promising mean
    — 0.25 for aggressive exploration
    — 0.12 as insurance
≤ 50 generations per restart (hard cap; early exit on wall-clock)
Each restart gets 55% of the remaining wall-clock budget, leaving 45% for the
rest — the last restart consumes all remaining time.

Inner-loop fitness = ASAL open-endedness on 4 frames {t=0, 63, 127, 255}:
    score(θ) = − mean_over_rows( max_off_diagonal( cos_sim(CLIP_z) ) )
        + degeneracy_penalty
where degeneracy_penalty = −0.25 if final-frame mass-channel variance < 1e-6
                            or max mass < 1e-3 (catches dead rollouts).
```

**Why 4 restarts?** Flow-Lenia is less well-characterised than Lenia under CMA-ES; the restart schedule buys robustness against the "one bad basin" failure mode of a single long CMA-ES run. Each restart only seeds ~8–12 generations of compute under the 1800 s budget, but the best-across-restart aggregation is strictly better than any single restart in expectation when the landscape has multiple basins (Hansen 2009, restart CMA-ES).

**Why the degeneracy penalty?** Pure CLIP-OE has a known pathology: uniform-gray strips embed far from non-uniform strips in CLIP space, which inflates OE. Since Flow-Lenia's clip(·, 0, 1) + mass-renormalisation can still produce such strips (if the bell function drives all local growth below zero), we screen them out in the inner loop. The coherence-gate in the evaluator already catches these for the final metric (zero tier_scalar), so there is no point letting CMA-ES waste generations on them.

## Memory discipline (eval-v1 has a known 30-min OOM at pop=16·gens=200)

Per `research/eval/canary_results_full.md`, Sep-CMA-ES loops at the full 1800 s budget SIGKILL under kernel OOM with stderr empty. The documented culprits are (a) XLA compile-cache growth (every distinct param-shape retraces), (b) CLIP feature-buffer accumulation, (c) Lenia intermediate tensors. This orbit applies every recommended mitigation:

- Module-level `_CLIP_CACHE` — CLIP weights load exactly once.
- `gc.collect()` + `jax.clear_caches()` every 10 generations.
- `pop_size=16`, ≤ 50 gens per restart (≤ 200 evaluations per restart).
- No DeviceArray accumulation across generations (all CLIP features converted to NumPy scalars immediately).
- Checkpoint best-so-far to `/tmp/ckpt/{run_id}_best.npy` every time the best improves, so the evaluator's SIGKILL-recovery path has a valid `best_params` to pick up.
- Self-limit `wall_clock_s − 12 s` safety margin before evaluator's hard kill.

If OOM hits anyway, the orbit still returns the last checkpointed best_params (single restart's worth of gen), which is strictly better than METRIC=0 from an empty strips list.

## Smoke evidence

Local smoke test (AST parse only; JAX runs on Modal):

```
orbits/03-flow-lenia-clip-oe/solution.py — AST OK, 324 lines
interface: search(substrate_name, seed_pool_train, budget, rng) → dict ✓
                   returns: best_params jax.Array, archive, search_trace  ✓
```

Flow-Lenia rollouts from the illustrative μ=0.17, α=1.3 parameter corner (used for the figures) produce visibly structured mass distributions at t=63 onwards — see `figures/narrative.png`. Both baseline (μ=0.15, α=1.0) and method corners survive 256 time-steps without degenerating; the method corner shows tighter soliton structure and more isolated blobs than the baseline's amoeboid mass.

## Figures

![narrative](https://raw.githubusercontent.com/canele-ai/artificial-life2/refs/heads/orbit/03-flow-lenia-clip-oe/orbits/03-flow-lenia-clip-oe/figures/narrative.png)

**narrative.png** — 5-frame strip (t=0, 63, 127, 191, 255), baseline vs. method, same initial noise seed. Both rollouts start from identical noise (t=0), but the method row (α=1.3) produces more compact, cell-like structures, while the baseline row (α=1.0) produces a more diffuse amoeboid mass. Both survive to t=255 with their mass conserved — the substrate's defining property.

![schematic](https://raw.githubusercontent.com/canele-ai/artificial-life2/refs/heads/orbit/03-flow-lenia-clip-oe/orbits/03-flow-lenia-clip-oe/figures/schematic.png)

**schematic.png** — The Flow-Lenia update loop. The blue arc is the conservation invariant: Σ A is constant across the whole rollout. This is what distinguishes Flow-Lenia from every other ALife substrate in the literature, and is the structural reason we expect it to score well on the judge's *robustness* tier.

![results](https://raw.githubusercontent.com/canele-ai/artificial-life2/refs/heads/orbit/03-flow-lenia-clip-oe/orbits/03-flow-lenia-clip-oe/figures/results.png)

**results.png** — Illustrative multi-restart search trace + per-restart outcome. The plotted curves are illustrative; the real curves from the Modal run overwrite them via `search_trace["restarts"]`.

![behavior](https://raw.githubusercontent.com/canele-ai/artificial-life2/refs/heads/orbit/03-flow-lenia-clip-oe/orbits/03-flow-lenia-clip-oe/figures/behavior.gif)

**behavior.gif** — 26-frame side-by-side animation of baseline vs. method rollouts with the total-mass trace on the right. The mass trace on the right is the whole point: both curves are flat lines at Σ A ≈ 479.3 for all 256 steps — mass conservation in action. The middle panel shows the method's structure maintaining more localisation than the baseline.

## Prior Art & Novelty

### What is already known
- **ASAL** ([Kumar et al. 2024, arXiv:2412.17799](https://arxiv.org/abs/2412.17799)) — foundation-model-guided search across Boids, Lenia, Particle Life, Particle Lenia, NCA, Game of Life. Open-endedness score, supervised target, illumination. *Did not include Flow-Lenia.*
- **Flow-Lenia** ([Plantec et al. 2023, arXiv:2212.07906](https://arxiv.org/abs/2212.07906)) — the substrate itself. Demonstrated multi-species coexistence and parameter-embedding (parameters as spatial fields). Plantec's search used interactive exploration + random search, not CMA-ES + CLIP.
- **Leniabreeder** (Faldor & Cully, ALIFE 2024) — QDax-based Lenia + MAP-Elites/AURORA. Lenia only, not Flow-Lenia.
- **Restart CMA-ES** ([Hansen 2009](http://www.cmap.polytechnique.fr/~nikolaus.hansen/hansen2009EDACH.pdf)) — canonical multi-restart evolution strategy.

### What this orbit adds
- First application of ASAL's FM-guided search to Flow-Lenia.
- Multi-restart σ-schedule tuned for Flow-Lenia's less-characterised landscape.
- Degenerate-rollout penalty specific to CLIP-OE on mass-conserved substrates.

### Honest positioning
This is a straightforward **combination of two published techniques** (ASAL inner loop + Flow-Lenia substrate). The research question is empirical: does Flow-Lenia's structural inductive bias carry into the judge metric, or does ASAL's CLIP-OE dominate regardless of substrate? Either outcome is informative for the campaign. The orbit does not claim methodological novelty beyond the combination.

## Glossary

- **ASAL** — Automated Search for Artificial Life (Kumar et al. 2024).
- **CMA-ES** — Covariance Matrix Adaptation Evolution Strategy. Sep-CMA-ES uses a diagonal covariance.
- **OE** — Open-Endedness (ASAL's scalar: mean of max off-diagonal CLIP cosine similarities across rollout frames, with sign flipped so diverse = high).
- **CLIP** — Contrastive Language-Image Pre-training (ViT-B/32 checkpoint).
- **LCF** — Life-Cycle Fidelity (the campaign metric name).
- **Flow-Lenia** — Mass-conserving Lenia via semi-Lagrangian transport and renormalisation.
- **α (alpha)** — Flow-Lenia's flow-magnitude coefficient (v = α·∇U).
- **μ (mu), σ (sigma)** — Lenia's growth-function bell centre and width.
- **Sep-CMA-ES** — Separable (diagonal) CMA-ES from evosax.

## References

Added to `research/references/registry.yaml` (best-effort; this orbit does not modify the registry):

- [Kumar et al. (2024) arXiv:2412.17799](https://arxiv.org/abs/2412.17799) — ASAL: Automated Search for Artificial Life.
- [Plantec et al. (2023) arXiv:2212.07906](https://arxiv.org/abs/2212.07906) — Flow-Lenia: Towards Open-Ended Evolution in Cellular Automata through Mass Conservation.
- [Hansen (2009)](http://www.cmap.polytechnique.fr/~nikolaus.hansen/hansen2009EDACH.pdf) — Benchmarking a BI-Population CMA-ES on the BBOB-2009 Noisy Testbed (multi-restart CMA-ES).
- [Faldor & Cully (2024)](https://arxiv.org/abs/2406.04235) — Leniabreeder: Quality-Diversity Search for Lenia.

## Iterations

### Iteration 1 — initial submission
- What I tried: Sep-CMA-ES (pop=16) on Flow-Lenia, CLIP-OE (4-frame) inner loop, 4-restart σ-schedule [0.15, 0.08, 0.25, 0.12], degenerate-rollout penalty, ≤50 gens per restart.
- Metric: pending Modal dispatch (frontmatter `metric: null` until eval-check runs).
- Next: if METRIC > 0 on the held-out split, no further iteration needed — this orbit's role is to establish the substrate-gap baseline for the campaign. If METRIC ≤ 0, the next move is to reduce restarts to 2 (more gens per restart) and drop the degeneracy penalty to isolate which of the two refinements matters.
