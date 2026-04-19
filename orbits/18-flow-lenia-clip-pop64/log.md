---
issue: 19
parents: [15-flow-lenia-retry]
eval_version: eval-v2
metric: -0.118516
---

# Research Notes — Flow-Lenia × Sep-CMA-ES pop=64

## Hypothesis — does Flow-Lenia re-open the compute scaling that saturated on Lenia?

On plain Lenia, Sep-CMA-ES population scaling saturated flat:

| pop | LCF improvement |
|-----|-----------------|
| 16  | +0.319 |
| 64  | +0.247 |
| 128 | +0.325 |

More CMA-ES compute did not translate to better judge scores on Lenia — the
substrate manifold appears to be small enough that pop=16 already covers it
and pop=64/128 just burns budget. The question for this orbit: does
**Flow-Lenia's larger/richer manifold** (semi-Lagrangian transport +
Σ-renormalisation giving multi-species coexistence, soft boundaries,
persistent identity) re-open that scaling?

Parent orbit 15 (`15-flow-lenia-retry`) leads the campaign at **+0.339 LCF
at pop=16**. If Flow-Lenia's manifold is genuinely richer than Lenia's,
quadrupling the Sep-CMA-ES population should find candidates that pop=16
missed — e.g. rare multi-species equilibria that the judge rewards on
existence/robustness/reproduction.

## Approach — inherit orbit 15 verbatim, flip one knob

- Substrate: `flow_lenia` via the frozen `_flow_lenia_rollout` +
  `_render_flow_lenia` in `research/eval/evaluator.py`.
- Search: Sep-CMA-ES (evosax) over the same 8-dim Flow-Lenia vector.
- Inner loop: CLIP-OE (ViT-B/32), same 4 frames, same degeneracy penalty.
- Same module-level `_CLIP_CACHE`, same SIGKILL-safe checkpointing.

Single knob change vs. orbit 15: **`pop_size` 16 → 64**. Budget-balancing:

- `max_gens_per_restart`: 50 → 15 (1500 s / (64 × ~1.5 s/cand) ~ 15–20 gens
  per restart; keep 15 with wall-clock margin).
- `σ_init` schedule: 4 restarts → 2 restarts (`0.10`, `0.20`). pop=64 needs
  more gens per restart to converge, so fewer restarts is the right trade.
- `clear_caches()` cadence: every 10 gens → every 5 gens (4× more CLIP
  forward passes per gen — OOM risk is higher).

Provenance tag: `search_trace["algorithm"] = "sep_cma_es_flow_lenia_clip_pop64_eval_v2"`.

## Prior Art & Novelty

### What is already known
- Flow-Lenia (Plantec et al. 2023, arXiv:2212.07906) — mass-conserving Lenia.
- ASAL (Kumar et al. 2024) — CLIP-OE inner loop + Sep-CMA-ES outer loop.
- Orbit 15 established +0.339 LCF for Flow-Lenia × Sep-CMA-ES × CLIP at pop=16.

### What this orbit adds
- Empirical test of whether the Lenia compute-saturation curve generalises to
  Flow-Lenia, or whether the richer substrate breaks that plateau.
- No algorithmic novelty — pure scaling study.

### Honest positioning
This is a compute-scaling ablation on the leading orbit. If pop=64 is within
noise of pop=16 on Flow-Lenia too, the saturation is substrate-independent.
If pop=64 materially beats +0.339, Flow-Lenia is genuinely richer than Lenia
and downstream orbits should continue pushing population.

## References
- Plantec et al. 2023 — [Flow-Lenia (arXiv:2212.07906)](https://arxiv.org/abs/2212.07906)
- Kumar et al. 2024 — ASAL
- Orbit 15 (`15-flow-lenia-retry`) — current leader (+0.339 LCF)
