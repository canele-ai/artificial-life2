---
issue: 18
parents: [15-flow-lenia-retry]
eval_version: eval-v2
metric: null
---

# Research Notes — Multi-Seed Robustness Check

## Purpose

Orbit 15 (`orbit/15-flow-lenia-retry`) produced a leader metric of **+0.339 at seed=1 only**.
Every orbit in this campaign to date has been evaluated on a single seed (seed=1).
The problem_spec calls for a paired Wilcoxon over 16 seeds; before we can make any claim
about the leader being real, we need to characterise seed-variance.

This orbit contributes **no algorithmic novelty**. It is a deliberate replica of orbit 15:
the same Flow-Lenia substrate + Sep-CMA-ES + CLIP open-endedness solver, loaded under a
new orbit identity so the external Modal eval dispatch can track seeds 2, 3, … separately
and report dispersion on this orbit's row of the landscape.

## Method

- `solution.py` is a **verbatim copy** of `orbits/15-flow-lenia-retry/solution.py` at
  branch `orbit/15-flow-lenia-retry` (326 lines, byte-identical).
- No hyperparameter changes, no algorithmic changes. `search_trace["algorithm"]` matches
  the parent exactly so cross-seed comparison is apples-to-apples.
- The evaluator harness (eval-v2 + infra fix commit 08e6131) dispatches this solution
  across seed_pool={1,2,3} externally; the orbit code itself is seed-agnostic through
  the `rng` argument.

## Prior Art & Novelty

### What is already known
- Orbit 15 — Flow-Lenia + Sep-CMA-ES + CLIP-OE → +0.339 at seed=1.
- Seed variance on CMA-ES-driven evolutionary search is well documented (Hansen 2016);
  a single-seed peak of this magnitude warrants ≥ 3-seed replication before acceptance.

### What this orbit adds
- **No method novelty.** Contribution is purely statistical: variance characterisation
  of the current leader.

### Honest positioning
This is bookkeeping, not a new idea. If seeds 2-3 land near +0.339, the leader holds;
if they regress to baseline, orbit 15's result was a lucky seed and the campaign needs
to revisit.

## Results

(Populated after eval dispatch across seeds 1/2/3.)

| Seed | Metric | Time |
|------|--------|------|
| 1    | TBD    |      |
| 2    | TBD    |      |
| 3    | TBD    |      |
| **Mean** | **TBD** | |

## Glossary

- **Sep-CMA-ES** — Separable Covariance Matrix Adaptation Evolution Strategy
- **CLIP-OE** — CLIP-based open-endedness objective (ASAL family)
- **Flow-Lenia** — mass-conserving continuous cellular automaton variant of Lenia
