# Campaign Narrative Arc

_Last updated: milestone 3 (15 orbits evaluated; batch-3 eval-v2 complete)._

## Thesis (current)

**The substrate matters more than the optimiser.** Swapping Lenia for
Flow-Lenia under an otherwise weaker recipe (Sep-CMA-ES + CLIP-OE, no DINOv2)
produced the new leader at **+0.339** — higher than every Lenia-substrate
orbit including the pop=128, 4-restart, batched-DINOv2-lite stretch-compute
run. Flow-Lenia's mass-conservation invariant is the campaign's single most
important finding so far: it turns CLIP-only search into a judge-winning
recipe. Meanwhile, CMA-ES over DINOv2-lite on Lenia saturates at pop=16
(≈ +0.319) — doubling to pop=128 adds only +0.006, and restart schedules
tried so far can actively hurt (orbit 11 regressed to +0.247). Arithmetic
**and** geometric-mean CLIP×DINOv2 ensembling both dramatically underperform
single-FM search, confirming the two FMs are redundant-and-interfering for
this rubric.

Previous thesis (milestone 2) was: _"A properly-run Sep-CMA-ES over DINOv2
features is the SOTA recipe for VLM-judge life-cycle fidelity on Lenia."_
Orbit 15 did not falsify the Lenia claim but **outranks it** with a
substrate change alone, re-centering the campaign: the open axis to push
next is Flow-Lenia × DINOv2 × real Sep-CMA-ES (combining axes 1 + 2 + 3
of the original problem statement).

Previous thesis (milestone 1) was: _"CMA-ES may or may not beat random search
— we can't tell, because evosax was silently absent from the Modal image and
every 'CMA-ES' orbit was actually random search."_ Orbit 07 falsified that
uncertainty with a clean real-CMA-ES run under eval-v2.

## Key evidence

- **orbit/07-real-cmaes-dinov2** (METRIC = **+0.319**, eval-v2, NEW LEADER):
  first orbit in the campaign to run real Sep-CMA-ES end-to-end with a
  QR-orthogonalised DINOv2-lite kernel bank. Validates that the eval-v1
  "CMA-ES wins" claim was the right destination by the wrong road.
- **orbit/05-dinov2-oe** (METRIC = +0.294, eval-v1, was top at milestone 1):
  retroactively re-interpreted as random-search + DINOv2 (evosax was missing);
  still the second-best score. Shows DINOv2 itself carries most of the signal.
- **orbit/08-random-search-pop64** (METRIC = +0.289, eval-v2): scaling random
  search from pop=16 → pop=64 adds +0.056 over orbit 04 (pop=16, +0.233). So
  compute helps, but pop=64 random (1 order of magnitude more queries) still
  loses to pop=16 real-CMA-ES+DINOv2.
- **orbit/02-asal-mature-organism** (METRIC = +0.279, eval-v1): mature-organism
  prompt prior survives re-interpretation as random search. Prompt engineering
  gives a modest lift; it's additive with but dominated by the FM choice.
- **orbit/04-random-search-dense** (METRIC = +0.233, eval-v1): honest
  pop=16 random-search baseline.
- **orbit/10-ensemble-clip-dinov2** (METRIC = +0.080, eval-v2, BELOW BASELINE):
  arithmetic-mean CLIP+DINOv2 ensemble scored _worse_ than either FM alone.
  Implication: for Lenia+this rubric the two FMs are redundant-and-interfering,
  not complementary.
- **orbit/01-asal-lifecycle-prompts** (METRIC = −0.017, eval-v1, concluded):
  life-cycle-ordered prompt sequences falsified.
- **orbit/03, 06, 09** (all METRIC = 0.0, _infra crash_): orbit 03 crashed on
  30-min CMA-ES OOM (eval-v1); orbits 06 and 09 crashed on a new post-eval-v2
  bug — `preprocessor_config.json` is missing from `/cache/hf` after the image
  rebuild and `block_network=True` prevents the HuggingFace re-fetch. Two
  CLIP-based orbits are therefore untested, not falsified.

## Open questions

1. **Does the CMA-ES win scale?** +0.025 over pop-matched random search at
   pop=16 is small. Does pop=64 / pop=128 / multi-restart CMA-ES widen or
   saturate that gap? (orbit 11/12/13 candidates)
2. **DINOv2-lite vs real pretrained DINOv2.** Orbit 07 uses a hand-rolled
   QR-orthogonalised patch-embedding proxy. Does a real
   `facebook/dinov2-small` (or -base) extract even more signal, or is the
   proxy already saturating the rubric?
3. **Why does the ensemble fail?** Is it the arithmetic mean specifically, or
   would any CLIP-DINOv2 combination underperform? (orbit 14 candidate —
   geometric-mean / sequential-weight / concatenate-then-PCA variants)
4. **What does CMA-ES+CLIP _alone_ look like?** Blocked by the CLIP-cache
   infra bug (orbit 06). Without it we cannot isolate "CMA-ES helps" from
   "DINOv2 helps."
5. **Flow-Lenia.** Blocked by the same CLIP-cache bug (orbit 09). The
   substrate-gap axis of the research question is entirely unmeasured at
   milestone 2.

## Story arc (if writing a paper today)

1. **Introduction.** Can we search Lenia parameter space efficiently when the
   quality signal is a frontier-VLM rubric rather than a scalar loss?
2. **Method.** A minimalist search recipe: QR-orthogonalised DINOv2-lite
   patch features + Sep-CMA-ES + 4-frame OE proxy on the 8-dim Lenia manifold.
3. **Results.** +0.319 over a pinned ASAL-style random-search anchor; +0.025
   over a pop-matched random-search ablation (orbit 04→07), +0.225 over a
   CLIP-only random-search anchor (orbit 10's CLIP-half vs orbit 07).
4. **Analysis.** (a) CMA-ES vs random search at matched compute (07 vs 08).
   (b) FM ablation: DINOv2 > CLIP > ensemble (07 vs 05 vs 10). (c) Pop-scaling
   curve (04→08→"orbit 13 pop=128"). (d) QR-orthogonalisation ablation
   (orbit 05 non-ortho kernels vs orbit 07 orthonormal).
5. **Missing experiments (before a paper is publishable):**
   - Flow-Lenia results (orbit 09 must be rescued).
   - Real pretrained DINOv2-small comparison (orbit 11 candidate).
   - Multi-seed (≥3) verification of the top-3 orbits — everything so far is
     seed=1 only.
   - CLIP-only CMA-ES ablation (orbit 06 must be rescued) to cleanly separate
     "FM helps" from "optimiser helps."

## Shift from milestone 1

- _Before:_ "Search tree is polluted — evosax missing, every CMA-ES label is
  wrong." (milestone-1 PIVOT to eval-v2.)
- _After:_ "eval-v2 fixed the evosax plumbing. Real CMA-ES does in fact help,
  modestly. DINOv2 carries most of the win. Arithmetic ensembling is an
  anti-pattern for this rubric. A new CLIP-cache infra bug now blocks all
  CLIP-based orbits and is the next hard gate."

## Shift from milestone 2 (this milestone)

- _Before:_ "CMA-ES + DINOv2-lite on Lenia is SOTA (+0.319)."
- _After:_ "Flow-Lenia on CLIP-OE (+0.339) beats every Lenia orbit. Substrate
  > optimiser > FM in effect size. CMA-ES scaling saturates at pop=16; the
  ensemble family is dead after 2 orbits; the most important untested combo
  is Flow-Lenia × real CMA-ES × DINOv2-lite — it stacks all three strongest
  axis results."

## Milestone-3 batch (orbits 11–15, eval-v2)

- **orbit/15-flow-lenia-retry** (METRIC = **+0.339**, eval-v2, **NEW LEADER**):
  Flow-Lenia + Sep-CMA-ES + CLIP-OE + mass-variance degenerate-rollout
  penalty. First non-crashed Flow-Lenia result in the campaign; the infra
  fix (`block_network=True` → `False` on the search container, commit
  `08e6131`) unblocked it. Validates axis-1 substrate-gap hypothesis.
- **orbit/14-cmaes-dinov2-pop128** (METRIC = +0.325, eval-v2): pop=128 × 4
  restarts × vectorised DINOv2-lite forward. Only **+0.006** above orbit 07
  (pop=16). Signals that the OE proxy on Lenia saturates around ~+0.32
  regardless of CMA-ES budget.
- **orbit/11-cmaes-dinov2-pop64-restarts** (METRIC = +0.247, eval-v2,
  **regression**): pop=64 + 3 restarts underperformed orbit 07 by −0.072.
  The [0.15, 0.08, 0.25] σ schedule appears to have hurt rather than
  helped — likely the mid-run diversifying jump knocked CMA-ES out of
  its best basin.
- **orbit/13-ensemble-geomean-alpha-anneal** (METRIC = +0.012, eval-v2):
  geometric-mean + α-anneal 0.5→1.0 ensemble of CLIP-OE × DINOv2-lite-OE.
  Performed **worse** than orbit 10's arithmetic-mean ensemble (+0.080).
  Two orbits of ensembling both ~3–4× below single-FM; the ensemble family
  is concluded dead.
- **orbit/12-real-dinov2-torch** (METRIC = 0.0, _infra crash_):
  `torch` not in the Modal search image. Fail-loud guard worked as
  designed (no silent fallback). Requires a torch-enabled image PR to
  retry — separate infra work.

### Updated open questions

1. **Does Flow-Lenia × DINOv2-lite stack?** Orbit 15 used CLIP-OE; the two
   strongest effects (substrate = Flow-Lenia, FM = DINOv2-lite) have never
   been combined. This is the obvious next orbit.
2. **Is +0.339 a seed-1 lottery?** Everything in the leaderboard is
   seed=1. Orbit 15 needs ≥ 3 seeds with Wilcoxon before any paper claim.
3. **Is CMA-ES saturation compute-bound or search-space-bound?** pop=16
   and pop=128 differ by ≤ +0.006 on Lenia. Same question on Flow-Lenia:
   does the larger substrate parameter space re-open compute scaling?
4. **CLIP-OE vs DINOv2-lite-OE on Flow-Lenia.** Orbit 15 has only the
   CLIP half of the FM ablation. Need a Flow-Lenia + DINOv2-lite point
   to complete the 2×2.
5. **What killed orbit 11?** The restart schedule looks like the culprit,
   but we don't have a pop=64, single-restart, σ=0.15 control. An
   ablation would disambiguate "pop=64 doesn't help" from "restart
   schedule hurts."
