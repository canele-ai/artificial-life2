<!-- metric: lcf_judge_heldout target: null eval: eval-v3 -->

# Improve on ASAL: life-cycle-aware search for artificial life

## Problem Statement

ASAL (Kumar et al. 2024, arXiv:2412.17799) reframed artificial-life search as
foundation-model-guided optimization: embed a simulation's rollout frames in a
frozen vision-language model (CLIP ViT-B/32), then optimize one of three scalar
objectives — **supervised target** (match a text prompt), **open-endedness**
(trajectory never repeats in FM space), or **illumination** (population-level
diversity) — with Sep-CMA-ES or a custom novelty archive. ASAL covers Boids,
Lenia, Particle Life, Particle Lenia, NCA, and Game of Life.

We aim to **improve on ASAL along three orthogonal axes** and produce results +
visualizations of the same character (species atlases, life-cycle GIFs,
illumination maps, cross-method comparisons):

1. **Substrate gap:** add Flow-Lenia (Plantec et al. 2023, arXiv:2212.07906),
   the mass-conserving Lenia variant that uniquely enables multi-species
   coexistence, soft self-boundaries, and intrinsic evolution within one world.
   ASAL did not cover Flow-Lenia.

2. **Foundation-model gap:** ASAL uses per-frame CLIP ViT-B/32. Investigate
   whether stronger image FMs (DINOv2, SigLIP-2) or temporal FMs (V-JEPA,
   VideoMAE, InternVideo2) yield a richer search signal — especially for
   behaviors whose essence is temporal (motion, development, reproduction).

3. **Objective gap:** ASAL's open-endedness rewards any trajectory that keeps
   moving in FM space — including chaotic noise. Replace with **life-cycle
   fidelity:** given a fixed bank of ordered prompt-sequences describing life
   stages (e.g. *"seed" → "developing organism" → "mature self-sustaining
   creature" → "reproducing creature"*), optimize for simulations whose rollout
   matches the ordered trajectory under the chosen FM. Structured novelty, not
   any novelty.

The research output is both quantitative (single-scalar leaderboard metric, see
below) and qualitative (GIFs, species atlases, perturbation-recovery videos,
multi-species ecology visualizations, cross-substrate comparisons).

## Solution Interface

A candidate "solution" is a search procedure that, given a fixed substrate
(Lenia / Flow-Lenia / Particle Lenia / NCA) and fixed FM, discovers a set of
simulation parameters whose rollouts score well on the life-cycle fidelity
metric.

Concretely, a solution is a Python module implementing:

```python
def search(substrate_name: str,
           fm_name: str,
           prompt_bank: list[list[str]],   # life-cycle prompt sequences
           budget: dict,                    # {"generations": int, "pop_size": int}
           rng: jax.Array) -> dict:
    """
    Returns:
      {
        "best_params": jax.Array,          # flat params of best simulation
        "best_rollout": jax.Array,         # [T, H, W, C] frames of best
        "archive": list[dict],             # top-K solutions for illumination
        "trace": dict,                     # per-generation metric curves
      }
    """
```

Candidates may use any search algorithm (Sep-CMA-ES baseline, custom novelty
archive à la ASAL, QDax MAP-Elites, AURORA, hierarchical QD, learned proposal
distributions, etc.). They must consume the same substrate and FM interfaces
(adapted from the ASAL repo; Flow-Lenia adapter to be written in Phase 2).

## Success Metric

**Life-cycle fidelity score** (single scalar, direction = maximize).

On a fixed bank of K life-cycle prompt sequences (e.g. 5 sequences × 5 prompts
each), for each sequence:
1. Run the discovered simulation for T steps, sample 5 equally-spaced frames.
2. Embed frames with the chosen FM → `z_i`, embed prompts → `z_txt_i`.
3. Compute **softmax-target score** (ASAL's collapse-resistant variant):
   symmetric cross-entropy on `softmax(z_txt @ z_img.T / τ)`.

Final score = mean over K sequences, minus an ASAL-OE baseline computed on a
held-out seed pool to anchor zero. Range: roughly `[0, 1]` in practice;
direction `maximize`.

**Why not use raw ASAL-OE or supervised-target score alone:** raw OE is gamed
by chaotic trajectories; raw supervised-target (ASAL's existing metric) only
scores static matching, not ordered temporal trajectories. Life-cycle fidelity
is the minimal extension that (a) stays a single differentiable scalar, (b) is
evaluable from the raw rollout tensor with a frozen FM, and (c) rewards
structured temporal behavior.

**Target:** no hard target set; this is an exploration-first campaign. We will
measure improvement relative to an ASAL-exact baseline (Sep-CMA-ES with CLIP
ViT-B/32 optimizing supervised target, scored on our life-cycle prompts).

## Known Approaches

Starting points to inherit / adapt / beat:

- **ASAL** (arXiv:2412.17799, github.com/SakanaAI/asal): JAX-native substrates,
  `asal_metrics.py` (60 LOC, directly reusable), Sep-CMA-ES via evosax, custom
  novelty-archive GA. This is the baseline to beat.
- **Flow-Lenia** (arXiv:2212.07906, github.com/erwanplantec/FlowLenia): JAX +
  Equinox. Needs a ~200-LOC adapter to plug into the ASAL substrate interface.
- **Leniabreeder** (Faldor & Cully, ALIFE 2024): QDax-based Lenia + MAP-Elites
  + AURORA. The canonical QD-for-Lenia reference.
- **QDax** (github.com/adaptive-intelligent-robotics/QDax): JAX-native QD
  library, actively maintained.
- **Foundation models:** HuggingFace `FlaxCLIPModel` (CLIP), `FlaxDinov2Model`
  (DINOv2). SigLIP via big_vision (weekend of integration work). V-JEPA /
  VideoMAE require PyTorch + JAX interop.
- **Classic metrics survey:** Bedau–Packard evolutionary activity, Channon's
  unbounded-evolution criteria, MODES toolbox — all require genealogy
  instrumentation; deferred to later campaigns.

## Visualization Targets

The campaign outputs are as much the figures as the metric. Required per
promising orbit:

- **Species atlas GIF:** grid of discovered patterns running in parallel.
- **Life-cycle GIF:** single pattern from seed through mature dynamics.
- **Perturbation-recovery GIF:** pattern → damage → regeneration timeline.
- **Multi-species ecology GIF** (Flow-Lenia only): multiple patterns
  interacting in a shared world.
- **Illumination map:** 2D projection of discovered parameter space, colored
  by FM-embedding descriptor.
- **Comparison GIF:** same initial conditions across substrates (Lenia vs.
  Flow-Lenia vs. NCA) or across FMs (CLIP vs. DINOv2 vs. V-JEPA).

## Open Questions (to be resolved by Phase 1.5 refinement panel)

1. Exact formula for life-cycle fidelity: softmax-target only, or combined
   with an OE penalty? How to weight?
2. Fixed prompt bank vs. evolving prompt bank? Human-written vs. LLM-generated?
3. What counts as the "held-out ASAL baseline" anchor — is it a fixed score
   frozen at init, or re-computed per orbit?
4. Unit of comparison across orbits: per-generation budget equalized, or
   wall-clock equalized, or GPU-hour equalized?
5. Does a single-scalar life-cycle score suffice, or do we need a
   multi-objective Pareto front (e.g. fidelity × robustness × diversity)?

## Metric history

Eval has been re-frozen once during this campaign. Each tag below records what
changed and why; metric semantics carry forward across versions but the
provenance/quality signals around them have improved.

- **eval-v1** (frozen 2026-04-18) — initial freeze. Inner-loop CLIP-OE
  + Sep-CMA-ES (`good.py`) + 5-tier VLM judge (Claude Sonnet 4.6) +
  geomean-then-trimmed-mean over 16 held-out seeds. Baseline anchored at
  LCF_judge=0.1185 (`baseline_score.json`, SHA `15c5b5d0…`).

- **eval-v2** (frozen 2026-04-19, commit `7fbd935` + infra fix `08e6131`) —
  metric semantics unchanged (still `lcf_judge_heldout`); honesty fixes
  surfaced at milestone 1's cross-orbit synthesis:
  - **`evosax==0.1.6` added to Modal `search_image`.** At eval-v1 it was absent;
    `good.py`'s silent `try/except ImportError` fallback meant every orbit
    claiming Sep-CMA-ES actually ran random search wearing a CMA-ES label.
    The frozen eval-v1 baseline 0.1185 was honest random-search + CLIP-OE,
    just mislabeled. eval-v2 keeps the same numerical anchor (same bytes)
    and updates metadata to HONEST_ANCHOR. BASELINE_SHA256 bumped to
    `a72d63e0…`.
  - **`good.py` made fail-loud.** Removed the `ImportError` fallback;
    a missing dep now crashes honestly instead of silently degrading.
  - **`evosax_version` captured in `env_audit`.** Provenance signal for
    a future Guard 11 (algorithm-provenance check); search_trace claims
    can be cross-checked against actually-resolved imports.
  - **`block_network=True` removed from `search_container`.** Defense-in-depth
    that broke legitimate HF cache resolution after the image rebuild
    (preprocessor_config.json missing); the real judge-oracle guard is the
    anthropic secret being judge-only, plus subprocess SIGKILL.

- **eval-v3** (frozen 2026-04-20, commit `65090f9`) — additive quality
  signals; no metric-semantics change, no re-anchor:
  - **`status="judge_parse_starvation"`** when `parse_success_rate < 0.5`.
    Closes the milestone-3 silent-collapse failure mode where 48/48 judge
    parse failures produced a 0-LCF metric without flagging the cause.
  - **`parse_success_rate`** surfaced as a top-level field in
    `METRIC_COMPONENTS`. Future leaderboard rendering can downweight
    orbits whose judge-success rate is degraded.
  - **`torch==2.5.1` added to Modal `search_image`.** Unblocks orbit-12
    family (pretrained-FM swap; previously honest-crashed on
    ModuleNotFoundError). Image rebuilds on next dispatch.
  - Baseline value unchanged (still 0.1185, HONEST_ANCHOR);
    `BASELINE_SHA256 → 4ceff810…` reflects metadata-only `eval_version`
    field bump.

### Open eval-v4 candidates (NOT yet frozen)

Tracked here so the spec records the work-in-progress contract.

- **Multi-seed contract.** Single-seed metrics are provisional; treat
  `best_orbit` as null until ≥3 seeds confirm. (Open framework issue:
  https://github.com/canele-ai/git-evolve/issues/9.) Costs ~3× Modal
  spend per orbit but eliminates the orbit-15-style false-positive class.
- **Algorithm-provenance assertion.** Solutions claiming algorithm `X` whose
  `search_trace["algorithm"]` references a module not in `env_audit`
  should be auto-flagged as `algorithm_provenance_warning`.
- **Re-anchor under multi-seed real CMA-ES.** Once the multi-seed
  contract lands, re-freeze the baseline against actual Sep-CMA-ES
  output (not random search). Estimated cost ~$30-50 Modal.
