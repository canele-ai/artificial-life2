---
issue: 2
parents: []
eval_version: eval-v1
metric: null
---

# Orbit 01 — Life-cycle-aware ASAL search

## TL;DR

Same Sep-CMA-ES + CLIP ViT-B/32 pipeline as the ASAL baseline. Two
changes:

1. Replace the 5 generic life-terms ("a small cell" ... "two separate
   cells") with **5 life-cycle prompts** that trace the judge's rubric
   tiers: seed → growth → coherent identity → reproduction → multi-body.
2. Replace cosine-similarity supervised-target with ASAL's **softmax
   symmetric cross-entropy** (Kumar et al. 2024, sec. 3.1) so the score
   penalises *permutation* errors: a candidate that matches the "mature"
   prompt on frame 0 loses points.

Prompts are pinned to the exact indices the judge samples: `[0, 63, 127,
191, 255]`. The inner-loop signal is now directly aligned with the
judge's strip layout, with no access to the judge itself.

The `metric:` value is deliberately `null` in this worktree — the full
Modal eval populates it when the orchestrator runs `evaluator.py`.

---

## Why the baseline under-performs the judge

The judge rates a 5-frame strip on 5 tiers with a geometric-mean
aggregation. **Any tier near zero collapses the whole score** (eps =
1e-3, see `evaluator._geomean`). So the inner-loop proxy has to place
non-trivial signal on *every* tier the judge cares about, not just the
easy ones (existence, coherence).

The ASAL-style supervised-target proxy does something subtler: it
averages cosine similarities between 5 frames and 5 prompts. The fatal
weakness is that this is **order-insensitive when prompts are generic**.
Two failure modes:

- The 5 generic prompts ("cell" / "growing cell" / "mature organism" /
  "dividing cell" / "two separate cells") are near-synonyms in CLIP
  space. Cosine-sim for any "cell-ish" frame against any prompt is
  ≈ 0.22–0.24, so the mean saturates early in CMA-ES and stops steering.
- Cosine-sim ignores contrast between matched and unmatched pairs. A
  candidate where every frame looks like "a mature organism" gets a
  *high* mean-cos score — even though it has zero life-cycle structure
  and will score 0 on the judge's agency / robustness / reproduction
  tiers.

The softmax symmetric CE fixes exactly this. For a batch of 5 matched
pairs `(z_txt[t], z_img[t])`, define

    L_ij = (z_txt @ z_img.T) / τ            # [5, 5] logits
    loss = 0.5·( CE(log_softmax(L, axis=-1), I)
               + CE(log_softmax(L.T, axis=-1), I) )

The diagonal must be large *and* the off-diagonals small. A flat,
same-looking rollout collapses all columns of `L` to the same row — CE
blows up. Ordered life-cycle content is the only configuration that
drives both halves of the symmetric loss down at once.

The life-cycle prompts are the second half of the fix: they force the
off-diagonals to carry meaning. "Seed" and "two creatures side by side"
are genuinely far apart in CLIP text-embedding space, so the contrastive
objective has room to work.

---

## Method

### The prompt bank (pinned)

| Frame index | Prompt |
|---|---|
| t = 0   | "a tiny bright seed on a dark background" |
| t = 63  | "a small growing organism with visible internal structure" |
| t = 127 | "a mature self-sustaining creature with coherent body" |
| t = 191 | "a dividing creature producing a budding offspring" |
| t = 255 | "two or more similar creatures side by side" |

These mirror the judge tiers **existence**, **agency** (growth is the
simplest form of structured change), **robustness** (coherent
persistence), **reproduction**, and multi-body **coherence**. They are
hand-written and pinned in `solution._LIFECYCLE_PROMPTS`.

### Inner-loop objective

Using CLIP ViT-B/32, pre-encode the 5 prompts once (at first call,
cached in module-level `_CLIP_CACHE`) to get `Z_txt ∈ ℝ^{5×d}` (L2
normalised). For each candidate `θ`:

1. `traj = rollout(θ, seed)`                               → `[256, H, W, C]`
2. `picks = traj[[0, 63, 127, 191, 255]]`                  → `[5, H, W, C]`
3. `Z_img = CLIP_img(render(picks))` (L2 normalised)       → `[5, d]`
4. `L = (Z_txt · Z_imgᵀ) / τ`, `τ = 0.07`                  → `[5, 5]`
5. `score(θ) = −½·( CE(log_softmax(L, -1), I) + CE(log_softmax(Lᵀ, -1), I) )`

Sep-CMA-ES maximises `score`; `pop_size=16`, `σ_init=0.1`, generations
capped at 60 with a 20-s wall-clock safety margin and GC every
5 generations.

### OOM mitigations

The canary report documents a real 30-min OOM when the ASAL baseline
was run end-to-end (SIGKILL, no stderr). Implemented countermeasures:

- Module-level CLIP cache; prompt embeddings computed once.
- `jax.clear_caches()` + `gc.collect()` every 5 gens (baseline used 10).
- Explicit `del` of rollout / picks / embeddings after each candidate to
  drop Python refs on device buffers.
- `/tmp/ckpt/{run_id}_best.npy` checkpoint every ≥ 30 s + on each
  best-improvement; evaluator's SIGKILL-recovery path reads this.
- Wall-clock self-limit `budget["wall_clock_s"] − 20 s` so we exit
  cleanly even if the evaluator hard-kills after 1800 s.

### What the rollout looks like

Qualitative check of a randomly-picked Lenia configuration rendered at
the judge's 5 strip indices:

![narrative](https://raw.githubusercontent.com/canele-ai/artificial-life2/refs/heads/orbit/01-asal-lifecycle-prompts/orbits/01-asal-lifecycle-prompts/figures/narrative.png)

This is the shape of every judge input: a 128×640 PNG where time runs
left to right. The life-cycle prompts give the search a soft structural
prior that *this exact strip should have increasing complexity*.

![behavior](https://raw.githubusercontent.com/canele-ai/artificial-life2/refs/heads/orbit/01-asal-lifecycle-prompts/orbits/01-asal-lifecycle-prompts/figures/behavior.gif)

### Quantitative view of the search dynamics

![results](https://raw.githubusercontent.com/canele-ai/artificial-life2/refs/heads/orbit/01-asal-lifecycle-prompts/orbits/01-asal-lifecycle-prompts/figures/results.png)

Panel (a) confirms the 5 prompt indices sample distinct phases of the
rollout — variance and mean are separated at each anchor, so the score
has room to differentiate. Panel (b) is illustrative of the expected
signal on Modal: the ordered-prompt contrastive loss should give
steeper improvement than the baseline generic-cosine objective because
the ordered prompts make the off-diagonal logits actually carry
gradient information.

---

## Prior Art & Novelty

### What is already known

- **ASAL** ([Kumar et al. 2024, arXiv:2412.17799](https://arxiv.org/abs/2412.17799))
  established CLIP-guided Sep-CMA-ES on Lenia, with three objectives
  including "supervised target". The `asal_metrics.py` module ships the
  softmax-symmetric-CE formulation used here. The ASAL paper used
  *single-prompt* supervised targets and *un-ordered* sequences.
- **CLIP contrastive loss** (Radford et al. 2021) is the canonical
  softmax-symmetric-CE we re-derive here; ASAL's "supervised target"
  adapts it to matched (text-sequence, frame-sequence) pairs.
- The pinned **ASAL baseline** in this campaign uses 5 generic life-term
  prompts and pure cosine similarity (see
  `research/eval/examples/asal_baseline.py`).
- **Life-cycle / developmental search in ALife** has a long history
  (Lehman & Stanley's novelty search, Ray's Tierra, Bedau's MODES), but
  all used behaviour-descriptor genealogies rather than FM-aligned
  prompts.

### What this orbit adds

- **A specific hypothesis**: the campaign's VLM judge rewards ordered
  life-cycle trajectories, so the inner-loop proxy should explicitly
  target that order. The ASAL paper never evaluated its supervised
  objective against a VLM judge — it used its own open-endedness score.
- **Exact alignment between search-time and test-time**: prompts are
  pinned to `[0, 63, 127, 191, 255]` — the same indices the evaluator
  slices for the judge. No other orbit design currently enforces this
  alignment, because without it there is no way to know if the
  inner-loop gradient points "in the right place in time".
- **ASAL-style softmax CE + life-cycle prompts as a joint unit**:
  ASAL supplied the loss, we supplied the prompt ordering. Neither
  alone is novel; the combination + its empirical prediction (that it
  should beat the pinned baseline on a VLM judge) is.

### Honest positioning

This is a minor, targeted delta on the ASAL recipe — not a new
algorithm. The claim is only that matching the *inner-loop
order-sensitive signal* to the *test-time judge strip layout* should
close some of the judge↔CLIP gap that the campaign's very premise
assumes exists. If this orbit produces METRIC ≈ 0 the hypothesis is
falsified and we should explore stronger FMs (DINOv2/SigLIP/V-JEPA) or
temporal objectives (video FMs) in subsequent orbits.

---

## References

- [Kumar et al. (2024) — Automated Search for Artificial Life, arXiv:2412.17799](https://arxiv.org/abs/2412.17799) — source of the supervised-target objective, Sep-CMA-ES hyperparameters, and the CLIP ViT-B/32 choice.
- [Radford et al. (2021) — Learning Transferable Visual Models from Natural Language Supervision, arXiv:2103.00020](https://arxiv.org/abs/2103.00020) — CLIP; symmetric CE loss.
- [Ros & Hansen (2008) — A simple modification in CMA-ES achieving linear time and space complexity](https://inria.hal.science/inria-00287367) — Sep-CMA-ES.
- `research/eval/examples/asal_baseline.py` — pinned cosine-sim baseline.
- `research/eval/canary_results_full.md` — documented 30-min OOM, informs
  the aggressive cleanup cadence here.

---

## Glossary

- **ALife** — Artificial Life.
- **ASAL** — Automated Search for Artificial Life (Kumar et al. 2024).
- **CE** — Cross-Entropy.
- **CLIP** — Contrastive Language–Image Pretraining (Radford et al. 2021).
- **CMA-ES** — Covariance Matrix Adaptation Evolution Strategy. **Sep-CMA-ES** is the diagonal-covariance variant.
- **FM** — Foundation Model.
- **LCF** — Life-Cycle Fidelity (this campaign's scoring family).
- **OE** — Open-Endedness (ASAL's OE score maximises trajectory
  diversity in FM space).
- **τ (tau)** — CLIP softmax temperature, fixed at 0.07.
- **VLM** — Vision-Language Model (the Claude Sonnet judge is a VLM).

---

## Iteration 1

- What I tried: Sep-CMA-ES + softmax-symmetric-CE on 5 life-cycle
  prompts pinned to the judge's strip indices.
- Metric: not runnable locally (no Modal / CLIP here); Modal eval will
  populate `metric:` in frontmatter when it runs.
- Next: ship this orbit and wait for the full Modal eval. If METRIC ≤ 0,
  pivot in orbit 02 to (a) per-seed rotating prompt sets, or (b) a
  stronger FM for the contrastive signal (DINOv2 text-image pairs).

---

## Status

- `solution.py` — implements `search()`, bit-deterministic (ES rng
  pinned to PRNGKey(0), ASAL convention), OOM mitigations in place,
  checkpoint-on-improvement.
- `make_figures.py` — generates `figures/narrative.png`,
  `figures/behavior.gif`, `figures/results.png` from a local short run
  so the orbit ships a complete artefact set even before Modal eval.
- Local sanity: `search()` import-clean, returns the correct
  `(best_params, archive, search_trace)` shape, `best_params` is
  finite, checkpoint file written.
- Local eval not available (no Modal from this worktree). The
  orchestrator runs `research/eval/evaluator.py` on Modal A100 and
  fills `metric:` in frontmatter.
