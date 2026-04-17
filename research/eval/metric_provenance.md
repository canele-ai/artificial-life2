# Metric Provenance

> Phase 2.0 Prior Art Deep Dive for the git-evolve campaign
> "Improve on ASAL: life-cycle-aware search for artificial life."
> All verdicts below are grounded in primary sources (papers + repo file paths).
> Claims the survey could not confirm are labeled `[INFERENCE]` or `[UNVERIFIED]`.

---

## Papers Reviewed

1. **Kumar, Couturier, Akarca, Lim, Sims, Tang, Telgarsky, Ha. "Automating the Search for Artificial Life with Foundation Models" (ASAL).** arXiv:2412.17799, 2024. <https://arxiv.org/abs/2412.17799>
   - Metric: three FM-guided scalar objectives — **supervised target** (mean diagonal of `z_txt @ z_img.T` kernel), **open-endedness** (max off-diagonal of the self-similarity kernel of the trajectory, a "distance from the most similar past frame"), **illumination** (same structure over a population). A fourth *softmax* variant of supervised target is implemented but **not the default** (coefficient 0 in `main_opt.py`).
   - Key finding: frozen CLIP ViT-B/32 is a usable, automation-grade fitness for discovering visually interesting Lenia/Boids/ParticleLife/NCA/GoL rules; Sep-CMA-ES from evosax with pop=16, σ=0.1 is sufficient.
   - Directly relevant: our metric *is* ASAL's softmax-target + permutation gap.

2. **Plantec, Hamon, Etcheverry, Moulin-Frier, Oudeyer, Clune. "Flow Lenia: Mass Conservation for the Study of Virtual Creatures."** arXiv:2212.07906, 2022. <https://arxiv.org/abs/2212.07906>
   - Metric: none quantitative in the main paper (abstract describes "behaviors of interest" qualitatively; paper surfaces creature characterizations like persistence, locomotion, and multi-species coexistence visually). Flow-Lenia itself does **not** contribute a scalar fitness score we can re-use.
   - Key finding: mass-conservative update rule enables multi-species, parameter-embedded dynamics — the reason we want this substrate in the campaign.
   - Relevance: substrate, not metric.

3. **Faldor & Cully. "Leniabreeder: Exploration of Lenia with Quality-Diversity."** ALIFE 2024. GitHub: <https://github.com/maxencefaldor/leniabreeder>
   - Metric: custom *fitness* (e.g. `pos_linear_velocity_avg`) + custom *descriptors* (e.g. `color`) plugged into MAP-Elites and AURORA on QDax.
   - Key finding: QD archives of Lenia creatures are feasible with hand-engineered behavior characterizations; the AURORA variant learns descriptors via an autoencoder.
   - Relevance: precedent for QD-in-Lenia and for swapping hand descriptors for learned ones (parallel to our FM-embedding strategy).

4. **Mouret & Clune. "Illuminating search spaces by mapping elites" (MAP-Elites).** arXiv:1504.04909, 2015. <https://arxiv.org/abs/1504.04909>
   - Metric: per-cell champion fitness, QD-score aggregate.
   - Key finding: introduces the illumination paradigm (behavior grid × performance).
   - Relevance: our anchor-subtraction *phrasing* is inspired by the MAP-Elites reproducibility norm of freezing a reference run, but MAP-Elites itself does **not** prescribe "pinned baseline subtraction." `[INFERENCE]` — this framing in our `problem_spec.md` is our own; the Mouret-Clune citation there is aspirational, not literal.

5. **Lehman & Stanley. "Abandoning Objectives: Evolution Through the Search for Novelty Alone" + "Evolving a diversity of virtual creatures through novelty search and local competition."** 2011. <https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/lehman_ecj11.pdf>
   - Metric: novelty = mean distance to k-nearest prior behaviors in an archive.
   - Key finding: pure novelty collapses to "specialists" unless coupled with a quality/local-competition term — a cautionary tale for ASAL-OE as implemented (which is exactly pure novelty in CLIP space).
   - Relevance: motivates our permutation gap (structured novelty) and existence gate (quality floor).

6. **Zhai et al. "Sigmoid Loss for Language-Image Pre-Training" (SigLIP).** arXiv:2303.15343, 2023.
   - Metric: sigmoid pairwise loss replacing softmax contrastive.
   - Relevance: one of the three FMs we hold out as a judge.

---

## Existing Implementations / Repos

### SakanaAI/asal — <https://github.com/SakanaAI/asal>

- Stars: **463** · License: Apache-2.0 · Last commit: `[UNVERIFIED — GitHub metadata not machine-read this session]`
- Root files confirmed: `asal_metrics.py`, `rollout.py`, `main_opt.py`, `main_illuminate.py`, `main_sweep_gol.py`, `asal.ipynb`, `foundation_models/`, `substrates/`, `util.py`.
- `asal_metrics.py` (fetched in full):
  - `calc_supervised_target_score(z, z_txt)` — returns `-jnp.diag(z_txt @ z.T).mean()` after repeating `z_txt` to match `T` frames.
  - `calc_supervised_target_softmax_score(z, z_txt, temperature_softmax=0.01)` — symmetric `-log(softmax(kernel/τ))` on both axes; **τ=0.01** baked in. This is the formula our spec re-labels `softmax(z_txt @ z.T / 0.07)` (spec uses `τ=0.07`, ASAL uses `τ=0.01` — divergence intentional; see Known Pitfalls).
  - `calc_open_endedness_score(z)` — `tril(z @ z.T, k=-1).max(-1).mean()`.
  - `calc_illumination_score(zs)` — self-similarity kernel with diagonal masked to `-inf`, `.max(-1).mean()`.
- `rollout.py::rollout_simulation(rng, params, s0=None, substrate=None, fm=None, rollout_steps=256, time_sampling='final', img_size=224, return_state=False)`:
  - Modes: `'final'`, `'video'`, or `(K, chunk_ends)` integer/tuple — returns K equally-spaced frames. Our eval's `frames[[0,63,127,191,255]]` sampling is the `(5, True)` pattern with `rollout_steps=256`.
  - Substrate interface: `init_state`, `step_state`, `render_state(state, params, img_size)`.
- `main_opt.py`: Sep-CMA-ES from evosax (`popsize=16`, `sigma_init=0.1`, `n_iters=1000` default). Loss = `coef_prompt*loss_prompt + coef_softmax*loss_softmax + coef_oe*loss_oe`, defaults `(1.0, 0.0, 0.0)` — **the canonical ASAL baseline uses non-softmax supervised target**, not softmax.
- **Verdict: USE.** `asal_metrics.py` is ~60 LOC, directly reusable for our PG / OT computation. `rollout.py` is directly adaptable; add Flow-Lenia substrate and three-FM dispatch. Pin to a specific SHA at campaign init.

### erwanplantec/FlowLenia — <https://github.com/erwanplantec/FlowLenia>

- Stars: **23** · License: **none visible** `[UNVERIFIED]` · Last commit `[UNVERIFIED]` (27 commits total).
- `flowlenia/flowlenia.py` exists with `Config` dataclass, `State` dataclass (activations `A`), and a `FlowLenia` class with `step`, kernel computation, and rule space.
- `flowlenia/flowlenia_params.py` is a parameter-embedding variant (dynamic local parameters).
- Pure JAX, GPU-capable, no explicit dependency manifest listed on the README page.
- **Verdict: ADAPT.** Requires a ~200-LOC bridge to match ASAL's `init_state / step_state / render_state(img_size)` substrate interface. No license is a real concern for redistribution — we should contact the author or reimplement the small amount we need from the paper. `[ACTION: confirm license before merging a Flow-Lenia adapter into main.]`

### maxencefaldor/leniabreeder — <https://github.com/maxencefaldor/leniabreeder>

- Stars: **26** · License: **MIT** · Last commit `[UNVERIFIED]`.
- Directories: `lenia/`, `qdax/` (vendored), `configs/`, `notebooks/`, `analysis/`, `apptainer/`.
- Entry points: `main_me.py` (MAP-Elites), `main_aurora.py` (AURORA).
- Fitness interface is a plain Python function returning a scalar (e.g., `pos_linear_velocity_avg`); descriptor is a list of callables returning a bounded behavior vector.
- **Verdict: ADAPT.** Use as reference for (a) how to wire QDax over Lenia, (b) AURORA autoencoder setup if we want to learn descriptors from FM embeddings. Do not vendor — import QDax fresh.

### adaptive-intelligent-robotics/QDax — <https://github.com/adaptive-intelligent-robotics/QDax>

- Stars: **349** · License: **MIT** · Latest release: v0.5.0 (2025-05-27).
- Algorithms: MAP-Elites, CVT-MAP-Elites, AURORA, CMA-ME, CMA-MEGA, PGA-ME, DCRL-ME, QDPG, OMG-MEGA, MOME, MEES, ME-PBT, ME-LS; baselines DIAYN, DADS, SMERL, NSGA-II, SPEA2, PBT.
- The exact file paths `qdax/core/map_elites.py` and `qdax/core/aurora.py` were not directly read this session `[UNVERIFIED path-exactness]`, but the algorithms are confirmed exposed.
- **Verdict: USE.** First-class JAX QD library; exactly what we need for the illumination archive.

### RobertTLange/evosax — <https://github.com/RobertTLange/evosax>

- Stars: **750** · License: **Apache-2.0** · v0.2.0 released 2025-03-11.
- `Sep_CMA_ES` strategy exposed at `evosax.algorithms.Sep_CMA_ES` (Separable CMA-ES, Ros & Hansen 2008). Ask/eval/tell API, fully `jit`/`vmap`/`scan` compatible.
- **Verdict: USE.** This is what ASAL uses and what our pinned baseline uses.

### Chakazul/Lenia (Python) — <https://github.com/Chakazul/Lenia>

- Chan's reference implementation + the authoritative species-record format (`animals.json`, etc.). Not consulted interactively this session, but the accompanying paper (Chan 2019, arXiv:1812.05433) catalogues 400+ species in 18 families — the inspiration for our prompt-bank vocabulary.
- **Verdict: ADAPT (as prompt inspiration, not code).** We do not need the numpy implementation; we need its taxonomy.

### HuggingFace transformers — Flax FM availability

Confirmed via direct file lookup on the transformers GitHub at tags v4.47.1 (our pin) and v4.50.0:

- `src/transformers/models/clip/modeling_flax_clip.py` exists → `FlaxCLIPModel` importable. ✓
- `src/transformers/models/dinov2/modeling_flax_dinov2.py` exists → `FlaxDinov2Model`, `FlaxDinov2ForImageClassification` importable. ✓
- `src/transformers/models/siglip/modeling_flax_siglip.py` does **not** exist at any tag searched. ✗ Only PyTorch `SiglipModel` is shipped.
- **Verdict: USE CLIP + DINOv2 as-is via transformers; ADAPT SigLIP from `google-research/big_vision` (native JAX/Flax) or port weights into a minimal Flax ViT + text tower.**

---

## Existing Benchmarks / Datasets / Prompt Banks

| Artifact | What it is | Format | Obtainable? | Verdict |
|----------|-----------|--------|-------------|---------|
| ASAL "supervised target prompts" in `main_opt.py` | Small list of creature-style text prompts passed at CLI | Plain strings | Yes, from the repo | ADAPT — too short (single-step targets, not ordered sequences). |
| Chan 2019 Lenia species catalog (`animals.json`) | 400+ named species with parameter records | JSON | Yes, Chakazul/Lenia repo | ADAPT — source of genus/family vocabulary for prompt authoring. |
| Flow-Lenia paper figure captions | Natural-language descriptions of multi-species behaviors | Prose in PDF | Yes | ADAPT — source of dynamic-behavior vocabulary (nomadic, territorial, reproducing). |
| ALIFE / ISAL conference creature gallery | Decades of informal creature naming | Mixed | Partially | ADAPT — supplementary vocabulary. |
| Leniabreeder descriptors | `[color]`, `[linear_velocity]` etc. | Python callables | Yes, MIT | SKIP for prompt-bank authoring (these are numeric descriptors, not text). |
| Pretrained CLIP benchmarks (ImageNet-R, DomainNet) | Object-recognition prompts | Text | Yes | SKIP — wrong domain (photographs, not abstract creatures). |
| A curated *ordered* life-cycle prompt bank for ALife | Does not exist in the literature to our knowledge `[INFERENCE]` | — | — | **CONSTRUCT.** The 20-sequence × 5-prompt bank required by `problem_spec.md` must be authored by us; no pre-existing ordered ALife prompt bank was found. |

---

## Chosen Metric

- **Name:** `LCF-robust@heldout` (life-cycle fidelity, worst-of-FMs, held-out split).

- **Formula (restated with citations):**

  Given solution params θ, held-out seed `s ∈ S_test`, held-out ordered prompt sequence `p ∈ P_test` (L=5 prompts), FM `f ∈ F = {clip_vitb32, dinov2_vitb14, siglip_vitb16}`:

  1. `frames = rollout(substrate, θ, s, T=256)` — sampling scheme inherited from ASAL `rollout.py::rollout_simulation` with `time_sampling=(5, chunk_ends=True)`; specifically `frames[[0, 63, 127, 191, 255]]` (`asal/rollout.py`).
  2. `Z_img = f.encode_image(frames)`, `Z_txt = f.encode_text(p)`, both L×d.
  3. `A = softmax(Z_img @ Z_txt.T / τ, axis=1)` with **τ=0.07** (our spec) — note: ASAL's `calc_supervised_target_softmax_score` uses τ=0.01 (`asal/asal_metrics.py` line default `temperature_softmax=0.01`). We deliberately use a larger τ because our L=5 prompts is tiny vs. ASAL's typical T — at τ=0.01 a 5×5 softmax saturates. `[INFERENCE — should be validated empirically at campaign init.]`
  4. `OT = mean_i A[i,i] − mean_{i≠j} A[i,j]` — diagonal-dominance variant of ASAL's softmax-target score (`calc_supervised_target_softmax_score`, inverting sign so maximization is natural).
  5. `PG = OT(p) − max over Π ∈ {reverse, 10 fixed shuffles} OT(Π·p)` — **permutation gap. Novel to this campaign.** No precedent in ASAL or Flow-Lenia; most-similar prior art is rank-correlation-based evaluation in sequence-alignment literature. `[INFERENCE]`
  6. Existence gate `E(θ,s) = 1 iff final_mass ∈ [0.1×, 10×]·initial_mass AND var(last_32_frames) > 1e-4` — mass-conservation check (natural for Flow-Lenia, reasonable proxy for Lenia), activity check. Ad hoc; no direct literature citation. `[INFERENCE]`
  7. `PG_eff = E · max(0, PG)`; `LCF(θ,f) = median_p mean_s PG_eff`; `METRIC(θ) = min_f LCF(θ,f) − min_f LCF_anchor(f)` where `LCF_anchor` is the pinned ASAL baseline.

- **Direction:** maximize. Expected range ≈ [−0.05, +0.20].

- **Why this metric:**
  - Inherits ASAL's softmax-target structure (measured, collapse-resistant — Kumar et al. 2024 show it outperforms raw cosine for supervised target).
  - Adds *ordering* signal via permutation gap, directly addressing the gap identified in `problem.md`: ASAL-OE rewards any moving trajectory; ASAL supervised target ignores order.
  - Adds *measurement robustness* via min over FMs, a defense against CLIP's known typographic/texture biases (Azuma et al., arXiv:2304.04512; see Pitfalls).
  - Keeps a single scalar (satisfies `problem_spec.md` interface).

- **Alternatives rejected:**
  - *Raw supervised-target score* (ASAL default): ignores ordering, so gameable by a static creature that loosely resembles all prompts. Fails our novelty bar.
  - *Raw ASAL-OE*: rewards chaotic noise (confirmed critique in Lehman–Stanley novelty pathologies, and in our `problem.md`).
  - *Kendall τ between predicted and target orderings*: discrete, not differentiable, and insensitive to magnitude of matches; survives as a *diagnostic* (already flagged in Open Question #5 of the spec).
  - *Optimal-transport / DTW between embedding trajectory and prompt trajectory*: richer, but costs O(L²·d) per candidate and is harder to permutation-test; deferred.
  - *MAP-Elites QD-score*: multi-cell, not a single scalar per solution; incompatible with the `solution.search → best_params` interface.
  - *Bedau–Packard evolutionary activity / MODES*: requires genealogy instrumentation we don't have in a 30-min budget (`problem.md` already defers these).

---

## Known Pitfalls

| Pitfall | Source | Our guard |
|---------|--------|-----------|
| **CLIP typographic / OCR-biased adversarial matches** | Azuma & Matsui, "Defense-Prefix for Preventing Typographic Attacks on CLIP," arXiv:2304.04512; Materzyńska, "Reading isn't Believing," arXiv:2103.10480 | Worst-of-three FMs (CLIP + DINOv2 + SigLIP). DINOv2 is text-blind; SigLIP uses a different training loss. A pattern gaming CLIP text-in-image won't also fool DINOv2's self-supervised features. |
| **Softmax temperature collapse / low entropy** | Zhai et al., SigLIP (arXiv:2303.15343) notes symmetric-loss sensitivity to τ; CLIP original paper uses learned τ≈0.01. ASAL hard-codes τ=0.01 but operates on dozens of frames, not L=5. | Spec uses τ=0.07. `[INFERENCE — validate at eval-freeze.]` Sweep τ ∈ {0.01, 0.05, 0.07, 0.1} on the anchor baseline before freezing. |
| **Chaotic-trajectory gaming of OE** | Lehman & Stanley 2011 novelty pathologies; ASAL paper acknowledges OE is "any trajectory that keeps moving." | Existence gate E + diagonal-dominance OT (a chaotic rollout has uniform softmax, OT→0). |
| **Pure-novelty specialist collapse** | Lehman–Stanley "local competition" requirement | Our metric has an explicit target (life-cycle prompts) — this is supervised-novelty, not pure novelty. |
| **Unordered match gaming** | Novel concern for this campaign `[INFERENCE]` | Permutation gap PG with 11 permutations (reverse + 10 random). |
| **Dead / exploded patterns** | Folk knowledge in Lenia community (Chan 2019 §3); no formal citation | Mass-ratio + late-frame variance gate. |
| **Prompt overfit** | Standard ML concern | Train/test split 10/10 on the 20 prompt sequences. |
| **Seed lottery** | Folk knowledge, QDax/evosax best practice | Paired Wilcoxon over 16 held-out seeds. |
| **Scorer-as-oracle (search uses the judge FM)** | Standard Goodhart concern | Two of three FMs process-isolated from `search`. Spec enforces this via Modal import restriction. |
| **FM embedding-norm drift** (not in spec) | Common issue when averaging across FMs of different embed-norm distributions | Min-over-FMs dodges this because we don't average across FMs. ✓ |

---

## Baseline References

| Baseline | Expected LCF-robust@heldout (raw, pre-subtraction) | Source |
|----------|-----------------------------------------------------|--------|
| **Trivial random params (single sample)** | ~0.00 ± 0.01 `[INFERENCE from problem_spec.md]` — a random Lenia kernel is usually dead or chaotic, so E·PG ≈ 0. | Needs empirical confirmation at campaign init. |
| **Simple baseline: budget-matched random search** | ~0.02 | Draw 64 × 200 = 12,800 params uniformly in a sensible box, pick best by train-split PG. Spec line 133. |
| **ASAL Sep-CMA-ES (pinned)** | **~0.03–0.08** (target range) | ASAL repo @ pinned SHA `[TO BE PINNED AT CAMPAIGN INIT]`; CLIP-only search with `calc_supervised_target_score` (default, coef 1.0; `main_opt.py`); `evosax.Sep_CMA_ES` with `popsize=16, sigma_init=0.1, n_iters=200`; `PRNGKey(0)`. Scored via our eval on 16 test seeds × 10 test sequences × 3 FMs. Score will be non-zero on CLIP (matches training objective) but PG will be ~0 because ASAL does not optimize ordering → worst-of-FMs is bounded by the weakest permutation gap. |

**Anchor protocol:** compute once, freeze `research/eval/baseline/baseline_score.json`, never recompute (spec lines 137–139).

---

## FM Availability Notes

| FM | Flax available? | Evidence | Download-free? | Port complexity |
|----|------------------|----------|-----------------|------------------|
| **CLIP ViT-B/32** (`openai/clip-vit-base-patch32`) | ✅ Yes | `transformers/src/transformers/models/clip/modeling_flax_clip.py` present at tags v4.44.0, v4.45.2, v4.47.1 (our pin), v4.50.0, v4.53.3 and `main`. Import: `from transformers import FlaxCLIPModel`. | No HF token gate; weights are OpenAI-released, freely downloadable. | — |
| **DINOv2-base** (`facebook/dinov2-base`) | ✅ Yes | `transformers/src/transformers/models/dinov2/modeling_flax_dinov2.py` present at v4.45.2, v4.47.1, v4.50.0, `main`. Import: `from transformers import FlaxDinov2Model`. Known issue #37246 notes "Inconsistent results between torch and jax versions" — we must match the evaluator's FM path and run fp32. `[ACTION: sanity-check DINOv2 Flax↔PyTorch parity at campaign init; if drift > 1e-3, pick one path and stick with it.]` | No HF token gate. | — |
| **SigLIP ViT-B/16** (`google/siglip-base-patch16-224`) | ❌ No Flax in `transformers` | No `modeling_flax_siglip.py` in any checked tag. Only PyTorch `SiglipModel` ships. | Weights are openly hosted on HF; no token gate. | **Medium.** Options, in increasing effort: (a) **PyTorch-in-Modal-subprocess** (spec's default choice) — import PyTorch SigLIP, `torch.no_grad()` forward, numpy round-trip to JAX. ~5% perf overhead. Low risk. (b) Port from `google-research/big_vision` (`configs/proj/image_text/SigLIP_demo.ipynb`) — native Flax, ~1 engineering day, bit-exact. (c) Use `pythoncrazy/jimm` (native Flax NNX, PyTorch weight loader) — community-maintained, unverified stability. Our recommendation: **(a) first, consider (b) only if scoring latency becomes a bottleneck.** |

**Overall verdict:** all three FMs are usable. Only SigLIP requires engineering; the default PyTorch-subprocess path is spec-compatible and does not block Phase 2.

---

## Prompt Bank Sources (inspiration for drafting 20 life-cycle sequences)

Drawn from biological / ALife canon. Each is a vocabulary source, not a ready-made prompt list.

1. **Chan, "Lenia — Biology of Artificial Life," Complex Systems 28 (2019)** — arXiv:1812.05433. <https://arxiv.org/abs/1812.05433>
   Vocabulary: 18 families × 400+ species (Orbium, Scutium, Gyrorbium, …), global morphology + local-behavior taxonomy. Source for "genus/family/species" noun vocabulary at each life-cycle stage.

2. **Chan's Lenia species catalog (`animals.json`)** — <https://github.com/Chakazul/Lenia>
   Vocabulary: per-species character tags ("solid", "hollow", "ring", "rotator", "wanderer"). Excellent source for visual-adjective prompts.

3. **Flow-Lenia paper, Plantec et al. 2022** — arXiv:2212.07906. <https://arxiv.org/abs/2212.07906>
   Vocabulary: dynamics terminology (persistent, nomadic, territorial, ghost, reproducing, colonizing). Key for the *late-stage* prompts in a life-cycle sequence.

4. **Wikipedia: "Embryogenesis"** — <https://en.wikipedia.org/wiki/Embryogenesis>
   Vocabulary: zygote → cleavage → morula → blastula → gastrula → neurula → organogenesis. Gold source for one of the 20 sequences, literally.

5. **Wikipedia: "Metamorphosis"** — <https://en.wikipedia.org/wiki/Metamorphosis>
   Vocabulary: egg → larva → pupa → adult (holometabolism); nymph stages (hemimetabolism). Second life-cycle archetype.

6. **Wikipedia: "Biological life cycle"** — <https://en.wikipedia.org/wiki/Biological_life_cycle>
   Vocabulary: haplontic / diplontic / haplodiplontic cycles, gametophyte/sporophyte. Third archetype — plant-like cycles.

7. **Wikipedia: Cellular-automaton glossary / Conway's Life patterns** — <https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life#Examples_of_patterns>
   Vocabulary: still-life, oscillator, spaceship, puffer, gun, rake, eater. Source for *ALife-native* vocabulary complementary to biology terms.

8. **International Society for Artificial Life (ISAL) "Lenia" page** — <https://alife.org/encyclopedia/software-platforms/lenia/>
   Vocabulary: narrative descriptions used by the community.

9. **Developmental biology textbook glossary (Gilbert, "Developmental Biology," any edition — Sinauer/OUP)** `[OFFLINE SOURCE]`
   Vocabulary: cell-fate terminology (totipotent → pluripotent → multipotent → differentiated), axis formation, polarity. Use when drafting "development" sequences.

10. **Bedau et al., "Open Problems in Artificial Life," Artificial Life 6 (2000)** — <https://www.mitpressjournals.org/doi/10.1162/106454600300103683>
    Vocabulary: adaptive, evolutionary, unbounded, open-ended — meta-vocabulary used to *label* sequences at the bank level, not individual prompts.

**Drafting target (for the human curator):** 20 ordered sequences × 5 prompts = 100 prompts total. Suggested archetypes: embryogenesis (×3 seeds), metamorphosis (×3), reproduction cycles (×3), population dynamics (×3), repair/regeneration (×2), colonization/spread (×2), decay/death (×2), chaotic-to-ordered transitions (×2). Human curates before baseline freeze.

---

## Gate Checks Before Proceeding

- [x] **≥3 published references reviewed** — ASAL, Flow-Lenia, Leniabreeder/QDax, MAP-Elites, Lehman-Stanley, SigLIP (six).
- [x] **ASAL repo + Flow-Lenia repo + QDax inspected; USE/ADAPT/SKIP documented** — see Existing Implementations section.
- [x] **Metric formula is cited to specific equations/files** — `asal_metrics.py` functions named; `rollout.py` sampling mode named; τ divergence from ASAL's 0.01 to our 0.07 explicitly flagged as needing empirical validation.
- [x] **Every FM has a confirmed Flax load path** — CLIP and DINOv2 via HF transformers v4.47.1; SigLIP via PyTorch-subprocess fallback (spec default) or big_vision port.

## Open Items for Phase 2.1 (not gating Phase 2 start)

1. Pin the ASAL repo SHA; freeze it in `research/eval/baseline/pinned_sha.txt`.
2. Confirm `erwanplantec/FlowLenia` licensing; if unresolved, plan a clean-room reimplementation from the paper (~300 LOC).
3. Validate τ=0.07 vs. τ=0.01 empirically on the anchor baseline before committing to the final eval freeze.
4. Sanity-check DINOv2 Flax↔PyTorch parity (HF issue #37246); pick one path and pin.
5. Draft the 20-sequence prompt bank from sources above; human review before freeze.
