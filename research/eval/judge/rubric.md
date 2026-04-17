# Judge Rubric — FROZEN at eval-v1

> This rubric is the research contract. It is the single source of truth for
> what "life-like" means in this campaign. Any change bumps the eval version.
>
> SHA-pinned at eval-v1 freeze. Evaluator hashes this file at startup and
> refuses to score if the hash drifts.

---

## System prompt (passed verbatim to the judge)

You are a strict scientific rater evaluating artificial-life simulations.
You will see a single horizontal strip of 5 frames showing the temporal
evolution of a simulation (left-to-right = time steps 0, 63, 127, 191, 255
out of 256). The simulation is run on a continuous cellular automaton
substrate (Lenia or Flow-Lenia) at 128×128 resolution per frame.

Rate the trajectory on each of 5 dimensions. Each score MUST be a float in
[0.0, 1.0]. Be strict: uniform blobs, dissolved patterns, chaotic noise,
and empty fields should all score below 0.2 on most dimensions.

Return ONLY a JSON object on one line, with no prose, no code fences, no
explanation:

{"existence": <float>, "agency": <float>, "robustness": <float>, "reproduction": <float>, "coherence": <float>}

## Tier definitions (for the rater's reference; NOT shown to the judge at call time)

### existence ∈ [0, 1]
1.0 — A coherent, spatially localized pattern is clearly distinct from the
      background, present and well-formed in all 5 frames.
0.5 — A pattern is present but ill-defined, or exists only in some frames,
      or occupies most of the field without clear boundaries.
0.0 — Empty field, uniform noise, uniform solid blob covering everything,
      or only isolated single-pixel flickers.

### agency ∈ [0, 1]
1.0 — The pattern demonstrates structured motion or deformation across
      frames: translation (gliders), rotation (rotators), pulsation
      (breathers), or systematic shape change.
0.5 — The pattern changes between frames but the change lacks clear
      structure; flickering or partial drift.
0.0 — The pattern is static, or changes are indistinguishable from noise.

### robustness ∈ [0, 1]
1.0 — The pattern maintains a recognizable identity across all 5 frames —
      the frame-5 object is clearly a developed version of the frame-1
      object.
0.5 — The pattern partially persists but morphs into something different,
      or shrinks/grows significantly without preserving identity.
0.0 — The pattern dissolves, explodes into chaos, or is replaced by
      something entirely unrelated between early and late frames.

### reproduction ∈ [0, 1]
1.0 — Clear offspring: later frames contain ≥ 2 similar patterns where
      earlier frames contained 1, OR the pattern visibly divides/buds.
0.5 — Possible budding or splitting that is not clearly resolved, or a
      secondary pattern appears but of different type.
0.0 — No offspring, no division, no secondary patterns.
      NOTE: absence of reproduction is normal and does not penalize other
      tiers. Score is multiplicative, so reproduction=0 only drags the
      overall score via the geometric-mean aggregation.

### coherence ∈ [0, 1]
1.0 — Crisp geometric structure with smooth boundaries and recognizable
      visual form (like a cell, creature, or organized wave).
0.5 — Some structure visible but partially disordered; fuzzy boundaries,
      mixed clean-and-noisy regions.
0.0 — Pure noise, random speckle, uniform mush, or aliasing artifacts with
      no discernible structure.

## Rationale (not shown to judge; for human reviewers)

- **5 tiers, not 6.** The original Phase 1 taxonomy had Tier 5 Ecology and
  Tier 6 Evolution. Both require multi-rollout or multi-species data and
  are not evaluable from a single 5-frame strip. They are deferred to
  campaign 2.
- **Geometric mean aggregation.** A pattern must score > 0 on every tier
  to score well overall. This prevents gaming a single tier.
- **Strictness bias.** Explicitly instructed to score below 0.2 for
  trivial failure modes. We measured during adversarial iteration that
  Sonnet without this bias tends to score 0.3–0.4 on random noise.
- **No code fences, no prose.** Constrains output to a parseable JSON line.
  The evaluator refuses any response that doesn't parse; the judge is
  re-called up to 3 times on parse failure.
- **No access to raw tensor or metadata.** Judge sees only the 128×640 PNG
  strip. Cannot see parameters, substrate name, or orbit identity —
  prevents orbit-identity bias.
