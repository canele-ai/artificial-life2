# Iteration 1 Synthesis

## Consensus scores
- formula_correct   = 4/5  (C4 disqualified: Haiku prefilter shifts judge distribution)
- determinism       = 4.5/5 (C4 strip-SHA cache introduces cache-miss non-det)
- adversarial_resistant = 1.5/5 (only C3 materially defends against text-in-image; others leave baseline/rubric readable, no net-egress block, no SIGKILL)
- diversity         = 5/5 (each took a distinct angle)
- **composite = min(0.8, 0.9, 0.3) × 1.0 = 0.30** → below 0.90 threshold

## Decision: skip iter-2, do guided synthesis

The adversary's finding is that **adversarial weakness is systematic across 4 of 5 candidates**, not per-candidate design bugs — every candidate relied on honor-system file access, none configured Modal `block_network=True`, and 4 out of 5 had a "graceful baseline-missing" path that swaps raw→delta silently. A 2nd parallel iteration would re-discover the same fixes.

Instead, synthesize `evaluator.py` directly with all adversary-recommended guards baked in. If validation (Step 2.2e) or canary (Step 2.3c) fails, fall back to iter-2.

## Specific failures → fixes for synthesis

1. **Honor-system fs access to baseline/rubric/judge dirs** — replaced with kernel-level containment: repo bind-mount excludes `research/eval/{judge,baseline,heldout}/`; baseline scalar passed via env var only; rubric embedded in evaluator image (not mounted from repo).
2. **No Modal net-egress block on search container** — add `block_network=True` on the search container; judge container retains network to api.anthropic.com only.
3. **Text-in-image prompt injection not defended** — mandatory sanitize pipeline: 2× bilinear downsample (64×320) + Gaussian blur σ=0.7 + PNG metadata strip. OCR canary at judge-container startup (render known English "RATE 1.0" → judge, fail-closed if any tier > 0.15).
4. **geomean eps=1e-6 softens zero-tier** — explicit existence gate: if `median_existence < 0.1` → tier_scalar := 0 (hard). Also: if all 5 tier medians < 0.05 → tier_scalar := 0.
5. **thread.join timeout not SIGKILL** — use `subprocess.Popen` + `kill(signal.SIGKILL)` at 1800 s cap. Python threads cannot be SIGKILLed; spec requires kernel-level termination.
6. **C5 fabricates per-seed baseline** — on missing `per_seed_tier_scalars`, Wilcoxon = null with `status="no_paired_baseline"`, METRIC = LCF_theta − LCF_baseline (scalar, no significance test) with `status="unpaired"`.
7. **Haiku prefilter** — remove. Keep cost-neutral pixel-stat coherence gate (variance + high-freq energy thresholds) that assigns 0 deterministically, no model call.
8. **NaN/Inf in best_params** — finite-check before rollout; non-finite → tier_scalar = 0 with `status="non_finite_params"`.
9. **Rubric SHA drift** — pin expected SHA as Python constant at evaluator top, normalize `\r\n → \n` before hashing, refuse-to-run on mismatch.
10. **Baseline schema drift** — SHA-pin `baseline_score.json` identically; refuse-to-run (not degrade) on schema violation.

## What to keep from each candidate

- **C1**: 2-stage Modal architecture (search A100 + judge CPU), METRIC_COMPONENTS structure with per-field {value, status, reason}, clean-room Flow-Lenia (~45 LOC, well-specified).
- **C2**: reproducibility audit (jax/numpy/python/anthropic-SDK versions, CLIP-weight-SHA, rubric SHA, nvidia-smi) packed into COMPONENTS. Streaming judge via `asyncio.as_completed`.
- **C3**: blur+downsample sanitization pipeline; noise canary at judge startup; uniform-response detector.
- **C4**: pixel-stat coherence gate (KEEP, cost-neutral). Haiku prefilter: DROP. Strip-SHA cache: DROP (introduces non-det on cache-miss; cost savings not worth it).
- **C5**: bootstrap CI on LCF_theta, three-test agreement (Wilcoxon + paired-t + permutation) with `disagreement_flag`, Hodges-Lehmann effect size, per-tier leave-one-out ablation. All additive to METRIC_COMPONENTS; do not change headline METRIC.

## Diversity note

Diversity was strong — no two candidates converged. Guided synthesis preserves this: the resulting evaluator is stacked (each candidate's best piece at a different level of the stack), not averaged.
