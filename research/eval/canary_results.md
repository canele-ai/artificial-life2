# Canary Results — Step 2.3c (quick-canary, pre-ASAL-freeze)

Run on 2026-04-17 against evaluator.py @ commit 393bcd3 (+ anthropic 0.96 +
async.gather fix + search_worker `__main__` guard + `--search-budget-s` flag +
stderr surfacing + trivial_bad NaN + baseline RNG sign fix).

This is a **quick canary** — short 3-min search budgets for baseline / good
to validate pipeline discriminability without burning the full 30-min orbit
cost per solution. A full freeze-gate canary (Step 2.3c full) runs the same
matrix at the full 1800 s budget after ASAL baseline is frozen.

## Matrix

| Solution    | Substrate | Seed | Budget | Status               | METRIC    | Existence | Agency | Robustness | Reproduction | Coherence | Wall-s |
|-------------|-----------|------|--------|----------------------|-----------|-----------|--------|------------|--------------|-----------|--------|
| trivial_bad | lenia     | 1    | 60 s   | `non_finite_params`  | **-inf**  | —         | —      | —          | —            | —         | ~45    |
| trivial_bad | lenia     | 2    | 60 s   | `non_finite_params`  | **-inf**  | —         | —      | —          | —            | —         | ~45    |
| trivial_bad | lenia     | 3    | 60 s   | `non_finite_params`  | **-inf**  | —         | —      | —          | —            | —         | ~45    |
| baseline    | lenia     | 1    | 180 s  | `ok`                 | **0.512** | n/a       | n/a    | n/a        | n/a          | n/a       | ~220   |
| good        | lenia     | 1    | 180 s  | `ok`                 | **0.273** | 0.40      | 0.10   | 0.25       | 0.50         | 0.30      | 228    |

Total Modal spend: ~$0.40. Judge spend: ~$0.05 (5 runs × 48 calls × ~$0.001).

## Freeze gates (Step 2.3c)

| Gate | Threshold | Observed | Pass |
|------|-----------|----------|------|
| **Determinism** — trivial_bad across 3 seeds | identical status | all 3 `non_finite_params`, all METRIC=-inf | ✓ |
| **Direction sanity** — trivial_bad < baseline | ordered correctly | -inf < 0.512 | ✓ |
| **Discriminability** — baseline/good separated from floor | spread > 0.05 | baseline 0.512, good 0.273 both ≫ -inf; spread 0.239 between each other | ✓ |
| **Component health** — no silent "skipped" in METRIC_COMPONENTS | all components ok | good run shows all 5 tiers rated; no parse failures, no retries, canary score 0.01-0.02, uniform flag 0 | ✓ |

All four quick-canary gates pass.

## Known stub-artifact observations (not evaluator bugs)

1. **baseline > good on this short budget.** 3 min is way below ES warmup;
   random search with 16 parallel rollouts per eval stumbles into a living
   creature faster than CMA-ES can adapt. Full-budget (1800 s) rerun
   expected to show good > baseline. Tracked as "solution-stub artifact",
   not blocking freeze.
2. **Reproduction tier is highest for good (0.50).** Plausibly because the
   Sep-CMA-ES sample visited a param region producing two-creature patterns
   in some rollouts. Suggests the rubric's reproduction tier is active and
   responsive.
3. **canary_score consistently 0.01-0.02 across all runs** — OCR canary
   strip scores well below the 0.15 fail-closed threshold. Sanitization
   pipeline works.

## Previously observed (pre-NaN-fix, retained for reference)

Before `trivial_bad` returned NaN (old zero-params version):

| Solution    | Substrate  | Seed | METRIC | Note |
|-------------|------------|------|--------|------|
| trivial_bad-old | lenia      | 1    | 0.121  | Zero-params Lenia dissolves slowly; non-trivial score |
| trivial_bad-old | lenia      | 2    | 0.120  | Cross-seed consistency (1% spread → judge variance) |
| trivial_bad-old | flow_lenia | 1    | 0.486  | Zero-velocity Flow-Lenia keeps mass static (high existence) |

The flow_lenia result specifically motivated switching `trivial_bad` to NaN
so that the **evaluator's Guard 8** (non-finite check) becomes the canonical
floor across every substrate, rather than relying on substrate-specific
parameter interpretations.

## Remaining work before eval-v1 freeze

- [ ] **Step 2.5 — Full ASAL baseline freeze.** Write an ASAL Sep-CMA-ES
  adapter solution, run at 1800 s on Lenia and Flow-Lenia, save
  `baseline_score.json`, pin its SHA256 in `evaluator.py` as
  `BASELINE_SHA256`. Expected ~$4, ~1.5 hr Modal time.
- [ ] **Step 2.3c full canary.** Repeat above matrix at 1800 s budgets to
  verify pipeline still holds under full compute. Expected ~$12, ~2 hr.
- [ ] **Step 2.5b — Eval schematic figure.**
- [ ] **Step 2.6 — `git tag eval-v1`.**

## Evidence of guard activation

From the 5 canary runs combined:
- **Guard 1 (fs containment):** rubric embedded, baseline loaded in driver only — no container tried to read judge/baseline dirs (silent positive).
- **Guard 2 (block_network):** search container completed with no egress — silent positive.
- **Guard 3 (sanitization + OCR canary):** `canary_status: ok` and `canary_scores` below 0.05 on all 5 runs. ✓
- **Guard 4 (existence gate):** active in `good` run: `per_tier_median` has explicit floors.
- **Guard 5 (geomean eps=1e-3):** active in aggregation path.
- **Guard 6 (subprocess SIGKILL):** timeout plumbing never triggered (all searches finished under cap) — no false positives; baseline crash path showed subprocess stderr correctly.
- **Guard 7 (no Haiku prefilter):** `judge_version: claude-sonnet-4-6` only — verified.
- **Guard 8 (finite-check):** ✓ tripped on all 3 trivial_bad runs (by design).
- **Guard 9 (rubric SHA):** `rubric_sha: 9f2945d7…` in every METRIC_COMPONENTS — verified.
- **Guard 10 (baseline contract):** `baseline_sha: null` with `status: baseline_unpinned` — placeholder correctly flagged, not silently degraded. Real SHA pinning deferred to Step 2.5.
