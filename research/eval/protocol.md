# Evaluation Protocol

> Frozen at eval-v1. Describes WHAT the evaluator measures and HOW —
> the research contract that all orbits must satisfy.
> For the full formal spec, see `problem_spec.md`. For the judge rubric,
> see `judge/rubric.md`.

## What We Measure

`lcf_judge_heldout` — a single scalar in approximately [−0.1, +0.3]:

> "How well do this orbit's discovered simulation parameters score on a
>  5-tier life-likeness rubric, as judged by Claude Sonnet 4.6 across 16
>  held-out initialization seeds, measured against a pinned ASAL baseline?"

Direction: **maximize**. Pass = METRIC > 0 at Wilcoxon p < 0.05.

The 5 tiers (operationalized from the user's 6-tier Phase 1 taxonomy,
dropping Ecology + Evolution which need multi-rollout data):
1. **Existence** — is there a coherent localized pattern?
2. **Agency** — does it move or deform in a structured way?
3. **Robustness** — does it persist with recognizable identity?
4. **Reproduction** — does it bud, divide, or produce offspring?
5. **Coherence** — is it non-chaotic, geometrically structured?

## How to Measure

```
1. Load orbit's solution.py.
2. Instantiate two Modal containers:
   - search container: A100, CLIP weights at /cache/hf, network egress to
     api.anthropic.com / api.openai.com blocked.
   - judge container: CPU, anthropic-api-key mounted, Sonnet 4.6 SDK.

3. search container:
   rng = jax.random.PRNGKey(SEEDS.root)
   seed_pool_train = jnp.arange(16) + 1000           # training seeds
   seed_pool_test  = jnp.arange(16) + 9000           # held-out, disjoint
   result = solution.search(
       substrate_name = <config>,                    # "lenia" | "flow_lenia"
       seed_pool_train = seed_pool_train,
       budget          = {"wall_clock_s": 1800},     # 30-min SIGKILL cap
       rng             = rng,
   )
   best_params = result["best_params"]               # [D] float32

4. search container — produce per-seed frame strips:
   for s in seed_pool_test:
       rollout = substrate(best_params, s, T=256)    # [256, 128, 128, C]
       picks   = rollout[[0, 63, 127, 191, 255]]     # 5 ordered frames
       rgb     = render_to_rgb(picks, substrate)     # substrate-specific
       strip   = concat_horizontal(rgb)              # [128, 640, 3] uint8
       save_png(strip, f"/tmp/strips/{run_id}_{s}.png")

5. judge container — rate each strip:
   for strip_path in strips:
       for k in range(3):
           resp = anthropic.messages.create(
               model       = "claude-sonnet-4-6",
               temperature = 0.0,
               max_tokens  = 256,
               system      = open("research/eval/judge/rubric.md").read(),
               messages    = [{"role": "user", "content": [
                   {"type": "image", "source": {"type": "base64", ...}},
                   {"type": "text",  "text": "Rate this rollout."}
               ]}],
           )
           scores[k] = parse_json(resp.content[0].text)
       median_scores = {tier: median([s[tier] for s in scores]) for tier in TIERS}
       tier_scalars[s] = geometric_mean(median_scores.values())

6. Aggregate:
   LCF_theta    = trimmed_mean_10pct(tier_scalars across 16 seeds)
   LCF_baseline = json.load("research/eval/baseline/baseline_score.json")["LCF_judge"]
   metric       = LCF_theta - LCF_baseline
   p_value      = wilcoxon_signed_rank(
                     [tier_scalars_theta[s] for s in seeds],
                     [tier_scalars_baseline[s] for s in seeds],
                     alternative="greater"
                  )

7. Print:
   print(f"METRIC={metric:.6f}")
   print(f"METRIC_COMPONENTS={json.dumps({
     'lcf_theta':    LCF_theta,
     'lcf_baseline': LCF_baseline,
     'wilcoxon_p':   p_value,
     'per_tier_median': {tier: median over seeds of median_scores[tier] for tier in TIERS},
     'judge_version':  'claude-sonnet-4-6',
     'rubric_sha':     rubric_sha256,
     'status':         'ok'  # or 'timeout' / 'crash' / 'judge_parse_fail'
   })}")
```

## Acceptance Criteria

- **Metric direction:** maximize.
- **Determinism:** same solution + same rng → identical CLIP-based
  inner-loop trace. Judge calls use temperature=0 but are non-deterministic
  at the API level; median-of-3 bounds variance. Canary gate (Step 2.3c)
  requires baseline spread < 1% across 3 seed replays.
- **Seeds:** 16 test seeds, disjoint from 16 train seeds. Seeds are
  `jnp.arange(16) + 9000` (test) and `jnp.arange(16) + 1000` (train),
  hard-coded in the evaluator.
- **Timeouts:**
  - Orbit `search()`: 1800 s (30 min) SIGKILL cap.
  - Per judge API call: 60 s client-side timeout, 3 retries.
  - Total orbit submission: 2400 s (40 min) evaluator-level hard cap.

## What Counts as a Solution

Orbits provide a `solution.py` module exposing `search(...)` as specified
in `problem_spec.md`. Inside `search()`:

- **May** use any CLIP-based scalar as inner-loop fitness proxy.
- **May** use any search algorithm (ES, CMA-ES, MAP-Elites, random search,
  gradient-based for differentiable substrates, LLM-local-proposed θ,
  hybrids — anything).
- **Must** return `{"best_params": θ*, "archive": [...], "search_trace": {...}}`.
- **Must** complete within 1800 s wall-clock (SIGKILL enforced).
- **Must NOT** call `api.anthropic.com` or `api.openai.com` during search.
- **Must NOT** read files under `research/eval/judge/` or
  `research/eval/baseline/` or `research/eval/heldout/`.

## Known Pitfalls

- **Judge variance:** Sonnet @ temperature=0 still has ~3-5% score
  variance run-to-run. Mitigated by median-of-3 + geometric-mean tier
  aggregation.
- **Judge prompt-injection via images:** orbits could try to embed
  text-like patterns into the rendered frames. Mitigated by structured
  JSON-only output format + non-parseable-response rejection + retry.
- **Flow-Lenia 2-channel rendering:** evaluator freezes
  `rgb = stack([mass, 0.5·|flow|, 0.2·mass], -1) ∈ [0,1] → uint8`.
  Substrate adapter cannot change this.
- **Inner-loop / judge signal divergence:** orbits optimizing CLIP-OE may
  find chaotic trajectories that score low on Coherence and Robustness
  tiers. This IS the research question — good orbits find proxies that
  correlate with the rubric.
- **Judge drift:** when Anthropic releases a new Sonnet, the judge score
  distribution changes. Policy: bump to eval-vN, re-run ASAL baseline,
  all orbits re-scored; published scores are only comparable within an
  eval-vN tag.
