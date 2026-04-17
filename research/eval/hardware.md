# Hardware Inference

## Problem signals

From `research/problem.md`:
> "[Substrates] Lenia / Flow-Lenia / Particle Lenia / NCA"
> "stronger image FMs (DINOv2, SigLIP-2) or temporal FMs (V-JEPA, VideoMAE, InternVideo2)"
> "QDax MAP-Elites, AURORA"
> "no compute budget limit" (user, Phase 1)

The eval inner loop is: **simulate (JAX scan, 128²–1024² grid, ~256–10k steps)
→ embed frames with a frozen vision(-language) FM → compute scalar metric →
repeat for population of 256–1024 per generation**. Temporal FMs (V-JEPA)
meaningfully raise memory and bandwidth requirements over per-frame CLIP.

## Inferred needs

- **Evaluation:** H100 — single rollout + FM embedding scoring fits on A100,
  but Flow-Lenia at 512²+ plus video FMs pushes into H100 territory and the
  user has no budget constraint, so default to the better tier.
- **Experiments:** H100 — full search sweeps (Sep-CMA-ES pop=256 × 500 gens,
  or QDax MAP-Elites archives), especially for Flow-Lenia + V-JEPA
  combinations. A100 would work for CPU-cheap substrates (GoL, small Lenia)
  but we want uniform hardware across orbits for clean leaderboard
  comparisons.
- **Estimated eval duration:** 5–30 minutes per orbit candidate solution
  depending on substrate grid size, rollout length, and FM. Full orbit
  (search loop with 500 generations × 256 pop) is 2–6 H100-hours.

## Config

```yaml
compute:
  gpu: H100
```

## Notes

- Multi-GPU (H100:2 / H100:4) is **not** required for campaign 1 — substrates
  are single-device JAX and we get parallelism for free via `jax.vmap` over
  the population. Leave room to upgrade later if we add V-JEPA-giant or
  Flow-Lenia 2048² runs.
- Fallback to A100-80GB is acceptable if H100 availability on Modal is poor;
  expect ~1.5–2× slower wall-clock but no functional difference.
- Modal backend (not local) — orbits need isolated reproducible containers.
