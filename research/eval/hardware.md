# Hardware Inference

## Problem signals

From `research/problem.md` and the refined `research/eval/problem_spec.md`
(Phase 1.5 synthesis):
> Substrates: Lenia + Flow-Lenia (dual), 128² grid, T=256 steps
> FMs: CLIP ViT-B/32, DINOv2-base, SigLIP ViT-B/16 (image-only; temporal FMs deferred)
> Search: 30-min wall-clock cap per `search()`
> User preference (Phase 1.6): cost-effective GPU

The eval inner loop is: **simulate (JAX scan, 128² grid, 256 steps)
→ embed 5 frames with up to 3 image FMs (≤ 1 GB combined weights)
→ compute scalar metric → vmap over population of 32–256 per generation**.

## Inferred needs

- **Evaluation and experiments (uniform):** A100 (40 GB default).
  - Memory budget: 3 FMs (~1 GB) + Flow-Lenia state + population of 64
    rollouts (~4 GB headroom) + JAX workspace. Sits comfortably in 40 GB.
  - Expected per-orbit wall-clock: ~32 min (30-min search + ~30 s held-out
    scoring + container overhead).
  - Full 50-orbit campaign: ~30 A100-hours ≈ $45–60 at Modal pricing.
- **Why not H100:** ~1.3–1.5× faster wall-clock, ~2× cost. The 30-min search
  cap is not CPU-bound — compile-time + FFT convolutions dominate, where
  H100's bandwidth advantage is modest. Cost-effectiveness wins.
- **Why not A10G / L4 (24 GB):** tight VRAM margin with 3 FMs + Flow-Lenia
  + CMA-ES pop. Workable but fragile; not worth saving ~$0.50/hr.
- **Why not CPU:** FM forward passes would dominate, ~50–100× slower; eval
  inner loop becomes infeasible within a 30-min orbit budget.
- **Estimated eval duration:** ~32 minutes per orbit submission (hard-capped
  at 40 min by `eval.timeout` with safety slack).

## Config

```yaml
compute:
  backend: modal
  gpu: A100
  cpu: 8
  memory: 32768    # 32 GB RAM
  timeout: 3600    # 1-hour Modal container hard kill
```

## Notes

- **Multi-GPU not needed** for campaign 1 — single-device JAX with vmap over
  population handles parallelism. Reserve multi-GPU for a future campaign
  upgrade (V-JEPA-giant, Flow-Lenia 2048², or QD archives > 10k cells).
- **A100-80GB fallback** acceptable if 40GB availability on Modal is poor;
  same wall-clock, ~1.3× cost.
- **Modal backend (not local)** — orbits need isolated reproducible
  containers; local is reserved for debug-only trivial orbits.
- **No GPU oversubscription:** each orbit gets its own A100 container;
  `execution.parallel_orbits = 5` ⇒ peak 5 A100s concurrent.
