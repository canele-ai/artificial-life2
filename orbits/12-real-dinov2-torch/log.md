---
issue: 13
parents: [07-real-cmaes-dinov2]
eval_version: eval-v2
metric: null
---

# Research Notes

## Hypothesis

EXTEND of orbit 07 (`07-real-cmaes-dinov2`, current campaign leader at
**+0.319**). Orbit 07 uses a hand-rolled QR-orthogonal multi-scale patch
encoder ("DINOv2-lite") because the Modal search image was missing
`torch`. This orbit swaps that proxy for the **real** `facebook/dinov2-small`
via PyTorch + HuggingFace `transformers`, holding everything else fixed
(Sep-CMA-ES, popsize=16, 60 generations, sigma_init=0.1, lenia substrate,
same 4 frame picks {0, 63, 127, 255}, same ASAL OE score formula).

**Claim under test:** if orbit 07's +0.319 signal is driven by the
pretrained representation content (not just random orthogonal patch
statistics), a *real* pretrained DINOv2 should match or exceed it.

## Design

* `from evosax import Sep_CMA_ES` and `import torch` at module top —
  **no `try/except`**. If the Modal image lacks torch, the orbit crashes
  and the evaluator records `search_crash`. That is the correct, honest
  signal (silent fallback to DINOv2-lite would collapse this orbit into
  orbit 07 and hide the comparison).
* `facebook/dinov2-small` loaded once per process via module-level
  `_FM_CACHE`; `AutoImageProcessor` + `AutoModel.from_pretrained(...,
  cache_dir="/cache/hf")` (HF cache volume).
* Determinism: `torch.use_deterministic_algorithms(True, warn_only=True)`,
  `torch.manual_seed(0)`, `torch.inference_mode()` around every forward pass.
* CLS-token embeddings (`last_hidden_state[:, 0, :]`, D=384) used for the
  OE-similarity proxy.
* Gen-level `gc.collect() + jax.clear_caches()` every 10 gens.
* Wall-clock self-limit at `wall_clock_s - 15 s`.
* Returns `{best_params, best_rollout, archive, search_trace}` (worker
  only consumes `best_params` but the full dict stays available for
  downstream tooling).
* `search_trace["algorithm"] = "sep_cma_es_real_dinov2_small_eval_v2[<backend>]"`.

## Risk / known unknowns

* **Torch in the Modal image.** The search_image in `modal_app.py` does
  **not** list torch in its `pip_install`. This run will likely crash at
  module import — that crash is itself the experiment result. If torch
  needs to land in the image, that is a separate infra PR.
* **HF weight resolution.** `block_network=False` was set in the
  eval-v2 infra fix (commit 08e6131), so `from_pretrained` can reach
  `huggingface.co` for missing files. The HF cache volume is populated
  by prior orbits' CLIP runs; DINOv2 weights may need a first-touch
  download.
* **Compute.** dinov2-small is ~22M params; a CPU-only forward on 4
  frames per candidate × 16 pop × 60 gens is manageable; GPU (if
  `cuda.is_available()`) is used when present.

## What a positive result looks like

* Clean completion, `algorithm="sep_cma_es_real_dinov2_small_eval_v2[real_dinov2_small]"`
  in the trace (NOT a crash, NOT a fallback tag).
* Metric comparable to or above orbit 07's +0.319. Outperforming orbit 07
  by >1% would confirm the FM-content hypothesis.

## What a negative result looks like

* `search_crash` at module import → torch missing; signal is that the
  campaign needs a torch-enabled image (actionable infra follow-up).
* Metric ≤ orbit 07 despite real FM → orbit 07's lite proxy captures
  most of the usable signal; pretraining not load-bearing here.

## Status

Implementation committed. Awaiting Modal eval dispatch via `/autorun`.
No local eval — no GPU on this host and the whole point is to test on
the campaign's search_container.
