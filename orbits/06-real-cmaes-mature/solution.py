"""solution.py — orbit 06-real-cmaes-mature (eval-v2, REAL Sep-CMA-ES).

Hypothesis (canonical EXTEND of orbit 02)
-----------------------------------------
Orbit 02 (`02-asal-mature-organism`) hit LCF_judge Δ = +0.279 under eval-v1
while silently running random search, because its `from evosax import ...`
was wrapped in `try/except ImportError` AND `evosax` was missing from the
Modal `search_image`. The batch-1 provenance bug was fixed in eval-v2:

  1. `evosax==0.1.6` is now installed in the search container.
  2. This solution has NO `except ImportError` fallback — if evosax is
     unavailable, the orbit must crash honestly so no silent regression
     can recur. (Hypothesis is unfalsifiable if fallbacks are allowed.)

Research question
~~~~~~~~~~~~~~~~~
Does **real** Sep-CMA-ES beat random-search at the mature-organism single-
prompt targeting hypothesis? Expected:
  - +5 to +15% LCF_judge over orbit 02's (silently-random) 0.279 if the
    Lenia 8-D parameter landscape is smooth enough that CMA-ES's
    covariance adaptation actually helps.
  - No significant lift if the landscape is effectively flat and random
    pop=32 already saturates.

Inheritance from orbit 02
~~~~~~~~~~~~~~~~~~~~~~~~~
Same 3-prompt bank, same single median-frame (t=127) scoring, same soft-CE
variant of ASAL's supervised-target objective, same pop_size=32, same
sigma_init=0.12, same CLIP cache + periodic gc strategy. The ONLY
differences vs. orbit 02 are:
  (a) fail-loud `from evosax import Sep_CMA_ES` at function entry.
  (b) `algorithm` provenance tag = "sep_cma_es_mature_target_REAL_eval_v2".
  (c) Tightened wall-clock safety margin to 15 s (same value; emphasised).

Prompt bank (inherited from orbit 02 verbatim)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  P1: "a mature self-sustaining organism with crisp geometric structure
       on a dark background"
  P2: "a bilaterally symmetric living cell with smooth clean boundaries"
  P3: "a soft glowing biological organism clearly distinct from empty
       space"

Each candidate is scored as `TAU * (logsumexp(cos / TAU) - log(n_prompts))`,
which is a smooth, collapse-resistant approximation of `max_i cos(z_img, z_txt[i])`.

Budget discipline
~~~~~~~~~~~~~~~~~
  - pop_size=32, n_gens ≤ 50, wall_clock 1800 s with 15 s safety margin.
  - Single-frame CLIP eval per candidate ≈ 0.5 s on A100 → fits comfortably.
  - Module-level CLIP cache keyed on `"model"`.
  - `gc.collect() + jax.clear_caches()` every 10 gens.
  - Best-params checkpoint to `/tmp/ckpt/<run_id>_best.npy` every 30 s.

Determinism
~~~~~~~~~~~
Sep-CMA-ES is seeded from the caller's `rng`. Training-seed rotation via
`seed_pool_train[gen % n_train]` is deterministic given rng + gen schedule.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp

# ─── Hyperparameters (inherited from orbit 02) ────────────────────────────────
_DIM = {"lenia": 8, "flow_lenia": 8}
_POP_SIZE = 32
_SIGMA_INIT = 0.12
_N_GENERATIONS = 50
_CLEAR_CACHE_EVERY = 10
_WALL_CLOCK_MARGIN_S = 15.0
_CKPT_EVERY_S = 30.0
_MEDIAN_FRAME_IDX = 127  # the "mature organism" timestep
_TAU = 0.07              # CLIP-standard temperature for softmax-CE

# Frozen prompt bank (inherited from orbit 02 verbatim).
_TARGET_PROMPTS: list[str] = [
    "a mature self-sustaining organism with crisp geometric structure on a dark background",
    "a bilaterally symmetric living cell with smooth clean boundaries",
    "a soft glowing biological organism clearly distinct from empty space",
]

# ─── Module-level CLIP cache ──────────────────────────────────────────────────
_CLIP_CACHE: dict = {}


def _get_clip():
    """Lazy-init CLIP + pre-compute text embeddings for the prompt bank."""
    if "model" in _CLIP_CACHE:
        return _CLIP_CACHE["processor"], _CLIP_CACHE["model"], _CLIP_CACHE["z_txt"]

    from transformers import FlaxCLIPModel, CLIPProcessor

    _CLIP_CACHE["processor"] = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32", cache_dir="/cache/hf",
    )
    _CLIP_CACHE["model"] = FlaxCLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", cache_dir="/cache/hf",
    )
    txt_inputs = _CLIP_CACHE["processor"](
        text=_TARGET_PROMPTS, return_tensors="jax", padding=True,
    )
    z_txt = _CLIP_CACHE["model"].get_text_features(**txt_inputs)
    z_txt = z_txt / (jnp.linalg.norm(z_txt, axis=-1, keepdims=True) + 1e-8)
    _CLIP_CACHE["z_txt"] = z_txt  # [n_prompts, d]
    return _CLIP_CACHE["processor"], _CLIP_CACHE["model"], z_txt


def _mature_target_score(params_batch: jax.Array, seed: int,
                         substrate_name: str) -> jax.Array:
    """Score each candidate by smooth-max prompt ASAL soft-CE at t=127.

    Returns [pop] float32; higher is better. No silent fallback — any
    failure inside this function propagates to the caller.
    """
    from evaluator import (
        _lenia_rollout, _flow_lenia_rollout,
        _render_lenia, _render_flow_lenia,
    )
    from PIL import Image as PILImage
    import numpy as np

    processor, model, z_txt = _get_clip()
    rollout_fn = _lenia_rollout if substrate_name == "lenia" else _flow_lenia_rollout
    render_fn  = _render_lenia if substrate_name == "lenia" else _render_flow_lenia

    scores = []
    for i in range(params_batch.shape[0]):
        traj = rollout_fn(params_batch[i], seed)                        # [T,H,W,C]
        pick = np.asarray(traj[_MEDIAN_FRAME_IDX:_MEDIAN_FRAME_IDX+1])  # [1,H,W,C]
        rgb  = render_fn(pick)                                          # [1,H,W,3]
        pil  = [PILImage.fromarray(rgb[0])]
        img_inputs = processor(images=pil, return_tensors="jax", padding=True)
        z_img = model.get_image_features(**{
            k: v for k, v in img_inputs.items() if k != "input_ids"
        })                                                              # [1, d]
        z_img = z_img / (jnp.linalg.norm(z_img, axis=-1, keepdims=True) + 1e-8)

        cos = (z_img @ z_txt.T).reshape(-1)                             # [n_prompts]
        lse = jax.nn.logsumexp(cos / _TAU) - jnp.log(cos.shape[0])
        score = _TAU * lse  # smooth-max approximation of max_prompt cos
        scores.append(float(score))
    return jnp.array(scores, dtype=jnp.float32)


def _ckpt_save(params: jax.Array, tag: str) -> None:
    """Write best-so-far to /tmp/ckpt for SIGKILL recovery (best-effort)."""
    try:
        import numpy as np
        ckpt_dir = Path("/tmp/ckpt")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        np.save(ckpt_dir / f"{tag}_best.npy", np.asarray(params))
    except Exception:
        pass


def search(
    substrate_name: Literal["lenia", "flow_lenia"],
    seed_pool_train: jax.Array,
    budget: dict,
    rng: jax.Array,
) -> dict:
    """Real Sep-CMA-ES on ASAL supervised-target with mature-organism prompts.

    NO ImportError fallback — fails loud if evosax is missing, so the
    provenance bug from eval-v1 cannot silently recur.
    """
    # Fail-loud import at the top of search() so the crash message is
    # captured in the search_worker stderr_tail (Guard 6).
    from evosax import Sep_CMA_ES  # noqa: F401  (intentional fail-loud)

    wall_clock_s = float(budget.get("wall_clock_s", 1800))
    K = _DIM.get(substrate_name, 8)
    t0 = time.monotonic()
    t_last_ckpt = t0
    run_tag = os.environ.get("RUN_ID", f"{substrate_name}_s{int(seed_pool_train[0])}")

    n_train = int(seed_pool_train.shape[0])
    best_proxy_per_gen: list[float] = []
    archive: list[jax.Array] = []

    strategy = Sep_CMA_ES(popsize=_POP_SIZE, num_dims=K, sigma_init=_SIGMA_INIT)
    es_state = strategy.initialize(rng)

    best_params = jnp.zeros(K, dtype=jnp.float32)
    best_score = float("-inf")

    for gen in range(_N_GENERATIONS):
        if time.monotonic() - t0 >= wall_clock_s - _WALL_CLOCK_MARGIN_S:
            break

        rng, sub = jax.random.split(rng)
        params_batch, es_state = strategy.ask(sub, es_state)            # [pop, K]

        s = int(seed_pool_train[gen % n_train])
        fitnesses = _mature_target_score(params_batch, s, substrate_name)

        # evosax minimises internally → negate to maximise fitness.
        es_state = strategy.tell(params_batch, -fitnesses, es_state)

        gen_best_idx = int(jnp.argmax(fitnesses))
        gen_best = float(fitnesses[gen_best_idx])
        if gen_best > best_score:
            best_score = gen_best
            best_params = params_batch[gen_best_idx]
            archive.append(best_params)

        best_proxy_per_gen.append(best_score)

        # OOM mitigation.
        if (gen + 1) % _CLEAR_CACHE_EVERY == 0:
            gc.collect()
            jax.clear_caches()

        # Periodic checkpoint for SIGKILL recovery.
        now = time.monotonic()
        if now - t_last_ckpt >= _CKPT_EVERY_S:
            _ckpt_save(best_params, run_tag)
            t_last_ckpt = now

    _ckpt_save(best_params, run_tag)  # final checkpoint

    archive_arr = (
        jnp.stack(archive[-16:]) if archive else jnp.zeros((1, K), dtype=jnp.float32)
    )
    return {
        "best_params": best_params,
        "archive": archive_arr,
        "search_trace": {
            "best_proxy_per_gen": best_proxy_per_gen,
            "n_generations": len(best_proxy_per_gen),
            # Provenance tag: unambiguous signal that this ran REAL CMA-ES
            # under eval-v2 (no silent random-search fallback possible).
            "algorithm": "sep_cma_es_mature_target_REAL_eval_v2",
            "pop_size": _POP_SIZE,
            "sigma_init": _SIGMA_INIT,
            "median_frame_idx": _MEDIAN_FRAME_IDX,
            "prompts": _TARGET_PROMPTS,
            "parent_orbit": "02-asal-mature-organism",
        },
    }
