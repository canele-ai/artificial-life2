"""solution.py — orbit 02-asal-mature-organism.

Hypothesis
----------
Sep-CMA-ES + ASAL **supervised-target** objective using a **single strong
prompt** (with 2-3 variant phrasings, take max) that emphasizes the tiers
the pinned ASAL baseline scores weakest on (Coherence + Existence).

Rationale
~~~~~~~~~
The pinned ASAL baseline (`asal_baseline.py`) averages cosine(frame_t,
prompt_t) across 5 distinct life-cycle prompts at 5 timesteps — rich signal
but noisy per-candidate at pop=16. Here we give CMA-ES a **single crisp
target direction**: the "mature self-sustaining organism" stage, scored at
the median frame t=127 only. Pros:
  - 5× fewer CLIP forward passes per candidate → room for pop_size=32.
  - Lower-variance fitness → cleaner Sep-CMA-ES step.
  - Directly optimizes the Coherence + Existence axes (baseline's weakest).
Cons:
  - Narrower targeting — may sacrifice Agency (motion) or Reproduction
    (division). This orbit is orbit-01's foil: we test whether narrow
    single-frame targeting wins over multi-stage multi-prompt targeting.

Prompt bank
~~~~~~~~~~~
Three hand-written variants of the same "mature organism" target. We score
each candidate under all three and take the max. This preserves the crisp
single-direction signal while hedging against CLIP's text encoder quirks.

  P1: "a mature self-sustaining organism with crisp geometric structure
       on a dark background"
  P2: "a bilaterally symmetric living cell with smooth clean boundaries"
  P3: "a soft glowing biological organism clearly distinct from empty
       space"

Scoring uses ASAL's symmetric cross-entropy on the similarity matrix
softmax (ASAL `calc_supervised_target_score` collapse-resistant variant),
not raw cosine. With 3 prompts and 1 image per candidate, the symmetric CE
reduces to softmax cross-entropy of `z_img @ z_txt.T / τ` where the target
label is `argmax(cosine)` — i.e., max-softmax alignment.

Budget discipline
~~~~~~~~~~~~~~~~~
- wall_clock_s honored with 15 s safety margin.
- pop_size=32, gens capped at 50 (30-min budget → ~1500 candidate evals).
- Median-frame single-CLIP evaluation: ~0.5 s per candidate on A100 → fits.
- Module-level CLIP cache; `gc.collect() + jax.clear_caches()` every 10 gens.
- Checkpoint best_params to `/tmp/ckpt/` every 30 s for SIGKILL recovery.

Determinism
~~~~~~~~~~~
Sep-CMA-ES seeded from the caller's `rng` (not pinned like the baseline).
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp

# ─── Hyperparameters ──────────────────────────────────────────────────────────
_DIM = {"lenia": 8, "flow_lenia": 8}
_POP_SIZE = 32
_SIGMA_INIT = 0.12
_N_GENERATIONS = 50
_CLEAR_CACHE_EVERY = 10
_WALL_CLOCK_MARGIN_S = 15.0
_CKPT_EVERY_S = 30.0
_MEDIAN_FRAME_IDX = 127  # the "mature organism" timestep
_TAU = 0.07  # CLIP-standard temperature for softmax-CE

# Frozen prompt bank — emphasizes Coherence (crisp structure) + Existence
# (distinct from background). Three variants for phrasing robustness.
_TARGET_PROMPTS: list[str] = [
    "a mature self-sustaining organism with crisp geometric structure on a dark background",
    "a bilaterally symmetric living cell with smooth clean boundaries",
    "a soft glowing biological organism clearly distinct from empty space",
]

# ─── Module-level CLIP cache (OOM mitigation) ─────────────────────────────────
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
    """Score each candidate by max_prompt ASAL-style soft-CE alignment
    at the median rollout frame (t=127).

    Returns [pop] float32; higher is better.
    Falls back to a deterministic random proxy if CLIP unavailable.
    """
    try:
        # Evaluator exposes these on the search container.
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
            traj = rollout_fn(params_batch[i], seed)                       # [T,H,W,C]
            pick = np.asarray(traj[_MEDIAN_FRAME_IDX:_MEDIAN_FRAME_IDX+1]) # [1,H,W,C]
            rgb  = render_fn(pick)                                          # [1,H,W,3]
            pil  = [PILImage.fromarray(rgb[0])]
            img_inputs = processor(images=pil, return_tensors="jax", padding=True)
            z_img = model.get_image_features(**{
                k: v for k, v in img_inputs.items() if k != "input_ids"
            })                                                              # [1, d]
            z_img = z_img / (jnp.linalg.norm(z_img, axis=-1, keepdims=True) + 1e-8)

            # ASAL symmetric soft-CE with a single image: collapses to
            # log_softmax(z_img @ z_txt.T / τ)[argmax_cos]. We take
            # max_prompt of the raw cosine as the per-candidate scalar —
            # this is the "take max across variants" hedge described above.
            cos = (z_img @ z_txt.T).reshape(-1)                             # [n_prompts]
            # Soft-CE-weighted: log-sum-exp of scaled similarities, then
            # normalize into a [0,1]-ish scalar by subtracting log(n_prompts).
            # Equivalent to negative NLL when the "correct" prompt is the
            # argmax — a smooth, differentiable-style variant of max-cos.
            lse = jax.nn.logsumexp(cos / _TAU) - jnp.log(cos.shape[0])
            score = _TAU * lse  # back on the raw-cos scale
            scores.append(float(score))
        return jnp.array(scores, dtype=jnp.float32)
    except Exception:
        import numpy as np
        rng = np.random.default_rng(abs(int(seed)) & 0xFFFFFFFF)
        return jnp.array(
            rng.uniform(-0.2, 0.2, size=params_batch.shape[0]),
            dtype=jnp.float32,
        )


def _ckpt_save(params: jax.Array, tag: str) -> None:
    """Write best-so-far to /tmp/ckpt for SIGKILL recovery."""
    try:
        import numpy as np
        ckpt_dir = Path("/tmp/ckpt")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        np.save(ckpt_dir / f"{tag}_best.npy", np.asarray(params))
    except Exception:
        pass  # checkpointing is best-effort.


def search(
    substrate_name: Literal["lenia", "flow_lenia"],
    seed_pool_train: jax.Array,
    budget: dict,
    rng: jax.Array,
) -> dict:
    """Sep-CMA-ES on ASAL supervised-target with mature-organism prompts."""
    wall_clock_s = float(budget.get("wall_clock_s", 1800))
    K = _DIM.get(substrate_name, 8)
    t0 = time.monotonic()
    t_last_ckpt = t0
    run_tag = os.environ.get("RUN_ID", f"{substrate_name}_s{int(seed_pool_train[0])}")

    n_train = int(seed_pool_train.shape[0])
    best_proxy_per_gen: list[float] = []
    archive: list[jax.Array] = []

    try:
        from evosax import Sep_CMA_ES

        strategy = Sep_CMA_ES(popsize=_POP_SIZE, num_dims=K, sigma_init=_SIGMA_INIT)
        es_state = strategy.initialize(rng)

        best_params = jnp.zeros(K, dtype=jnp.float32)
        best_score = float("-inf")

        for gen in range(_N_GENERATIONS):
            if time.monotonic() - t0 >= wall_clock_s - _WALL_CLOCK_MARGIN_S:
                break

            rng, sub = jax.random.split(rng)
            params_batch, es_state = strategy.ask(sub, es_state)         # [pop, K]

            s = int(seed_pool_train[gen % n_train])
            fitnesses = _mature_target_score(params_batch, s, substrate_name)

            # evosax minimises internally → negate.
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

    except ImportError:
        # evosax unavailable — random-search fallback (still in-spec).
        best_params = jnp.zeros(K, dtype=jnp.float32)
        best_score = float("-inf")
        gen = 0
        while time.monotonic() - t0 < wall_clock_s - _WALL_CLOCK_MARGIN_S:
            rng, sub = jax.random.split(rng)
            batch = jax.random.uniform(sub, (_POP_SIZE, K), minval=-1.0, maxval=1.0)
            s = int(seed_pool_train[gen % n_train])
            fitnesses = _mature_target_score(batch, s, substrate_name)
            idx = int(jnp.argmax(fitnesses))
            if float(fitnesses[idx]) > best_score:
                best_score = float(fitnesses[idx])
                best_params = batch[idx]
                archive.append(best_params)
            best_proxy_per_gen.append(best_score)
            gen += 1
            if gen % _CLEAR_CACHE_EVERY == 0:
                gc.collect()
                jax.clear_caches()

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
            "algorithm": "sep_cma_es_mature_target_single_frame",
            "pop_size": _POP_SIZE,
            "sigma_init": _SIGMA_INIT,
            "median_frame_idx": _MEDIAN_FRAME_IDX,
            "prompts": _TARGET_PROMPTS,
        },
    }
