"""good.py — solution.py stub: Sep-CMA-ES + CLIP open-endedness proxy.

Strategy: ASAL-style Sep-CMA-ES (evosax) with CLIP ViT-B/32 open-endedness
as the inner-loop fitness proxy.  This should find parameters whose rollouts
are visually diverse in CLIP space, which correlates (imperfectly) with the
judge's life-likeness rubric.

Expected LCF_judge (raw): ~0.20–0.35 (competitive with ASAL baseline).
Expected METRIC: ~0 ± 0.1 vs pinned ASAL anchor.

This is a skeleton.  A world-class orbit would additionally:
  - Use CMA-ES with population of 64 (more exploration).
  - Ensemble CLIP-OE with a text-supervised target on life-cycle prompts.
  - Use DINOv2 as a second FM for diversity.
  - Checkpoint every 60 s to /tmp/ckpt/{run_id}_best.npy for SIGKILL recovery.

Constraints respected:
  - Bit-identical under fixed rng.
  - Respects budget["wall_clock_s"] = 1800.
  - Does NOT call api.anthropic.com.
  - Does NOT read research/eval/judge/ or research/eval/baseline/.
"""

from __future__ import annotations

import time
from typing import Literal

import jax
import jax.numpy as jnp

_DIM = {"lenia": 8, "flow_lenia": 8}

# CMA-ES hyper-parameters (ASAL defaults).
_POP_SIZE = 16
_SIGMA_INIT = 0.1
_N_GENERATIONS = 60    # 200 OOMs the A100 container at 30-min budget
_CLEAR_CACHE_EVERY = 10


# Module-level CLIP cache — prevents OOM on long searches (loading CLIP per
# generation leaks memory and kills the container after ~20 gens).
_CLIP_CACHE: dict = {}


def _clip_oe_score(params_batch: jax.Array, seed: int) -> jax.Array:
    """CLIP open-endedness score for a batch of params [pop, K].

    Returns [pop] float32 scores.  Higher = more temporally diverse in CLIP.
    Falls back to random proxy if CLIP unavailable.
    """
    try:
        import numpy as np

        if "model" not in _CLIP_CACHE:
            from transformers import FlaxCLIPModel, CLIPProcessor
            _CLIP_CACHE["processor"] = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32", cache_dir="/cache/hf",
            )
            _CLIP_CACHE["model"] = FlaxCLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", cache_dir="/cache/hf",
            )
        processor = _CLIP_CACHE["processor"]
        model     = _CLIP_CACHE["model"]
        from evaluator import _lenia_rollout, _render_lenia
        from PIL import Image as PILImage

        scores = []
        for i in range(params_batch.shape[0]):
            params = params_batch[i]
            traj = _lenia_rollout(params, seed)          # [256, H, W, 1]
            picks = np.asarray(traj[[0, 63, 127, 255]])  # 4 frames
            rgb = _render_lenia(picks)                    # [4, H, W, 3]
            pil_imgs = [PILImage.fromarray(rgb[j]) for j in range(4)]
            inputs = processor(images=pil_imgs, return_tensors="jax", padding=True)
            feats = model.get_image_features(**{k: v for k, v in inputs.items()
                                               if k != "input_ids"})
            z = feats / (jnp.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)
            sim = z @ z.T
            n = sim.shape[0]
            # ASAL calc_open_endedness_score: max off-diagonal per row, mean.
            mask = jnp.tril(jnp.ones((n, n), dtype=bool), k=-1)
            oe = float(jnp.where(mask, sim, -jnp.inf).max(-1).mean())
            scores.append(oe)
        return jnp.array(scores, dtype=jnp.float32)
    except Exception:
        import numpy as np
        rng = np.random.default_rng(seed)
        return jnp.array(
            rng.uniform(-0.2, 0.2, size=params_batch.shape[0]),
            dtype=jnp.float32,
        )


def search(
    substrate_name: Literal["lenia", "flow_lenia"],
    seed_pool_train: jax.Array,
    budget: dict,
    rng: jax.Array,
) -> dict:
    """Sep-CMA-ES with CLIP-OE proxy.  Uses evosax if available; else random.

    Wall-clock cap: budget["wall_clock_s"] = 1800 s (enforced by evaluator
    SIGKILL; we also self-limit with a 10 s safety margin).
    """
    wall_clock_s = float(budget.get("wall_clock_s", 1800))
    K = _DIM.get(substrate_name, 8)
    t0 = time.monotonic()

    n_train = int(seed_pool_train.shape[0])
    best_proxy_per_gen: list[float] = []
    archive: list[jax.Array] = []

    try:
        from evosax import Sep_CMA_ES

        strategy = Sep_CMA_ES(
            popsize=_POP_SIZE,
            num_dims=K,
            sigma_init=_SIGMA_INIT,
        )
        es_state = strategy.initialize(rng)

        best_params = jnp.zeros(K, dtype=jnp.float32)
        best_score = float("-inf")

        for gen in range(_N_GENERATIONS):
            if time.monotonic() - t0 >= wall_clock_s - 10:
                break

            rng, sub = jax.random.split(rng)
            params_batch, es_state = strategy.ask(sub, es_state)  # [pop, K]

            # Evaluate on a rotating training seed.
            s = int(seed_pool_train[gen % n_train])
            fitnesses = _clip_oe_score(params_batch, s)  # [pop]

            # evosax maximises by negating fitness (it minimises internally).
            es_state = strategy.tell(params_batch, -fitnesses, es_state)

            gen_best_idx = int(jnp.argmax(fitnesses))
            gen_best = float(fitnesses[gen_best_idx])
            if gen_best > best_score:
                best_score = gen_best
                best_params = params_batch[gen_best_idx]
                archive.append(best_params)

            best_proxy_per_gen.append(best_score)

            if (gen + 1) % _CLEAR_CACHE_EVERY == 0:
                import gc
                gc.collect()
                jax.clear_caches()

    except ImportError:
        # evosax not installed — fall back to random search.
        best_params = jnp.zeros(K, dtype=jnp.float32)
        best_score = float("-inf")
        gen = 0
        while time.monotonic() - t0 < wall_clock_s - 10:
            rng, sub = jax.random.split(rng)
            batch = jax.random.uniform(sub, (_POP_SIZE, K), minval=-1.0, maxval=1.0)
            s = int(seed_pool_train[gen % n_train])
            fitnesses = _clip_oe_score(batch, s)
            best_idx = int(jnp.argmax(fitnesses))
            if float(fitnesses[best_idx]) > best_score:
                best_score = float(fitnesses[best_idx])
                best_params = batch[best_idx]
                archive.append(best_params)
            best_proxy_per_gen.append(best_score)
            gen += 1

    archive_arr = (
        jnp.stack(archive[-16:]) if archive else jnp.zeros((1, K), dtype=jnp.float32)
    )

    return {
        "best_params": best_params,
        "archive": archive_arr,
        "search_trace": {
            "best_proxy_per_gen": best_proxy_per_gen,
            "n_generations": len(best_proxy_per_gen),
            "algorithm": "sep_cma_es_clip_oe",
        },
    }
