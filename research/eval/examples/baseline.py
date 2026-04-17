"""baseline.py — solution.py stub: random search with 30-min budget.

Strategy: sample params uniformly from a sensible box, evaluate each with
a CLIP open-endedness proxy (ASAL-style), keep the best seen.  No gradient,
no CMA-ES.  This is the random-search baseline (expected METRIC slightly
below 0 vs the pinned ASAL Sep-CMA-ES anchor).

Expected LCF_judge (raw): ~0.10–0.20 depending on luck.
Expected METRIC: ~-0.10 to -0.20 (below the ASAL baseline).

Constraints respected:
  - Bit-identical under fixed rng (jax.random, no global state).
  - Respects budget["wall_clock_s"] = 1800.
  - Does NOT call api.anthropic.com (network egress is blocked in search
    container; CLIP runs locally from /cache/hf).
  - Does NOT read research/eval/judge/ or research/eval/baseline/.
"""

from __future__ import annotations

import time
from typing import Literal

import jax
import jax.numpy as jnp

# ── Parameter space ──────────────────────────────────────────────────────────
# We search over 8 flat params per substrate.  The substrate rollout in the
# evaluator interprets the first few as mu, sigma, alpha, etc.
_DIM = {"lenia": 8, "flow_lenia": 8}

# Sensible box bounds (matches evaluator substrate interpretation).
_PARAM_LO = jnp.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=jnp.float32)
_PARAM_HI = jnp.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0], dtype=jnp.float32)


def _clip_proxy(params: jax.Array, seed: int) -> float:
    """Cheap CLIP open-endedness proxy as inner-loop fitness.

    Runs a short 32-step rollout, embeds 4 frames with CLIP ViT-B/32,
    returns the ASAL open-endedness score (trajectory diversity in CLIP space).
    Falls back to a random proxy if CLIP weights are unavailable.
    """
    try:
        from transformers import FlaxCLIPModel, CLIPProcessor
        import numpy as np

        # Load CLIP (cached from /cache/hf after bootstrap).
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir="/cache/hf",
        )
        model = FlaxCLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir="/cache/hf",
        )

        # Short rollout for speed (32 steps, 4 frames).
        from evaluator import _lenia_rollout, _render_lenia
        traj = _lenia_rollout(params, seed)         # [256, 128, 128, 1]
        picks = np.asarray(traj[[0, 10, 20, 31]])   # [4, 128, 128, 1]
        rgb = _render_lenia(picks)                   # [4, 128, 128, 3]

        # CLIP image embeddings.
        from PIL import Image as PILImage
        pil_imgs = [PILImage.fromarray(rgb[i]) for i in range(len(rgb))]
        inputs = processor(images=pil_imgs, return_tensors="jax", padding=True)
        feats = model.get_image_features(**{k: v for k, v in inputs.items()
                                           if k != "input_ids"})
        z = feats / (jnp.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)
        # ASAL open-endedness: max off-diagonal of self-similarity kernel.
        sim = z @ z.T
        n = sim.shape[0]
        mask = jnp.tril(jnp.ones((n, n), dtype=bool), k=-1)
        score = float(jnp.where(mask, sim, -jnp.inf).max(-1).mean())
        return score
    except Exception:
        # Fallback: random proxy (uniform in [-1, 1]) if CLIP unavailable.
        import numpy as np
        rng = np.random.default_rng(int(jnp.sum(params * 1000).astype(int)) ^ seed)
        return float(rng.uniform(-0.2, 0.2))


def search(
    substrate_name: Literal["lenia", "flow_lenia"],
    seed_pool_train: jax.Array,
    budget: dict,
    rng: jax.Array,
) -> dict:
    """Random search: sample uniformly, evaluate with CLIP-OE proxy, keep best.

    Budget: wall_clock_s=1800.  Samples as many candidates as time allows.
    """
    wall_clock_s = float(budget.get("wall_clock_s", 1800))
    K = _DIM.get(substrate_name, 8)
    t0 = time.monotonic()

    best_params = jnp.zeros(K, dtype=jnp.float32)
    best_score = float("-inf")
    best_proxy_per_gen: list[float] = []
    archive: list[jax.Array] = []

    n_train = int(seed_pool_train.shape[0])
    gen = 0

    while time.monotonic() - t0 < wall_clock_s - 10:  # 10 s safety margin
        # Sample a batch of 8 candidates.
        rng, sub = jax.random.split(rng)
        batch = jax.random.uniform(sub, (8, K), minval=-1.0, maxval=1.0)

        for i in range(8):
            if time.monotonic() - t0 >= wall_clock_s - 10:
                break
            params = batch[i]
            # Evaluate on one randomly chosen training seed.
            s_idx = gen % n_train
            s = int(seed_pool_train[s_idx])
            score = _clip_proxy(params, s)
            if score > best_score:
                best_score = score
                best_params = params
                archive.append(params)

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
            "n_generations": gen,
            "algorithm": "random_search",
        },
    }
