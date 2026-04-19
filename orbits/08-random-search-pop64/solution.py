"""solution.py — Orbit 08: random search pop=64 (scale test of orbit 04).

Refines `orbit/04-random-search-dense` by scaling the implicit population
from 1-at-a-time CLIP scoring up to **pop_size = 64 per generation**:
rollouts are `jax.vmap`'d over 64 candidates (one JIT trace), and the
CLIP image encoder forwards all 256 images (64 candidates × 4 frames)
in a single call per generation.

Research question
-----------------
Does random-search CLIP-OE improve with a larger parallel population at
a fixed wall-clock budget? Three plausible outcomes vs orbit 04
(METRIC = +0.233 at pop≈1 CLIP-serial):

  * **Linear scaling** → ~+0.27: exploration was bottlenecked.
  * **Sublinear** → ~+0.245: saturated at orbit 04's sample count.
  * **Flat / worse**: CLIP batch overhead cancels the extra coverage.

Key changes vs orbit 04
-----------------------
1. Numpy candidate sampling unchanged (3-component mixture).
2. Rollout is `jax.vmap`'d → one fixed-shape call evaluates pop=64
   candidates on one fixed seed per generation.
3. CLIP call takes a `[256, 3, 224, 224]` image batch (64 × 4 frames)
   once per generation instead of 64 separate 4-image calls. This is
   the real scaling lever: the 64× larger batch amortises Python/JAX
   launch overhead and saturates the A100's matmul units better.
4. `_N_RESTARTS=40` × `_POP=64` = 2560 candidate ceiling (close to
   orbit 04's 2466 actual, so we isolate the effect of *batching* from
   the effect of *more samples*).

Interface identical to orbit 04: `search(substrate_name, seed_pool_train,
budget, rng)`. `search_trace["algorithm"] = "random_search_pop64_eval_v2"`.
"""
from __future__ import annotations

import gc
import os
import time
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────
_DIM = {"lenia": 8, "flow_lenia": 8}
_POP = 64                        # candidates scored per generation
_N_RESTARTS = 40                 # 40 × 64 = 2560 candidate ceiling
_CLIP_EVERY = 4                  # 4 frames per rollout (shape-fixed)
_GC_EVERY_GEN = 5                # gc.collect() every 5 generations
_WALL_SAFETY_S = 30.0            # exit margin (pop=64 last-iter is larger)

# Orbium prior — inherited verbatim from orbit 04.
_PRIOR_MEAN = np.zeros(8, dtype=np.float32)
_PRIOR_STD_NARROW = 0.35
_PRIOR_STD_WIDE = 1.2

# Module-level caches (one load per worker).
_CLIP_CACHE: dict = {}
_ROLLOUT_CACHE: dict = {}


# ──────────────────────────────────────────────────────────────────────────────
# Lenia rollout — single-candidate body (unchanged), vmap'd over pop=64
# ──────────────────────────────────────────────────────────────────────────────
def _get_lenia_rollout_fn(T: int = 256, H: int = 128, W: int = 128):
    """Returns `f(params_batch[POP,8], seed_i32) -> [POP,T,H,W,1]`.

    Single-trace JIT: inputs always `(POP, 8)` float32 + scalar seed.
    """
    if "rollout_batch" in _ROLLOUT_CACHE:
        return _ROLLOUT_CACHE["rollout_batch"]

    ys, xs = jnp.mgrid[:H, :W].astype(jnp.float32)
    yc, xc = ys - H / 2, xs - W / 2

    def bell(x, m, s):
        return jnp.exp(-((x - m) / s) ** 2 / 2)

    r = jnp.sqrt(xc ** 2 + yc ** 2) / 13.0
    K = bell(r, 0.5, 0.15) * (r < 1)
    K = K / (K.sum() + 1e-9)
    Kf_const = jnp.fft.fft2(jnp.fft.ifftshift(K))

    def single_rollout(params, seed):
        p = jnp.asarray(params, jnp.float32).ravel()
        mu = jnp.clip(p[0] * 0.1 + 0.15, 0.05, 0.45)
        sigma = jnp.clip(p[1] * 0.02 + 0.015, 0.003, 0.06)
        dt = 0.1
        noise = jax.random.uniform(
            jax.random.PRNGKey(seed), (H, W), jnp.float32
        )
        A0 = jnp.where(yc ** 2 + xc ** 2 < (13.0 * 1.5) ** 2, noise, 0.)

        def step(A, _):
            U = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(A) * Kf_const))
            An = jnp.clip(A + dt * (bell(U, mu, sigma) * 2 - 1), 0, 1)
            return An, An

        _, traj = jax.lax.scan(step, A0, None, length=T)
        return traj[..., None]   # [T,H,W,1]

    # vmap over the leading params axis; seed is broadcast (scalar).
    batched = jax.jit(jax.vmap(single_rollout, in_axes=(0, None)))
    _ROLLOUT_CACHE["rollout_batch"] = batched
    return batched


def _render_lenia_np(picks_np):
    """`[N,H,W,1]` float in [0,1] -> `[N,H,W,3]` uint8 via 8-stop viridis."""
    vir = np.array(
        [[0.267, 0.005, 0.329], [0.230, 0.322, 0.546], [0.128, 0.567, 0.551],
         [0.191, 0.719, 0.455], [0.369, 0.789, 0.383], [0.586, 0.836, 0.292],
         [0.815, 0.878, 0.144], [0.993, 0.906, 0.144]],
        dtype=np.float32,
    )
    x = np.clip(np.asarray(picks_np).squeeze(-1), 0, 1)
    idx = np.round(x * (len(vir) - 1)).astype(np.int32)
    return (vir[idx] * 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# CLIP-OE scoring — BATCHED over pop=64 candidates per call
# ──────────────────────────────────────────────────────────────────────────────
def _load_clip():
    if "model" in _CLIP_CACHE:
        return _CLIP_CACHE["processor"], _CLIP_CACHE["model"]
    from transformers import CLIPProcessor, FlaxCLIPModel

    cache_dir = os.environ.get("HF_HOME", "/cache/hf")
    _CLIP_CACHE["processor"] = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32", cache_dir=cache_dir,
    )
    _CLIP_CACHE["model"] = FlaxCLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", cache_dir=cache_dir,
    )
    return _CLIP_CACHE["processor"], _CLIP_CACHE["model"]


_PICKS_IDX = np.array([0, 63, 127, 255], dtype=np.int32)


def _clip_oe_batch(traj_batch_jnp) -> np.ndarray:
    """Score pop candidates in ONE CLIP forward pass.

    traj_batch_jnp: [POP, T, H, W, 1].
    Returns: np.ndarray of length POP (float32 OE scores; NaN on failure).
    Fallback (no transformers): per-candidate pixel-novelty proxy.
    """
    pop = int(traj_batch_jnp.shape[0])
    picks = np.asarray(traj_batch_jnp[:, _PICKS_IDX])  # [POP, 4, H, W, 1]
    try:
        processor, model = _load_clip()
        from PIL import Image as PILImage

        rgb = _render_lenia_np(picks.reshape(pop * 4, picks.shape[2],
                                              picks.shape[3], 1))
        pil_imgs = [PILImage.fromarray(rgb[j]) for j in range(pop * 4)]
        inputs = processor(images=pil_imgs, return_tensors="jax", padding=True)
        feats = model.get_image_features(
            **{k: v for k, v in inputs.items() if k != "input_ids"}
        )  # [POP*4, D]
        feats = feats / (
            jnp.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8
        )
        z = feats.reshape(pop, 4, -1)                     # [POP,4,D]
        sim = jnp.einsum("pid,pjd->pij", z, z)            # [POP,4,4]
        mask = jnp.tril(jnp.ones((4, 4), dtype=bool), k=-1)
        scores = jnp.where(mask[None], sim, -jnp.inf).max(-1).mean(-1)
        return np.asarray(scores, dtype=np.float32)
    except Exception:
        # Pixel-novelty proxy (local-only fallback).
        pix = picks.squeeze(-1).reshape(pop, 4, -1).mean(axis=2)  # [POP,4]
        stds = pix.std(axis=1)
        stds = np.where(np.isfinite(stds), stds, -1.0)
        return stds.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Proposal sampling — 3-component mixture (inherited from orbit 04)
# ──────────────────────────────────────────────────────────────────────────────
def _sample_batch(rng_np: np.random.Generator, batch: int, K: int) -> np.ndarray:
    u = rng_np.uniform(0, 1, size=batch)
    out = np.empty((batch, K), dtype=np.float32)

    narrow = u < 0.6
    wide = (u >= 0.6) & (u < 0.9)
    explode = u >= 0.9

    if narrow.any():
        out[narrow] = (
            _PRIOR_MEAN[:K]
            + rng_np.standard_normal((narrow.sum(), K)).astype(np.float32)
            * _PRIOR_STD_NARROW
        )
    if wide.any():
        out[wide] = rng_np.uniform(-1.0, 1.0, size=(wide.sum(), K)).astype(
            np.float32
        )
    if explode.any():
        out[explode] = (
            _PRIOR_MEAN[:K]
            + rng_np.standard_normal((explode.sum(), K)).astype(np.float32)
            * _PRIOR_STD_WIDE
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Search entrypoint
# ──────────────────────────────────────────────────────────────────────────────
def search(
    substrate_name: Literal["lenia", "flow_lenia"],
    seed_pool_train: jax.Array,
    budget: dict,
    rng: jax.Array,
) -> dict:
    """Stateless random search with pop=64 batched rollout + CLIP scoring."""
    wall_clock_s = float(budget.get("wall_clock_s", 1800))
    safety_s = min(_WALL_SAFETY_S, max(1.0, wall_clock_s * 0.1))
    K = _DIM.get(substrate_name, 8)
    t0 = time.monotonic()

    rollout_fn = _get_lenia_rollout_fn()

    # Deterministic numpy RNG derived from JAX key.
    jkey = jax.random.fold_in(rng, 0)
    np_seed = int(jax.random.randint(jkey, (), 0, 2**31 - 1))
    rng_np = np.random.default_rng(np_seed)

    n_train = int(seed_pool_train.shape[0])

    best_params = jnp.zeros(K, dtype=jnp.float32)
    best_score = float("-inf")
    archive: list[np.ndarray] = []
    archive_scores: list[float] = []
    best_so_far_trace: list[float] = []
    all_scores: list[float] = []
    gen_elapsed_s: list[float] = []

    total_candidates = _N_RESTARTS * _POP
    c = 0
    gen = 0
    for gen in range(_N_RESTARTS):
        if time.monotonic() - t0 >= wall_clock_s - safety_s:
            break
        t_gen0 = time.monotonic()
        batch_np = _sample_batch(rng_np, _POP, K)
        params_b = jnp.asarray(batch_np, dtype=jnp.float32)      # [POP,K]
        train_seed = int(seed_pool_train[gen % n_train])

        # One vmap'd JIT call → [POP,T,H,W,1]
        traj_b = rollout_fn(params_b, train_seed)
        traj_b.block_until_ready()

        scores = _clip_oe_batch(traj_b)                          # [POP]
        # Release the big device buffer eagerly.
        del traj_b

        for j in range(_POP):
            s = float(scores[j])
            all_scores.append(s)
            if np.isfinite(s) and s > best_score:
                best_score = s
                best_params = params_b[j]
                archive.append(np.asarray(params_b[j]))
                archive_scores.append(s)
            best_so_far_trace.append(best_score)
            c += 1

        gen_elapsed_s.append(time.monotonic() - t_gen0)
        if (gen + 1) % _GC_EVERY_GEN == 0:
            gc.collect()

    # Trim archive to the last 16 improving checkpoints.
    if archive:
        archive_arr = jnp.stack(
            [jnp.asarray(a, jnp.float32) for a in archive[-16:]]
        )
    else:
        archive_arr = jnp.zeros((1, K), dtype=jnp.float32)

    elapsed = time.monotonic() - t0
    return {
        "best_params": best_params,
        "archive": archive_arr,
        "search_trace": {
            "best_proxy_per_gen": best_so_far_trace,
            "all_scores": all_scores,
            "n_candidates": len(all_scores),
            "n_target_candidates": total_candidates,
            "n_generations_completed": gen + (1 if all_scores else 0),
            "pop_size": _POP,
            "elapsed_s": elapsed,
            "gen_elapsed_s": gen_elapsed_s,
            "algorithm": "random_search_pop64_eval_v2",
            "proposal": {
                "narrow_frac": 0.6,
                "wide_frac": 0.3,
                "explode_frac": 0.1,
                "narrow_std": _PRIOR_STD_NARROW,
                "wide_std": _PRIOR_STD_WIDE,
            },
            "archive_scores": archive_scores[-16:],
        },
    }
