"""solution.py — REAL facebook/dinov2-small via PyTorch + transformers (orbit 12).

EXTEND of orbit 07 (`07-real-cmaes-dinov2`, leader at +0.319). Identical
search strategy (Sep-CMA-ES, popsize=16, 60 gens, sigma=0.1) — the only
thing we change is the foundation-model backend driving the OE proxy.

Hypothesis
----------

Orbit 07 uses a hand-rolled "DINOv2-lite" QR-orthogonal multi-scale patch
encoder because the Modal search image was missing torch. Orbit 12 swaps
that proxy for the real `facebook/dinov2-small` pretrained model via
PyTorch + HuggingFace transformers. A genuinely pretrained FM should
produce a richer, more semantically meaningful OE signal than random
orthogonal kernels — so if orbit 07's +0.319 signal is real (not just an
artifact of the lite proxy), a real FM should match or exceed it.

Fail-loud policy
----------------

Per the relaunch plan, we do NOT try/except the torch import. If the
Modal image lacks torch, the orbit crashes at module load time and the
evaluator records `search_crash`. That is the correct, honest outcome —
a silent fallback to DINOv2-lite would collapse orbit 12 into orbit 07
and hide the real comparison we want to make.

Determinism
-----------

* `torch.use_deterministic_algorithms(True, warn_only=True)`
* `torch.manual_seed(0)` at module load
* `torch.inference_mode()` for every forward pass
* Module-level cache for torch, processor, model — loaded once per process

Interface (unchanged from orbit 07)
-----------------------------------

    search(substrate_name, seed_pool_train, budget, rng) ->
        {"best_params", "best_rollout", "archive", "search_trace"}
"""

from __future__ import annotations

import gc
import math
import os
import time
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

# ── Fail-loud imports. No try/except. If either is missing, the orbit
#    crashes and the evaluator logs `search_crash` — honest signal that
#    the Modal image needs torch. ─────────────────────────────────────────
from evosax import Sep_CMA_ES  # noqa: E402
import torch  # noqa: E402
from transformers import AutoImageProcessor, AutoModel  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS — identical to orbit 07 to isolate the FM change.
# ══════════════════════════════════════════════════════════════════════════

_DIM = {"lenia": 8, "flow_lenia": 8}

_POP_SIZE = 16
_SIGMA_INIT = 0.1
_N_GENERATIONS = 60
_CLEAR_CACHE_EVERY = 10

_FRAME_PICKS = (0, 63, 127, 255)      # ASAL-style early/mid/late picks
_GRID = 128

_ALGO_TAG = "sep_cma_es_real_dinov2_small_eval_v2"

# Module-level FM cache.
_FM_CACHE: dict = {}


# ══════════════════════════════════════════════════════════════════════════
# SUBSTRATE ROLLOUTS — inlined from orbit 07 verbatim.
# ══════════════════════════════════════════════════════════════════════════

def _lenia_rollout(params: jax.Array, seed: int) -> jax.Array:
    p = jnp.concatenate(
        [jnp.asarray(params, jnp.float32).ravel(),
         jnp.zeros(max(0, 8 - jnp.asarray(params).size), jnp.float32)]
    )
    mu = jnp.clip(p[0] * 0.1 + 0.15, 0.05, 0.45)
    sigma = jnp.clip(p[1] * 0.02 + 0.015, 0.003, 0.06)
    R = 13.0
    dt = 0.1
    H = W = _GRID
    ys, xs = jnp.mgrid[:H, :W].astype(jnp.float32)
    yc, xc = ys - H / 2, xs - W / 2

    def bell(x, m, s):
        return jnp.exp(-((x - m) / s) ** 2 / 2)

    r = jnp.sqrt(xc ** 2 + yc ** 2) / R
    K = bell(r, 0.5, 0.15) * (r < 1)
    K = K / (K.sum() + 1e-9)
    Kf = jnp.fft.fft2(jnp.fft.ifftshift(K))
    noise = jax.random.uniform(jax.random.PRNGKey(seed), (H, W), jnp.float32)
    A0 = jnp.where(yc ** 2 + xc ** 2 < (R * 1.5) ** 2, noise, 0.0)

    def step(A, _):
        U = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(A) * Kf))
        A_new = jnp.clip(A + dt * (bell(U, mu, sigma) * 2 - 1), 0, 1)
        return A_new, A_new

    _, traj = jax.lax.scan(step, A0, None, length=256)
    return traj[..., None]


def _flow_lenia_rollout(params: jax.Array, seed: int) -> jax.Array:
    p = jnp.concatenate(
        [jnp.asarray(params, jnp.float32).ravel(),
         jnp.zeros(max(0, 8 - jnp.asarray(params).size), jnp.float32)]
    )
    mu = jnp.clip(p[0] * 0.1 + 0.15, 0.05, 0.45)
    sigma = jnp.clip(p[1] * 0.02 + 0.015, 0.003, 0.06)
    alpha = jnp.clip(p[2] * 0.5 + 1.0, 0.5, 3.0)
    R = 13.0
    dt = 0.2
    H = W = _GRID
    ys, xs = jnp.mgrid[:H, :W].astype(jnp.float32)
    yc, xc = ys - H / 2, xs - W / 2

    def bell(x, m, s):
        return jnp.exp(-((x - m) / s) ** 2 / 2)

    r = jnp.sqrt(xc ** 2 + yc ** 2) / R
    K = bell(r, 0.5, 0.15) * (r < 1)
    K = K / (K.sum() + 1e-9)
    Kf = jnp.fft.fft2(jnp.fft.ifftshift(K))
    noise = jax.random.uniform(jax.random.PRNGKey(seed), (H, W), jnp.float32)
    A0 = jnp.where(yc ** 2 + xc ** 2 < (R * 1.5) ** 2, noise * 0.8, 0.0)

    def step(A, _):
        U = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(A) * Kf))
        vy = alpha * (jnp.roll(U, -1, 0) - jnp.roll(U, 1, 0)) * 0.5
        vx = alpha * (jnp.roll(U, -1, 1) - jnp.roll(U, 1, 1)) * 0.5
        sy = ys - vy * dt
        sx = xs - vx * dt
        y0 = jnp.floor(sy).astype(jnp.int32)
        x0 = jnp.floor(sx).astype(jnp.int32)
        fy, fx = sy - y0, sx - x0

        def g_(yy, xx):
            return A[jnp.mod(yy, H), jnp.mod(xx, W)]

        Aa = (
            (1 - fy) * (1 - fx) * g_(y0, x0)
            + (1 - fy) * fx * g_(y0, x0 + 1)
            + fy * (1 - fx) * g_(y0 + 1, x0)
            + fy * fx * g_(y0 + 1, x0 + 1)
        )
        An = jnp.clip(Aa + dt * (bell(U, mu, sigma) * 2 - 1), 0, 1)
        An = An * (A.sum() + 1e-9) / (An.sum() + 1e-9)
        return An, jnp.stack([An, jnp.sqrt(vx ** 2 + vy ** 2)], axis=-1)

    _, traj = jax.lax.scan(step, A0, None, length=256)
    return traj.astype(jnp.float32)


def _render_lenia_rgb(picks: jax.Array) -> np.ndarray:
    x = np.clip(np.asarray(picks).squeeze(-1), 0, 1)
    r = np.clip(1.5 * x - 0.2, 0, 1)
    g = np.clip(1.5 * np.minimum(x, 1 - x) * 2, 0, 1)
    b = np.clip(1.5 * (1 - x) - 0.2, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _render_flow_lenia_rgb(picks: jax.Array) -> np.ndarray:
    x = np.asarray(picks)
    m = x[..., 0]
    f = x[..., 1]
    rgb = np.clip(np.stack([m, 0.5 * f, 0.2 * m], -1), 0, 1)
    return (rgb * 255).astype(np.uint8)


def _render_rgb(picks: jax.Array, substrate_name: str) -> np.ndarray:
    if substrate_name == "lenia":
        return _render_lenia_rgb(picks)
    return _render_flow_lenia_rgb(picks)


# ══════════════════════════════════════════════════════════════════════════
# REAL DINOv2 loader — facebook/dinov2-small via HuggingFace transformers.
# Fail-loud: no try/except on the model load. If the HF cache + network
# fallback cannot resolve the weights, the orbit crashes honestly.
# ══════════════════════════════════════════════════════════════════════════

def _ensure_fm_loaded() -> str:
    if "backend" in _FM_CACHE:
        return _FM_CACHE["backend"]

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(0)

    cache_dir = os.environ.get("HF_HOME", "/cache/hf")
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-small", cache_dir=cache_dir
    )
    model = AutoModel.from_pretrained(
        "facebook/dinov2-small", cache_dir=cache_dir
    ).eval()
    # CPU is fine — the embedding cost (4 frames × pop 16) is tiny vs rollout.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    _FM_CACHE["torch"] = torch
    _FM_CACHE["processor"] = processor
    _FM_CACHE["model"] = model
    _FM_CACHE["device"] = device
    _FM_CACHE["backend"] = "real_dinov2_small"
    return _FM_CACHE["backend"]


def _embed_frames(frames_uint8: np.ndarray) -> np.ndarray:
    """Embed [T, H, W, 3] uint8 frames with real facebook/dinov2-small.

    Returns:
      [T, D] float32 CLS embeddings (D=384 for dinov2-small).
    """
    from PIL import Image as PILImage

    processor = _FM_CACHE["processor"]
    model = _FM_CACHE["model"]
    device = _FM_CACHE["device"]

    pil_imgs = [PILImage.fromarray(f) for f in frames_uint8]
    inputs = processor(images=pil_imgs, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.inference_mode():
        out = model(pixel_values=pixel_values)
    # CLS token = last_hidden_state[:, 0, :] for DINOv2.
    cls = out.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
    return cls


# ══════════════════════════════════════════════════════════════════════════
# ASAL OPEN-ENDEDNESS SCORE — FM-agnostic.
# Given [F, D] per-frame embeddings: OE = -mean_i max_{j<i} cos(z_i, z_j).
# Higher = trajectory keeps moving through FM space (diverse rollout).
# ══════════════════════════════════════════════════════════════════════════

def _dinov2_oe_score(
    params_batch: jax.Array, seed: int, substrate_name: str
) -> np.ndarray:
    pop = params_batch.shape[0]
    scores = np.zeros(pop, dtype=np.float32)
    roll = _lenia_rollout if substrate_name == "lenia" else _flow_lenia_rollout
    pick_idx = jnp.asarray(_FRAME_PICKS)
    for i in range(pop):
        try:
            traj = roll(params_batch[i], seed)
            picks = traj[pick_idx]
            rgb = _render_rgb(picks, substrate_name)
            z = _embed_frames(rgb)
            if not np.all(np.isfinite(z)) or np.allclose(z, 0):
                scores[i] = -1.0
                continue
            z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
            sim = z @ z.T
            n = sim.shape[0]
            mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
            sim_masked = np.where(mask, sim, -np.inf)
            row_max = sim_masked.max(axis=-1)
            valid = row_max[row_max > -np.inf]
            if valid.size == 0:
                scores[i] = -1.0
                continue
            scores[i] = float(-valid.mean())
        except Exception:
            scores[i] = -1.0
    return scores


# ══════════════════════════════════════════════════════════════════════════
# SEARCH — real Sep-CMA-ES on real facebook/dinov2-small OE proxy.
# ══════════════════════════════════════════════════════════════════════════

def search(
    substrate_name: Literal["lenia", "flow_lenia"],
    seed_pool_train: jax.Array,
    budget: dict,
    rng: jax.Array,
) -> dict:
    """Sep-CMA-ES driven by real facebook/dinov2-small OE proxy.

    Fail-loud: `import torch` and `from evosax import Sep_CMA_ES` run at
    module load; any missing dep crashes the orbit and shows up in the
    eval record as `search_crash` (correct, honest signal).
    """
    wall_clock_s = float(budget.get("wall_clock_s", 1800))
    K = _DIM.get(substrate_name, 8)
    t0 = time.monotonic()
    safety_margin = 15.0

    n_train = int(seed_pool_train.shape[0])
    best_proxy_per_gen: list[float] = []
    mean_proxy_per_gen: list[float] = []
    gen_times: list[float] = []
    archive: list[jax.Array] = []

    backend_used = _ensure_fm_loaded()

    ckpt_dir = "/tmp/ckpt"
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
    except Exception:
        ckpt_dir = None

    best_params = jnp.zeros(K, dtype=jnp.float32)
    best_score = float("-inf")

    strategy = Sep_CMA_ES(
        popsize=_POP_SIZE,
        num_dims=K,
        sigma_init=_SIGMA_INIT,
    )
    es_state = strategy.initialize(rng)
    algo = f"{_ALGO_TAG}[{backend_used}]"

    try:
        for gen in range(_N_GENERATIONS):
            if time.monotonic() - t0 >= wall_clock_s - safety_margin:
                break
            t_gen = time.monotonic()

            rng, sub = jax.random.split(rng)
            params_batch, es_state = strategy.ask(sub, es_state)

            s = int(seed_pool_train[gen % n_train])
            fitnesses = _dinov2_oe_score(params_batch, s, substrate_name)

            # evosax minimises; negate so it maximises our OE score.
            es_state = strategy.tell(
                params_batch, -jnp.asarray(fitnesses), es_state
            )

            gen_best_idx = int(np.argmax(fitnesses))
            gen_best = float(fitnesses[gen_best_idx])
            if gen_best > best_score and np.isfinite(gen_best):
                best_score = gen_best
                best_params = params_batch[gen_best_idx]
                archive.append(best_params)
                if ckpt_dir is not None:
                    try:
                        np.save(
                            os.path.join(ckpt_dir, "orbit12_real_dinov2_best.npy"),
                            np.asarray(best_params, dtype=np.float32),
                        )
                    except Exception:
                        pass

            best_proxy_per_gen.append(best_score)
            mean_proxy_per_gen.append(float(np.mean(fitnesses)))
            gen_times.append(round(time.monotonic() - t_gen, 2))

            if (gen + 1) % _CLEAR_CACHE_EVERY == 0:
                gc.collect()
                jax.clear_caches()
    except Exception as exc:
        algo = (
            f"{_ALGO_TAG}[{backend_used}] "
            f"crashed@gen={len(best_proxy_per_gen)}: {type(exc).__name__}"
        )

    bp = np.asarray(best_params, dtype=np.float32)
    if not np.all(np.isfinite(bp)) or not math.isfinite(best_score):
        bp = np.zeros(K, dtype=np.float32)
        best_params = jnp.asarray(bp)

    archive_arr = (
        jnp.stack(archive[-16:])
        if archive
        else jnp.zeros((1, K), dtype=jnp.float32)
    )

    # Generate a single best_rollout for the evaluator's downstream use.
    roll = _lenia_rollout if substrate_name == "lenia" else _flow_lenia_rollout
    try:
        best_rollout = roll(best_params, int(seed_pool_train[0]))
    except Exception:
        best_rollout = jnp.zeros((1, _GRID, _GRID, 1), dtype=jnp.float32)

    return {
        "best_params": best_params,
        "best_rollout": best_rollout,
        "archive": archive_arr,
        "search_trace": {
            "best_proxy_per_gen": best_proxy_per_gen,
            "mean_proxy_per_gen": mean_proxy_per_gen,
            "gen_times": gen_times,
            "n_generations": len(best_proxy_per_gen),
            "algorithm": algo,
            "fm_backend": backend_used,
            "best_score": best_score if math.isfinite(best_score) else None,
            "fm_model": "facebook/dinov2-small",
        },
    }
