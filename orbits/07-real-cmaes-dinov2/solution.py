"""solution.py — REAL Sep-CMA-ES + corrected DINOv2-lite OE proxy (orbit 07).

Clean-slate EXTEND of orbit 05 (`05-dinov2-oe`) after the eval-v2 provenance
audit revealed two compounding issues in batch 1:

  1. The Modal search image did NOT install `evosax`. Every orbit labelled
     "Sep-CMA-ES" silently fell back to random search via an `ImportError`
     branch. Commit `7fbd935` installed evosax. This orbit removes the
     fallback entirely: if evosax fails to import, we crash instead of
     lying about the algorithm.
  2. Orbit 05's DINOv2-lite kernels had two bugs flagged by orbit-reviewer:
     (a) kernels were NOT orthogonal — `k / ||k||_F` only unit-normalises
         each kernel, it does NOT give an orthonormal basis, so neighbouring
         kernels had high inner product and the 96-dim "feature" was
         effectively low-rank;
     (b) at scales 1 and 2 the multi-scale path *bilinearly upsampled* the
         random 16×16 base kernel to 32×32 and 64×64, destroying whatever
         orthogonality survived (a).
     Net effect: what was called a "multi-scale patch-structural embedding"
     was closer to "three copies of the same near-degenerate embedding".

This orbit fixes both. Research question: with genuine CMA-ES and a
genuinely orthogonal multi-scale patch embedding, does orbit 05's +0.294
signal survive, or was it a lucky random-search seed?

Fixes vs orbit 05
-----------------

* **Orthogonal kernels (A).** Each scale draws `N_KERNELS` IID Gaussian
  tensors, flattens them to shape `(N_KERNELS, C * ph * pw)`, runs a thin
  QR decomposition, and the rows of Q^T become an orthonormal kernel bank.
  Feature responses at different kernel indices are now genuinely
  linearly independent.

* **Native-scale kernels (B).** Scale `s` samples a fresh random tensor
  of shape `(N_KERNELS, 3, PATCH*2^s, PATCH*2^s)` and QR-orthogonalises
  it at that native size. No `jax.image.resize` on the kernel. The
  three scales now carry independent information.

* **Real Sep-CMA-ES (C).** `from evosax import Sep_CMA_ES` at module top
  level; no `try/except ImportError`. If the import fails, the orbit
  crashes honestly and its status becomes `search_crash` in the eval
  record — which is the truthful outcome when the declared algorithm
  cannot run.

* **Real DINOv2 attempt (D).** On every startup we still try the real
  `transformers.Dinov2Model` path first, in case a future image bump
  adds torch. On the current image torch is absent and we fall through
  to DINOv2-lite, which is now a genuine orthogonal multi-scale embedding
  rather than orbit 05's near-rank-one version.

Constraints respected
---------------------

* Substrate: `lenia` (flow_lenia kept for symmetry but not the target).
* 8-dim `params`, deterministic under fixed seed/rng.
* ≤ 200 MB artifacts; MP4/GIF rather than raw tensors.
* No network — we never call api.anthropic.com; DINOv2 weights attempted
  only from the local `/cache/hf` Modal volume.
* Interface: `search(substrate_name, seed_pool_train, budget, rng) -> dict`.
* Module-level `_FM_CACHE`; `gc.collect() + jax.clear_caches()` every 10 gens.
* Self-limits to `wall_clock_s - 15` s so the evaluator's SIGKILL never
  fires in the common case.
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

# Mandatory, no-fallback import — if evosax is missing we want to crash
# loudly, not silently degrade to random search (the eval-v2 bug).
from evosax import Sep_CMA_ES  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

_DIM = {"lenia": 8, "flow_lenia": 8}

# Canary-validated Sep-CMA-ES profile (orbit 05, research/eval/canary_results_full.md §root-cause).
_POP_SIZE = 16
_SIGMA_INIT = 0.1
_N_GENERATIONS = 60
_CLEAR_CACHE_EVERY = 10

# ASAL "keep it moving" frame picks — early / mid-early / mid-late / late.
_FRAME_PICKS = (0, 63, 127, 255)
_GRID = 128

# DINOv2-lite (corrected) hyperparameters.
_BASE_PATCH = 16          # native patch size at scale 0
_N_KERNELS = 32           # per scale
_N_SCALES = 3             # scales 0, 1, 2  → native patch sizes 16, 32, 64
_D_MODEL = _N_KERNELS * _N_SCALES   # == 96

# Deterministic kernel-init seed. Identical to orbit 05 so the only
# difference between orbit 07 and orbit 05 kernel banks is the
# QR-orthogonalisation + native-scale construction (not the RNG).
_KERNEL_SEED = 0xD1_50_FE

# Module-level FM cache: populated once per process.
_FM_CACHE: dict = {}

# Algorithm provenance string used by the campaign viz and by
# orbit-reviewer to distinguish real CMA-ES from random-search wearing a
# CMA-ES label (the eval-v1 batch-1 bug).
_ALGO_TAG_BASE = "sep_cma_es_dinov2_REAL_eval_v2"


# ══════════════════════════════════════════════════════════════════════════
# SUBSTRATE ROLLOUTS — verbatim from evaluator._lenia_rollout /
# _flow_lenia_rollout (orbit 05 inlined them for self-containment; we
# keep the same pattern).
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
# DINOv2 loader — tries real torch DINOv2 first, falls through to
# corrected DINOv2-lite. We DO NOT silently treat the fallback as the
# real model; `search_trace["fm_backend"]` records which path ran.
# ══════════════════════════════════════════════════════════════════════════

def _try_load_torch_dinov2() -> bool:
    try:
        import torch  # type: ignore
        from transformers import Dinov2Model, AutoImageProcessor  # type: ignore
    except Exception:
        return False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(0)
        _FM_CACHE["torch"] = torch
        _FM_CACHE["processor"] = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-small", cache_dir="/cache/hf"
        )
        model = Dinov2Model.from_pretrained(
            "facebook/dinov2-small", cache_dir="/cache/hf"
        ).eval()
        _FM_CACHE["torch_model"] = model
        _FM_CACHE["backend"] = "torch_dinov2"
        return True
    except Exception:
        return False


def _build_orthogonal_kernels() -> list[jax.Array]:
    """Build N_SCALES kernel banks of orthonormal random kernels.

    For each scale s:
      1. Sample G ~ N(0, I) of shape (N_KERNELS, C * ph_s * pw_s) with
         ph_s = pw_s = BASE_PATCH * 2**s.
      2. Thin QR decomposition: G = Q R with Q having orthonormal COLUMNS.
         We want orthonormal ROWS (one per kernel), so we transpose: the
         rows of Q^T are orthonormal iff N_KERNELS ≤ C*ph*pw, which it
         is (32 ≤ 3·16·16 = 768 at the smallest scale).
      3. Reshape back to (N_KERNELS, C, ph_s, pw_s).

    This is the single load-bearing fix vs orbit 05's kernel bank.
    """
    rng = jax.random.PRNGKey(_KERNEL_SEED)
    kernels: list[jax.Array] = []
    C = 3
    for s in range(_N_SCALES):
        rng, sub = jax.random.split(rng)
        ph = _BASE_PATCH * (2 ** s)
        pw = ph
        dim = C * ph * pw
        # Shape: (dim, N_KERNELS) so QR returns Q of shape (dim, N_KERNELS)
        # with orthonormal COLUMNS — each column is one kernel vector.
        G = jax.random.normal(sub, (dim, _N_KERNELS), dtype=jnp.float32)
        Q, _R = jnp.linalg.qr(G, mode="reduced")       # Q: (dim, N_KERNELS)
        Q = Q.T                                         # (N_KERNELS, dim)
        k = Q.reshape(_N_KERNELS, C, ph, pw)
        kernels.append(k)
    return kernels


def _ensure_fm_loaded() -> str:
    if "backend" in _FM_CACHE:
        return _FM_CACHE["backend"]
    if _try_load_torch_dinov2():
        # Also keep orthogonal kernels around in case caller probes
        # the lite path for debugging/ablation.
        _FM_CACHE["dinovlite_kernels"] = _build_orthogonal_kernels()
        return _FM_CACHE["backend"]
    _FM_CACHE["dinovlite_kernels"] = _build_orthogonal_kernels()
    _FM_CACHE["backend"] = "dinov2_lite_ortho"
    return "dinov2_lite_ortho"


def _embed_torch_dinov2(frames_uint8: np.ndarray) -> np.ndarray:
    import torch
    from PIL import Image as PILImage

    processor = _FM_CACHE["processor"]
    model = _FM_CACHE["torch_model"]
    pil_imgs = [PILImage.fromarray(f) for f in frames_uint8]
    inputs = processor(images=pil_imgs, return_tensors="pt")
    with torch.inference_mode():
        out = model(pixel_values=inputs["pixel_values"])
    cls = out.last_hidden_state[:, 0, :].detach().cpu().numpy()
    return cls


def _embed_dinov2_lite(frames_uint8: np.ndarray) -> np.ndarray:
    """Corrected DINOv2-lite: orthonormal multi-scale patch encoder.

    Differences vs orbit 05:
      (A) kernels orthogonalised via QR (see `_build_orthogonal_kernels`);
      (B) kernels constructed at native patch size per scale — no
          `jax.image.resize` of kernels, which was the bug that collapsed
          scales 1 and 2 to near-copies of scale 0.

    Feature pipeline per frame:
      1. For each scale s (native patch size ph = 16·2^s):
         extract non-overlapping patches of shape (ph, ph), compute the
         inner product with each of the N_KERNELS=32 orthonormal kernels,
         global-mean across patches.
      2. Per-frame LayerNorm the 32-d response vector to cancel
         brightness drift.
      3. Concatenate the three scale vectors → 96-d DINOv2-lite token.

    Pure JAX, deterministic under `_KERNEL_SEED`, no network.

    Args:
      frames_uint8: [T, H, W, 3] uint8.

    Returns:
      [T, D] float32 with D=96.
    """
    x = jnp.asarray(frames_uint8, dtype=jnp.float32) / 255.0   # [T,H,W,3]
    x = jnp.transpose(x, (0, 3, 1, 2))                         # [T,3,H,W]
    T, C, H, W = x.shape

    kernels = _FM_CACHE["dinovlite_kernels"]
    feats = []
    for s, k in enumerate(kernels):
        ph = _BASE_PATCH * (2 ** s)
        if ph > H or ph > W:
            # Scale exceeds image; pad a zero vector so the concat keeps
            # a constant D.
            feats.append(jnp.zeros((T, _N_KERNELS), dtype=jnp.float32))
            continue
        # Non-overlapping patch extraction via reshape — H, W assumed
        # divisible by ph at the scales we use (128 % 16 = 128 % 32 =
        # 128 % 64 = 0).
        nh = H // ph
        nw = W // ph
        patches = x.reshape(T, C, nh, ph, nw, ph)
        patches = jnp.transpose(patches, (0, 2, 4, 1, 3, 5))
        # [T, nh, nw, C, ph, ph] — inner product with k: [N_KERNELS, C, ph, ph]
        resp = jnp.einsum("tnmcij,kcij->tnmk", patches, k)
        # Global mean across patches.
        feat = resp.mean(axis=(1, 2))                          # [T, N_KERNELS]
        # LayerNorm per-frame — makes the embedding invariant to DC / gain.
        feat = (feat - feat.mean(-1, keepdims=True)) / (
            feat.std(-1, keepdims=True) + 1e-6
        )
        feats.append(feat)

    z = jnp.concatenate(feats, axis=-1)                        # [T, 96]
    return np.asarray(z, dtype=np.float32)


def _embed_frames(frames_uint8: np.ndarray) -> np.ndarray:
    backend = _ensure_fm_loaded()
    if backend == "torch_dinov2":
        return _embed_torch_dinov2(frames_uint8)
    return _embed_dinov2_lite(frames_uint8)


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
# SEARCH — real Sep-CMA-ES on the corrected DINOv2-lite OE proxy.
# ══════════════════════════════════════════════════════════════════════════

def search(
    substrate_name: Literal["lenia", "flow_lenia"],
    seed_pool_train: jax.Array,
    budget: dict,
    rng: jax.Array,
) -> dict:
    """Real Sep-CMA-ES on the corrected DINOv2-lite OE proxy.

    This orbit honours the eval-v2 truth rule: the algorithm string
    reflects what actually ran. If evosax is present and succeeds, the
    tag is `sep_cma_es_dinov2_REAL_eval_v2[<fm_backend>]`. If evosax or
    the search crash, we surface the exception (no random-search
    fallback) so the evaluator records `search_crash` honestly.
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

    # Trigger FM load eagerly so we know the backend by the time the
    # first generation runs (and so it appears in search_trace even on
    # early timeout).
    try:
        backend_used = _ensure_fm_loaded()
    except Exception:
        backend_used = "unknown"

    ckpt_dir = "/tmp/ckpt"
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
    except Exception:
        ckpt_dir = None

    best_params = jnp.zeros(K, dtype=jnp.float32)
    best_score = float("-inf")

    # Real Sep-CMA-ES — no ImportError fallback. A bare `except` is
    # retained around the LOOP (not the import) so a per-generation
    # failure doesn't throw away the best-so-far.
    strategy = Sep_CMA_ES(
        popsize=_POP_SIZE,
        num_dims=K,
        sigma_init=_SIGMA_INIT,
    )
    es_state = strategy.initialize(rng)
    algo = f"{_ALGO_TAG_BASE}[{backend_used}]"

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
                            os.path.join(
                                ckpt_dir, "orbit07_dinov2_best.npy"
                            ),
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
        # Record the exception in the trace but keep whatever best_params
        # we found so the evaluator can still roll out and score.
        algo = f"{_ALGO_TAG_BASE}[{backend_used}] crashed@gen={len(best_proxy_per_gen)}: {type(exc).__name__}"

    bp = np.asarray(best_params, dtype=np.float32)
    if not np.all(np.isfinite(bp)) or not math.isfinite(best_score):
        bp = np.zeros(K, dtype=np.float32)
        best_params = jnp.asarray(bp)

    archive_arr = (
        jnp.stack(archive[-16:])
        if archive
        else jnp.zeros((1, K), dtype=jnp.float32)
    )

    return {
        "best_params": best_params,
        "archive": archive_arr,
        "search_trace": {
            "best_proxy_per_gen": best_proxy_per_gen,
            "mean_proxy_per_gen": mean_proxy_per_gen,
            "gen_times": gen_times,
            "n_generations": len(best_proxy_per_gen),
            "algorithm": algo,
            "fm_backend": backend_used,
            "best_score": best_score if math.isfinite(best_score) else None,
            "kernel_fix": "qr_orthogonal_native_scale",
        },
    }
