"""solution.py — Sep-CMA-ES + CLIP × DINOv2-lite geomean ensemble with α-anneal (orbit 13).

Hypothesis (REFINE of orbit 10):
Orbit 10 mixed CLIP-OE and DINOv2-lite-OE with a fixed arithmetic-mean
(α=0.5). It scored BELOW either FM alone. Two candidate root causes:

  (1) Arithmetic mean is forgiving — a candidate that fools ONE FM with a
      degenerate high score gets half-credit from the ensemble. The
      geometric mean requires BOTH FMs to like the candidate; a near-zero
      score on either collapses the product. That is the right inductive
      bias if we believe the FMs see complementary aspects of life-likeness.
  (2) Late in search, DINOv2-lite's randomly-kernelled patch signal may be
      a noisier teacher than CLIP, pulling the population off the CLIP
      optimum. Linearly annealing α from 0.5 → 1.0 uses DINOv2 only for
      early exploration and locks on CLIP in the exploitation phase.

Ensemble fitness per candidate:
    α_t   = 0.5 + 0.5 · (t / (N_GENERATIONS − 1))         # 0.5 → 1.0
    f_c   = (clip_z + ε)^α_t                              # ε = 1e-3
    f_d   = (dino_z + ε)^(1 − α_t)
    F(θ)  = f_c · f_d

Standardised scores (clip_z, dino_z) are first *shifted* onto a non-negative
domain before the geomean, so the power-product is well-defined:
    clip_z ← clip_z − clip_z.min() + ε_floor          (ε_floor = 1e-3)
    dino_z ← dino_z − dino_z.min() + ε_floor

where the min is taken over the frozen warmup batch — same stats as orbit
10's mean/std normalization, extended with a min so the shifted variable
is ≥ ε_floor everywhere under the warmup distribution. Subsequent
populations shift by the same frozen offset (bit-reproducible).

Bug fix vs orbit 10:
The parent relied on a MODULE-LEVEL `_NORM_CACHE` dict. Under the
search-worker subprocess pattern, that module is imported exactly once, so
in principle re-use is fine — but the cache persists across nested
`search()` calls within one process (e.g. if the evaluator re-enters), and
the warmup stats are then frozen from whatever seed ran first. Clear
`_NORM_CACHE` at the top of `search()` so each invocation freezes its own
warmup statistics from its own RNG-seeded first population. This makes the
algorithm deterministic per-seed rather than per-process.

Also added a *min-offset* to `_NORM_CACHE` (`clip_offset`, `dino_offset`)
so the geomean is mathematically well-formed. Orbit 10 did not need this.

FM stack (same as orbit 10, for a clean A/B):
  CLIP   : transformers.FlaxCLIPModel ViT-B/32 (module-level cached).
  DINOv2 : QR-orthogonalized multi-scale patch-kernel proxy in pure JAX
           (NOT real DINOv2). Keeping the same "DINOv2-lite" as orbit 10
           so any metric delta is attributable to the ensemble rule (geomean
           + α-anneal), not to a swap of the second FM.

Constraints respected (eval-v2):
  - No ImportError fallback on evosax; import is top-of-module.
  - Bit-identical under fixed rng (with the new per-call cache reset).
  - Respects budget["wall_clock_s"] (self-limits wall_clock − 15 s).
  - No api.anthropic.com calls (network blocked in search container).
  - Does not read research/eval/judge/ or research/eval/baseline/.
  - pop_size=16, ≤60 gens (OOM-safe profile).
  - search_trace["algorithm"] = "sep_cma_es_ensemble_geomean_alpha_anneal_eval_v2"
"""

from __future__ import annotations

import gc
import math
import os
import time
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from evosax import Sep_CMA_ES  # fail-loud per eval-v2 contract

# ═══════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

_DIM = {"lenia": 8, "flow_lenia": 8}

_POP_SIZE = 16
_SIGMA_INIT = 0.1
_N_GENERATIONS = 60
_CLEAR_CACHE_EVERY = 10
_SAFETY_MARGIN_S = 15.0
_FRAME_PICKS = (0, 63, 127, 255)
_GRID = 128

# α-anneal schedule: linearly from 0.5 → 1.0 across N_GENERATIONS.
_ALPHA_START = 0.5
_ALPHA_END = 1.0

# Geomean floor: ensures the powered factors are strictly positive.
_GEOMEAN_FLOOR = 1e-3

# DINOv2-lite hyperparameters (orbit 05's recipe, QR-orthonormal).
_PATCH = 16
_D_MODEL = 96
_N_KERNELS = 32
_N_SCALES = 3

# Module-level FM caches (populated lazily; prevents per-gen weight reloads).
_CLIP_CACHE: dict = {}
_DINOV2_CACHE: dict = {}
# NOTE: kept at module level for parity with orbit 10's structure, but
# `search()` clears it at entry so stats are per-call (bug fix).
_NORM_CACHE: dict = {}


# ═══════════════════════════════════════════════════════════════════════════
# FM LOADERS
# ═══════════════════════════════════════════════════════════════════════════

def _load_clip():
    """Load FlaxCLIPModel ViT-B/32 exactly once per process."""
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


def _init_dinov2_lite() -> None:
    """Pure-JAX DINOv2-lite: fixed multi-scale patch kernels, QR-orthonormal.
    Same recipe as orbit 10 (kept deliberately identical for clean A/B)."""
    rng = jax.random.PRNGKey(0xD1_50_FE)
    kernels = []
    for scale in range(_N_SCALES):
        rng, sub = jax.random.split(rng)
        ph = _PATCH * (2 ** scale)
        flat_dim = 3 * ph * ph
        n_k = min(_N_KERNELS, flat_dim)
        raw = jax.random.normal(sub, (flat_dim, n_k))
        q, _ = jnp.linalg.qr(raw)
        k = q.T.reshape(n_k, 3, ph, ph)
        if k.shape[0] < _N_KERNELS:
            pad = jnp.zeros((_N_KERNELS - k.shape[0], 3, ph, ph), k.dtype)
            k = jnp.concatenate([k, pad], axis=0)
        kernels.append(k)
    _DINOV2_CACHE["dinovlite_kernels"] = kernels
    _DINOV2_CACHE["backend"] = "dinov2_lite_jax"


def _ensure_dinov2_loaded() -> str:
    if "backend" in _DINOV2_CACHE:
        return _DINOV2_CACHE["backend"]
    _init_dinov2_lite()
    return _DINOV2_CACHE["backend"]


# ═══════════════════════════════════════════════════════════════════════════
# FM-SPECIFIC EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════

def _embed_clip(frames_uint8: np.ndarray) -> np.ndarray:
    """frames_uint8: [T, H, W, 3] → [T, 512] float32."""
    from PIL import Image as PILImage
    processor, model = _load_clip()
    pil = [PILImage.fromarray(f) for f in frames_uint8]
    inputs = processor(images=pil, return_tensors="jax", padding=True)
    feats = model.get_image_features(
        **{k: v for k, v in inputs.items() if k != "input_ids"}
    )
    return np.asarray(feats, dtype=np.float32)


def _embed_dinov2(frames_uint8: np.ndarray) -> np.ndarray:
    """DINOv2-lite pure-JAX. [T, H, W, 3] → [T, D]."""
    _ensure_dinov2_loaded()
    x = jnp.asarray(frames_uint8, dtype=jnp.float32) / 255.0
    x = jnp.transpose(x, (0, 3, 1, 2))
    T, C, H, W = x.shape
    kernels = _DINOV2_CACHE["dinovlite_kernels"]
    feats = []
    for s, k in enumerate(kernels):
        ph = _PATCH * (2 ** s)
        stride = max(ph // 2, 1)
        pad_h = (stride - H % stride) % stride
        pad_w = (stride - W % stride) % stride
        x_p = jnp.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
        Hn, Wn = x_p.shape[2], x_p.shape[3]
        if ph > Hn or ph > Wn:
            continue
        nh = max(Hn // stride - 1, 1)
        nw = max(Wn // stride - 1, 1)
        ys_idx = jnp.arange(nh) * stride
        xs_idx = jnp.arange(nw) * stride
        patches = jax.vmap(
            lambda yy: jax.vmap(
                lambda xx: jax.lax.dynamic_slice(x_p, (0, 0, yy, xx), (T, C, ph, ph))
            )(xs_idx)
        )(ys_idx)
        patches = jnp.transpose(patches, (2, 0, 1, 3, 4, 5))
        resp = jnp.einsum("tnmcij,kcij->tnmk", patches, k)
        feat = resp.mean(axis=(1, 2))
        feat = (feat - feat.mean(-1, keepdims=True)) / (
            feat.std(-1, keepdims=True) + 1e-6
        )
        feats.append(feat)
    if not feats:
        return np.zeros((T, _D_MODEL), dtype=np.float32)
    z = jnp.concatenate(feats, axis=-1)
    return np.asarray(z, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# PER-FM OE SCORE
# ═══════════════════════════════════════════════════════════════════════════

def _asal_oe(z: np.ndarray) -> float:
    """ASAL-style OE: −mean_i max_{j<i} cos(z_i, z_j). Higher = more novel trajectory."""
    if not np.all(np.isfinite(z)) or np.allclose(z, 0):
        return float("nan")
    zn = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    sim = zn @ zn.T
    n = sim.shape[0]
    mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    sim_masked = np.where(mask, sim, -np.inf)
    row_max = sim_masked.max(axis=-1)
    valid = row_max[row_max > -np.inf]
    if valid.size == 0:
        return float("nan")
    return float(-valid.mean())


def _per_candidate_scores(params: jax.Array, seed: int, substrate_name: str
                          ) -> tuple[float, float]:
    from evaluator import (_lenia_rollout, _render_lenia,
                           _flow_lenia_rollout, _render_flow_lenia)
    if substrate_name == "lenia":
        traj = _lenia_rollout(params, seed)
        picks = traj[jnp.asarray(_FRAME_PICKS)]
        rgb = _render_lenia(picks)
    else:
        traj = _flow_lenia_rollout(params, seed)
        picks = traj[jnp.asarray(_FRAME_PICKS)]
        rgb = _render_flow_lenia(picks)
    rgb_np = np.asarray(rgb, dtype=np.uint8)
    z_clip = _embed_clip(rgb_np)
    z_dino = _embed_dinov2(rgb_np)
    return _asal_oe(z_clip), _asal_oe(z_dino)


def _batch_scores(params_batch: jax.Array, seed: int, substrate_name: str
                  ) -> tuple[np.ndarray, np.ndarray]:
    pop = params_batch.shape[0]
    clip_raw = np.full(pop, np.nan, dtype=np.float32)
    dino_raw = np.full(pop, np.nan, dtype=np.float32)
    for i in range(pop):
        try:
            c, d = _per_candidate_scores(params_batch[i], seed, substrate_name)
            clip_raw[i] = c if math.isfinite(c) else np.nan
            dino_raw[i] = d if math.isfinite(d) else np.nan
        except Exception:
            pass
    return clip_raw, dino_raw


def _freeze_warmup_stats(clip_raw: np.ndarray, dino_raw: np.ndarray) -> None:
    """Freeze per-FM (mu, sd, offset) from the warmup population.

    After standardisation z = (raw − mu) / sd the values live on a scale
    comparable to a unit normal. We then shift by −min(z) + _GEOMEAN_FLOOR
    so the shifted variable is ≥ _GEOMEAN_FLOOR for every warmup candidate,
    which makes the powered geomean well-defined. The shift `offset` is
    frozen from warmup and reused — it is NOT recomputed per generation,
    so we preserve bit-identical reproducibility under fixed rng."""
    c_valid = clip_raw[np.isfinite(clip_raw)]
    d_valid = dino_raw[np.isfinite(dino_raw)]

    _NORM_CACHE["clip_mu"] = float(c_valid.mean()) if c_valid.size else 0.0
    _NORM_CACHE["clip_sd"] = float(c_valid.std()) if c_valid.size > 1 else 1.0
    _NORM_CACHE["dino_mu"] = float(d_valid.mean()) if d_valid.size else 0.0
    _NORM_CACHE["dino_sd"] = float(d_valid.std()) if d_valid.size > 1 else 1.0

    # Compute the implied z-min on warmup, then record `offset = −z_min + floor`
    # so `z_shifted = z + offset ≥ floor` for every warmup candidate.
    clip_sd = max(_NORM_CACHE["clip_sd"], 1e-6)
    dino_sd = max(_NORM_CACHE["dino_sd"], 1e-6)
    clip_z_warm = (c_valid - _NORM_CACHE["clip_mu"]) / clip_sd if c_valid.size else np.array([0.0])
    dino_z_warm = (d_valid - _NORM_CACHE["dino_mu"]) / dino_sd if d_valid.size else np.array([0.0])
    _NORM_CACHE["clip_offset"] = float(-clip_z_warm.min() + _GEOMEAN_FLOOR)
    _NORM_CACHE["dino_offset"] = float(-dino_z_warm.min() + _GEOMEAN_FLOOR)


def _shifted_z(raw: np.ndarray, mu: float, sd: float, offset: float) -> np.ndarray:
    """Standardise then shift. NaN → `_GEOMEAN_FLOOR` (minimum possible value),
    which under the powered geomean collapses the ensemble score to ≈0."""
    sd = max(float(sd), 1e-6)
    z = (raw - mu) / sd + offset
    z = np.where(np.isfinite(raw), z, _GEOMEAN_FLOOR).astype(np.float32)
    # Clip below at _GEOMEAN_FLOOR in case a later-gen candidate undershoots warmup.
    return np.maximum(z, _GEOMEAN_FLOOR)


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def search(
    substrate_name: Literal["lenia", "flow_lenia"],
    seed_pool_train: jax.Array,
    budget: dict,
    rng: jax.Array,
) -> dict:
    """Sep-CMA-ES on geomean(CLIP-z^α_t, DINOv2-z^(1−α_t)) with α_t ramped 0.5→1.0."""
    # BUG FIX: clear per-call norm cache so warmup stats are per-seed, not per-process.
    _NORM_CACHE.clear()

    wall_clock_s = float(budget.get("wall_clock_s", 1800.0))
    K = _DIM.get(substrate_name, 8)
    t0 = time.monotonic()
    deadline = t0 + wall_clock_s - _SAFETY_MARGIN_S

    dino_backend = _ensure_dinov2_loaded()

    run_id = os.environ.get("RE_RUN_ID") or os.environ.get("MODAL_TASK_ID") or "local"
    ckpt_path = Path(f"/tmp/ckpt/{run_id}_best.npy")

    strategy = Sep_CMA_ES(popsize=_POP_SIZE, num_dims=K, sigma_init=_SIGMA_INIT)
    es_state = strategy.initialize(rng)

    best_params = jnp.zeros(K, dtype=jnp.float32)
    best_score = float("-inf")

    best_proxy_per_gen: list[float] = []
    mean_proxy_per_gen: list[float] = []
    clip_raw_per_gen: list[float] = []
    dino_raw_per_gen: list[float] = []
    alpha_per_gen: list[float] = []
    archive: list[jax.Array] = []
    gen_times: list[float] = []

    scatter_clip: list[float] = []
    scatter_dino: list[float] = []

    n_train = int(seed_pool_train.shape[0])
    denom_alpha = max(_N_GENERATIONS - 1, 1)

    for gen in range(_N_GENERATIONS):
        if time.monotonic() >= deadline:
            break
        t_gen = time.monotonic()

        # α-anneal: 0.5 at gen=0, 1.0 at gen=N-1.
        alpha = _ALPHA_START + (_ALPHA_END - _ALPHA_START) * (gen / denom_alpha)
        alpha = float(min(max(alpha, 0.0), 1.0))

        rng, sub = jax.random.split(rng)
        params_batch, es_state = strategy.ask(sub, es_state)
        s = int(seed_pool_train[gen % n_train])

        clip_raw, dino_raw = _batch_scores(params_batch, s, substrate_name)

        for ci, di in zip(clip_raw, dino_raw):
            if math.isfinite(ci) and math.isfinite(di):
                scatter_clip.append(float(ci))
                scatter_dino.append(float(di))

        if gen == 0 and "clip_mu" not in _NORM_CACHE:
            _freeze_warmup_stats(clip_raw, dino_raw)

        clip_sh = _shifted_z(
            clip_raw,
            _NORM_CACHE["clip_mu"], _NORM_CACHE["clip_sd"], _NORM_CACHE["clip_offset"],
        )
        dino_sh = _shifted_z(
            dino_raw,
            _NORM_CACHE["dino_mu"], _NORM_CACHE["dino_sd"], _NORM_CACHE["dino_offset"],
        )
        # Geomean via power-product on the positive shifted variables.
        fitness = np.power(clip_sh, alpha) * np.power(dino_sh, 1.0 - alpha)
        fitness = fitness.astype(np.float32)

        # evosax minimises → pass negated fitness.
        es_state = strategy.tell(params_batch, -jnp.asarray(fitness), es_state)

        gen_best_idx = int(np.argmax(fitness))
        gen_best = float(fitness[gen_best_idx])
        if gen_best > best_score and math.isfinite(gen_best):
            best_score = gen_best
            best_params = params_batch[gen_best_idx]
            archive.append(best_params)
            try:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(ckpt_path), np.asarray(best_params, np.float32))
            except Exception:
                pass

        best_proxy_per_gen.append(best_score)
        mean_proxy_per_gen.append(float(np.nanmean(fitness)))
        clip_raw_per_gen.append(float(np.nanmean(clip_raw)) if np.any(np.isfinite(clip_raw)) else float("nan"))
        dino_raw_per_gen.append(float(np.nanmean(dino_raw)) if np.any(np.isfinite(dino_raw)) else float("nan"))
        alpha_per_gen.append(alpha)
        gen_times.append(round(time.monotonic() - t_gen, 2))

        if (gen + 1) % _CLEAR_CACHE_EVERY == 0:
            gc.collect()
            try:
                jax.clear_caches()
            except Exception:
                pass

    bp = np.asarray(best_params, dtype=np.float32)
    if not np.all(np.isfinite(bp)) or not math.isfinite(best_score):
        bp = np.zeros(K, dtype=np.float32)
        best_params = jnp.asarray(bp)

    try:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(ckpt_path), np.asarray(best_params, np.float32))
    except Exception:
        pass

    archive_arr = (
        jnp.stack(archive[-16:]) if archive
        else jnp.zeros((1, K), dtype=jnp.float32)
    )

    return {
        "best_params": best_params,
        "archive": archive_arr,
        "search_trace": {
            "algorithm": "sep_cma_es_ensemble_geomean_alpha_anneal_eval_v2",
            "n_generations": len(best_proxy_per_gen),
            "pop_size": _POP_SIZE,
            "alpha_start": _ALPHA_START,
            "alpha_end": _ALPHA_END,
            "alpha_schedule": "linear",
            "sigma_init": _SIGMA_INIT,
            "dinov2_backend": dino_backend,
            "geomean_floor": _GEOMEAN_FLOOR,
            "clip_norm_stats": {
                "mu": _NORM_CACHE.get("clip_mu"),
                "sd": _NORM_CACHE.get("clip_sd"),
                "offset": _NORM_CACHE.get("clip_offset"),
            },
            "dino_norm_stats": {
                "mu": _NORM_CACHE.get("dino_mu"),
                "sd": _NORM_CACHE.get("dino_sd"),
                "offset": _NORM_CACHE.get("dino_offset"),
            },
            "best_ensemble_score": best_score if math.isfinite(best_score) else None,
            "best_proxy_per_gen": best_proxy_per_gen,
            "mean_proxy_per_gen": mean_proxy_per_gen,
            "clip_raw_per_gen": clip_raw_per_gen,
            "dino_raw_per_gen": dino_raw_per_gen,
            "alpha_per_gen": alpha_per_gen,
            "gen_times": gen_times,
            "scatter_clip_sample": scatter_clip[::max(1, len(scatter_clip)//512)][:512],
            "scatter_dino_sample": scatter_dino[::max(1, len(scatter_dino)//512)][:512],
            "substrate": substrate_name,
            "wall_clock_s": round(time.monotonic() - t0, 1),
        },
    }
