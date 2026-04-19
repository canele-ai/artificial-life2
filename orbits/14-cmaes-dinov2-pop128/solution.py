"""solution.py — Sep-CMA-ES (pop=128, 4-restart σ-schedule) + DINOv2-lite (orbit 14).

EXTEND of orbit 07 (`07-real-cmaes-dinov2`, leader at METRIC=+0.319) via
orbit 11's pop=64 + 3-restart scaffold. Largest CMA-ES budget in the
campaign: stretch-compute test of the scaling ceiling.

Hypothesis
----------
Orbit 07 (pop=16) → +0.319. Orbit 11 (pop=64, 3 restarts) tested 4× lift.
Orbit 14 doubles pop again (128) and adds a 4th restart with a σ schedule
covering wide (0.15) → local (0.08) → explosive (0.25) → medium (0.12)
to probe whether the proxy's signal ceiling has been reached or whether
larger effective batches still yield meaningful lift on LCF.

Differences vs orbit 11
-----------------------

* **pop=128** (2× orbit 11, 8× orbit 07). Sep-CMA-ES per-gen cost is
  O(K · pop); at K=8 with a vectorised DINOv2-lite batch forward, the
  marginal cost per extra candidate is ~0.02s on A100.
* **4 restarts** on σ_init schedule `[0.15, 0.08, 0.25, 0.12]`:
    - R0 σ=0.15: wide exploration, ~30% wall clock.
    - R1 σ=0.08: local refinement warm-started from R0 mean, ~25%.
    - R2 σ=0.25: diversifying jump, cold re-init, ~25%.
    - R3 σ=0.12: moderate-σ refinement, cold re-init (or warm from R2
      if R2 improved global best), ~20%.
* **Vectorised DINOv2-lite batch forward** — all `pop` candidates'
  frames are embedded in one `jax.vmap`-able batched op rather than a
  Python loop (orbit 11's `_dinov2_oe_score` iterated candidates). At
  pop=128 this is the critical speed-up: ~128x fewer Python round-trips.
* **More aggressive GC** — `gc.collect() + jax.clear_caches()` every 5
  gens (orbit 11 did every 10) to manage high-pop memory churn.
* **Larger safety margin** — 20s (orbit 11: 15s) to protect against
  JIT-compile tail latency at high pop.
* **Total candidate budget**: pop=128 × ~8-10 gens × 4 restarts =
  4096-5120 candidates; at ~0.3s/CLIP-forward ≈ 1500s — tight against
  1800s budget, so wall-clock gating (not gen count) is binding.

Constraints respected
---------------------

* Substrate `lenia` primary, 8-dim params, deterministic under rng.
* ≤200 MB artifacts; FM cache module-level; gc+clear_caches every 5 gens.
* Fail-loud on evosax (`from evosax import Sep_CMA_ES` at top level).
* `/tmp/ckpt/{run_id}_best.npy` written on every improvement.
* Interface `search(substrate_name, seed_pool_train, budget, rng)`.
* `search_trace["algorithm"] = "sep_cma_es_dinov2_pop128_4restart_eval_v2[..]"`.
"""

from __future__ import annotations

import gc
import math
import os
import time
import uuid
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

# Fail-loud evosax import.
from evosax import Sep_CMA_ES  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

_DIM = {"lenia": 8, "flow_lenia": 8}

# Scaled profile — pop 8× orbit 07 / 2× orbit 11, four restarts.
_POP_SIZE = 128
_SIGMA_SCHEDULE = (0.15, 0.08, 0.25, 0.12)   # restart 0, 1, 2, 3
_BUDGET_FRACTIONS = (0.30, 0.25, 0.25, 0.20) # sum = 1.0
_MAX_GEN_PER_RESTART = 120
_CLEAR_CACHE_EVERY = 5

# ASAL "keep it moving" frame picks — early / mid-early / mid-late / late.
_FRAME_PICKS = (0, 63, 127, 255)
_GRID = 128

# DINOv2-lite (orbit 07 corrected) hyperparameters.
_BASE_PATCH = 16
_N_KERNELS = 32
_N_SCALES = 3
_D_MODEL = _N_KERNELS * _N_SCALES      # == 96

# Same deterministic kernel seed as orbit 07.
_KERNEL_SEED = 0xD1_50_FE

_FM_CACHE: dict = {}

_ALGO_TAG_BASE = "sep_cma_es_dinov2_pop128_4restart_eval_v2"


# ══════════════════════════════════════════════════════════════════════════
# SUBSTRATE ROLLOUTS — verbatim orbit 07/11.
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
# DINOv2-lite (QR-orthogonal) — orbit 07 kernel build.
# ══════════════════════════════════════════════════════════════════════════

def _build_orthogonal_kernels() -> list[jax.Array]:
    rng = jax.random.PRNGKey(_KERNEL_SEED)
    kernels: list[jax.Array] = []
    C = 3
    for s in range(_N_SCALES):
        rng, sub = jax.random.split(rng)
        ph = _BASE_PATCH * (2 ** s)
        pw = ph
        dim = C * ph * pw
        G = jax.random.normal(sub, (dim, _N_KERNELS), dtype=jnp.float32)
        Q, _R = jnp.linalg.qr(G, mode="reduced")
        Q = Q.T
        k = Q.reshape(_N_KERNELS, C, ph, pw)
        kernels.append(k)
    return kernels


def _ensure_fm_loaded() -> str:
    """Orbit 14: skip real-DINOv2 entirely and use QR-orthogonal DINOv2-lite.

    Matches orbit 07's kernel build exactly so results are compute-vs-compute
    comparable.
    """
    if "backend" in _FM_CACHE:
        return _FM_CACHE["backend"]
    _FM_CACHE["dinovlite_kernels"] = _build_orthogonal_kernels()
    _FM_CACHE["backend"] = "dinov2_lite_ortho"
    return "dinov2_lite_ortho"


# ══════════════════════════════════════════════════════════════════════════
# VECTORISED DINOv2-lite BATCH EMBEDDER
# ══════════════════════════════════════════════════════════════════════════

def _embed_dinov2_lite_batched_jax(frames_uint8_batch: jax.Array) -> jax.Array:
    """Batched DINOv2-lite forward.

    Input:  [B, T, H, W, 3] uint8  (B = pop * n_frames effectively)
    Output: [B, T, D_MODEL]        float32 patch features per frame
    """
    # Normalise + NCHW.
    x = jnp.asarray(frames_uint8_batch, dtype=jnp.float32) / 255.0
    x = jnp.transpose(x, (0, 1, 4, 2, 3))  # [B,T,C,H,W]
    B, T, C, H, W = x.shape

    kernels = _FM_CACHE["dinovlite_kernels"]
    feat_parts = []
    for s, k in enumerate(kernels):
        ph = _BASE_PATCH * (2 ** s)
        if ph > H or ph > W:
            feat_parts.append(jnp.zeros((B, T, _N_KERNELS), dtype=jnp.float32))
            continue
        nh = H // ph
        nw = W // ph
        patches = x.reshape(B, T, C, nh, ph, nw, ph)
        # [B,T,nh,nw,C,ph,pw]
        patches = jnp.transpose(patches, (0, 1, 3, 5, 2, 4, 6))
        # Einsum patch bank:  -> [B,T,nh,nw,Kernels]
        resp = jnp.einsum("btnmcij,kcij->btnmk", patches, k)
        feat = resp.mean(axis=(2, 3))  # [B,T,Kernels]
        feat = (feat - feat.mean(-1, keepdims=True)) / (
            feat.std(-1, keepdims=True) + 1e-6
        )
        feat_parts.append(feat)

    z = jnp.concatenate(feat_parts, axis=-1)  # [B,T,D_MODEL]
    return z


# ══════════════════════════════════════════════════════════════════════════
# ASAL OE SCORE — VECTORISED ACROSS POPULATION
# ══════════════════════════════════════════════════════════════════════════

def _oe_score_batch(
    params_batch: jax.Array, seed: int, substrate_name: str
) -> np.ndarray:
    """Compute ASAL OE ("keep it moving") score for every candidate.

    Rolls each candidate one-at-a-time (rollouts are the dominant cost
    and don't fit in a single VMAP due to FFT scratch), but batches the
    DINOv2-lite forward across all pop × n_frames frames in one JAX
    einsum — the population-size bottleneck.
    """
    pop = int(params_batch.shape[0])
    roll = _lenia_rollout if substrate_name == "lenia" else _flow_lenia_rollout
    pick_idx = jnp.asarray(_FRAME_PICKS)
    n_frames = pick_idx.shape[0]

    # Step 1 — rollouts + render → uint8 tensor [pop, n_frames, H, W, 3].
    rgb_stack = np.zeros((pop, n_frames, _GRID, _GRID, 3), dtype=np.uint8)
    valid = np.ones(pop, dtype=bool)
    for i in range(pop):
        try:
            traj = roll(params_batch[i], seed)
            picks = traj[pick_idx]
            rgb = _render_rgb(picks, substrate_name)
            if rgb.shape != (n_frames, _GRID, _GRID, 3):
                valid[i] = False
                continue
            rgb_stack[i] = rgb
        except Exception:
            valid[i] = False

    # Step 2 — one batched DINOv2-lite forward over [pop, n_frames, …].
    try:
        z = np.asarray(
            _embed_dinov2_lite_batched_jax(jnp.asarray(rgb_stack)),
            dtype=np.float32,
        )  # [pop, n_frames, D]
    except Exception:
        return np.full(pop, -1.0, dtype=np.float32)

    # Step 3 — per-candidate OE score (tril sim-max of frame embeddings).
    scores = np.zeros(pop, dtype=np.float32)
    for i in range(pop):
        if not valid[i]:
            scores[i] = -1.0
            continue
        zi = z[i]
        if not np.all(np.isfinite(zi)) or np.allclose(zi, 0):
            scores[i] = -1.0
            continue
        zi = zi / (np.linalg.norm(zi, axis=-1, keepdims=True) + 1e-8)
        sim = zi @ zi.T
        n = sim.shape[0]
        mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
        sim_masked = np.where(mask, sim, -np.inf)
        row_max = sim_masked.max(axis=-1)
        valid_rows = row_max[row_max > -np.inf]
        if valid_rows.size == 0:
            scores[i] = -1.0
            continue
        scores[i] = float(-valid_rows.mean())
    return scores


# ══════════════════════════════════════════════════════════════════════════
# SEARCH — Sep-CMA-ES pop=128, 4-restart σ-schedule, global-best tracked.
# ══════════════════════════════════════════════════════════════════════════

def _run_one_restart(
    substrate_name: str,
    seed_pool_train: jax.Array,
    rng: jax.Array,
    sigma_init: float,
    K: int,
    t_start: float,
    wall_budget: float,
    safety_margin: float,
    gen_offset: int,
    best_params_in: jax.Array,
    best_score_in: float,
    best_proxy_series: list,
    mean_proxy_series: list,
    gen_times: list,
    archive: list,
    run_id: str,
    ckpt_dir: str | None,
    mean_init: jax.Array | None,
) -> tuple[jax.Array, float, int, jax.Array]:
    """Run one Sep-CMA-ES restart; return (best_params, best_score, gens, terminal_mean)."""
    n_train = int(seed_pool_train.shape[0])
    strategy = Sep_CMA_ES(popsize=_POP_SIZE, num_dims=K, sigma_init=sigma_init)
    es_state = strategy.initialize(rng)
    if mean_init is not None:
        try:
            es_state = es_state.replace(mean=jnp.asarray(mean_init, jnp.float32))
        except Exception:
            pass

    best_params = best_params_in
    best_score = best_score_in
    gens_run = 0

    for gen in range(_MAX_GEN_PER_RESTART):
        if time.monotonic() - t_start >= wall_budget - safety_margin:
            break
        t_gen = time.monotonic()
        rng, sub = jax.random.split(rng)
        params_batch, es_state = strategy.ask(sub, es_state)

        s = int(seed_pool_train[(gen_offset + gen) % n_train])
        fitnesses = _oe_score_batch(params_batch, s, substrate_name)

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
                        os.path.join(ckpt_dir, f"{run_id}_best.npy"),
                        np.asarray(best_params, dtype=np.float32),
                    )
                except Exception:
                    pass

        best_proxy_series.append(best_score)
        mean_proxy_series.append(float(np.mean(fitnesses)))
        gen_times.append(round(time.monotonic() - t_gen, 2))
        gens_run += 1

        if (gen + 1) % _CLEAR_CACHE_EVERY == 0:
            gc.collect()
            jax.clear_caches()

    terminal_mean = jnp.asarray(es_state.mean, jnp.float32)
    return best_params, best_score, gens_run, terminal_mean


def search(
    substrate_name: Literal["lenia", "flow_lenia"],
    seed_pool_train: jax.Array,
    budget: dict,
    rng: jax.Array,
) -> dict:
    """Sep-CMA-ES pop=128 with 4 restarts on σ-schedule.

    Restart plan:
      R0: σ=0.15, ~30% budget, cold init (x=0).             — wide search
      R1: σ=0.08, ~25% budget, warm-started from R0 mean.   — local refine
      R2: σ=0.25, ~25% budget, cold re-init.                — diversify jump
      R3: σ=0.12, ~20% budget, cold re-init (fresh).        — mid-σ polish
    Global best tracked across all restarts.
    """
    wall_clock_s = float(budget.get("wall_clock_s", 1800))
    K = _DIM.get(substrate_name, 8)
    t0 = time.monotonic()
    safety_margin = 20.0

    best_proxy_series: list[float] = []
    mean_proxy_series: list[float] = []
    gen_times: list[float] = []
    archive: list[jax.Array] = []

    try:
        backend_used = _ensure_fm_loaded()
    except Exception:
        backend_used = "unknown"

    ckpt_dir = "/tmp/ckpt"
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
    except Exception:
        ckpt_dir = None

    run_id = f"orbit14_{uuid.uuid4().hex[:8]}"

    best_params = jnp.zeros(K, dtype=jnp.float32)
    best_score = float("-inf")
    restart_metadata: list[dict] = []
    algo = f"{_ALGO_TAG_BASE}[{backend_used}]"

    prev_mean: jax.Array | None = None

    try:
        for r_idx, (frac, sigma) in enumerate(
            zip(_BUDGET_FRACTIONS, _SIGMA_SCHEDULE)
        ):
            cum_frac = sum(_BUDGET_FRACTIONS[: r_idx + 1])
            sub_deadline = cum_frac * wall_clock_s
            if time.monotonic() - t0 >= sub_deadline - safety_margin:
                restart_metadata.append(
                    {"restart": r_idx, "sigma_init": sigma, "gens": 0,
                     "skipped": True, "best_at_end": best_score}
                )
                continue
            rng, sub_rng = jax.random.split(rng)
            # Warm-start only R1 from R0's terminal mean.
            mean_init = prev_mean if r_idx == 1 else None
            gen_offset = sum(m.get("gens", 0) for m in restart_metadata)
            bp, bs, gens_run, terminal_mean = _run_one_restart(
                substrate_name=substrate_name,
                seed_pool_train=seed_pool_train,
                rng=sub_rng,
                sigma_init=sigma,
                K=K,
                t_start=t0,
                wall_budget=sub_deadline,
                safety_margin=safety_margin,
                gen_offset=gen_offset,
                best_params_in=best_params,
                best_score_in=best_score,
                best_proxy_series=best_proxy_series,
                mean_proxy_series=mean_proxy_series,
                gen_times=gen_times,
                archive=archive,
                run_id=run_id,
                ckpt_dir=ckpt_dir,
                mean_init=mean_init,
            )
            best_params, best_score = bp, bs
            prev_mean = terminal_mean
            restart_metadata.append(
                {"restart": r_idx, "sigma_init": sigma, "gens": gens_run,
                 "skipped": False, "best_at_end": best_score}
            )
    except Exception as exc:
        algo = (
            f"{_ALGO_TAG_BASE}[{backend_used}] crashed@gen="
            f"{len(best_proxy_series)}: {type(exc).__name__}"
        )

    bp = np.asarray(best_params, dtype=np.float32)
    if not np.all(np.isfinite(bp)) or not math.isfinite(best_score):
        bp = np.zeros(K, dtype=np.float32)
        best_params = jnp.asarray(bp)

    archive_arr = (
        jnp.stack(archive[-32:])
        if archive
        else jnp.zeros((1, K), dtype=jnp.float32)
    )

    # best_rollout: rollout of best_params on seed_pool_train[0].
    try:
        roll = _lenia_rollout if substrate_name == "lenia" else _flow_lenia_rollout
        best_rollout = roll(best_params, int(seed_pool_train[0]))
    except Exception:
        best_rollout = jnp.zeros((1, _GRID, _GRID, 1), dtype=jnp.float32)

    total_candidates = sum(m.get("gens", 0) for m in restart_metadata) * _POP_SIZE

    return {
        "best_params": best_params,
        "best_rollout": best_rollout,
        "archive": archive_arr,
        "search_trace": {
            "best_proxy_per_gen": best_proxy_series,
            "mean_proxy_per_gen": mean_proxy_series,
            "gen_times": gen_times,
            "n_generations": len(best_proxy_series),
            "algorithm": algo,
            "fm_backend": backend_used,
            "best_score": best_score if math.isfinite(best_score) else None,
            "kernel_fix": "qr_orthogonal_native_scale",
            "pop_size": _POP_SIZE,
            "sigma_schedule": list(_SIGMA_SCHEDULE),
            "budget_fractions": list(_BUDGET_FRACTIONS),
            "restart_metadata": restart_metadata,
            "total_candidates": total_candidates,
            "run_id": run_id,
        },
    }
