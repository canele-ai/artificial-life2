"""solution.py — Flow-Lenia × Sep-CMA-ES × DINOv2-lite OE proxy (orbit 16).

Hypothesis — stack top two axes
-------------------------------
Orbit 15 (`15-flow-lenia-retry`) is the current leader at +0.339: Flow-Lenia
substrate + Sep-CMA-ES + CLIP-OE inner loop.  Orbit 07 (`07-real-cmaes-dinov2`)
was +0.319: plain Lenia + Sep-CMA-ES + DINOv2-lite (QR-orthogonal multi-scale
patch proxy) inner loop.  The two strongest effects — (1) mass-conserving
substrate and (2) patch-structural FM proxy — have never been combined.
This orbit does exactly that: Flow-Lenia rollout, DINOv2-lite embedding,
ASAL open-endedness score, Sep-CMA-ES pop=16 σ=0.1 ≤60 gens.

Provenance
----------
- Substrate + degeneracy penalty: orbit 15 (`orbits/15-flow-lenia-retry`).
- Flow-Lenia rollout + render_flow_lenia_rgb: inlined from orbit 07 so we
  do not depend on evaluator-private symbols that may shift across eval
  revisions.
- DINOv2-lite kernel bank (QR-orthogonal, native-scale per level): orbit 07.
- Sep-CMA-ES single-pass schedule (pop=16, σ_init=0.1, ≤60 gens,
  `_CLEAR_CACHE_EVERY=10`): orbit 07 (canary-validated).

Fail-loud contract (eval-v2, Guard 11)
--------------------------------------
`from evosax import Sep_CMA_ES` at module top level, no `try/except`.  If the
search container is mis-provisioned we crash honestly and the evaluator
records `status=search_crash`.

Memory discipline
-----------------
- Module-level `_FM_CACHE`: DINOv2-lite kernels built once per process.
- `gc.collect() + jax.clear_caches()` every 10 generations.
- Per-candidate scores converted to NumPy scalars immediately.
- Self-limits to `wall_clock_s - 15 s` so SIGKILL does not strand best-so-far.
- Ckpt best-so-far to `/tmp/ckpt/orbit16_best.npy` on every improve.

Interface (problem_spec.md)
---------------------------
search(substrate_name, seed_pool_train, budget, rng) → {
    "best_params":  jax.Array [8] float32,
    "best_rollout": jax.Array [T, H, W, 2] float32,
    "archive":      jax.Array [<=16, 8] float32,
    "search_trace": dict (scalars only).
}
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

# Fail-loud on evosax (eval-v2 Guard 11).
from evosax import Sep_CMA_ES  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

_DIM = {"lenia": 8, "flow_lenia": 8}
_POP_SIZE = 16
_SIGMA_INIT = 0.1
_N_GENERATIONS = 60
_CLEAR_CACHE_EVERY = 10
_SAFETY_MARGIN_S = 15.0

_FRAME_PICKS = (0, 63, 127, 255)  # ASAL-style 4 frames → 6 lower-triangle pairs
_GRID = 128

# DINOv2-lite (orbit 07 corrected) hyperparameters.
_BASE_PATCH = 16
_N_KERNELS = 32
_N_SCALES = 3
_D_MODEL = _N_KERNELS * _N_SCALES  # == 96
_KERNEL_SEED = 0xD1_50_FE

# Mass-variance degeneracy penalty (orbit 15).
_DEGEN_VAR_EPS = 1e-6
_DEGEN_PENALTY = 0.25

# Module-level FM cache — populated once per process.
_FM_CACHE: dict = {}

# Algorithm provenance tag — critical for audit / viz.
_ALGO_TAG = "sep_cma_es_flow_lenia_dinov2_lite_eval_v2"


# ══════════════════════════════════════════════════════════════════════════
# FLOW-LENIA ROLLOUT (inlined, identical dynamics to orbit 07/15 substrate).
# Mass-conserving via semi-Lagrangian transport + Σ-renormalisation.
# ══════════════════════════════════════════════════════════════════════════

def _flow_lenia_rollout(params: jax.Array, seed: int) -> jax.Array:
    """Returns [T=256, H, W, 2] trajectory: channel 0 = mass, channel 1 = flow-speed."""
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
        An = An * (A.sum() + 1e-9) / (An.sum() + 1e-9)  # Σ-renormalisation
        return An, jnp.stack([An, jnp.sqrt(vx ** 2 + vy ** 2)], axis=-1)

    _, traj = jax.lax.scan(step, A0, None, length=256)
    return traj.astype(jnp.float32)


def _render_flow_lenia_rgb(picks: jax.Array) -> np.ndarray:
    """[F, H, W, 2] → [F, H, W, 3] uint8 (mass→R, 0.5*flow→G, 0.2*mass→B)."""
    x = np.asarray(picks)
    m = x[..., 0]
    f = x[..., 1]
    rgb = np.clip(np.stack([m, 0.5 * f, 0.2 * m], -1), 0, 1)
    return (rgb * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════
# DINOv2-LITE: QR-orthogonal multi-scale patch encoder (orbit 07 corrected).
# ══════════════════════════════════════════════════════════════════════════

def _build_orthogonal_kernels() -> list[jax.Array]:
    """N_SCALES kernel banks of orthonormal random kernels (rows orthonormal).

    Scale s kernel has native patch size ph = _BASE_PATCH * 2^s; we QR at
    that native size (no bilinear upsampling — the orbit-05 bug).
    """
    rng = jax.random.PRNGKey(_KERNEL_SEED)
    kernels: list[jax.Array] = []
    C = 3
    for s in range(_N_SCALES):
        rng, sub = jax.random.split(rng)
        ph = _BASE_PATCH * (2 ** s)
        dim = C * ph * ph
        G = jax.random.normal(sub, (dim, _N_KERNELS), dtype=jnp.float32)
        Q, _ = jnp.linalg.qr(G, mode="reduced")  # Q: (dim, N_KERNELS)
        Q = Q.T                                   # rows orthonormal
        kernels.append(Q.reshape(_N_KERNELS, C, ph, ph))
    return kernels


def _ensure_fm_loaded() -> str:
    if "backend" in _FM_CACHE:
        return _FM_CACHE["backend"]
    _FM_CACHE["dinovlite_kernels"] = _build_orthogonal_kernels()
    _FM_CACHE["backend"] = "dinov2_lite_ortho"
    return _FM_CACHE["backend"]


def _embed_dinov2_lite(frames_uint8: np.ndarray) -> np.ndarray:
    """[T, H, W, 3] uint8 → [T, 96] float32 (orthogonal multi-scale patch feat)."""
    x = jnp.asarray(frames_uint8, dtype=jnp.float32) / 255.0
    x = jnp.transpose(x, (0, 3, 1, 2))  # [T, 3, H, W]
    T, C, H, W = x.shape

    kernels = _FM_CACHE["dinovlite_kernels"]
    feats = []
    for s, k in enumerate(kernels):
        ph = _BASE_PATCH * (2 ** s)
        if ph > H or ph > W:
            feats.append(jnp.zeros((T, _N_KERNELS), dtype=jnp.float32))
            continue
        nh, nw = H // ph, W // ph
        patches = x.reshape(T, C, nh, ph, nw, ph)
        patches = jnp.transpose(patches, (0, 2, 4, 1, 3, 5))  # [T,nh,nw,C,ph,ph]
        resp = jnp.einsum("tnmcij,kcij->tnmk", patches, k)
        feat = resp.mean(axis=(1, 2))                         # [T, N_KERNELS]
        feat = (feat - feat.mean(-1, keepdims=True)) / (
            feat.std(-1, keepdims=True) + 1e-6
        )
        feats.append(feat)
    z = jnp.concatenate(feats, axis=-1)                       # [T, 96]
    return np.asarray(z, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════
# OPEN-ENDEDNESS SCORE + degeneracy penalty.
# ══════════════════════════════════════════════════════════════════════════

def _oe_from_embeddings(z: np.ndarray) -> float:
    """ASAL OE: -mean_i max_{j<i} cos(z_i, z_j). Higher = more diverse trajectory."""
    if not np.all(np.isfinite(z)) or np.allclose(z, 0):
        return -1.0
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    sim = z @ z.T
    n = sim.shape[0]
    mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    sim_masked = np.where(mask, sim, -np.inf)
    row_max = sim_masked.max(axis=-1)
    valid = row_max[row_max > -np.inf]
    if valid.size == 0:
        return -1.0
    return float(-valid.mean())


def _score_candidate(params: jax.Array, seed: int) -> tuple[float, jax.Array]:
    """Rollout → DINOv2-lite embed → OE score − degeneracy penalty."""
    traj = _flow_lenia_rollout(params, seed)                 # [256, H, W, 2]
    pick_idx = jnp.asarray(_FRAME_PICKS)
    picks = traj[pick_idx]                                   # [4, H, W, 2]
    rgb = _render_flow_lenia_rgb(picks)                      # [4, H, W, 3] u8
    z = _embed_dinov2_lite(rgb)
    score = _oe_from_embeddings(z)

    # Mass-variance degeneracy penalty (orbit 15): Flow-Lenia's failure mode
    # is uniform/dead final frames that paradoxically inflate OE.
    final_mass = np.asarray(picks[-1, ..., 0])
    if final_mass.var() < _DEGEN_VAR_EPS or final_mass.max() < 1e-3:
        score -= _DEGEN_PENALTY
    return score, traj


def _batch_fitness(params_batch: jax.Array, seed: int) -> np.ndarray:
    """Evaluate a population. Per-candidate failures → score=-1.0."""
    pop = params_batch.shape[0]
    scores = np.zeros(pop, dtype=np.float32)
    for i in range(pop):
        try:
            s, _ = _score_candidate(params_batch[i], seed)
            scores[i] = s
        except Exception:
            scores[i] = -1.0
    return scores


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def search(
    substrate_name: Literal["lenia", "flow_lenia"],
    seed_pool_train: jax.Array,
    budget: dict,
    rng: jax.Array,
) -> dict:
    """Flow-Lenia + Sep-CMA-ES + DINOv2-lite OE search (orbit 16).

    Stacks the two strongest eval-v2 effects: mass-conserving substrate
    (orbit 15, +0.339) × QR-orthogonal patch-structural FM proxy
    (orbit 07, +0.319).
    """
    wall_clock_s = float(budget.get("wall_clock_s", 1800.0))
    K = _DIM.get(substrate_name, 8)
    t0 = time.monotonic()
    deadline = t0 + wall_clock_s - _SAFETY_MARGIN_S

    # Eager FM load so `fm_backend` is in the trace even on early timeout.
    try:
        backend_used = _ensure_fm_loaded()
    except Exception:
        backend_used = "unknown"

    run_id = os.environ.get("RE_RUN_ID") or os.environ.get("MODAL_TASK_ID") or "local"
    ckpt_path = f"/tmp/ckpt/orbit16_{run_id}_best.npy"
    try:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    except Exception:
        ckpt_path = None

    n_train = int(seed_pool_train.shape[0])
    best_proxy_per_gen: list[float] = []
    mean_proxy_per_gen: list[float] = []
    gen_times: list[float] = []
    archive: list[jax.Array] = []

    best_params = jnp.zeros(K, dtype=jnp.float32)
    best_rollout = None
    best_score = float("-inf")

    strategy = Sep_CMA_ES(popsize=_POP_SIZE, num_dims=K, sigma_init=_SIGMA_INIT)
    es_state = strategy.initialize(rng)
    algo = f"{_ALGO_TAG}[{backend_used}]"

    try:
        for gen in range(_N_GENERATIONS):
            if time.monotonic() >= deadline:
                break
            t_gen = time.monotonic()

            rng, sub = jax.random.split(rng)
            params_batch, es_state = strategy.ask(sub, es_state)

            s = int(seed_pool_train[gen % n_train])
            fitnesses = _batch_fitness(params_batch, s)

            # evosax minimises; negate for maximisation.
            es_state = strategy.tell(
                params_batch, -jnp.asarray(fitnesses), es_state
            )

            gbest_idx = int(np.argmax(fitnesses))
            gbest = float(fitnesses[gbest_idx])
            if gbest > best_score and np.isfinite(gbest):
                best_score = gbest
                best_params = params_batch[gbest_idx]
                archive.append(best_params)
                # Materialise best rollout for evaluator (overwritten on improve).
                try:
                    best_rollout = _flow_lenia_rollout(best_params, s)
                except Exception:
                    best_rollout = None
                if ckpt_path is not None:
                    try:
                        np.save(ckpt_path, np.asarray(best_params, np.float32))
                    except Exception:
                        pass

            best_proxy_per_gen.append(best_score)
            mean_proxy_per_gen.append(float(np.mean(fitnesses)))
            gen_times.append(round(time.monotonic() - t_gen, 2))

            if (gen + 1) % _CLEAR_CACHE_EVERY == 0:
                gc.collect()
                try:
                    jax.clear_caches()
                except Exception:
                    pass
    except Exception as exc:
        algo = (
            f"{_ALGO_TAG}[{backend_used}] "
            f"crashed@gen={len(best_proxy_per_gen)}: {type(exc).__name__}"
        )

    # Finite sanity check.
    bp = np.asarray(best_params, dtype=np.float32)
    if not np.all(np.isfinite(bp)) or not math.isfinite(best_score):
        bp = np.zeros(K, dtype=np.float32)
        best_params = jnp.asarray(bp)

    if best_rollout is None:
        try:
            best_rollout = _flow_lenia_rollout(
                best_params, int(seed_pool_train[0])
            )
        except Exception:
            best_rollout = jnp.zeros((256, _GRID, _GRID, 2), dtype=jnp.float32)

    archive_arr = (
        jnp.stack(archive[-16:]) if archive
        else jnp.zeros((1, K), dtype=jnp.float32)
    )

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
            "kernel_fix": "qr_orthogonal_native_scale",
            "degeneracy_penalty": _DEGEN_PENALTY,
            "pop_size": _POP_SIZE,
            "sigma_init": _SIGMA_INIT,
            "max_gens": _N_GENERATIONS,
            "substrate": substrate_name,
            "parent_orbits": ["15-flow-lenia-retry", "07-real-cmaes-dinov2"],
            "eval_version": "eval-v2",
        },
    }
