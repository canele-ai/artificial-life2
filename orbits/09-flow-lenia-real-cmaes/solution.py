"""solution.py — Flow-Lenia × Sep-CMA-ES × CLIP open-endedness (orbit 09).

eval-v2 RE-RUN of orbit 03's hypothesis.

Provenance story
----------------
Orbit 03 (`03-flow-lenia-clip-oe`, issue #4) was the ONLY eval-v1 orbit that
crashed loudly with `ModuleNotFoundError: evosax` instead of silently falling
back to random search.  Every "Sep-CMA-ES" orbit in batch-1 was actually
random search — `evosax` was never pip-installed into the Modal search
container, and the common `good.py` template caught the ImportError and
substituted a uniform draw.  Orbit 03's solution did NOT have that try/except,
so it failed instead of lying (and returned METRIC=0 via the evaluator's
search_crash path).

eval-v2 (commit 7fbd935) fixed the pipeline:
  - `modal_app.py` now pip-installs `evosax==0.1.6` into search_image.
  - `evaluator._env_audit()` records `evosax_version` (Guard 11 signal).
  - `examples/good.py` is fail-loud on missing evosax.
  - `BASELINE_SHA256` bumped; baseline relabelled HONEST_ANCHOR.

This orbit re-submits orbit 03's hypothesis *without modification* on
eval-v2.  Any delta from 03's METRIC=0 is attributable to Sep-CMA-ES now
actually running (rather than to a different algorithm or inner loop).

Hypothesis (campaign axis 1 — substrate gap)
--------------------------------------------
Flow-Lenia (Plantec et al. 2023, arXiv:2212.07906) enforces mass conservation
via semi-Lagrangian transport + Σ-renormalisation.  This uniquely enables
multi-species coexistence, soft self-boundaries, and persistent identity —
precisely the properties the 5-tier judge rewards for *existence*, *robustness*,
and *reproduction*.  ASAL did not cover Flow-Lenia.  Research question: does
the substrate's mass-conservation invariant translate into higher judge scores
now that the CMA-ES machinery actually works?

Search procedure
----------------
Sep-CMA-ES (evosax) over an 8-dim Flow-Lenia parameter vector, optimising
ASAL open-endedness (mean of max off-diagonal CLIP cosine similarity, sign
flipped so diverse=high) on K=4 frames at t ∈ {0, 63, 127, 255}.

Because Flow-Lenia is less well-characterised than Lenia under CMA-ES, we
issue up-to-4 restarts with σ_init ∈ [0.15, 0.08, 0.25, 0.12], capped at
≤50 generations per restart, and retain the best-ever candidate.

Memory discipline (canary_results_full.md documents a 30-min OOM on long
Sep-CMA-ES loops due to XLA retracing + CLIP leak):
  - Module-level `_CLIP_CACHE` — FM weights loaded exactly once.
  - `jax.clear_caches()` + `gc.collect()` every 10 generations.
  - pop_size = 16 (ASAL default), ≤ 50 gens per restart.
  - Per-candidate features converted to NumPy scalars immediately.
  - Checkpoint best-so-far to `/tmp/ckpt/{run_id}_best.npy` every improve.
  - Self-limit wall-clock budget minus 12 s margin for SIGKILL recovery.

Interface (problem_spec.md)
---------------------------
search(substrate_name, seed_pool_train, budget, rng) → {
    "best_params":  jax.Array [8] float32,
    "archive":      jax.Array [<=16, 8] float32,
    "search_trace": dict (scalars only).
}

Fail-loud contract (eval-v2, Guard 11)
--------------------------------------
This module does NOT catch ImportError from evosax or transformers at
`search()` entry.  If the Modal search container is mis-provisioned, the
subprocess crashes with a stderr-captured ModuleNotFoundError and the
evaluator returns status=search_crash.  This is the correct behaviour — we
want algorithm provenance to be legible, not papered over.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
_K = 8                              # Flow-Lenia param dim (padded; 3 active)
_POP_SIZE = 16                      # ASAL default
_MAX_GENS_PER_RESTART = 50
_CLEAR_CACHE_EVERY = 10             # per OOM-mitigation notes
_SAFETY_MARGIN_S = 12.0             # leave headroom before evaluator SIGKILL
_CLIP_FRAME_IDX = (0, 63, 127, 255)  # 4 frames → 6 pairs for OE score
_RESTART_SIGMAS = (0.15, 0.08, 0.25, 0.12)  # diverse σ_init across restarts

# Module-level cache — prevents repeated CLIP loads that leak FlaxCLIPModel
# params across generations (documented OOM trigger).
_CLIP_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# CLIP open-endedness inner-loop proxy
# ─────────────────────────────────────────────────────────────────────────────
def _load_clip():
    """Load CLIP ViT-B/32 exactly once per process. Fail-loud on missing deps."""
    if "model" in _CLIP_CACHE:
        return _CLIP_CACHE["processor"], _CLIP_CACHE["model"]
    # NO try/except here — we want algorithm provenance to fail legibly.
    from transformers import CLIPProcessor, FlaxCLIPModel  # deferred

    cache_dir = os.environ.get("HF_HOME", "/cache/hf")
    _CLIP_CACHE["processor"] = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32", cache_dir=cache_dir,
    )
    _CLIP_CACHE["model"] = FlaxCLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", cache_dir=cache_dir,
    )
    return _CLIP_CACHE["processor"], _CLIP_CACHE["model"]


def _oe_score(z: jnp.ndarray) -> float:
    """ASAL open-endedness: negative mean of max off-diagonal CLIP cos-sim.

    More diversity → higher score (sign flipped for maximisation).
    """
    z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    sim = z @ z.T
    n = sim.shape[0]
    mask = jnp.tril(jnp.ones((n, n), dtype=bool), k=-1)
    row_max = jnp.where(mask, sim, -jnp.inf).max(-1)
    valid = jnp.isfinite(row_max)
    mean_sim = jnp.where(valid, row_max, 0.0).sum() / (valid.sum() + 1e-8)
    return float(-mean_sim)


def _fitness_for_candidate(params: jax.Array, seed: int,
                           processor, model) -> float:
    """Rollout one Flow-Lenia trajectory, CLIP-embed 4 frames, return OE score.

    Imports the *frozen* substrate from evaluator.py so search and scoring
    share identical physics.  Penalise degenerate rollouts (uniform / dead).
    """
    from evaluator import _flow_lenia_rollout, _render_flow_lenia  # frozen
    from PIL import Image as PILImage

    traj = _flow_lenia_rollout(params, seed)                 # [256, H, W, 2]
    picks = np.asarray(traj[jnp.asarray(_CLIP_FRAME_IDX)])   # [4, H, W, 2]
    rgb = _render_flow_lenia(picks)                          # [4, H, W, 3] u8
    pil = [PILImage.fromarray(rgb[j]) for j in range(rgb.shape[0])]
    inputs = processor(images=pil, return_tensors="jax", padding=True)
    feats = model.get_image_features(
        **{k: v for k, v in inputs.items() if k != "input_ids"}
    )
    score = _oe_score(feats)

    # Degeneracy penalty — uniform strips CLIP-embed far from non-uniform
    # strips, which paradoxically inflates OE on dead rollouts.
    final_mass = np.asarray(picks[-1, ..., 0])
    if final_mass.var() < 1e-6 or final_mass.max() < 1e-3:
        score -= 0.25
    return float(score)


def _batch_fitness(params_batch: jax.Array, seed: int) -> jnp.ndarray:
    """Evaluate a population.  Per-candidate failures → score=-1.0."""
    processor, model = _load_clip()  # fail-loud — eval-v2 contract

    scores = np.zeros(params_batch.shape[0], dtype=np.float32)
    for i in range(params_batch.shape[0]):
        try:
            scores[i] = _fitness_for_candidate(
                params_batch[i], int(seed), processor, model,
            )
        except Exception:
            scores[i] = -1.0  # penalise rollouts that crash the substrate
    return jnp.asarray(scores, dtype=jnp.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Sep-CMA-ES restart loop
# ─────────────────────────────────────────────────────────────────────────────
def _run_restart(
    rng: jax.Array,
    seed_pool_train: jax.Array,
    sigma_init: float,
    gens_cap: int,
    wall_deadline: float,
    ckpt_path: Path | None,
    start_gen: int,
) -> tuple[jax.Array, float, list[float], int]:
    """One Sep-CMA-ES restart. Returns (best_params, best_score, curve, gens)."""
    from evosax import Sep_CMA_ES  # deferred; fail-loud on missing evosax

    strategy = Sep_CMA_ES(popsize=_POP_SIZE, num_dims=_K, sigma_init=sigma_init)
    state = strategy.initialize(rng)

    best_params = jnp.zeros(_K, dtype=jnp.float32)
    best_score = float("-inf")
    curve: list[float] = []
    n_train = int(seed_pool_train.shape[0])

    for g in range(gens_cap):
        if time.monotonic() >= wall_deadline:
            break
        rng, sub = jax.random.split(rng)
        params_batch, state = strategy.ask(sub, state)
        s = int(seed_pool_train[(start_gen + g) % n_train])
        fits = _batch_fitness(params_batch, s)

        # evosax minimises; we pass -fits so it maximises our OE score.
        state = strategy.tell(params_batch, -fits, state)

        gbest_idx = int(jnp.argmax(fits))
        gbest = float(fits[gbest_idx])
        if gbest > best_score:
            best_score = gbest
            best_params = params_batch[gbest_idx]
            if ckpt_path is not None:
                try:
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(ckpt_path), np.asarray(best_params, np.float32))
                except Exception:
                    pass

        curve.append(best_score)

        if (g + 1) % _CLEAR_CACHE_EVERY == 0:
            gc.collect()
            try:
                jax.clear_caches()
            except Exception:
                pass

    return best_params, best_score, curve, len(curve)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def search(
    substrate_name: Literal["lenia", "flow_lenia"],
    seed_pool_train: jax.Array,
    budget: dict,
    rng: jax.Array,
) -> dict:
    """Flow-Lenia + Sep-CMA-ES + CLIP-OE search (eval-v2 re-run of orbit 03)."""
    wall_clock_s = float(budget.get("wall_clock_s", 1800.0))
    t0 = time.monotonic()
    deadline = t0 + wall_clock_s - _SAFETY_MARGIN_S

    run_id = os.environ.get("RE_RUN_ID") or os.environ.get("MODAL_TASK_ID") or "local"
    ckpt_path = Path(f"/tmp/ckpt/{run_id}_best.npy")

    all_curves: list[float] = []
    archive: list[jax.Array] = []
    best_params = jnp.zeros(_K, dtype=jnp.float32)
    best_score = float("-inf")
    restart_stats: list[dict] = []

    n_restarts = len(_RESTART_SIGMAS)
    start_gen = 0

    for r, sigma_init in enumerate(_RESTART_SIGMAS):
        if time.monotonic() >= deadline:
            break
        time_remaining = deadline - time.monotonic()
        restarts_left = n_restarts - r
        restart_budget = time_remaining  # last restart absorbs the remainder
        if restarts_left > 1:
            restart_budget = time_remaining * 0.55
        restart_deadline = time.monotonic() + restart_budget

        rng, sub = jax.random.split(rng)
        bp, bs, curve, gens_done = _run_restart(
            rng=sub,
            seed_pool_train=seed_pool_train,
            sigma_init=float(sigma_init),
            gens_cap=_MAX_GENS_PER_RESTART,
            wall_deadline=min(restart_deadline, deadline),
            ckpt_path=ckpt_path,
            start_gen=start_gen,
        )
        start_gen += gens_done
        restart_stats.append({
            "restart_idx": r,
            "sigma_init": float(sigma_init),
            "n_generations": gens_done,
            "best_score": float(bs),
        })
        all_curves.extend(curve)
        if bp is not None and bs > -float("inf"):
            archive.append(bp)
        if bs > best_score:
            best_score = bs
            best_params = bp

    # Fallback — finite sanity check.
    if not np.all(np.isfinite(np.asarray(best_params))) or best_score == float("-inf"):
        best_params = jnp.zeros(_K, dtype=jnp.float32)

    # Persist final best one more time for SIGKILL recovery.
    try:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(ckpt_path), np.asarray(best_params, np.float32))
    except Exception:
        pass

    archive_arr = (
        jnp.stack(archive[-16:]) if archive
        else jnp.zeros((1, _K), dtype=jnp.float32)
    )

    return {
        "best_params": best_params,
        "archive": archive_arr,
        "search_trace": {
            "best_proxy_per_gen": all_curves,
            "n_generations": len(all_curves),
            # eval-v2 provenance tag: if logs show this string, the run used
            # real Sep-CMA-ES (evosax imported without ImportError).  A
            # random-search fallback would never carry this tag.
            "algorithm": "sep_cma_es_flow_lenia_REAL_eval_v2",
            "restarts": restart_stats,
            "sigma_schedule": list(_RESTART_SIGMAS),
            "pop_size": _POP_SIZE,
            "substrate": substrate_name,
            "wall_clock_budget_s": wall_clock_s,
            "parent_orbit": "03-flow-lenia-clip-oe",
            "eval_version": "eval-v2",
        },
    }
