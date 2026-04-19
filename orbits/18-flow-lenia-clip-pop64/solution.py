"""solution.py — Flow-Lenia × Sep-CMA-ES × CLIP open-endedness at pop=64 (orbit 18).

Parent: orbit/15-flow-lenia-retry (leader at +0.339 LCF at pop=16).
Hypothesis (campaign axis — compute scaling on Flow-Lenia):
On plain Lenia, Sep-CMA-ES scaling saturated (pop=16→64→128: +0.319 → +0.247
→ +0.325).  Flow-Lenia's larger/richer manifold (semi-Lagrangian transport +
Σ-renormalisation) may re-open that saturated scaling: more compute → better
judge score, instead of a flat curve.

Change vs. orbit 15: pop_size 16 → 64 (4×).  To fit the 1500 s search budget,
we halve gens per restart (50 → 15) and trim the σ schedule to 2 restarts.
Cache-clearing frequency is tightened (every 5 gens instead of 10) because
each gen allocates 4× more CLIP forward passes.

Everything else (Flow-Lenia rollout via frozen evaluator, degeneracy penalty,
module-level _CLIP_CACHE, checkpointing) is inherited verbatim from orbit 15.
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
# Hyper-parameters (pop=64 variant)
# ─────────────────────────────────────────────────────────────────────────────
_K = 8                              # Flow-Lenia param dim (padded; 3 active)
_POP_SIZE = 64                      # 4× orbit 15
_MAX_GENS_PER_RESTART = 15          # was 50 at pop=16; budget-balanced for pop=64
_CLEAR_CACHE_EVERY = 5              # tightened: 4× more CLIP calls per gen
_SAFETY_MARGIN_S = 12.0
_CLIP_FRAME_IDX = (0, 63, 127, 255)  # 4 frames → 6 pairs for OE score
_RESTART_SIGMAS = (0.10, 0.20)      # 2 restarts only — pop=64 needs more gens/restart

# Module-level cache — prevents repeated CLIP loads that leak FlaxCLIPModel
# params across generations (documented OOM trigger in canary_results_full.md).
_CLIP_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# CLIP open-endedness inner-loop proxy (unchanged from orbit 15)
# ─────────────────────────────────────────────────────────────────────────────
def _load_clip():
    """Load CLIP ViT-B/32 exactly once per process. Fail-loud on missing deps."""
    if "model" in _CLIP_CACHE:
        return _CLIP_CACHE["processor"], _CLIP_CACHE["model"]
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
    """ASAL open-endedness: negative mean of max off-diagonal CLIP cos-sim."""
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
    """Rollout one Flow-Lenia trajectory, CLIP-embed 4 frames, return OE score."""
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
            scores[i] = -1.0
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
    """Flow-Lenia + Sep-CMA-ES (pop=64) + CLIP-OE search."""
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
            # eval-v2 provenance tag — logs must show this exact string to
            # confirm algorithm=Sep-CMA-ES at pop=64 on Flow-Lenia.
            "algorithm": "sep_cma_es_flow_lenia_clip_pop64_eval_v2",
            "restarts": restart_stats,
            "sigma_schedule": list(_RESTART_SIGMAS),
            "pop_size": _POP_SIZE,
            "max_gens_per_restart": _MAX_GENS_PER_RESTART,
            "substrate": substrate_name,
            "wall_clock_budget_s": wall_clock_s,
            "parent_orbit": "15-flow-lenia-retry",
            "eval_version": "eval-v2",
        },
    }
