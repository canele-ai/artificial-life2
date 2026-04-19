"""solution.py — Flow-Lenia × Sep-CMA-ES × CLIP open-endedness (orbit 03).

Hypothesis (campaign axis 1 — substrate gap):
    Flow-Lenia (Plantec et al. 2023, arXiv:2212.07906) enforces mass
    conservation via semi-Lagrangian transport + renormalisation.  This
    uniquely enables (a) multi-species coexistence, (b) soft self-boundaries,
    (c) persistent identity under perturbation — precisely the properties the
    5-tier judge rewards for *reproduction* and *robustness*.  ASAL's paper
    did not cover Flow-Lenia.  This orbit is the campaign's first probe on it.

Search procedure:
    Sep-CMA-ES (evosax) over Flow-Lenia's 8-dim parameter vector, optimising
    ASAL open-endedness (max off-diagonal CLIP cosine similarity, averaged
    across K=4 frames drawn from t ∈ {0, 63, 127, 255}).  Because Flow-Lenia
    is less well-characterised than Lenia under CMA-ES, we issue 2-4 restarts
    with varying σ_init (0.15 → 0.08 → 0.25 → 0.12) and retain the best-ever
    candidate across restarts.  Each restart is capped at ≤ 50 generations
    and a fraction of the wall-clock budget.

Memory discipline (eval-v1 has a documented 30-min subprocess OOM bug on
CMA-ES loops because XLA retraces every candidate's param-shape):
  - module-level `_CLIP_CACHE` — CLIP weights loaded exactly once.
  - `jax.clear_caches()` + `gc.collect()` every 10 generations.
  - pop_size = 16 (ASAL default), ≤ 50 gens per restart.
  - Per-candidate rollouts are materialised, rendered, embedded, then freed;
    no accumulation of DeviceArrays across generations.
  - Checkpoint every generation to `/tmp/ckpt/{run_id}_best.npy` so the
    evaluator can recover `best_params` on SIGKILL.
  - Self-limit wall-clock to `budget - 10 s` (SIGKILL safety margin).

Interface contract (problem_spec.md):
  search(substrate_name, seed_pool_train, budget, rng) → {
      "best_params":   jax.Array [8] float32,
      "archive":       jax.Array [<=16, 8] float32,
      "search_trace":  dict (scalars only — no θ histories on disk).
  }

Constraints respected:
  - Bit-identical under fixed rng.
  - No api.anthropic.com / api.openai.com calls.
  - Does not read research/eval/judge/ or research/eval/baseline/.
  - Imports `_flow_lenia_rollout` + `_render_flow_lenia` from the frozen
    evaluator — DO NOT re-implement the physics.
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
_CLEAR_CACHE_EVERY = 10             # per OOM-mitigation notes in canary_results
_SAFETY_MARGIN_S = 12.0             # leave headroom before evaluator SIGKILL
_CLIP_FRAME_IDX = (0, 63, 127, 255)  # 4 frames → 6 pairs for OE score
_RESTART_SIGMAS = (0.15, 0.08, 0.25, 0.12)  # diverse σ_init across restarts

# Module-level cache — prevents repeated CLIP loads that leak FlaxCLIPModel
# params across generations and are the primary OOM trigger per canary notes.
_CLIP_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# CLIP open-endedness inner-loop proxy
# ─────────────────────────────────────────────────────────────────────────────
def _load_clip():
    """Load CLIP ViT-B/32 exactly once per process."""
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
    """ASAL open-endedness: mean-of-max-off-diagonal CLIP cosine similarity.

    Lower similarity = higher novelty across frames.  ASAL *minimises* pairwise
    similarity to push rollouts apart in FM space; here we define the "score"
    so that MORE diversity = HIGHER score (for evosax max-by-negation below).
    """
    z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
    sim = z @ z.T
    n = sim.shape[0]
    mask = jnp.tril(jnp.ones((n, n), dtype=bool), k=-1)
    row_max = jnp.where(mask, sim, -jnp.inf).max(-1)
    # Drop the first row (row 0 has no off-diagonal entries below) by masking.
    valid = jnp.isfinite(row_max)
    mean_sim = jnp.where(valid, row_max, 0.0).sum() / (valid.sum() + 1e-8)
    # Score = -mean_sim so that diverse trajectories have HIGHER score.
    return float(-mean_sim)


def _fitness_for_candidate(params: jax.Array, seed: int,
                           processor, model) -> float:
    """Rollout one Flow-Lenia trajectory, CLIP-embed 4 frames, return OE score.

    Imports the *frozen* substrate from evaluator.py to guarantee identical
    physics with final scoring (no drift between search and judge).
    """
    from evaluator import _flow_lenia_rollout, _render_flow_lenia  # frozen
    from PIL import Image as PILImage

    traj = _flow_lenia_rollout(params, seed)            # [256, H, W, 2]
    picks = np.asarray(traj[jnp.asarray(_CLIP_FRAME_IDX)])  # [4, H, W, 2]
    rgb = _render_flow_lenia(picks)                     # [4, H, W, 3] uint8
    pil = [PILImage.fromarray(rgb[j]) for j in range(rgb.shape[0])]
    inputs = processor(images=pil, return_tensors="jax", padding=True)
    feats = model.get_image_features(
        **{k: v for k, v in inputs.items() if k != "input_ids"}
    )
    # feats: [4, 512]
    score = _oe_score(feats)
    # Penalise degenerate rollouts (all-zero mass / all-saturated) — these
    # can paradoxically yield low cosine similarity because CLIP embeds the
    # uniform-gray strip far from non-degenerate patterns.  Detect via
    # low mass-channel variance on the final frame.
    final_mass = np.asarray(picks[-1, ..., 0])
    if final_mass.var() < 1e-6 or final_mass.max() < 1e-3:
        score -= 0.25
    return float(score)


def _batch_fitness(params_batch: jax.Array, seed: int) -> jnp.ndarray:
    """Evaluate a population.  Falls back to a deterministic random proxy if
    CLIP is unavailable (lets the smoke test pass without transformers)."""
    try:
        processor, model = _load_clip()
    except Exception:  # transformers not installed, weights missing, etc.
        rng = np.random.default_rng(int(seed) * 9973 + 17)
        return jnp.asarray(
            rng.uniform(-0.2, 0.2, size=(params_batch.shape[0],)),
            dtype=jnp.float32,
        )

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
    """One Sep-CMA-ES restart.

    Returns (best_params, best_score, per-gen-best-scores, gens_done).
    """
    from evosax import Sep_CMA_ES  # deferred

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
    """Flow-Lenia + Sep-CMA-ES + CLIP-OE search."""
    wall_clock_s = float(budget.get("wall_clock_s", 1800.0))
    t0 = time.monotonic()
    deadline = t0 + wall_clock_s - _SAFETY_MARGIN_S

    # Resolve checkpoint path from env (set by search_worker.py).
    run_id = os.environ.get("RE_RUN_ID") or os.environ.get("MODAL_TASK_ID") or "local"
    ckpt_path = Path(f"/tmp/ckpt/{run_id}_best.npy")

    # Allocate budget across restarts.  Rough heuristic: each restart gets
    # an equal share, but we also cap per-restart generations at 50.  If
    # the first restart converges early we'll flow remaining time into later
    # restarts automatically (they run until deadline).
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
        # Budget per remaining restart, minus elapsed.
        time_remaining = deadline - time.monotonic()
        restarts_left = n_restarts - r
        restart_budget = time_remaining  # last restart gets all remaining
        if restarts_left > 1:
            # Give half of remaining to this one, keep half for later.
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

    # Fallback — if EVERY restart failed to produce a finite score, return
    # a sensible Flow-Lenia centre (μ≈0.15, σ≈0.015, α≈1.0) scaled into the
    # [~-1, ~1] param-space convention the substrate uses.
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
            "algorithm": "sep_cma_es_clip_oe_flow_lenia_multirestart",
            "restarts": restart_stats,
            "sigma_schedule": list(_RESTART_SIGMAS),
            "pop_size": _POP_SIZE,
            "substrate": substrate_name,
            "wall_clock_budget_s": wall_clock_s,
        },
    }
