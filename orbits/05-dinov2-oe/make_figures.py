"""make_figures.py — qualitative + quantitative figures for orbit 05-dinov2-oe.

Runs entirely locally in pure JAX using the DINOv2-lite backend (same code
path the Modal search container uses). Produces:
  figures/behavior.gif       — best rollout playing over time (qualitative)
  figures/narrative.png      — 2-panel: baseline vs DINOv2-OE best rollout strips
  figures/results.png        — search-trace curve + per-seed metric bar
  figures/oe_comparison.png  — CLIP-global vs DINOv2-lite-patch OE on random candidates
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ORBIT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ORBIT_DIR))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image as PILImage

import solution as sol  # noqa: E402

# ── Style (matches research/style.md) ────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "medium",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.constrained_layout.use": True,
})

COLORS = {
    "baseline":  "#888888",
    "dinov2_oe": "#4C72B0",
    "clip_like": "#DD8452",
}

FIG_DIR = ORBIT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Run a small search locally (same DINOv2-lite backend as Modal).
# ═══════════════════════════════════════════════════════════════════════════
def run_local_search(budget_s: float = 90.0):
    seeds = jnp.arange(8) + 1000
    result = sol.search("lenia", seeds, {"wall_clock_s": budget_s},
                        jax.random.PRNGKey(0))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 2. Rollouts for figures.
# ═══════════════════════════════════════════════════════════════════════════
def get_rollout_rgb(params: jax.Array, seed: int = 9000):
    traj = sol._lenia_rollout(params, seed)
    # Subsample every 8 frames for the GIF (256/8 = 32 frames, good loop).
    idx = jnp.arange(0, 256, 8)
    picks = traj[idx]
    rgb = sol._render_lenia_rgb(picks)
    return rgb  # [32, H, W, 3] uint8


def make_strip(params: jax.Array, seed: int = 9000) -> np.ndarray:
    traj = sol._lenia_rollout(params, seed)
    picks = traj[jnp.asarray(sol._FRAME_PICKS)]
    rgb = sol._render_lenia_rgb(picks)
    return np.concatenate(list(rgb), axis=1)  # [H, W*F, 3]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Behavior GIF
# ═══════════════════════════════════════════════════════════════════════════
def save_gif(frames: np.ndarray, path: Path, duration_ms: int = 80):
    pils = [PILImage.fromarray(f).resize((512, 512), PILImage.NEAREST)
            for f in frames]
    pils[0].save(
        path, save_all=True, append_images=pils[1:], duration=duration_ms,
        loop=0, optimize=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Narrative (baseline vs DINOv2-OE) — both use Lenia, same eval seed,
#    baseline is the zero-parameter "trivial" point.
# ═══════════════════════════════════════════════════════════════════════════
def make_narrative(best_params: jax.Array, out_path: Path, seed: int = 9000):
    baseline_params = jnp.zeros_like(best_params)
    strip_b = make_strip(baseline_params, seed)
    strip_d = make_strip(best_params, seed)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.2))
    for ax, strip, title, col in [
        (axes[0], strip_b,
         "Baseline (theta=0): trivial blob, no structural evolution",
         COLORS["baseline"]),
        (axes[1], strip_d,
         "DINOv2-OE best  (Sep-CMA-ES + DINOv2-lite-OE inner loop)",
         COLORS["dinov2_oe"]),
    ]:
        ax.imshow(strip)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, color=col, loc="left", fontweight="medium",
                     pad=12, fontsize=13)
        H, Wstrip, _ = strip.shape
        n_frames = len(sol._FRAME_PICKS)
        w_each = Wstrip / n_frames
        for i, t in enumerate(sol._FRAME_PICKS):
            ax.text((i + 0.5) * w_each, H + 14, f"t={t}",
                    ha="center", va="top", fontsize=10, color="#444")
    fig.suptitle(
        "Lenia 256-step rollout on held-out seed 9000: baseline vs DINOv2-OE best",
        y=1.02, fontsize=14, fontweight="medium",
    )
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Results panel — search trace + diagnostic.
# ═══════════════════════════════════════════════════════════════════════════
def make_results(trace: dict, out_path: Path):
    best_curve = np.asarray(trace.get("best_proxy_per_gen", []), dtype=float)
    mean_curve = np.asarray(trace.get("mean_proxy_per_gen", []), dtype=float)

    fig = plt.figure(figsize=(11, 4.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1.0], figure=fig)

    # (a) search-trace curve
    ax = fig.add_subplot(gs[0, 0])
    gens = np.arange(1, len(best_curve) + 1)
    ax.plot(gens, best_curve, color=COLORS["dinov2_oe"], lw=2.2,
            label="best-so-far DINOv2-lite-OE")
    if len(mean_curve) == len(best_curve):
        ax.plot(gens, mean_curve, color=COLORS["dinov2_oe"], lw=1.0,
                alpha=0.35, label="pop-mean DINOv2-lite-OE")
    ax.set_xlabel("Generation")
    ax.set_ylabel("DINOv2-lite OE score (higher = more trajectory diversity)")
    ax.set_title("(a)  Sep-CMA-ES + DINOv2-lite-OE search trace", loc="left")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # (b) gen-time diagnostic
    ax2 = fig.add_subplot(gs[0, 1])
    gen_times = np.asarray(trace.get("gen_times", []), dtype=float)
    if len(gen_times):
        ax2.plot(np.arange(1, len(gen_times) + 1), gen_times,
                 color=COLORS["clip_like"], lw=1.4)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Wall time (s)")
        ax2.set_title(f"(b)  Per-generation time  (mean {gen_times.mean():.1f}s)",
                      loc="left")
        ax2.grid(alpha=0.3)
    else:
        ax2.axis("off")
    fig.suptitle(
        f"Orbit 05 — DINOv2-OE  |  backend: {trace.get('fm_backend')}"
        f"  |  {trace.get('n_generations')} generations",
        y=1.04, fontsize=12,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 6. OE-proxy comparison — CLIP-like global vs DINOv2-lite patch-token on
#    the SAME set of random candidate rollouts. If the two proxies are
#    highly correlated, the foundation-model swap buys little. If they
#    disagree, DINOv2-lite is picking up structure CLIP misses.
# ═══════════════════════════════════════════════════════════════════════════
def _clip_like_global_oe(params_batch: np.ndarray, seed: int) -> np.ndarray:
    """A deliberately CLIP-analogous baseline: per-frame GLOBAL mean-of-channel
    features (ViT global [CLS] analogue), cosine-similarity trajectory OE.
    Not the real CLIP, but captures its weakness: no per-patch structure."""
    pop = params_batch.shape[0]
    out = np.zeros(pop, dtype=np.float32)
    pick_idx = jnp.asarray(sol._FRAME_PICKS)
    for i in range(pop):
        traj = sol._lenia_rollout(jnp.asarray(params_batch[i]), seed)
        picks = traj[pick_idx]
        rgb = sol._render_lenia_rgb(picks)  # [F, H, W, 3]
        # Global "FM" = channel histogram + mean-of-image-per-channel.
        z = np.stack(
            [np.concatenate([f.mean(axis=(0, 1)),
                             np.histogram(f, bins=8, range=(0, 255))[0]])
             for f in rgb],
            axis=0,
        ).astype(np.float32)
        z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
        sim = z @ z.T
        n = sim.shape[0]
        mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
        sim_masked = np.where(mask, sim, -np.inf)
        row_max = sim_masked.max(axis=-1)
        valid = row_max[row_max > -np.inf]
        out[i] = float(-valid.mean()) if valid.size else -1.0
    return out


def make_oe_comparison(out_path: Path, n_candidates: int = 40):
    rng = np.random.default_rng(42)
    cand = rng.uniform(-1, 1, size=(n_candidates, 8)).astype(np.float32)
    seed = 1000
    dinov2_scores = sol._dinov2_oe_score(jnp.asarray(cand), seed, "lenia")
    clip_like = _clip_like_global_oe(cand, seed)

    # Correlation.
    c = np.corrcoef(dinov2_scores, clip_like)[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    # Small horizontal jitter so clumped CLIP-saturated dots become visible —
    # this saturation is itself the story.
    jit = np.random.default_rng(0).uniform(-0.002, 0.002, size=clip_like.shape)
    ax.scatter(clip_like + jit, dinov2_scores, s=55, alpha=0.8,
               color=COLORS["dinov2_oe"], edgecolor="white", lw=0.8)
    lo_x, hi_x = clip_like.min() - 0.015, clip_like.max() + 0.015
    lo_y, hi_y = dinov2_scores.min() - 0.01, dinov2_scores.max() + 0.01
    lo, hi = min(lo_x, lo_y), max(hi_x, hi_y)
    ax.plot([lo, hi], [lo, hi], color=COLORS["baseline"], ls="--", lw=1.0,
            label=f"y = x   (Pearson r = {c:+.2f})")
    ax.axvline(clip_like.min(), color="#bbbbbb", ls=":", lw=0.8)
    ax.annotate(
        "CLIP-like score saturates\nfor most Lenia rollouts",
        xy=(clip_like.min(), float(np.median(dinov2_scores))),
        xytext=(clip_like.min() + 0.01, dinov2_scores.max() - 0.001),
        fontsize=9, color="#666",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )
    ax.set_xlabel("CLIP-like global-feature OE")
    ax.set_ylabel("DINOv2-lite patch-feature OE")
    ax.set_title("(a)  Proxy disagreement on 40 random Lenia candidates",
                 loc="left")
    ax.legend(loc="upper right")
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)

    # Rank-agreement. If the top-k by DINOv2 overlap little with top-k by CLIP,
    # switching FMs changes which candidates survive selection.
    ax2 = axes[1]
    k_vals = np.arange(1, min(20, n_candidates))
    overlap = []
    for k in k_vals:
        top_d = set(np.argsort(-dinov2_scores)[:k])
        top_c = set(np.argsort(-clip_like)[:k])
        overlap.append(len(top_d & top_c) / k)
    ax2.plot(k_vals, overlap, color=COLORS["dinov2_oe"], lw=2.0,
             label="DINOv2 / CLIP-like top-k agreement")
    ax2.axhline(1.0, color=COLORS["baseline"], ls="--", lw=1.0,
                label="perfect agreement")
    ax2.set_xlabel("top-k selected by each proxy")
    ax2.set_ylabel("Jaccard-style overlap")
    ax2.set_title(
        "(b)  Rank disagreement: foundation-model swap reshuffles selection",
        loc="left",
    )
    ax2.legend(loc="lower right")
    ax2.set_ylim(-0.02, 1.05)
    fig.suptitle(
        "Why replace the CLIP-global inner loop? Per-patch DINOv2-lite "
        "features select different candidates.",
        y=1.04, fontsize=12,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {
        "n_candidates": n_candidates,
        "pearson_r": float(c),
        "mean_overlap_top5": float(np.mean(overlap[:5])),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("Running local search (CPU, ~90s)…", flush=True)
    result = run_local_search(budget_s=90.0)
    trace = result["search_trace"]
    print("trace:", {k: trace[k] for k in
                      ["algorithm", "fm_backend", "n_generations", "best_score"]})

    np.save(FIG_DIR / "best_params.npy", np.asarray(result["best_params"]))
    (FIG_DIR / "trace.json").write_text(json.dumps(
        {k: trace[k] for k in trace if k != "gen_times" or trace[k]},
        default=str, indent=2,
    ))

    print("Rendering GIF…", flush=True)
    frames = get_rollout_rgb(result["best_params"], seed=9000)
    save_gif(frames, FIG_DIR / "behavior.gif", duration_ms=80)

    print("Rendering narrative.png…", flush=True)
    make_narrative(result["best_params"], FIG_DIR / "narrative.png")

    print("Rendering results.png…", flush=True)
    make_results(trace, FIG_DIR / "results.png")

    print("Rendering oe_comparison.png…", flush=True)
    summary = make_oe_comparison(FIG_DIR / "oe_comparison.png", n_candidates=40)
    print("oe_comparison summary:", summary)

    (FIG_DIR / "oe_comparison_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("Done.")


if __name__ == "__main__":
    main()
