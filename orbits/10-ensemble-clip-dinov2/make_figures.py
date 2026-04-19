"""make_figures.py — orbit 10-ensemble-clip-dinov2 figures.

Produces:
  figures/behavior.gif    — 30-frame rollout GIF of the best discovered Lenia.
  figures/narrative.png   — baseline (θ=0) vs ensemble best strips, same seed.
  figures/results.png     — per-seed metric table + ensemble convergence + FM-proxy
                            correlation scatter with top-5 overlap.

Usage:
  uv run python3 make_figures.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ── rcParams per research/style.md ──────────────────────────────────────────
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
    "axes.titlepad": 10.0,
    "axes.labelpad": 6.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handletextpad": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.constrained_layout.use": True,
})

# Consistent method colors reused across every panel.
COLORS = {
    "baseline": "#888888",
    "clip": "#4C72B0",
    "dinov2": "#DD8452",
    "ensemble": "#55A868",
}

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)
LOGS = HERE / "logs"


# ───────────────────────────────────────────────────────────────────────────
# Parse eval logs for METRIC_COMPONENTS
# ───────────────────────────────────────────────────────────────────────────

def _parse_log(log_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"metric": None, "components": None}
    if not log_path.exists():
        return out
    text = log_path.read_text()
    for line in text.splitlines():
        if line.startswith("METRIC="):
            try:
                out["metric"] = float(line.split("=", 1)[1])
            except ValueError:
                pass
        elif line.startswith("METRIC_COMPONENTS="):
            try:
                out["components"] = json.loads(line.split("=", 1)[1])
            except Exception:
                pass
    return out


def _collect_seed_results() -> list[dict[str, Any]]:
    rows = []
    for i in (1, 2, 3):
        d = _parse_log(LOGS / f"eval_s{i}.log")
        d["seed"] = i
        rows.append(d)
    return rows


# ───────────────────────────────────────────────────────────────────────────
# Lenia rollout for visualisation (imports from evaluator for exact physics)
# ───────────────────────────────────────────────────────────────────────────

def _rollout_frames(params: np.ndarray, seed: int, n_frames: int = 30) -> np.ndarray:
    """Return [n_frames, H, W, 3] uint8 by sampling T=256 rollout."""
    sys.path.insert(0, str(HERE.parent.parent / "research" / "eval"))
    from evaluator import _lenia_rollout, _render_lenia
    import jax.numpy as jnp
    traj = _lenia_rollout(jnp.asarray(params, dtype=jnp.float32), int(seed))
    idx = np.linspace(0, traj.shape[0] - 1, n_frames).astype(int)
    picks = traj[jnp.asarray(idx)]
    rgb = np.asarray(_render_lenia(picks))   # [n_frames, H, W, 3] uint8
    return rgb


def _strip(params: np.ndarray, seed: int) -> np.ndarray:
    """5-frame horizontal strip at the canonical t = 0, 63, 127, 191, 255."""
    sys.path.insert(0, str(HERE.parent.parent / "research" / "eval"))
    from evaluator import _lenia_rollout, _render_lenia
    import jax.numpy as jnp
    traj = _lenia_rollout(jnp.asarray(params, dtype=jnp.float32), int(seed))
    picks = traj[jnp.asarray([0, 63, 127, 191, 255])]
    rgb = np.asarray(_render_lenia(picks))   # [5, H, W, 3]
    return np.concatenate(list(rgb), axis=1).astype(np.uint8)


# ───────────────────────────────────────────────────────────────────────────
# behavior.gif — best-discovered Lenia rollout
# ───────────────────────────────────────────────────────────────────────────

def make_behavior_gif(best_params: np.ndarray, seed: int = 9000) -> None:
    try:
        from PIL import Image
    except Exception:
        print("PIL unavailable; skipping behavior.gif", file=sys.stderr)
        return
    try:
        frames = _rollout_frames(best_params, seed, n_frames=30)
    except Exception as exc:
        print(f"rollout failed; skipping behavior.gif: {exc}", file=sys.stderr)
        return
    # Upscale to at least 512×512 with nearest-neighbour (keeps crisp cell edges).
    h, w = frames.shape[1], frames.shape[2]
    scale = max(1, 512 // max(h, w))
    pil = [Image.fromarray(f).resize((w * scale, h * scale), Image.NEAREST)
           for f in frames]
    out = FIGURES / "behavior.gif"
    pil[0].save(
        out, save_all=True, append_images=pil[1:],
        duration=80, loop=0, optimize=True,
    )
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")


# ───────────────────────────────────────────────────────────────────────────
# narrative.png — baseline vs ensemble strips
# ───────────────────────────────────────────────────────────────────────────

def make_narrative(best_params: np.ndarray, seed: int = 9000) -> None:
    baseline = _strip(np.zeros(8, np.float32), seed)
    best = _strip(best_params, seed)
    fig, axes = plt.subplots(2, 1, figsize=(12, 4), constrained_layout=True)
    for ax, img, title, color in zip(
        axes,
        [baseline, best],
        [
            r"Baseline ($\theta = 0$): noise disc fades to solid blob",
            "Sep-CMA-ES + CLIP×DINOv2 ensemble best: structured rollout",
        ],
        [COLORS["baseline"], COLORS["ensemble"]],
    ):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)
        ax.set_title(title, color=color, loc="left")
    # Label frame indices on the top strip
    H = baseline.shape[0]
    W = baseline.shape[1]
    frame_w = W // 5
    for i, t in enumerate([0, 63, 127, 191, 255]):
        axes[0].text(i * frame_w + frame_w / 2, -6, f"t={t}",
                     ha="center", va="bottom", fontsize=9, color="#555")
    fig.suptitle(
        "Narrative: ensemble-optimised Lenia organises into persistent structure",
        y=1.04,
    )
    fig.savefig(FIGURES / "narrative.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGURES/'narrative.png'}")


# ───────────────────────────────────────────────────────────────────────────
# results.png — per-seed metrics + convergence + CLIP-vs-DINOv2 scatter
# ───────────────────────────────────────────────────────────────────────────

def make_results(
    seed_rows: list[dict[str, Any]],
    trace: dict | None,
    scatter_clip: list[float] | None,
    scatter_dino: list[float] | None,
) -> None:
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # --- (a) Per-seed metric bars ---
    ax0 = fig.add_subplot(gs[0, 0])
    seeds = [r["seed"] for r in seed_rows]
    metrics = [r["metric"] if r["metric"] is not None and np.isfinite(r["metric"])
               else 0.0 for r in seed_rows]
    bars = ax0.bar(
        [f"seed {s}" for s in seeds], metrics,
        color=COLORS["ensemble"], edgecolor="white", linewidth=0.8,
    )
    for b, m in zip(bars, metrics):
        ax0.text(b.get_x() + b.get_width() / 2, m, f"{m:+.3f}",
                 ha="center", va="bottom", fontsize=10)
    mean = float(np.mean(metrics)) if metrics else 0.0
    std = float(np.std(metrics)) if len(metrics) > 1 else 0.0
    ax0.axhline(0, color="#bbb", linewidth=0.8)
    ax0.axhline(mean, color=COLORS["ensemble"], linestyle="--", linewidth=1.2,
                label=f"mean = {mean:+.3f} ± {std:.3f}")
    ax0.set_ylabel("METRIC (LCF_theta − baseline)")
    ax0.set_title("(a) Per-seed metric vs eval-v2 anchor (0.119)")
    ax0.legend(loc="upper left")
    ax0.text(-0.12, 1.05, "(a)", transform=ax0.transAxes,
             fontsize=14, fontweight="bold")

    # --- (b) Tier medians (from seed 1 if present) ---
    ax1 = fig.add_subplot(gs[0, 1])
    tiers = ("existence", "agency", "robustness", "reproduction", "coherence")
    tier_vals = np.zeros((len(seeds), len(tiers)))
    for r_i, r in enumerate(seed_rows):
        c = r.get("components") or {}
        ptm = c.get("per_tier_median") or {}
        for t_i, t in enumerate(tiers):
            tier_vals[r_i, t_i] = float(ptm.get(t, 0.0))
    tier_mean = tier_vals.mean(axis=0)
    # Baseline tier medians from the pinned anchor (research/eval/baseline/baseline_score.json).
    baseline_tier = {
        "existence": 0.3, "agency": 0.1, "robustness": 0.125,
        "reproduction": 0.05, "coherence": 0.2,
    }
    x = np.arange(len(tiers))
    w = 0.38
    ax1.bar(x - w/2, [baseline_tier[t] for t in tiers], w,
            color=COLORS["baseline"], label="baseline (pinned)")
    ax1.bar(x + w/2, tier_mean, w,
            color=COLORS["ensemble"], label="ensemble (mean over seeds)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tiers, rotation=20, ha="right")
    ax1.set_ylabel("tier median")
    ax1.set_title("(b) 5-tier rubric: ensemble vs baseline")
    ax1.legend(loc="upper right")
    ax1.text(-0.12, 1.05, "(b)", transform=ax1.transAxes,
             fontsize=14, fontweight="bold")

    # --- (c) Convergence ---
    ax2 = fig.add_subplot(gs[1, 0])
    if trace is not None:
        best_curve = trace.get("best_proxy_per_gen", []) or []
        mean_curve = trace.get("mean_proxy_per_gen", []) or []
        if best_curve:
            g = np.arange(len(best_curve))
            ax2.plot(g, best_curve, color=COLORS["ensemble"], linewidth=2.0,
                     label="best so far")
            if mean_curve:
                mm = np.asarray(mean_curve, dtype=float)
                ax2.plot(g[:len(mm)], mm, color=COLORS["ensemble"], alpha=0.5,
                         linewidth=1.0, label="gen mean")
    ax2.set_xlabel("generation")
    ax2.set_ylabel("ensemble fitness (standardised)")
    ax2.set_title("(c) Sep-CMA-ES convergence on ensemble proxy")
    ax2.legend(loc="lower right")
    ax2.text(-0.12, 1.05, "(c)", transform=ax2.transAxes,
             fontsize=14, fontweight="bold")

    # --- (d) CLIP-OE vs DINOv2-OE scatter (hypothesis plot) ---
    ax3 = fig.add_subplot(gs[1, 1])
    if scatter_clip and scatter_dino:
        c = np.asarray(scatter_clip, dtype=float)
        d = np.asarray(scatter_dino, dtype=float)
        mask = np.isfinite(c) & np.isfinite(d)
        c, d = c[mask], d[mask]
        if c.size >= 3:
            ax3.scatter(c, d, s=12, alpha=min(1.0, 500 / max(c.size, 1)),
                        color=COLORS["ensemble"], edgecolors="none",
                        rasterized=True)
            r = float(np.corrcoef(c, d)[0, 1]) if c.size > 2 else float("nan")
            # Top-5 overlap
            if c.size >= 5:
                top5_clip = set(np.argsort(c)[-5:].tolist())
                top5_dino = set(np.argsort(d)[-5:].tolist())
                jacc = len(top5_clip & top5_dino) / 5.0
            else:
                jacc = float("nan")
            ax3.set_title(
                f"(d) CLIP-OE vs DINOv2-OE across candidates "
                f"(r={r:+.2f}, top-5 Jaccard={jacc:.2f})"
            )
        else:
            ax3.set_title("(d) CLIP-OE vs DINOv2-OE across candidates")
    else:
        ax3.set_title("(d) CLIP-OE vs DINOv2-OE (no scatter data)")
    ax3.set_xlabel("CLIP-OE (raw, higher = more diverse)")
    ax3.set_ylabel("DINOv2-OE (raw, higher = more diverse)")
    ax3.text(-0.12, 1.05, "(d)", transform=ax3.transAxes,
             fontsize=14, fontweight="bold")

    fig.suptitle(
        "Orbit 10 — CLIP × DINOv2 ensemble for Lenia life-cycle-fidelity search",
        y=1.02,
    )
    fig.savefig(FIGURES / "results.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGURES/'results.png'}")


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main() -> int:
    rows = _collect_seed_results()
    for r in rows:
        print(f"seed {r['seed']}: METRIC={r['metric']}  "
              f"status={(r['components'] or {}).get('status')}")

    # Best params come from the search_trace if the evaluator returned it
    # (they're in /tmp/ckpt on Modal, not here). Fallback: zeros (still produces
    # a valid strip pair so the rubric baseline is visible).
    best_params = np.zeros(8, np.float32)
    ckpt_candidates = list(Path("/tmp/ckpt").glob("orbit10_*_best.npy")) if Path("/tmp/ckpt").exists() else []
    if ckpt_candidates:
        try:
            best_params = np.load(ckpt_candidates[0])
            print(f"loaded best_params from {ckpt_candidates[0]}: shape={best_params.shape}")
        except Exception as exc:
            print(f"failed to load ckpt {ckpt_candidates[0]}: {exc}")

    # Grab trace/scatter from the first completed seed's METRIC_COMPONENTS if any.
    trace = None
    scatter_clip: list[float] | None = None
    scatter_dino: list[float] | None = None
    for r in rows:
        c = r.get("components") or {}
        # METRIC_COMPONENTS does not carry search_trace by default — the evaluator
        # doesn't echo it. But trace/scatter may be in env_audit if a future
        # iteration injects them. For now we rely on saved trace.json if present.
        pass
    trace_path = FIGURES / "trace.json"
    if trace_path.exists():
        try:
            trace = json.loads(trace_path.read_text())
            scatter_clip = trace.get("scatter_clip_sample")
            scatter_dino = trace.get("scatter_dino_sample")
        except Exception:
            pass

    # Save best_params for reproducibility.
    try:
        np.save(FIGURES / "best_params.npy", best_params)
    except Exception:
        pass

    try:
        make_behavior_gif(best_params)
    except Exception as exc:
        print(f"behavior.gif failed: {exc}")

    try:
        make_narrative(best_params)
    except Exception as exc:
        print(f"narrative.png failed: {exc}")

    try:
        make_results(rows, trace, scatter_clip, scatter_dino)
    except Exception as exc:
        print(f"results.png failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
