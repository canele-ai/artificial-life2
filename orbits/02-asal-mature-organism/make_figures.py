"""make_figures.py — post-hoc figure generation for orbit 02-asal-mature-organism.

Builds:
  - figures/results.png — metric + trace + per-seed bars (uses log_*.json dumps)
  - figures/narrative.png — method schematic + best-rollout strip
  - figures/search_trace.png — per-generation best-proxy curve

Run from worktree root:
  uv run python3 orbits/02-asal-mature-organism/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True, parents=True)

COL = {
    "ours": "#4C72B0",      # steel blue — this orbit
    "baseline": "#888888",  # ASAL baseline
    "good": "#DD8452",      # CLIP-OE reference
}


def _load_all_traces() -> list[dict]:
    """Load every *.json result we have from Modal runs (dumped by make_figures_result)."""
    results = []
    for p in sorted(FIG.glob("result_*.json")):
        try:
            results.append(json.loads(p.read_text()))
        except Exception:
            pass
    return results


def fig_search_trace(results: list[dict]) -> None:
    """Single-panel: best-proxy curve across generations for each seed run."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for r in results:
        trace = r.get("search_trace", {})
        ys = trace.get("best_proxy_per_gen") or []
        if not ys:
            continue
        label = f"seed={r.get('seed', '?')} (n={len(ys)} gen)"
        ax.plot(range(1, len(ys) + 1), ys, color=COL["ours"], alpha=0.7,
                linewidth=1.5, label=label)
    ax.set_xlabel("generation")
    ax.set_ylabel("best mature-target soft-CE score")
    ax.set_title("Sep-CMA-ES convergence — mature-organism single-prompt target")
    if len(results) > 0:
        ax.legend(loc="lower right")
    fig.savefig(FIG / "search_trace.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_results(results: list[dict]) -> None:
    """Results panel: metric per seed + trace overlay + bar comparison to anchors."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2),
                             gridspec_kw={"width_ratios": [1.0, 1.4, 1.0]})

    ax = axes[0]
    metrics = [r.get("metric") for r in results if r.get("metric") is not None]
    seeds   = [r.get("seed", i) for i, r in enumerate(results) if r.get("metric") is not None]
    if metrics:
        bars = ax.bar(range(len(metrics)), metrics, color=COL["ours"], edgecolor="white")
        ax.axhline(0.0, color=COL["baseline"], linestyle="--", linewidth=1, label="ASAL anchor = 0")
        for i, m in enumerate(metrics):
            ax.text(i, m + (0.002 if m >= 0 else -0.004), f"{m:+.3f}",
                    ha="center", va="bottom" if m >= 0 else "top", fontsize=9)
        mean = float(np.mean(metrics))
        std  = float(np.std(metrics))
        ax.set_title(f"METRIC per seed\nmean = {mean:+.4f} ± {std:.4f}")
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([f"s{s}" for s in seeds])
        ax.set_ylabel("METRIC (Δ vs ASAL baseline)")
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "(no Modal runs yet)", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("METRIC per seed")
        ax.set_xticks([])

    ax = axes[1]
    for r in results:
        ys = (r.get("search_trace") or {}).get("best_proxy_per_gen") or []
        if ys:
            ax.plot(range(1, len(ys) + 1), ys, alpha=0.7, linewidth=1.5,
                    color=COL["ours"], label=f"seed={r.get('seed','?')}")
    ax.set_xlabel("generation")
    ax.set_ylabel("best proxy (soft-CE max-prompt)")
    ax.set_title("Search trace (inner-loop proxy)")
    if results:
        ax.legend(loc="lower right")

    ax = axes[2]
    labels = ["random\n(3-min canary)", "ASAL base\n(3-min canary)", "this orbit\n(mature-target)"]
    vals = [0.512, 0.273, float(np.mean(metrics)) if metrics else np.nan]
    colors = [COL["good"], COL["baseline"], COL["ours"]]
    xs = np.arange(len(labels))
    for x, v, c in zip(xs, vals, colors):
        if not np.isnan(v):
            ax.bar(x, v, color=c, edgecolor="white")
            ax.text(x, v + 0.01, f"{v:+.3f}", ha="center", va="bottom", fontsize=9)
        else:
            ax.bar(x, 0, color=c, alpha=0.3, edgecolor="white")
            ax.text(x, 0.02, "n/a", ha="center", va="bottom",
                    fontsize=9, color="gray")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_ylabel("METRIC")
    ax.set_title("Campaign-context comparison")

    fig.suptitle("Orbit 02 — ASAL-mature-organism (single-prompt Sep-CMA-ES)",
                 fontsize=14, y=1.04)
    fig.savefig(FIG / "results.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_narrative(results: list[dict]) -> None:
    """Method schematic: prompt-bank → CLIP → Sep-CMA-ES → best-rollout strip."""
    fig = plt.figure(figsize=(12, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9], hspace=0.25)

    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.set_title("Method: single-stage mature-organism target (vs baseline's 5-stage life-cycle)",
                 fontsize=12, loc="left")
    ax.text(0.02, 0.60,
            "ASAL baseline (pinned): 5 prompts × 5 frames → 25 CLIP forwards/candidate, "
            "softmax-target averaged.\n"
            "This orbit:              3 prompt variants × 1 frame (t=127) → 3 CLIP forwards/candidate, "
            "max-prompt soft-CE.\n"
            "Rationale: narrower target direction + 5× cheaper candidate → pop 32 instead of 16.",
            transform=ax.transAxes, fontsize=10, family="monospace", va="center")
    ax.text(0.02, 0.15,
            "Prompts (Coherence + Existence bias):\n"
            '  1. "a mature self-sustaining organism with crisp geometric structure on a dark background"\n'
            '  2. "a bilaterally symmetric living cell with smooth clean boundaries"\n'
            '  3. "a soft glowing biological organism clearly distinct from empty space"',
            transform=ax.transAxes, fontsize=9, family="monospace", va="center", color="#333")

    strip_loaded = False
    for r in results:
        path = r.get("strip_png")
        if path and Path(path).exists():
            img = plt.imread(path)
            ax_s = fig.add_subplot(gs[1, :])
            ax_s.imshow(img)
            ax_s.set_title(
                f"Best-rollout strip (seed={r.get('seed','?')})  — "
                f"t ∈ {{0, 63, 127, 191, 255}} → CLIP targets frame 3 (mature)",
                fontsize=11, loc="left")
            ax_s.set_xticks([]); ax_s.set_yticks([])
            strip_loaded = True
            break
    if not strip_loaded:
        for axi in range(3):
            ax_b = fig.add_subplot(gs[1, axi])
            ax_b.text(0.5, 0.5, "(strip not yet available)", ha="center", va="center",
                      fontsize=10, color="gray", transform=ax_b.transAxes)
            ax_b.set_xticks([]); ax_b.set_yticks([])

    fig.suptitle("Orbit 02-asal-mature-organism — method narrative",
                 fontsize=14, y=1.02)
    fig.savefig(FIG / "narrative.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    results = _load_all_traces()
    fig_search_trace(results)
    fig_results(results)
    fig_narrative(results)
    print(f"Generated figures: {sorted(p.name for p in FIG.glob('*.png'))}")
    print(f"(based on {len(results)} result file(s))")


if __name__ == "__main__":
    main()
