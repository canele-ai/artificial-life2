"""teaser.py — Campaign Issue teaser figure.

Two-panel before/after for the research question:
  (a) what's broken: ASAL's open-endedness + supervised-target CLIP metrics
      reward chaotic trajectories or static-prompt-match; they don't capture
      structured life-like dynamics.
  (b) what success looks like: a VLM-judge rubric rewarding 5 tiers
      (existence, agency, robustness, reproduction, coherence), gated by
      multiplicative geometric mean.

Output: research/figures/teaser.png  (~300 KB target)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIG_PATH = REPO / "research" / "figures" / "teaser.png"


def _make_fake_trajectory(kind: str, rng: np.random.Generator) -> np.ndarray:
    """Return [5, 64, 64] float arrays simulating 5 life-cycle frames."""
    frames = np.zeros((5, 64, 64), dtype=np.float32)
    if kind == "chaos":
        for i in range(5):
            frames[i] = rng.random((64, 64))
    elif kind == "static":
        blob = np.zeros((64, 64), dtype=np.float32)
        yy, xx = np.mgrid[:64, :64]
        blob[(xx - 32) ** 2 + (yy - 32) ** 2 < 200] = 0.8
        for i in range(5):
            frames[i] = blob + 0.01 * rng.random((64, 64))
    elif kind == "lifecycle":
        yy, xx = np.mgrid[:64, :64]
        for i in range(5):
            r = 4 + i * 3
            cx, cy = 32 + (i - 2) * 2, 32
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            frames[i] = np.exp(-d2 / (r ** 2 * 1.5)) * 0.9
            if i >= 3:
                cx2 = cx + 8 + (i - 3) * 6
                d2b = (xx - cx2) ** 2 + (yy - cy) ** 2
                frames[i] += np.exp(-d2b / (r ** 2)) * 0.7
            frames[i] = np.clip(frames[i], 0, 1)
    return frames


def _render_strip(frames: np.ndarray, cmap="viridis") -> np.ndarray:
    cm = plt.get_cmap(cmap)
    strip = np.concatenate([cm(frames[i])[..., :3] for i in range(5)], axis=1)
    return strip


def main() -> None:
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(13, 5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])

    # ── Left column: "what's broken" ──────────────────────────────────────
    ax_l_strip = fig.add_subplot(gs[0, 0])
    ax_l_text = fig.add_subplot(gs[1, 0])

    chaos = _render_strip(_make_fake_trajectory("chaos", rng))
    static = _render_strip(_make_fake_trajectory("static", rng))
    combo = np.concatenate([chaos, np.ones_like(chaos[:2]) * 1.0, static], axis=0)[:-2]
    ax_l_strip.imshow(combo, aspect="auto")
    ax_l_strip.set_xticks([])
    ax_l_strip.set_yticks([])
    ax_l_strip.set_title(
        "(a) what ASAL metrics reward\n"
        "top: chaotic noise (high open-endedness) · bottom: static blob (high prompt-match)",
        fontsize=10,
    )
    ax_l_text.axis("off")
    ax_l_text.text(
        0.0, 0.95,
        "Problem\n\n"
        "ASAL's open-endedness score rewards any rollout that keeps moving\n"
        "in CLIP embedding space, including pure chaos.  The supervised-target\n"
        "score rewards matching a single image prompt, not a developmental\n"
        "trajectory.  Neither captures structured life-like dynamics.",
        transform=ax_l_text.transAxes,
        fontsize=10, va="top", fontfamily="sans-serif",
    )

    # ── Right column: "what success looks like" ───────────────────────────
    ax_r_strip = fig.add_subplot(gs[0, 1])
    ax_r_text = fig.add_subplot(gs[1, 1])

    life = _render_strip(_make_fake_trajectory("lifecycle", rng))
    ax_r_strip.imshow(life, aspect="auto")
    ax_r_strip.set_xticks([])
    ax_r_strip.set_yticks([])
    ax_r_strip.set_title(
        "(b) what we want to find\n"
        "ordered developmental trajectory: seed → growth → maturity → reproduction",
        fontsize=10,
    )

    ax_r_text.axis("off")
    ax_r_text.text(
        0.0, 0.95,
        "Approach\n\n"
        "Replace the scalar fitness with a VLM-as-judge rubric:\n"
        "  5 tiers — existence · agency · robustness · reproduction · coherence\n"
        "  Claude Sonnet 4.6 rates each 128×640 frame strip on each tier\n"
        "  geometric mean → any tier = 0 zeroes the score (no gaming)\n"
        "  paired Wilcoxon vs pinned ASAL baseline on 16 held-out seeds",
        transform=ax_r_text.transAxes,
        fontsize=10, va="top", fontfamily="sans-serif",
    )

    fig.suptitle(
        "Life-Cycle Fidelity Under a VLM Judge — eval-v1",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(FIG_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)

    size_kb = FIG_PATH.stat().st_size / 1024
    print(f"[teaser] wrote {FIG_PATH}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
