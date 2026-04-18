"""eval_schematic.py — Step 2.5b: generate the eval-v1 schematic figure.

Produces `research/eval/figures/eval_schematic.png` — a 3-panel image:
  (a) discriminability ladder: trivial_bad / baseline / good / target
  (b) eval pipeline diagram: solution → A100 rollout → CPU judge → METRIC
  (c) metric card: formula, direction, citation, dataset, pitfalls

Values read from `research/eval/baseline/baseline_score.json` and
`research/eval/canary_results.md`. Run AFTER Step 2.5 baseline freeze.

Usage:
    uv run python3 scripts/eval_schematic.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FIG_PATH = REPO / "research" / "eval" / "figures" / "eval_schematic.png"
BASELINE_JSON = REPO / "research" / "eval" / "baseline" / "baseline_score.json"
CANARY_MD = REPO / "research" / "eval" / "canary_results.md"


def _extract_canary_metrics() -> dict[str, float]:
    text = CANARY_MD.read_text()
    wanted = {"trivial_bad": None, "baseline": None, "good": None}
    for row in re.finditer(
        r"\|\s*(trivial_bad|baseline|good)\s*\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*\*?\*?([-\d.+\w]+)\*?\*?\s*\|",
        text,
    ):
        name, val = row.group(1), row.group(2)
        if wanted[name] is None:
            try:
                wanted[name] = float("-inf") if val in ("-inf", "-∞") else float(val)
            except ValueError:
                pass
    return wanted


def main() -> None:
    canary = _extract_canary_metrics()
    if BASELINE_JSON.exists():
        baseline_payload = json.loads(BASELINE_JSON.read_text())
        lcf_baseline = baseline_payload.get("lcf_judge", 0.0)
    else:
        baseline_payload = {"asal_commit_sha": "(pending)"}
        lcf_baseline = 0.0

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9])
    ax_ladder = fig.add_subplot(gs[0, 0])
    ax_pipe = fig.add_subplot(gs[0, 1])
    ax_card = fig.add_subplot(gs[1, :])

    # ── Panel (a): discriminability ladder ─────────────────────────────────
    names = ["trivial_bad\n(NaN params)", "random search\n(3-min budget)",
             "Sep-CMA-ES\n(3-min budget)", "ASAL baseline\n(30-min, frozen)", "target\n(> 0)"]
    floor = -0.35  # plotting floor for -inf
    vals_raw = [
        canary.get("trivial_bad", float("-inf")),
        canary.get("baseline", 0.0),
        canary.get("good", 0.0),
        lcf_baseline,  # raw LCF_judge, not delta — to show absolute positions
        lcf_baseline + 0.05,  # target = beat by ≥ 0.05
    ]
    vals = [floor if v == float("-inf") else v for v in vals_raw]
    colors = ["#b03030", "#c08030", "#7090c0", "#2060a0", "#209060"]
    bars = ax_ladder.bar(range(len(names)), vals, color=colors, edgecolor="black", linewidth=0.8)
    for i, (b, v, vr) in enumerate(zip(bars, vals, vals_raw)):
        label = "−∞" if vr == float("-inf") else f"{vr:.3f}"
        ax_ladder.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                       label, ha="center", va="bottom", fontsize=9)
    ax_ladder.axhline(lcf_baseline, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_ladder.text(len(names) - 0.3, lcf_baseline + 0.005, "baseline anchor",
                   fontsize=8, color="black", va="bottom", ha="right")
    ax_ladder.set_xticks(range(len(names)))
    ax_ladder.set_xticklabels(names, fontsize=8)
    ax_ladder.set_ylabel("LCF_judge  (↑ maximize)")
    ax_ladder.set_title("(a) Discriminability ladder", fontsize=11, loc="left")
    ax_ladder.set_ylim(floor - 0.05, max(vals) + 0.08)
    ax_ladder.grid(True, axis="y", alpha=0.3)

    # ── Panel (b): pipeline diagram ────────────────────────────────────────
    ax_pipe.set_xlim(0, 10); ax_pipe.set_ylim(0, 10); ax_pipe.set_aspect("equal")
    ax_pipe.set_title("(b) eval pipeline", fontsize=11, loc="left")
    ax_pipe.axis("off")

    def _box(x, y, w, h, label, fc="#f0f0f0"):
        ax_pipe.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                                                   boxstyle="round,pad=0.1",
                                                   facecolor=fc, edgecolor="black"))
        ax_pipe.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    def _arrow(x1, y1, x2, y2):
        ax_pipe.annotate("", xy=(x2, y2), xytext=(x1, y1),
                         arrowprops=dict(arrowstyle="->", lw=1.5))

    _box(0.2, 7.5, 2.3, 1.5, "solution.py\nsearch()", fc="#d8e8f8")
    _box(3.5, 7.5, 2.8, 1.5, "A100\nrollout+render\n(block_network)", fc="#e8f0d8")
    _box(7.2, 7.5, 2.5, 1.5, "16 × 5-frame\nPNG strips\n+ sanitize", fc="#e8f0d8")
    _box(3.5, 4.5, 2.8, 1.5, "CPU judge\nSonnet 4.6\n× 3 × 16", fc="#f8e0d0")
    _box(7.2, 4.5, 2.5, 1.5, "aggregate\nmedian × gmean\n× trim", fc="#f8e0d0")
    _box(3.5, 1.5, 6.2, 1.5, "METRIC = LCF_theta − LCF_baseline\n+ Wilcoxon p  + per-tier medians + CI",
         fc="#fff0c0")

    _arrow(2.5, 8.25, 3.5, 8.25)
    _arrow(6.3, 8.25, 7.2, 8.25)
    _arrow(8.45, 7.5, 8.45, 6.0); _arrow(8.45, 6.0, 6.3, 5.25)
    _arrow(6.3, 5.25, 7.2, 5.25)
    _arrow(8.45, 4.5, 8.45, 3.0); _arrow(8.45, 3.0, 6.6, 3.0)

    # ── Panel (c): metric card ─────────────────────────────────────────────
    ax_card.axis("off")
    ax_card.set_title("(c) metric card — lcf_judge_heldout (eval-v1)", fontsize=11, loc="left")
    card_text = (
        "Name:         lcf_judge_heldout                     Direction: maximize\n"
        "Formula:      per-seed  tier_scalar = gmean(median_{k=1..3} judge(strip))\n"
        "              LCF_theta  = trimmed_mean_10pct_{s ∈ S_test} tier_scalar\n"
        "              METRIC     = LCF_theta − LCF_baseline        # paired-Wilcoxon p-value reported\n"
        "Direction:    maximize (pass bar: METRIC > 0 at p < 0.05 over 16 held-out seeds)\n"
        "Judge:        Claude Sonnet 4.6 @ temperature=0, median-of-3, rubric SHA-pinned\n"
        "Baseline:     pinned ASAL Sep-CMA-ES + supervised-target-CLIP, PRNGKey(0), pop=16\n"
        "Dataset:      16 held-out seeds (jnp.arange(16)+9000) × 5 frames per rollout × 128×128\n"
        "Rubric tiers: existence · agency · robustness · reproduction · coherence   (geometric-mean)\n"
        "Pitfalls:     text-in-image (sanitize+blur+OCR canary) · rubric drift (SHA-pin) · non-finite params\n"
        "              (Guard 8) · judge drift (model-version bumps eval-vN) · prompt collapse (rubric structured JSON)\n"
        "Eval-v1:      10 adversary guards active; quick-canary 4/4 gates pass; baseline anchor frozen."
    )
    ax_card.text(0.01, 0.98, card_text, transform=ax_card.transAxes,
                 fontfamily="monospace", fontsize=8.5, va="top")

    fig.suptitle("Eval-v1 — Life-Cycle Fidelity under VLM Judge",
                 fontsize=13, fontweight="bold")
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"[schematic] wrote {FIG_PATH}  (baseline lcf_judge = {lcf_baseline:.4f})")


if __name__ == "__main__":
    main()
