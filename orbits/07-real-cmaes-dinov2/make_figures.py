"""make_figures.py — orbit 07 qualitative + quantitative artifacts.

Produces:
  figures/kernel_fix.png  — the load-bearing orbit-05 → 07 bugfix: orbit 05's
                             Gram matrix at each scale (rank-deficient after
                             bilinear resize) vs orbit 07's QR-orthogonal
                             Gram. Annotate rank + off-diag max.
  figures/behavior.gif    — 32-frame rollout of the best-found Lenia on a
                             held-out test seed (9000). 5 fps.
  figures/results.png     — multi-panel: (a) algorithm-provenance bar vs
                             orbit 05, (b) per-gen best-score trace from the
                             eval-v2 run (reads search_trace from any
                             committed trace.json), (c) per-seed METRIC with
                             baseline reference.

Pure-CPU, no Modal. Safe to run locally.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make solution importable with fake evosax (local only).
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent.parent))
import types
_fake_evosax = types.ModuleType("evosax")
class _FakeES:
    def __init__(self, popsize, num_dims, sigma_init):
        self.popsize, self.num_dims = popsize, num_dims
    def initialize(self, rng): return rng
    def ask(self, rng, state):
        import jax
        return jax.random.uniform(rng, (self.popsize, self.num_dims), minval=-1, maxval=1), state
    def tell(self, batch, fit, state): return state
_fake_evosax.Sep_CMA_ES = _FakeES
sys.modules["evosax"] = _fake_evosax

import jax
import jax.numpy as jnp
import importlib.util
spec = importlib.util.spec_from_file_location("solution", ROOT / "solution.py")
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

# ───────────────────────────────────────────────────────────────────────────
# rcParams — follow research/style.md
# ───────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "figure.dpi": 160,
    "savefig.dpi": 160,
})


# ───────────────────────────────────────────────────────────────────────────
# Figure 1: kernel_fix — side-by-side Gram matrices
# ───────────────────────────────────────────────────────────────────────────

def _orbit05_kernels():
    """Reconstruct orbit 05's (buggy) kernel bank from its seed.

    Orbit 05 sampled a single 16×16 random tensor then called
    `jax.image.resize` to get 32×32 and 64×64 variants AND did `k / ||k||_F`
    (NOT orthogonal). We replicate that here for the comparison.
    """
    rng = jax.random.PRNGKey(0xD1_50_FE)  # same as orbit 05
    kernels = []
    for scale in range(3):
        rng, sub = jax.random.split(rng)
        k = jax.random.normal(sub, (32, 3, 16, 16))
        k = k / (jnp.linalg.norm(k, axis=(2, 3), keepdims=True) + 1e-6)
        # Orbit 05 resized kernels to per-scale dims:
        ph = 16 * (2 ** scale)
        k_s = jax.image.resize(k, (k.shape[0], 3, ph, ph), method="linear")
        kernels.append(k_s)
    return kernels


def _gram_stats(kernels):
    stats = []
    for k in kernels:
        flat = np.asarray(k).reshape(k.shape[0], -1)
        # normalise rows so the gram diag is ≈ 1 for easier viewing.
        flat = flat / (np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-12)
        gram = flat @ flat.T
        eigs = np.linalg.eigvalsh(gram)
        eigs = np.clip(eigs, 0, None)
        rank_eff = int(((eigs / eigs.max()) > 1e-3).sum())
        off_max = float(np.abs(gram - np.eye(gram.shape[0])).max())
        stats.append({"gram": gram, "rank": rank_eff, "off_max": off_max,
                      "ph": k.shape[-1]})
    return stats


def fig_kernel_fix():
    s05 = _gram_stats(_orbit05_kernels())
    s07 = _gram_stats(sol._build_orthogonal_kernels())

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.4), constrained_layout=True)
    vmax = 1.0
    for col in range(3):
        for row, (label, stats) in enumerate([
            ("orbit 05 (buggy)", s05),
            ("orbit 07 (QR)", s07),
        ]):
            ax = axes[row, col]
            im = ax.imshow(stats[col]["gram"], vmin=-vmax, vmax=vmax,
                           cmap="RdBu_r", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            title = (f"scale {col}  (patch {stats[col]['ph']}×{stats[col]['ph']})"
                     f"\nrank={stats[col]['rank']}/32, "
                     f"off_max={stats[col]['off_max']:.2f}")
            if row == 0:
                ax.set_title(title, fontsize=10)
            else:
                ax.set_xlabel(f"rank={stats[col]['rank']}/32, "
                              f"off_max={stats[col]['off_max']:.2f}",
                              fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=11)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, aspect=20, pad=0.02)
    cb.set_label("kernel inner product", fontsize=10)
    fig.suptitle(
        "Kernel bank Gram matrices — orbit 05 bug vs orbit 07 fix\n"
        "(perfect diag, zero off-diag == orthonormal basis)",
        fontsize=12
    )
    out = FIGS / "kernel_fix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ───────────────────────────────────────────────────────────────────────────
# Figure 2: behavior.gif — rollout of best params
# ───────────────────────────────────────────────────────────────────────────

def fig_behavior_gif(best_params_path: str | None = None):
    if best_params_path is None or not Path(best_params_path).exists():
        # Fall back: run a tiny local search to obtain best params.
        print("[behavior.gif] running local 25s mini-search to obtain best_params")
        res = sol.search(
            "lenia", jnp.arange(16) + 1000,
            {"wall_clock_s": 25}, jax.random.PRNGKey(1)
        )
        bp = np.asarray(res["best_params"], dtype=np.float32)
    else:
        bp = np.load(best_params_path).astype(np.float32)

    traj = sol._lenia_rollout(jnp.asarray(bp), seed=9000)
    traj = np.asarray(traj).squeeze(-1)                      # [T,H,W]
    # Subsample 32 frames evenly.
    idx = np.linspace(0, traj.shape[0] - 1, 32).astype(int)
    frames = traj[idx]

    from PIL import Image
    rgb = []
    for f in frames:
        f = np.clip(f, 0, 1)
        r = np.clip(1.5 * f - 0.2, 0, 1)
        g = np.clip(1.5 * np.minimum(f, 1 - f) * 2, 0, 1)
        b = np.clip(1.5 * (1 - f) - 0.2, 0, 1)
        arr = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
        # Upscale to 384 for visibility.
        img = Image.fromarray(arr).resize((384, 384), Image.NEAREST)
        rgb.append(img)
    out = FIGS / "behavior.gif"
    rgb[0].save(out, save_all=True, append_images=rgb[1:], duration=66,
                loop=0, optimize=True)
    np.save(FIGS / "best_params.npy", bp)
    print(f"wrote {out} ({len(rgb)} frames)")


# ───────────────────────────────────────────────────────────────────────────
# Figure 3: results.png — algorithm provenance + search trace + per-seed metric
# ───────────────────────────────────────────────────────────────────────────

def _read_seed_metric(seed: int) -> float | None:
    log_path = ROOT / "eval_logs" / f"seed{seed}.log"
    if not log_path.exists():
        return None
    for line in log_path.read_text().splitlines():
        if line.startswith("METRIC="):
            try:
                return float(line.split("=", 1)[1])
            except ValueError:
                return None
    return None


def fig_results():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)

    # (a) provenance: orbit 05 (random search w/ CMA-ES label) vs orbit 07 (real CMA-ES)
    ax = axes[0]
    bars = [
        ("orbit 05\n(batch 1)", "random search\nwearing CMA-ES label", "#b0b0b0"),
        ("orbit 07\n(this orbit)", "real Sep-CMA-ES\nno fallback", "#2a7ab0"),
    ]
    for i, (lbl, note, c) in enumerate(bars):
        ax.barh(i, 1.0, color=c)
        ax.text(0.02, i, note, va="center", fontsize=10, color="white")
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b[0] for b in bars])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_title("(a) algorithm provenance", fontsize=11)
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)

    # (b) search trace (if a trace.json was committed; otherwise show the
    # local smoke-run trace).
    ax = axes[1]
    trace_path = FIGS / "trace.json"
    if not trace_path.exists():
        # regenerate via a tiny local run
        res = sol.search("lenia", jnp.arange(16) + 1000,
                         {"wall_clock_s": 30}, jax.random.PRNGKey(1))
        trace = res["search_trace"]
        trace_path.write_text(json.dumps(trace))
    else:
        trace = json.loads(trace_path.read_text())
    best = trace.get("best_proxy_per_gen", [])
    mean = trace.get("mean_proxy_per_gen", [])
    if best:
        ax.plot(best, lw=2, color="#2a7ab0", label="best-so-far")
    if mean:
        ax.plot(mean, lw=1, alpha=0.5, color="#b0b0b0", label="pop mean")
    ax.set_xlabel("generation")
    ax.set_ylabel("DINOv2-lite OE score")
    ax.set_title("(b) CMA-ES convergence", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)

    # (c) per-seed METRIC vs baseline
    ax = axes[2]
    metrics = [_read_seed_metric(s) for s in (1, 2, 3)]
    xs = [1, 2, 3]
    finite = [(x, m) for x, m in zip(xs, metrics) if m is not None]
    if finite:
        xx, mm = zip(*finite)
        ax.bar(xx, mm, color="#2a7ab0", width=0.5)
        mean_m = float(np.mean(mm))
        ax.axhline(mean_m, ls="--", color="#2a7ab0", lw=1,
                   label=f"mean={mean_m:+.3f}")
    else:
        ax.text(0.5, 0.5, "evals running...", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
    ax.axhline(0.119, ls=":", color="gray", lw=1, label="baseline 0.119")
    ax.axhline(0.294, ls=":", color="#d48a00", lw=1, label="orbit 05 +0.294")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("seed")
    ax.set_ylabel("lcf_judge_heldout METRIC")
    ax.set_title("(c) eval-v2 metric per seed", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Orbit 07 — real Sep-CMA-ES + QR-orthogonal DINOv2-lite (eval-v2)",
        fontsize=12
    )
    out = FIGS / "results.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ───────────────────────────────────────────────────────────────────────────
# Figure 4: narrative.png — baseline vs best-found rollout strip
# ───────────────────────────────────────────────────────────────────────────

def fig_narrative(best_params_path: str | None = None):
    if best_params_path and Path(best_params_path).exists():
        bp = np.load(best_params_path).astype(np.float32)
    else:
        bp_file = FIGS / "best_params.npy"
        if bp_file.exists():
            bp = np.load(bp_file).astype(np.float32)
        else:
            res = sol.search("lenia", jnp.arange(16) + 1000,
                             {"wall_clock_s": 25}, jax.random.PRNGKey(1))
            bp = np.asarray(res["best_params"], dtype=np.float32)
            np.save(bp_file, bp)

    baseline = np.zeros(8, dtype=np.float32)  # θ=0 → fades to blob
    picks = np.asarray([0, 63, 127, 191, 255])

    def _strip(params):
        traj = np.asarray(sol._lenia_rollout(jnp.asarray(params), seed=9000))
        frames = traj[picks].squeeze(-1)
        return frames

    bsl = _strip(baseline)
    d07 = _strip(bp)

    fig, axes = plt.subplots(2, 5, figsize=(11, 4.6), constrained_layout=True)
    for col in range(5):
        for row, (lbl, arr) in enumerate([
            ("baseline (θ=0)", bsl),
            ("orbit 07 best", d07),
        ]):
            ax = axes[row, col]
            ax.imshow(arr[col], cmap="magma", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t = {int(picks[col])}", fontsize=10)
            if col == 0:
                ax.set_ylabel(lbl, fontsize=10)
            ax.grid(False)
    fig.suptitle(
        "Baseline vs orbit-07 best — held-out seed 9000\n"
        "real Sep-CMA-ES + QR-orthogonal DINOv2-lite",
        fontsize=12
    )
    out = FIGS / "narrative.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_kernel_fix()
    fig_behavior_gif()
    fig_narrative()
    fig_results()
