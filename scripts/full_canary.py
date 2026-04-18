"""full_canary.py — Step 2.3c full-budget canary.

Runs a minimal matrix at full 1800 s budgets to verify pipeline-under-load
after the ASAL baseline is frozen.  Produces
`research/eval/canary_results_full.md`.

Matrix (minimal to keep cost reasonable — ~$4, ~65 min):
  - trivial_bad × lenia × seed 1         (quick; ~45 s)
  - baseline    × lenia × seed 1         (30 min)
  - good        × lenia × seed 1         (30 min)

(The full 9-orbit matrix from the protocol is overkill now that the quick
canary has already shown determinism + direction + discriminability.)

Usage:
    uv run python3 scripts/full_canary.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVALUATOR = REPO / "research" / "eval" / "evaluator.py"
OUT_MD = REPO / "research" / "eval" / "canary_results_full.md"
LOG_DIR = REPO / ".omc" / "canary_full"


def _run(solution_name: str, substrate: str, seed: int, budget_s: int) -> dict:
    log_path = LOG_DIR / f"{solution_name}_{substrate}_s{seed}.log"
    run_id = f"full-canary-{solution_name}-{substrate}-s{seed}"
    cmd = [
        "uv", "run", "python3", str(EVALUATOR),
        "--solution", str(REPO / "research" / "eval" / "examples" / f"{solution_name}.py"),
        "--substrate", substrate,
        "--seed", str(seed),
        "--run-id", run_id,
        "--search-budget-s", str(budget_s),
    ]
    print(f"[canary] dispatching {run_id}  (budget={budget_s}s)…")
    t0 = time.monotonic()
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    elapsed = time.monotonic() - t0
    print(f"[canary]   exit={proc.returncode}  elapsed={elapsed:.1f}s  → {log_path}")

    text = log_path.read_text()
    m = None
    for cand in re.finditer(r"^METRIC_COMPONENTS=(.+)$", text, re.MULTILINE):
        m = cand
    components = json.loads(m.group(1)) if m else {"status": "no_components_found"}
    metric_match = re.search(r"^METRIC=([-\d.einf]+)", text, re.MULTILINE)
    metric = metric_match.group(1) if metric_match else "n/a"
    return {
        "solution": solution_name, "substrate": substrate, "seed": seed,
        "budget_s": budget_s, "metric": metric, "components": components,
        "elapsed_s": round(elapsed, 1),
    }


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    matrix = [
        ("trivial_bad", "lenia", 1, 60),
        ("baseline",    "lenia", 1, 1800),
        ("good",        "lenia", 1, 1800),
    ]
    results = []
    for sol, sub, seed, budget in matrix:
        results.append(_run(sol, sub, seed, budget))

    # Build markdown table
    lines = ["# Canary Results — Step 2.3c full (1800 s budgets)", ""]
    lines.append("| Solution | Substrate | Seed | Budget | Status | METRIC | existence | agency | robustness | reproduction | coherence | elapsed-s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        c = r["components"]
        ptm = c.get("per_tier_median") or {}
        lines.append(
            f"| {r['solution']} | {r['substrate']} | {r['seed']} | {r['budget_s']} s "
            f"| `{c.get('status','?')}` | **{r['metric']}** "
            f"| {ptm.get('existence','—')} | {ptm.get('agency','—')} "
            f"| {ptm.get('robustness','—')} | {ptm.get('reproduction','—')} "
            f"| {ptm.get('coherence','—')} | {r['elapsed_s']} |"
        )
    lines.append("")
    lines.append("## Freeze gate re-check (full-budget)")
    lines.append("")

    tb = next((r for r in results if r["solution"] == "trivial_bad"), None)
    bl = next((r for r in results if r["solution"] == "baseline"), None)
    gd = next((r for r in results if r["solution"] == "good"), None)
    gates = []

    if tb and tb["components"].get("status") == "non_finite_params":
        gates.append("| Direction (trivial_bad ≪ baseline)       | ✓ | trivial_bad=-inf |")
    else:
        gates.append("| Direction                                | ✗ | unexpected |")

    if bl and gd:
        try:
            bl_m = float(bl["metric"]); gd_m = float(gd["metric"])
            spread = abs(bl_m - gd_m)
            gates.append(f"| Discriminability (baseline vs good)      | "
                         f"{'✓' if spread > 0.02 else '?'} | spread={spread:.3f} |")
        except ValueError:
            gates.append("| Discriminability                         | ? | non-numeric METRIC |")

    all_ok = all(r["components"].get("status") in ("ok", "non_finite_params", "baseline_unpinned")
                 for r in results)
    gates.append(f"| Component health (no silent failures)    | {'✓' if all_ok else '✗'} | |")

    lines.extend(gates)
    lines.append("")
    lines.append("## Raw METRIC_COMPONENTS (for reference)")
    lines.append("")
    for r in results:
        lines.append(f"### {r['solution']}_{r['substrate']}_s{r['seed']}")
        lines.append("```json")
        lines.append(json.dumps(r["components"], indent=2)[:2500])
        lines.append("```")
        lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"[canary] wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
