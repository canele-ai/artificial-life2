"""freeze_baseline.py — Step 2.5: convert an ASAL baseline run's output into
the frozen `research/eval/baseline/baseline_score.json` and pin its SHA256
into evaluator.py as `BASELINE_SHA256`.

Usage:
    uv run python3 scripts/freeze_baseline.py .omc/freeze/asal_baseline_lenia.log

Reads the last METRIC_COMPONENTS line from the log, extracts lcf_theta and
per_seed_scalars, writes baseline_score.json with the canonical schema,
computes SHA256, updates evaluator.py in-place.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO / "research" / "eval" / "baseline" / "baseline_score.json"
EVALUATOR_PATH = REPO / "research" / "eval" / "evaluator.py"


def _extract_last_components(log_path: Path) -> dict:
    text = log_path.read_text()
    match = None
    for m in re.finditer(r"^METRIC_COMPONENTS=(.+)$", text, re.MULTILINE):
        match = m
    if match is None:
        raise SystemExit(f"[freeze] no METRIC_COMPONENTS line found in {log_path}")
    return json.loads(match.group(1))


def _write_baseline(components: dict, asal_note: str) -> tuple[str, dict]:
    if components.get("status") not in ("ok", "baseline_unpinned"):
        raise SystemExit(
            f"[freeze] baseline run did not succeed: status={components.get('status')!r}"
        )
    lcf_judge = float(components["lcf_theta"])
    per_seed = [float(x) for x in components["per_seed_scalars"]]
    if len(per_seed) != 16:
        raise SystemExit(f"[freeze] expected 16 per-seed scalars, got {len(per_seed)}")

    payload = {
        "lcf_judge": lcf_judge,
        "per_seed_tier_scalars": per_seed,
        "asal_commit_sha": asal_note,
        "eval_version": "eval-v1",
        "judge_version": components.get("judge_version", "claude-sonnet-4-6"),
        "rubric_sha": components.get("rubric_sha"),
        "substrate": components.get("substrate"),
        "wall_clock_s": components.get("wall_clock_s"),
        "per_tier_median": components.get("per_tier_median"),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_bytes(serialized)
    sha = hashlib.sha256(serialized).hexdigest()
    return sha, payload


def _pin_sha_in_evaluator(sha: str) -> None:
    text = EVALUATOR_PATH.read_text()
    pattern = re.compile(r'^BASELINE_SHA256:\s*str\s*\|\s*None\s*=\s*(None|"[^"]*")\s*$',
                         re.MULTILINE)
    new_line = f'BASELINE_SHA256: str | None = "{sha}"'
    new_text, n = pattern.subn(new_line, text, count=1)
    if n != 1:
        raise SystemExit("[freeze] failed to locate BASELINE_SHA256 line in evaluator.py")
    EVALUATOR_PATH.write_text(new_text)


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <baseline_run_log>")
        return 1
    log_path = Path(sys.argv[1]).resolve()
    components = _extract_last_components(log_path)

    asal_note = (
        "reimplemented_from_arxiv_2412.17799_asal_metrics.calc_supervised_target_score; "
        "sep_cma_es(popsize=16,sigma=0.1,gens=200); "
        "PRNGKey(0); prompts=pinned_in_asal_baseline.py"
    )
    sha, payload = _write_baseline(components, asal_note)
    _pin_sha_in_evaluator(sha)

    print(f"[freeze] baseline_score.json written")
    print(f"[freeze]   lcf_judge = {payload['lcf_judge']:.6f}")
    print(f"[freeze]   per_seed  min/max = {min(payload['per_seed_tier_scalars']):.4f} / "
          f"{max(payload['per_seed_tier_scalars']):.4f}")
    print(f"[freeze]   wall_clock_s = {payload['wall_clock_s']}")
    print(f"[freeze]   rubric_sha  = {payload['rubric_sha']}")
    print(f"[freeze] SHA256        = {sha}")
    print(f"[freeze] BASELINE_SHA256 pinned in evaluator.py ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
