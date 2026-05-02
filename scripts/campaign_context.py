#!/usr/bin/env python3
"""
campaign_context.py — Core state engine for git-evolve.

Derives all campaign state live from git branches + GitHub Issue comments.
No persistent cache — every read queries fresh.

Usage:
    python scripts/campaign_context.py context [--offline] [--repo-root .]
    python scripts/campaign_context.py orbit <name> [--offline] [--repo-root .]
    python scripts/campaign_context.py audit [--repo-root .]

Back-compat shims (may be deprecated in the future):
    python scripts/campaign_context.py rebuild [--repo-root .]  # alias for context; no file write
    python scripts/campaign_context.py refresh <orbit-name>     # no-op in live mode
    python scripts/campaign_context.py read                     # alias for context

Offline mode (--offline) skips all gh calls. Eval/review/crossval/labels are
empty; orbits default to status="running". Use for landscape visualization,
OpenCode prompt injection, or when GitHub is unreachable.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def run(cmd, cwd=None, check=True):
    """Run a shell command and return stdout stripped."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=cwd
    )
    if check and result.returncode != 0:
        return None
    return result.stdout.strip()


def git(cmd, repo_root):
    """Run a git command in the repo root."""
    return run(f"git {cmd}", cwd=repo_root)


def gh(cmd, repo_root):
    """Run a gh CLI command in the repo root."""
    return run(f"gh {cmd}", cwd=repo_root, check=False)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _synthesize_default_config(repo_root):
    """Best-effort synthesis of research/config.yaml when the campaign has an
    evaluator but no config.

    Written for OpenCode / hookless sessions where session-start.mjs didn't fire.
    Copies `templates/config.yaml` into place (idempotent first-run fix), infers
    `metric.direction` + `metric.name` from the evaluator source if possible.
    Returns the resulting config dict, or None if synthesis couldn't proceed
    (no template available or no evaluator present).
    """
    repo_path = Path(repo_root)
    evaluator = None
    for candidate in (
        repo_path / "research" / "eval" / "evaluator.py",
        repo_path / "evaluator.py",
        repo_path / "eval_harness.py",
    ):
        if candidate.exists():
            evaluator = candidate
            break
    if evaluator is None:
        return None

    template_path = Path(__file__).resolve().parent.parent / "templates" / "config.yaml"
    if not template_path.exists():
        return None

    body = template_path.read_text()
    try:
        src = evaluator.read_text()
    except OSError:
        src = ""

    # Direction — literal in a string/arg, else default minimize.
    direction = "minimize"
    m = re.search(r"direction[\s:=\"']+(minimize|maximize)\b", src, re.IGNORECASE)
    if not m:
        m = re.search(
            r"--direction[\s\S]{0,120}?default\s*=\s*[\"'](minimize|maximize)[\"']",
            src,
        )
    if m:
        direction = m.group(1).lower()

    # Metric name — METRIC_NAME = "..." or "metric: name" line.
    name = ""
    nm = re.search(r"METRIC_NAME\s*=\s*[\"']([^\"']+)[\"']", src)
    if nm:
        name = nm.group(1)
    else:
        nm = re.search(r"metric\s*[:=]\s*[\"']?([a-z0-9_\-]+)", src, re.IGNORECASE)
        if nm and nm.group(1).lower() != "value":
            name = nm.group(1)

    if direction != "minimize":
        body = re.sub(r"(^\s*direction:\s*)minimize", rf"\g<1>{direction}", body, count=1, flags=re.MULTILINE)
    if name:
        body = re.sub(r'(^\s*name:\s*)""', rf'\g<1>"{name}"', body, count=1, flags=re.MULTILINE)

    config_path = repo_path / "research" / "config.yaml"
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(body)
        print(
            f"[campaign_context] Synthesized {config_path.relative_to(repo_path)} "
            f"(direction={direction}{', name=' + name if name else ''}).",
            file=sys.stderr,
        )
    except OSError as e:
        print(f"[campaign_context] Failed to write synthesized config: {e}", file=sys.stderr)
        # Still return the parsed dict so the in-memory path works even if disk is read-only.

    return yaml.safe_load(body) or {}


def load_config(repo_root):
    """Load research/config.yaml with RE_* env overrides."""
    config_path = Path(repo_root) / "research" / "config.yaml"
    if not config_path.exists():
        # OpenCode fallback (#1): synthesize from template + evaluator metadata so
        # the first /autorun call doesn't fail on a missing config.
        synthesized = _synthesize_default_config(repo_root)
        if synthesized is None:
            return {}
        config = synthesized
        _normalize_config(config)
        return config

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    # Normalize config: handle both v2 schema (metric/execution) and
    # agent-created schema (problem/search/budget)
    _normalize_config(config)

    # Apply RE_* env overrides to execution section
    execution = config.setdefault("execution", {})
    env_map = {
        "RE_PARALLEL_AGENTS": ("parallel_agents", int),
        "RE_BUDGET": ("budget", float),
        "RE_MAX_ORBITS": ("max_orbits", int),
        "RE_MILESTONE_INTERVAL": ("milestone_interval", int),
        "RE_DESIGN_ITERATIONS": ("design_iterations", int),
        "RE_BRAINSTORM_DEBATE_ROUNDS": ("brainstorm_debate_rounds", int),
        "RE_AUTORUN": ("mode", lambda v: "autorun" if v.lower() in ("1", "true") else "interactive"),
    }
    for env_key, (config_key, transform) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            execution[config_key] = transform(val)

    return config


def _normalize_config(config):
    """Normalize config from agent-created schema to v2 schema.

    Handles:
      problem.direction → metric.direction
      problem.target → metric.target
      problem.metric → metric.name
      search.max_orbits → execution.max_orbits
      budget.max_iterations → execution.max_orbits (approx)
      eval.command → (preserved as-is)
    """
    # metric section: prefer metric.*, fallback to problem.*
    metric = config.setdefault("metric", {})
    problem = config.get("problem", {})
    if not metric.get("direction") and problem.get("direction"):
        metric["direction"] = problem["direction"]
    if not metric.get("target") and problem.get("target"):
        metric["target"] = problem["target"]
    if not metric.get("name") and problem.get("metric"):
        metric["name"] = problem["metric"]

    # execution section: prefer execution.*, fallback to search.*/budget.*
    execution = config.setdefault("execution", {})
    search = config.get("search", {})
    budget_section = config.get("budget", {})
    if not execution.get("max_orbits") and search.get("max_orbits"):
        execution["max_orbits"] = search["max_orbits"]
    if not execution.get("parallel_agents") and search.get("parallel_orbits"):
        execution["parallel_agents"] = search["parallel_orbits"]
    if not execution.get("budget") and budget_section.get("max_iterations"):
        execution["budget"] = budget_section["max_iterations"]


# ---------------------------------------------------------------------------
# Orbit data extraction
# ---------------------------------------------------------------------------

def list_orbit_branches(repo_root):
    """List all orbit/* branches."""
    output = git("branch --list 'orbit/*' --format='%(refname:short)'", repo_root)
    if not output:
        return []
    return [b.strip() for b in output.splitlines() if b.strip()]


def read_orbit_log(branch, repo_root):
    """Read log.md frontmatter from an orbit branch via git show.

    Replicas (orbit/<name>.rN) share the primary's workspace dir
    `orbits/<name>/`, so we look up log.md under the primary name.
    Falls back to the direct `orbits/<branch-name>/log.md` path if
    the workspace doesn't exist at the primary location.
    """
    workspace = primary_name(branch)
    content = git(f"show {branch}:orbits/{workspace}/log.md", repo_root)
    if not content:
        # Back-compat: non-replica branches where name == workspace
        raw = branch.removeprefix("orbit/")
        content = git(f"show {branch}:orbits/{raw}/log.md", repo_root)
    if not content:
        return None

    # Parse YAML frontmatter
    match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return None

    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None

    return frontmatter


def get_last_commit_time(branch, repo_root):
    """Get the last commit time on a branch as ISO-8601."""
    return git(f"log -1 --format=%aI {branch}", repo_root)


def is_replica_branch(branch):
    """Check if this is a cross-validation replica branch."""
    return bool(re.search(r"\.r\d+$", branch))


def primary_name(branch):
    """Get the primary orbit name (strip .rN suffix)."""
    name = branch.removeprefix("orbit/")
    return re.sub(r"\.r\d+$", "", name)


# ---------------------------------------------------------------------------
# Issue comment parsing
# ---------------------------------------------------------------------------

RE_EVAL_PATTERN = re.compile(
    r"<!--\s*RE:EVAL\s+orbit=(\S+?)(?:\s+eval_version=\S+?)?\s*-->.*?"
    r"\*\*Result:\*\*\s*(VERIFIED|MISMATCH).*?"
    # Capture measured value: accept floats AND infinity/nan tokens.
    # Alternation must test word tokens BEFORE the digit class so that
    # "inf" is consumed fully instead of the fallback matching just ".".
    # Without this, ".inf" → group(3)="." → float(".") raises ValueError,
    # and bare "inf" → no match → orbit silently dropped from eval dict.
    # Field name (issue #14): accept the canonical **Measured:** plus the
    # variants that have appeared in the wild — **Metric:** (re-eval scripts),
    # **<N>-seed mean:** (multi-seed re-eval scripts that lead with the
    # aggregate). The canonical-only regex was a footgun: every RE:EVAL with
    # a non-canonical field name silently fell through to has_eval=False.
    # The value itself may be wrapped in optional formatting:
    #   - backticks: `1.23e-2`     (reeval_orbits_v2.py prefers this)
    #   - bold: **0.9559**          (some hand-written / older RE:EVALs)
    #   - italic / plain: 0.9559    (vanilla)
    # All three render fine on GitHub; the parser must not silently care.
    # See `templates/re_eval_comment.md` for the canonical body new scripts
    # should use; the alternation is for backwards-compatibility.
    r"\*\*(?:Measured|Metric|\d+-seed mean):\*\*\s*[`*_]*(\.?inf|\.?nan|null|none|[\d.eE+-]+)[`*_]*"
    # Seeds info is optional and can appear two ways. The strict original
    # regex required `**Seeds:** N/M` on its own line, which caused
    # has_eval to silently stay False whenever agents posted the newer
    # inline form (e.g. `**Measured:** 2.156 (3/3 seeds, ...)`). Accept
    # either form — groups 4/5 hold the inline match, 6/7 hold the
    # explicit-line match, and both being absent still registers the
    # comment as a VERIFIED eval.
    r"(?:[^\n]*?\((\d+)\s*/\s*(\d+)\s+seeds?[^)]*\))?"
    r"(?:.*?\*\*Seeds:\*\*\s*(\d+)\s*/\s*(\d+))?",
    re.DOTALL | re.IGNORECASE,
)

# Per-seed: 0.339, -0.118, -0.121    (or any float list, optionally bracketed)
# Multi-seed contract: the orchestrator reads this array to flag orbits as
# provisional (fewer seeds than required) and to surface seed variance so a
# single outlier seed can't silently become the campaign leader.
RE_PER_SEED_PATTERN = re.compile(
    r"\*\*Per-seed:\*\*\s*\[?([^\n\]]+)\]?",
)

# RE:REVIEW marker. Two forms accepted:
#   Legacy (pre-PR-B): <!-- RE:REVIEW orbit=<n> -->
#   Modern (PR-B):     <!-- RE:REVIEW orbit=<n> verdict=<v> rounds=<n> major=<n> minor=<n> -->
# We accept either by allowing arbitrary `<key>=<value>` tokens after orbit.
# The body match (`**Code quality:**`) is from a legacy reviewer format that the
# current build_re_complete.py no longer emits — kept for compatibility but
# expected to fail-soft (no advisory_notes populated). PR-C will replace this
# with a markdown-section parser that buckets findings by category.
RE_REVIEW_PATTERN = re.compile(
    r"<!--\s*RE:REVIEW\s+orbit=(\S+)(?:\s+\S+=\S+)*\s*-->.*?"
    r"\*\*Code quality:\*\*\s*(.+?)(?:\n|\r)",
    re.DOTALL,
)

# Lightweight marker-only matcher for verdict-aware tooling (label gate,
# milestone synthesizer). Captures the 4 verdict fields when present.
RE_REVIEW_MARKER_PATTERN = re.compile(
    r"<!--\s*RE:REVIEW\s+orbit=(\S+)"
    r"(?:\s+verdict=(ACCEPT|REVISE|BLOCKED)\s+rounds=(\d+)\s+major=(\d+)\s+minor=(\d+))?"
    r"\s*-->"
)

RE_CROSSVAL_PATTERN = re.compile(
    r"<!--\s*RE:CROSSVAL\s+orbit=(\S+)\s*-->",
)


def parse_issue_comments(issue_number, repo_root):
    """Parse structured comments from a GitHub Issue.

    Returns dict with keys: eval, review, crossval (each keyed by orbit name).

    The RE_EVAL / RE_REVIEW patterns are multi-line (HTML comment marker on
    line 1, **Result:** / **Measured:** / **Seeds:** on subsequent lines).
    We therefore run finditer over the whole gh blob — splitting by "\\n"
    first would break every multi-line match.
    """
    result = {"eval": {}, "review": {}, "crossval": set()}

    raw = gh(
        f"issue view {issue_number} --json comments --jq '.comments[].body'",
        repo_root,
    )
    if not raw:
        return result

    # Eval check (multi-line pattern)
    for m in RE_EVAL_PATTERN.finditer(raw):
        orbit_name = m.group(1)
        # Seeds may come from the inline `(N/M seeds, ...)` form (groups
        # 4/5) or the explicit `**Seeds:** N/M` line (groups 6/7); fall
        # back to None when neither is present so downstream code can
        # tell "unknown" apart from "0 of N passed".
        passed_raw = m.group(4) or m.group(6)
        total_raw = m.group(5) or m.group(7)

        # Per-seed array (optional, multi-seed contract). If present, campaign
        # leader promotion gates on len(per_seed) >= search_seeds_required.
        per_seed = []
        block = raw[m.start():m.end() + 400]  # look in a window after the match
        ps_match = RE_PER_SEED_PATTERN.search(block)
        if ps_match:
            for token in ps_match.group(1).split(","):
                token = token.strip()
                if not token:
                    continue
                coerced = _coerce_metric(token)
                if coerced is not None:
                    per_seed.append(coerced)

        result["eval"][orbit_name] = {
            "result": m.group(2),
            "measured": _coerce_metric(m.group(3)),
            "seeds_passed": int(passed_raw) if passed_raw else None,
            "seeds_total": int(total_raw) if total_raw else None,
            "per_seed": per_seed,
        }

    # Advisory review (multi-line pattern)
    for m in RE_REVIEW_PATTERN.finditer(raw):
        orbit_name = m.group(1)
        result["review"][orbit_name] = {
            "code_quality": m.group(2).strip(),
        }

    # Cross-validation (single-line marker)
    for m in RE_CROSSVAL_PATTERN.finditer(raw):
        result["crossval"].add(m.group(1))

    return result


def fetch_all_orbit_comments(orbit_issues, repo_root):
    """Fetch and parse comments for all orbit Issues.

    Args:
        orbit_issues: dict mapping orbit_name -> issue_number
    Returns:
        Aggregated dict with eval/review/crossval keyed by orbit name.
    """
    aggregated = {"eval": {}, "review": {}, "crossval": set()}

    seen_issues = set()
    for orbit_name, issue_num in orbit_issues.items():
        if issue_num in seen_issues or issue_num is None:
            continue
        seen_issues.add(issue_num)

        parsed = parse_issue_comments(issue_num, repo_root)
        aggregated["eval"].update(parsed["eval"])
        aggregated["review"].update(parsed["review"])
        aggregated["crossval"].update(parsed["crossval"])

    return aggregated


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------

STALE_RUNNING_MINUTES = 30
STALE_REVISE_HOURS = 2


def compute_staleness(orbit_name, status, last_commit_at, has_eval, has_review):
    """Compute staleness for an orbit."""
    now = datetime.now(timezone.utc)

    if last_commit_at:
        try:
            last_commit = datetime.fromisoformat(last_commit_at)
            age_minutes = (now - last_commit).total_seconds() / 60
        except (ValueError, TypeError):
            age_minutes = 0
    else:
        age_minutes = float("inf")

    if status == "running" and age_minutes > STALE_RUNNING_MINUTES:
        return {
            "stale": True,
            "stale_reason": f"no commits for {int(age_minutes)}min while running",
            "action": "check if agent died, auto-complete if dead",
        }

    if status == "complete" and not has_eval:
        return {
            "stale": True,
            "stale_reason": "complete but no RE:EVAL comment",
            "action": "re-run eval-check",
        }

    if status == "complete" and has_eval and not has_review:
        # Surface orbits whose eval-check passed but never received an
        # advisory review. /evolve and /autorun both gate "done" on this;
        # exposing it here lets the orchestrator dispatch orbit-reviewer
        # without re-deriving the predicate.
        return {
            "stale": True,
            "stale_reason": "eval-check passed but no RE:REVIEW comment",
            "action": "dispatch orbit-reviewer",
        }

    return {"stale": False, "stale_reason": None, "action": None}


# ---------------------------------------------------------------------------
# Orbit state assembly
# ---------------------------------------------------------------------------

def _coerce_metric(value):
    """Coerce a frontmatter metric value to float (or None).

    Agents write `metric: inf` (bare string) or `metric: .inf` (YAML float)
    inconsistently. PyYAML returns the former as the str "inf" and the latter
    as float('inf'); mixing them later breaks min()/sort with TypeError.
    Normalize here so downstream comparisons are always float-vs-float.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    if s in {"inf", "+inf", ".inf", "+.inf"}:
        return float("inf")
    if s in {"-inf", "-.inf"}:
        return float("-inf")
    if s in {"nan", ".nan"}:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return None


def _init_orbit_record(branch, frontmatter, last_commit):
    """Build the default orbit record from git-only data."""
    name = branch.removeprefix("orbit/")
    orbit = {
        "name": name,
        "branch": branch,
        "is_replica": is_replica_branch(branch),
        "primary": primary_name(branch),
        "metric": None,
        "issue": None,
        "parents": [],
        "eval_version": None,
        "last_commit_at": last_commit,
        "status": "running",  # default, overridden by Issue comments
        "has_eval": False,
        "has_review": False,
        "cross_validated": False,
        "advisory_notes": None,
        "labels": [],
        "stale": False,
        "stale_reason": None,
        # Multi-seed contract (#9): the hook writes per-seed values into RE:EVAL;
        # provisional=True blocks promotion to best_orbit until enough seeds land.
        "per_seed": [],
        "provisional": True,
        # Quality signals (e.g. parse_failures_pct) — surface evaluator health
        # alongside the metric so a leaderboard can flag "high metric but
        # evaluator is broken" without reading the components section by hand.
        "quality_signals": {},
        # env_audit: resolved versions of imported modules, captured by evaluator.
        # Used to detect algorithm-provenance forgery (claims Sep_CMA_ES but
        # evosax is missing from the environment and the orbit silently fell
        # back to random search).
        "env_audit": {},
    }
    if frontmatter:
        orbit["metric"] = _coerce_metric(frontmatter.get("metric"))
        orbit["issue"] = frontmatter.get("issue")
        orbit["parents"] = frontmatter.get("parents", [])
        orbit["eval_version"] = frontmatter.get("eval_version")
    return orbit


def read_eval_verdict_sidecar(orbit_name, repo_root):
    """Fetch `orbits/<name>/eval-verdict.json` from origin/orbit/<name>.

    Written by `scripts/hooks/agent-complete.mjs` after eval-check runs. Carries
    quality_signals, env_audit, and full-precision per_seed that don't travel
    through the RE:EVAL regex. Returns {} if the blob is missing or malformed.
    """
    blob = git(
        f"show origin/orbit/{orbit_name}:orbits/{orbit_name}/eval-verdict.json",
        repo_root,
    )
    if not blob:
        return {}
    try:
        return json.loads(blob) or {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _apply_issue_state(orbit, comments, labels, config=None, repo_root=None):
    """Apply GitHub-derived state to an orbit record in place."""
    name = orbit["name"]
    if name in comments["eval"]:
        eval_entry = comments["eval"][name]
        orbit["has_eval"] = True
        orbit["status"] = "complete"
        if eval_entry["result"] == "MISMATCH":
            orbit["status"] = "mismatch"
        # Multi-seed contract: RE:EVAL comment carries per_seed array; the
        # provisional flag blocks promotion to best_orbit until enough
        # independent seeds have measured this orbit.
        per_seed = eval_entry.get("per_seed") or []
        if per_seed:
            orbit["per_seed"] = per_seed
        required = 1
        if config:
            required = int(((config.get("metric") or {}).get("search_seeds_required") or 1))
        # Effective seed count: prefer the per_seed array length (has full
        # numeric precision), else fall back to the parsed `Seeds: N/M`
        # field from the RE:EVAL comment. Issue #12: orbits with `Seeds: 1/1`
        # but no `Per-seed:` array were silently treated as 0 seeds, so
        # eval_verified stayed False even when search_seeds_required was 1.
        effective_seeds = len(orbit["per_seed"])
        if effective_seeds == 0:
            sp = eval_entry.get("seeds_passed")
            if isinstance(sp, int) and sp > 0:
                effective_seeds = sp
        orbit["provisional"] = effective_seeds < required
        # eval_verified: has a VERIFIED eval AND enough seeds to satisfy
        # search_seeds_required. Used by /doctor Stage 5 and label-apply.
        orbit["eval_verified"] = (
            eval_entry["result"] == "VERIFIED"
            and effective_seeds >= required
        )
    else:
        orbit["eval_verified"] = False
    if name in comments["review"]:
        orbit["has_review"] = True
        orbit["advisory_notes"] = comments["review"][name]
    if name in comments["crossval"]:
        orbit["cross_validated"] = True
    orbit["labels"] = labels
    # Sidecar: eval-verdict.json on orbit branch carries quality_signals + env_audit
    # (emitted by agent-complete.mjs). Only read when the orbit has an eval — no
    # point hitting git for running orbits.
    if repo_root and orbit["has_eval"]:
        sidecar = read_eval_verdict_sidecar(name, repo_root)
        if sidecar:
            if isinstance(sidecar.get("quality_signals"), dict):
                orbit["quality_signals"] = sidecar["quality_signals"]
            if isinstance(sidecar.get("env_audit"), dict):
                orbit["env_audit"] = sidecar["env_audit"]
            # Prefer sidecar per_seed (it has full precision) over parsed RE:EVAL
            if isinstance(sidecar.get("per_seed"), list) and sidecar["per_seed"]:
                orbit["per_seed"] = [float(v) for v in sidecar["per_seed"] if isinstance(v, (int, float))]
                required = 1
                if config:
                    required = int(((config.get("metric") or {}).get("search_seeds_required") or 1))
                orbit["provisional"] = len(orbit["per_seed"]) < required
                # eval_verified re-derived from the richer sidecar view: only
                # True if the RE:EVAL verdict was VERIFIED (not MISMATCH).
                eval_entry = comments["eval"].get(name) or {}
                orbit["eval_verified"] = (
                    eval_entry.get("result") == "VERIFIED"
                    and len(orbit["per_seed"]) >= required
                )
    staleness = compute_staleness(
        name, orbit["status"], orbit["last_commit_at"],
        orbit["has_eval"], orbit["has_review"],
    )
    orbit.update(staleness)


def _fetch_orbit_labels(issue, repo_root):
    """Fetch labels for a single orbit issue via gh CLI."""
    if not issue:
        return []
    labels_raw = gh(
        f"issue view {issue} --json labels --jq '[.labels[].name] | join(\",\")'",
        repo_root,
    )
    if not labels_raw:
        return []
    return [l.strip() for l in labels_raw.split(",") if l.strip()]


def _empty_comments():
    return {"eval": {}, "review": {}, "crossval": set()}


def _has_degraded_quality_signal(orbit):
    """True if any `quality.<name>_range: [lo, hi]` band is violated.

    Convention (#9.2): the evaluator emits paired entries in METRIC_COMPONENTS
    — `quality.parse_failures_pct: 0.9` alongside `quality.parse_failures_pct_range: [0, 0.1]`.
    The hook peels `quality.*` entries into orbit["quality_signals"]; paired
    `_range` entries declare the acceptable band. Out-of-band means the
    evaluator produced an unreliable signal (e.g. 90% judge parse failures);
    the orbit's metric is not research-quality and must not lead.
    """
    signals = orbit.get("quality_signals") or {}
    if not isinstance(signals, dict):
        return False
    for name, value in signals.items():
        if name.endswith("_range"):
            continue
        if not isinstance(value, (int, float)):
            continue
        band = signals.get(f"{name}_range")
        if not (isinstance(band, (list, tuple)) and len(band) == 2):
            continue
        try:
            lo, hi = float(band[0]), float(band[1])
        except (TypeError, ValueError):
            continue
        if value < lo or value > hi:
            return True
    return False


def _compute_aggregates(orbits, config, repo_root):
    """Compute campaign-level derived fields (best, leaderboard, tags, etc.)."""
    direction = config.get("metric", {}).get("direction", "minimize")

    # Finite-float guard: integrator blow-up writes metric=inf, which must not
    # become `best` and must not crash min()/sort (mixed str/float or +/-inf).
    def _is_finite_metric(o):
        m = o.get("metric")
        return isinstance(m, (int, float)) and not isinstance(m, bool) and math.isfinite(m)

    completed = [o for o in orbits.values() if o["status"] == "complete" and _is_finite_metric(o)]
    if not completed:
        # Fallback: any orbit with a finite metric (even if eval-check hasn't run)
        completed = [o for o in orbits.values() if _is_finite_metric(o)]

    # Multi-seed contract gate: orbits measured on too few seeds are provisional
    # and cannot be promoted to `best`. The leaderboard still surfaces them so
    # humans can see pending leader candidates; they just aren't the official
    # campaign leader until CROSSVAL adds the missing seeds.
    # Quality-signal gate (#9.2): if ANY declared quality signal is out-of-band
    # (the evaluator's `quality.<name>_range` was violated), the orbit is also
    # excluded — a high metric from a broken evaluator isn't research signal.
    def _is_promotable(o):
        if o.get("provisional", False):
            return False
        if _has_degraded_quality_signal(o):
            return False
        return True

    promotable = [o for o in completed if _is_promotable(o)]
    pool = promotable or completed  # fall back so we still surface *some* best if all are provisional

    if pool:
        if direction == "minimize":
            best = min(pool, key=lambda o: o["metric"])
        else:
            best = max(pool, key=lambda o: o["metric"])
        best_metric = best["metric"]
        best_orbit = best["name"]
        # Flag when the "best" we could find isn't actually promotable.
        best_provisional = not _is_promotable(best)
    else:
        best_metric = None
        best_orbit = None
        best_provisional = False

    # Leaderboard still surfaces non-finite orbits (so failed runs are visible),
    # but they sort after the finite entries and never become `best_orbit`.
    finite_board = sorted(
        [o for o in orbits.values() if _is_finite_metric(o)],
        key=lambda o: o["metric"],
        reverse=(direction == "maximize"),
    )
    nonfinite_board = [o for o in orbits.values()
                       if o["metric"] is not None and not _is_finite_metric(o)]
    leaderboard = finite_board + nonfinite_board
    # Unconcluded = orbit has not reached a quality-tier verdict AND has not
    # been merged to main. Labels after consolidation: quality tier is one of
    # {winner, promising, dead-end}; `graduated` is applied by /merge. Any
    # other state (no quality tier yet, or needs-human-review) means the
    # orbit is still unconcluded. The retired `concluded` label is ignored.
    _CONCLUSION_LABELS = {"winner", "promising", "dead-end", "graduated"}
    unconcluded = [o for o in orbits.values()
                   if not (_CONCLUSION_LABELS & set(o["labels"]))]
    pending_eval = [o["name"] for o in orbits.values()
                    if o["status"] == "running" or (o["status"] != "mismatch" and not o["has_eval"])]
    action_required = [
        {"orbit": o["name"], "action": o["action"]}
        for o in orbits.values()
        if o["stale"]
    ]

    milestones_raw = git("tag --list 'milestone/*' --sort=-creatordate", repo_root) or ""
    milestones = [t.strip() for t in milestones_raw.splitlines() if t.strip()]
    graduations_raw = git("tag --list 'graduated/*'", repo_root) or ""
    graduations = [t.strip() for t in graduations_raw.splitlines() if t.strip()]

    problem_path = Path(repo_root) / "research" / "problem.md"
    research_question = ""
    if problem_path.exists():
        content = problem_path.read_text()
        for line in content.splitlines():
            if line.strip():
                research_question = line.strip().lstrip("# ")
                break

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_question": research_question,
        "config": {
            "direction": direction,
            "target": config.get("metric", {}).get("target"),
            "budget": config.get("execution", {}).get("budget"),
            "parallel_agents": config.get("execution", {}).get("parallel_agents", 1),
            "search_seeds_required": int(
                (config.get("metric") or {}).get("search_seeds_required") or 1
            ),
            "aggregation": (config.get("metric") or {}).get("aggregation") or "mean",
        },
        "best_metric": best_metric,
        "best_orbit": best_orbit,
        # True when the nominal leader is still provisional (too few seeds) or
        # has a degraded quality signal. Skills read this to decide whether to
        # issue CROSSVAL / pause vs. EXTEND from the current leader.
        "best_provisional": best_provisional,
        "leaderboard": [
            {
                "orbit": o["name"],
                "metric": o["metric"],
                "status": o["status"],
                "cross_validated": o["cross_validated"],
                "labels": o["labels"],
                "provisional": o.get("provisional", False),
                "per_seed": o.get("per_seed", []),
                "quality_signals": o.get("quality_signals", {}),
            }
            for o in leaderboard
        ],
        "active_orbits": [o["name"] for o in orbits.values() if o["status"] == "running"],
        "unconcluded_count": len(unconcluded),
        "pending_eval": pending_eval,
        "milestones": milestones,
        "graduations": graduations,
        "action_required": action_required,
        "orbits": {name: orbit for name, orbit in orbits.items()},
    }


# ---------------------------------------------------------------------------
# Public reader API (live, no disk cache)
# ---------------------------------------------------------------------------

def get_context(repo_root, *, offline=False):
    """Derive the full campaign state packet live from git + GitHub Issues.

    This function does NOT write to disk. It is the primary read path.
    The old rebuild() wrapper keeps its file-write side effect for back-compat,
    but new code should call get_context() directly.

    Args:
        repo_root: path to the campaign repo.
        offline: if True, skip all gh calls. Eval, review, cross-val, and
                 labels will be empty; orbits default to status="running".
                 Use when GitHub is unreachable or unneeded (e.g., landscape
                 visualization, OpenCode prompt injection).
    """
    config = load_config(repo_root)
    branches = list_orbit_branches(repo_root)

    orbits = {}
    orbit_issues = {}
    for branch in branches:
        frontmatter = read_orbit_log(branch, repo_root)
        last_commit = get_last_commit_time(branch, repo_root)
        orbit = _init_orbit_record(branch, frontmatter, last_commit)
        if orbit["issue"]:
            orbit_issues[orbit["name"]] = orbit["issue"]
        orbits[orbit["name"]] = orbit

    if offline:
        comments = _empty_comments()
        labels_map = {name: [] for name in orbits}
    else:
        comments = fetch_all_orbit_comments(orbit_issues, repo_root)
        labels_map = {
            name: _fetch_orbit_labels(issue, repo_root)
            for name, issue in orbit_issues.items()
        }

    for name, orbit in orbits.items():
        _apply_issue_state(
            orbit,
            comments,
            labels_map.get(name, []),
            config=config,
            repo_root=None if offline else repo_root,
        )

    return _compute_aggregates(orbits, config, repo_root)


def get_orbit(name, repo_root, *, offline=False):
    """Derive a single orbit's state packet live.

    Uses a single-issue gh fetch (not the batched cross-orbit path). Returns
    None if the orbit branch does not exist.
    """
    branch = f"orbit/{name}"
    if branch not in list_orbit_branches(repo_root):
        return None

    config = load_config(repo_root)
    frontmatter = read_orbit_log(branch, repo_root)
    last_commit = get_last_commit_time(branch, repo_root)
    orbit = _init_orbit_record(branch, frontmatter, last_commit)

    if offline or not orbit["issue"]:
        comments = _empty_comments()
        labels = []
    else:
        comments = parse_issue_comments(orbit["issue"], repo_root)
        labels = _fetch_orbit_labels(orbit["issue"], repo_root)

    _apply_issue_state(
        orbit,
        comments,
        labels,
        config=config,
        repo_root=None if offline else repo_root,
    )
    return orbit


# ---------------------------------------------------------------------------
# Resume planning
# ---------------------------------------------------------------------------

RESUME_ACTION_COMPLETE = "complete"
RESUME_ACTION_RESPAWN_AGENT = "respawn-agent"
RESUME_ACTION_CONTINUE_AGENT = "continue-agent"
RESUME_ACTION_EVAL_CHECK = "eval-check"
RESUME_ACTION_REVIEW_LOOP = "review-loop"


def _orbit_has_solution(branch, repo_root):
    """Check whether the orbit branch has committed a solution.py.

    Replica branches (orbit/<name>.rN) share their primary's workspace
    directory `orbits/<primary>/`, so we look up solution.py under the
    primary name. A replica without its primary's solution is still
    "respawn" territory.
    """
    workspace = primary_name(branch)
    return bool(git(f"show {branch}:orbits/{workspace}/solution.py", repo_root))


def _orbit_has_critic_verdict(branch, repo_root):
    """Check whether the orbit branch has committed a figure-critic verdict.

    Uses the shared workspace dir (primary name) same as _orbit_has_solution.
    """
    workspace = primary_name(branch)
    return bool(git(f"show {branch}:orbits/{workspace}/figures/.critic-verdict.json", repo_root))


def _orbit_has_reviewer_verdict(branch, repo_root):
    """Check whether the orbit branch has committed an orbit-reviewer verdict.

    /autorun writes this in step 4h alongside .critic-verdict.json. Used as
    an on-branch fallback signal for skills (like /autorun) that defer the
    Issue post to a consolidated RE:COMPLETE comment in step 4j.
    """
    workspace = primary_name(branch)
    return bool(git(f"show {branch}:orbits/{workspace}/figures/.reviewer-verdict.json", repo_root))


def get_resume_plan(repo_root, *, offline=False):
    """Inspect every orbit/* branch and decide what action would move it
    forward. Returns a list of {orbit, branch, issue, action, reason} dicts,
    one per orbit, plus a summary of how many are already complete.

    Actions (in order of "how far the orbit has gotten"):
        respawn-agent   — branch exists but has no committed solution.py
        continue-agent  — solution.py committed but log.md has no metric
        eval-check      — metric in log.md but no RE:EVAL comment yet
        review-loop     — has_eval but missing critic OR orbit-reviewer artifact
        complete        — eval, both reviewer artifacts, and a posted review

    /autorun's Step 0 uses this to skip orbits that are already finished
    and restart only the incomplete ones.
    """
    context = get_context(repo_root, offline=offline)
    plan = []
    for name, orbit in context["orbits"].items():
        branch = orbit["branch"]
        action = None
        reason = ""

        if not _orbit_has_solution(branch, repo_root):
            action = RESUME_ACTION_RESPAWN_AGENT
            reason = "no solution.py committed on orbit branch"
        elif orbit.get("metric") is None:
            action = RESUME_ACTION_CONTINUE_AGENT
            reason = "solution.py exists but log.md has no metric"
        elif not orbit["has_eval"]:
            action = RESUME_ACTION_EVAL_CHECK
            reason = "metric claimed but no RE:EVAL comment on Issue"
        elif not _orbit_has_critic_verdict(branch, repo_root):
            action = RESUME_ACTION_REVIEW_LOOP
            reason = "eval-check verified but no figure-critic verdict file"
        elif not (orbit["has_review"] or _orbit_has_reviewer_verdict(branch, repo_root)):
            # Cross-skill review gate: /evolve posts <!-- RE:REVIEW --> directly
            # (sets has_review); /autorun writes .reviewer-verdict.json on the
            # orbit branch and folds it into the consolidated RE:COMPLETE
            # comment (which embeds the RE:REVIEW marker, also setting
            # has_review). Either signal counts; missing both means the
            # advisory review never landed.
            action = RESUME_ACTION_REVIEW_LOOP
            reason = "no orbit-reviewer verdict (file or RE:REVIEW comment)"
        else:
            action = RESUME_ACTION_COMPLETE
            reason = "has solution, metric, RE:EVAL, critic verdict, and review"

        plan.append({
            "orbit": name,
            "branch": branch,
            "issue": orbit.get("issue"),
            "action": action,
            "reason": reason,
            "metric": orbit.get("metric"),
        })

    # Sort so complete orbits come first (skippable), actionable last
    order = {
        RESUME_ACTION_COMPLETE: 0,
        RESUME_ACTION_REVIEW_LOOP: 1,
        RESUME_ACTION_EVAL_CHECK: 2,
        RESUME_ACTION_CONTINUE_AGENT: 3,
        RESUME_ACTION_RESPAWN_AGENT: 4,
    }
    plan.sort(key=lambda p: order.get(p["action"], 99))
    return {
        "orbits": plan,
        "complete_count": sum(1 for p in plan if p["action"] == RESUME_ACTION_COMPLETE),
        "actionable_count": sum(1 for p in plan if p["action"] != RESUME_ACTION_COMPLETE),
    }


# ---------------------------------------------------------------------------
# Rebuild (back-compat alias for get_context)
# ---------------------------------------------------------------------------

def rebuild(repo_root):
    """Back-compat alias for get_context(). No file is written.

    Retained so existing Python imports keep working during migration.
    New code should call get_context() directly.
    """
    return get_context(repo_root)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit(repo_root):
    """Detect breakpoints from interrupted sessions.

    Reads campaign state live via get_context() and writes a snapshot to
    .re/cache/audit.json (the snapshot is the only disk artifact).
    """
    context = get_context(repo_root)
    breakpoints = []

    for name, orbit in context["orbits"].items():
        branch = orbit["branch"]

        # 1. Dirty worktree check
        worktree_path = Path(repo_root) / ".worktrees" / name
        if worktree_path.exists():
            status = run(f"git -C {worktree_path} status --porcelain", check=False)
            if status:
                breakpoints.append({
                    "type": "dirty_worktree",
                    "orbit": name,
                    "auto_fixable": True,
                    "detail": f"{len(status.splitlines())} uncommitted files",
                    "action": "commit + push",
                })

            # Check if pushed
            local_sha = run(f"git -C {worktree_path} rev-parse HEAD", check=False)
            remote_sha = git(f"rev-parse origin/{branch}", repo_root)
            if local_sha and remote_sha and local_sha != remote_sha:
                breakpoints.append({
                    "type": "unpushed_branch",
                    "orbit": name,
                    "auto_fixable": True,
                    "detail": f"local {local_sha[:8]} != remote {(remote_sha or 'none')[:8]}",
                    "action": "git push",
                })

        # 2. log.md integrity
        frontmatter = read_orbit_log(branch, repo_root)
        if not frontmatter:
            breakpoints.append({
                "type": "missing_log",
                "orbit": name,
                "auto_fixable": False,
                "detail": "log.md missing or invalid frontmatter",
                "action": "investigate orbit branch",
            })
        else:
            required = ["issue", "parents", "eval_version"]
            missing = [f for f in required if f not in frontmatter or frontmatter[f] is None]
            if missing:
                breakpoints.append({
                    "type": "incomplete_log",
                    "orbit": name,
                    "auto_fixable": False,
                    "detail": f"missing frontmatter fields: {', '.join(missing)}",
                    "action": "fix log.md frontmatter",
                })

        # 3. Missing eval-check
        if orbit["status"] == "running" and orbit["stale"]:
            has_metric = orbit.get("metric") is not None
            breakpoints.append({
                "type": "stale_orbit",
                "orbit": name,
                "issue": orbit.get("issue"),       # needed by /evolve step 0 and session-start
                "has_metric": has_metric,
                "auto_fixable": has_metric,        # can rerun eval-check if metric is known
                "detail": orbit["stale_reason"],
                "action": "run eval-check + post RE:EVAL" if has_metric else orbit.get("action", "investigate"),
            })

        # 4. Label sync — orbit has RE:EVAL but label not applied yet
        if orbit["has_eval"] and "evaluated" not in orbit.get("labels", []):
            breakpoints.append({
                "type": "stale_label",
                "orbit": name,
                "issue": orbit.get("issue"),       # needed by session-start auto-fix
                "add_labels": "evaluated",
                "remove_labels": None,
                "auto_fixable": True,
                "detail": "eval-check passed but 'evaluated' label missing",
                "action": "add 'evaluated' label",
            })

    audit_result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "breakpoints": breakpoints,
        "resume_safe": all(bp["auto_fixable"] for bp in breakpoints) if breakpoints else True,
        "auto_fixable_count": sum(1 for bp in breakpoints if bp["auto_fixable"]),
        "manual_count": sum(1 for bp in breakpoints if not bp["auto_fixable"]),
    }

    cache_dir = Path(repo_root) / ".re" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "audit.json", "w") as f:
        json.dump(audit_result, f, indent=2)

    return audit_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="git-evolve campaign context engine")
    parser.add_argument(
        "command",
        choices=["context", "orbit", "resume", "rebuild", "refresh", "audit", "read"],
    )
    parser.add_argument("orbit_name", nargs="?", help="Orbit name (for refresh or orbit)")
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    parser.add_argument("--format", choices=["json", "summary"], default="summary")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip all gh calls. Useful when GitHub is unreachable or unneeded.",
    )
    # Polling args for `resume --wait-for <state> --orbits a,b,c`
    parser.add_argument(
        "--wait-for",
        choices=["solution-ready", "complete"],
        help="For `resume`: block until every listed orbit reaches the given "
             "state. 'solution-ready' = orbit-agent committed solution.py + "
             "metric (action is eval-check / review-loop / complete). "
             "'complete' = figure-critic verdict also present.",
    )
    parser.add_argument(
        "--orbits",
        default="",
        help="For `resume --wait-for`: comma-separated orbit names to watch.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="For `resume --wait-for`: max seconds to poll before erroring out.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="For `resume --wait-for`: seconds between polls.",
    )
    parser.add_argument(
        "--min-ready",
        type=int,
        default=None,
        help="For `resume --wait-for`: exit successfully as soon as at least "
             "this many of the watched orbits reach the target state. Defaults "
             "to len(--orbits), meaning 'all must reach'. Set below the count "
             "to tolerate partial failures in a batch (e.g., K=3, min-ready=2). "
             "Exits 0 on partial success; the remaining orbits are printed to "
             "stderr as 'stragglers' so the caller can mark them needs-human-review.",
    )

    args = parser.parse_args()
    repo_root = os.path.abspath(args.repo_root)

    if args.command == "context":
        # Primary live-read path. Emits JSON to stdout, never writes to disk.
        context = get_context(repo_root, offline=args.offline)
        print(json.dumps(context, indent=2, default=str))

    elif args.command == "orbit":
        if not args.orbit_name:
            print("Error: orbit_name required for 'orbit' command", file=sys.stderr)
            sys.exit(1)
        orbit = get_orbit(args.orbit_name, repo_root, offline=args.offline)
        if orbit is None:
            print(f"Error: orbit/{args.orbit_name} not found", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(orbit, indent=2, default=str))

    elif args.command == "resume":
        # In wait mode, block until every listed orbit reaches the target state.
        if args.wait_for:
            import time
            watch = [n.strip() for n in args.orbits.split(",") if n.strip()]
            if not watch:
                print("Error: --wait-for requires --orbits <a,b,c>", file=sys.stderr)
                sys.exit(2)
            # 'solution-ready' accepts any action past the orbit-agent stage.
            ready_actions_solution = {
                RESUME_ACTION_COMPLETE,
                RESUME_ACTION_REVIEW_LOOP,
                RESUME_ACTION_EVAL_CHECK,
            }
            ready_actions_complete = {RESUME_ACTION_COMPLETE}
            accept = (
                ready_actions_solution
                if args.wait_for == "solution-ready"
                else ready_actions_complete
            )
            # Default: all orbits must reach target state.
            min_ready = args.min_ready if args.min_ready is not None else len(watch)
            min_ready = max(1, min(min_ready, len(watch)))
            deadline = time.monotonic() + args.timeout
            last_plan = None
            while True:
                plan = get_resume_plan(repo_root, offline=args.offline)
                last_plan = plan
                by_name = {e["orbit"]: e for e in plan["orbits"]}
                ready = []
                pending = []
                for name in watch:
                    entry = by_name.get(name)
                    if entry is not None and entry["action"] in accept:
                        ready.append(name)
                    else:
                        pending.append((name, entry["action"] if entry else "missing"))
                if len(ready) >= min_ready:
                    # Done. Print the final plan JSON on stdout, and the ready
                    # vs. straggler lists on stderr for the caller to branch on.
                    print(json.dumps(plan, indent=2, default=str))
                    if pending:
                        print(
                            f"[WAIT] {len(ready)}/{len(watch)} reached '{args.wait_for}' "
                            f"(min-ready={min_ready}). STRAGGLERS: "
                            + ", ".join(f"{n}({a})" for n, a in pending),
                            file=sys.stderr,
                        )
                        print(
                            "[WAIT] ready=" + ",".join(ready),
                            file=sys.stderr,
                        )
                        print(
                            "[WAIT] stragglers=" + ",".join(n for n, _ in pending),
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"[WAIT] all {len(watch)} orbits reached '{args.wait_for}'",
                            file=sys.stderr,
                        )
                    sys.exit(0)
                if time.monotonic() >= deadline:
                    print(
                        f"[WAIT] TIMEOUT after {args.timeout}s — "
                        f"ready={len(ready)}/{len(watch)} (min-ready={min_ready}). "
                        f"Pending: " + ", ".join(f"{n}({a})" for n, a in pending),
                        file=sys.stderr,
                    )
                    print(json.dumps(plan, indent=2, default=str))
                    sys.exit(1)
                print(
                    f"[WAIT] ready={len(ready)}/{len(watch)} "
                    f"(min-ready={min_ready}); pending: "
                    + ", ".join(f"{n}({a})" for n, a in pending),
                    file=sys.stderr,
                )
                time.sleep(args.poll_interval)
        # Non-wait mode: emit a one-shot per-orbit action plan.
        plan = get_resume_plan(repo_root, offline=args.offline)
        print(json.dumps(plan, indent=2, default=str))
        print(
            f"[RESUME] {plan['complete_count']} complete, "
            f"{plan['actionable_count']} actionable",
            file=sys.stderr,
        )
        for entry in plan["orbits"]:
            if entry["action"] != RESUME_ACTION_COMPLETE:
                print(
                    f"  [{entry['action']}] orbit/{entry['orbit']}: {entry['reason']}",
                    file=sys.stderr,
                )

    elif args.command == "rebuild":
        # Back-compat shim: behaves like `context`, no file write.
        print("[REBUILD] live mode — no file written (use `context` directly)", file=sys.stderr)
        context = get_context(repo_root, offline=args.offline)
        if args.format == "json":
            print(json.dumps(context, indent=2, default=str))
        else:
            print(f"[REBUILD] {len(context.get('orbits', {}))} orbits")
            if context["best_orbit"]:
                print(f"[BEST] {context['best_orbit']}: {context['best_metric']}")
            if context["action_required"]:
                print(f"[ACTION REQUIRED] {len(context['action_required'])} stale orbits")

    elif args.command == "refresh":
        # Back-compat shim: no-op in live mode. Every subsequent `context`
        # call reads fresh data from git + GitHub directly.
        if not args.orbit_name:
            print("Error: orbit_name required for refresh", file=sys.stderr)
            sys.exit(1)
        print(f"[REFRESH] {args.orbit_name} (no-op in live mode)", file=sys.stderr)

    elif args.command == "audit":
        result = audit(repo_root)
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            if not result["breakpoints"]:
                print("[AUDIT] Clean — no breakpoints detected")
            else:
                print(f"[AUDIT] {len(result['breakpoints'])} breakpoints:")
                for bp in result["breakpoints"]:
                    fix = "auto-fix" if bp["auto_fixable"] else "MANUAL"
                    print(f"  [{fix}] {bp['orbit']}: {bp['type']} — {bp['detail']}")
                print(f"  Resume safe: {result['resume_safe']}")

    elif args.command == "read":
        # Back-compat shim: routes to live `context` derivation.
        context = get_context(repo_root, offline=args.offline)
        print(json.dumps(context, indent=2, default=str))


if __name__ == "__main__":
    main()
