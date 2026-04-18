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

def load_config(repo_root):
    """Load research/config.yaml with RE_* env overrides."""
    config_path = Path(repo_root) / "research" / "config.yaml"
    if not config_path.exists():
        return {}

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
    """Read log.md frontmatter from an orbit branch via git show."""
    name = branch.removeprefix("orbit/")
    content = git(f"show {branch}:orbits/{name}/log.md", repo_root)
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
    r"<!--\s*RE:EVAL\s+orbit=(\S+)\s*-->.*?"
    r"\*\*Result:\*\*\s*(VERIFIED|MISMATCH).*?"
    r"\*\*Measured:\*\*\s*([\d.eE+-]+).*?"
    r"\*\*Seeds:\*\*\s*(\d+)/(\d+)",
    re.DOTALL,
)

RE_REVIEW_PATTERN = re.compile(
    r"<!--\s*RE:REVIEW\s+orbit=(\S+)\s*-->.*?"
    r"\*\*Code quality:\*\*\s*(.+?)(?:\n|\r)",
    re.DOTALL,
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
        result["eval"][orbit_name] = {
            "result": m.group(2),
            "measured": float(m.group(3)),
            "seeds_passed": int(m.group(4)),
            "seeds_total": int(m.group(5)),
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

    return {"stale": False, "stale_reason": None, "action": None}


# ---------------------------------------------------------------------------
# Orbit state assembly
# ---------------------------------------------------------------------------

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
    }
    if frontmatter:
        orbit["metric"] = frontmatter.get("metric")
        orbit["issue"] = frontmatter.get("issue")
        orbit["parents"] = frontmatter.get("parents", [])
        orbit["eval_version"] = frontmatter.get("eval_version")
    return orbit


def _apply_issue_state(orbit, comments, labels):
    """Apply GitHub-derived state to an orbit record in place."""
    name = orbit["name"]
    if name in comments["eval"]:
        orbit["has_eval"] = True
        orbit["status"] = "complete"
        if comments["eval"][name]["result"] == "MISMATCH":
            orbit["status"] = "mismatch"
    if name in comments["review"]:
        orbit["has_review"] = True
        orbit["advisory_notes"] = comments["review"][name]
    if name in comments["crossval"]:
        orbit["cross_validated"] = True
    orbit["labels"] = labels
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


def _compute_aggregates(orbits, config, repo_root):
    """Compute campaign-level derived fields (best, leaderboard, tags, etc.)."""
    direction = config.get("metric", {}).get("direction", "minimize")
    completed = [o for o in orbits.values() if o["status"] == "complete" and o["metric"] is not None]
    if not completed:
        # Fallback: any orbit with a metric (even if eval-check hasn't run)
        completed = [o for o in orbits.values() if o["metric"] is not None]

    if completed:
        if direction == "minimize":
            best = min(completed, key=lambda o: o["metric"])
        else:
            best = max(completed, key=lambda o: o["metric"])
        best_metric = best["metric"]
        best_orbit = best["name"]
    else:
        best_metric = None
        best_orbit = None

    leaderboard = sorted(
        [o for o in orbits.values() if o["metric"] is not None],
        key=lambda o: o["metric"],
        reverse=(direction == "maximize"),
    )
    unconcluded = [o for o in orbits.values()
                   if "concluded" not in o["labels"] and "graduated" not in o["labels"]]
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
        },
        "best_metric": best_metric,
        "best_orbit": best_orbit,
        "leaderboard": [
            {
                "orbit": o["name"],
                "metric": o["metric"],
                "status": o["status"],
                "cross_validated": o["cross_validated"],
                "labels": o["labels"],
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
        _apply_issue_state(orbit, comments, labels_map.get(name, []))

    return _compute_aggregates(orbits, config, repo_root)


def get_orbit(name, repo_root, *, offline=False):
    """Derive a single orbit's state packet live.

    Uses a single-issue gh fetch (not the batched cross-orbit path). Returns
    None if the orbit branch does not exist.
    """
    branch = f"orbit/{name}"
    if branch not in list_orbit_branches(repo_root):
        return None

    frontmatter = read_orbit_log(branch, repo_root)
    last_commit = get_last_commit_time(branch, repo_root)
    orbit = _init_orbit_record(branch, frontmatter, last_commit)

    if offline or not orbit["issue"]:
        comments = _empty_comments()
        labels = []
    else:
        comments = parse_issue_comments(orbit["issue"], repo_root)
        labels = _fetch_orbit_labels(orbit["issue"], repo_root)

    _apply_issue_state(orbit, comments, labels)
    return orbit


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
        choices=["context", "orbit", "rebuild", "refresh", "audit", "read"],
    )
    parser.add_argument("orbit_name", nargs="?", help="Orbit name (for refresh or orbit)")
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    parser.add_argument("--format", choices=["json", "summary"], default="summary")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip all gh calls. Useful when GitHub is unreachable or unneeded.",
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
