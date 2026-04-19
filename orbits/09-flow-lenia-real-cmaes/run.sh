#!/usr/bin/env bash
# run.sh — reproduce orbit 09 from seed.
#
# Assumes: Modal authenticated, repo at worktree root, `uv` installed.
# This orbit's solution inherits orbit 03 verbatim; the delta is purely
# pipeline (evosax pip-installed in Modal search_image under eval-v2).

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Regenerate local figures (Flow-Lenia rollout is CPU-fast).
uv run python3 orbits/09-flow-lenia-real-cmaes/make_figures.py

# Eval — 3 seeds, parallel via evaluator's internal Modal dispatch.
# The evaluator emits METRIC=<float> per line; caller aggregates.
for SEED in 1 2 3; do
  uv run python3 research/eval/evaluator.py \
    --solution orbits/09-flow-lenia-real-cmaes/solution.py \
    --substrate flow_lenia \
    --seed "$SEED" \
    --run-id "orbit09_seed${SEED}"
done
