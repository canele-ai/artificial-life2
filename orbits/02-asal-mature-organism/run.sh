#!/usr/bin/env bash
# run.sh — reproduce orbit 02-asal-mature-organism from seed.
#
# Dispatches `solution.search()` via the frozen evaluator on Modal (A100),
# rolls out on 16 held-out seeds, judges with Claude Sonnet 4.6, prints
# METRIC=<float>. Expected wall-clock ~32 min per seed (includes 30-min
# search budget + ~90 s rollout + judge).
#
# Usage:
#   bash orbits/02-asal-mature-organism/run.sh          # full 30-min budget, seed 1
#   SEED=2 bash orbits/02-asal-mature-organism/run.sh   # seed 2
#   BUDGET=180 bash orbits/02-asal-mature-organism/run.sh   # 3-min canary budget

set -euo pipefail

SEED="${SEED:-1}"
BUDGET="${BUDGET:-1800}"
SUBSTRATE="${SUBSTRATE:-lenia}"
RUN_ID="${RUN_ID:-mature_${SUBSTRATE}_s${SEED}_b${BUDGET}}"

cd "$(git rev-parse --show-toplevel)"
uv run python3 research/eval/evaluator.py \
  --solution orbits/02-asal-mature-organism/solution.py \
  --substrate "$SUBSTRATE" \
  --seed "$SEED" \
  --run-id "$RUN_ID" \
  --search-budget-s "$BUDGET"
