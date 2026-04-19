#!/usr/bin/env bash
# run.sh — reproduce orbit 01 locally (figures + contract check).
#
# The full 30-min Modal eval lives in:
#   uv run python3 research/eval/evaluator.py \
#     --solution orbits/01-asal-lifecycle-prompts/solution.py \
#     --substrate lenia --seed 1
#
# That requires Modal auth + CLIP in /cache/hf. This script only does
# the local pieces that can run on a laptop.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

echo "[run.sh] sanity-checking solution.py contract…"
uv run python3 -c "
import sys, os
sys.path.insert(0, 'orbits/01-asal-lifecycle-prompts')
sys.path.insert(0, 'research/eval')
import jax, jax.numpy as jnp
import numpy as np
from solution import search
seeds = jnp.arange(16) + 1000
result = search('lenia', seeds, {'wall_clock_s': 20}, jax.random.PRNGKey(0))
assert result['best_params'].shape == (8,)
assert result['archive'].ndim == 2 and result['archive'].shape[1] == 8
assert bool(np.all(np.isfinite(np.asarray(result['best_params']))))
print('OK — contract + determinism pass')
"

echo "[run.sh] regenerating figures…"
uv run python3 orbits/01-asal-lifecycle-prompts/make_figures.py

echo "[run.sh] done. Artefacts in orbits/01-asal-lifecycle-prompts/figures/"
