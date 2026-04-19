#!/usr/bin/env bash
# run.sh — reproduce the local CPU smoke test + figures from scratch.
# Modal eval is out of scope for this script; it's invoked by the campaign
# orchestrator via research/eval/evaluator.py.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

cd "$ROOT"

# 1. Smoke test + save /tmp artifacts.
uv run --with "jax[cpu]" --with numpy --with scipy python3 -c "
import sys, time, platform, resource, gc
sys.path.insert(0, 'orbits/04-random-search-dense')
import jax, jax.numpy as jnp
import numpy as np
from solution import search
seeds = jnp.arange(16) + 1000
scale = 1024*1024 if platform.system()=='Darwin' else 1024
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t0 = time.monotonic()
r = search('lenia', seeds, {'wall_clock_s': 60}, jax.random.PRNGKey(0))
elapsed = time.monotonic()-t0
gc.collect()
rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
tr = r['search_trace']
print('algo:', tr['algorithm'])
print('n_candidates:', tr['n_candidates'])
print(f'elapsed: {elapsed:.1f}s, rss_delta_MB: {(rss1-rss0)/scale:.1f}')
print('best_score:', tr['best_proxy_per_gen'][-1])
np.save('/tmp/smoke_all_scores.npy', np.array(tr['all_scores'], np.float32))
np.save('/tmp/smoke_best_proxy.npy', np.array(tr['best_proxy_per_gen'], np.float32))
np.save('/tmp/smoke_best_params.npy', np.asarray(r['best_params']))
"

# 2. Figures.
uv run --with "jax[cpu]" --with numpy --with scipy --with matplotlib --with pillow \
  python3 orbits/04-random-search-dense/make_figures.py

echo "done — see orbits/04-random-search-dense/figures/"
