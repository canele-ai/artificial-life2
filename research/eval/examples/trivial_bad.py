"""trivial_bad.py — solution.py stub: returns all-zero params (trivial baseline).

Expected METRIC: well below 0 (dead/blank rollouts score near 0 on all tiers;
baseline is the pinned ASAL result ~0.25–0.40, so METRIC ≈ -0.30).

This is the floor-of-floors example: a solution that does no search at all.
The evaluator's existence gate will assign tier_scalar := 0 for every seed
because blank frames have existence ≈ 0.0.
"""

from __future__ import annotations
from typing import Literal

import jax
import jax.numpy as jnp

# Substrate-dependent parameter dimensionality.
_DIM = {"lenia": 8, "flow_lenia": 8}


def search(
    substrate_name: Literal["lenia", "flow_lenia"],
    seed_pool_train: jax.Array,
    budget: dict,
    rng: jax.Array,
) -> dict:
    """Return all-zero params immediately without doing any search.

    Expected judged score: existence ≈ 0 for every seed → tier_scalar = 0
    for all seeds → LCF_theta ≈ 0 → METRIC ≈ -baseline ≈ -0.30.
    """
    K = _DIM.get(substrate_name, 8)
    best_params = jnp.zeros(K, dtype=jnp.float32)
    return {
        "best_params": best_params,
        "archive": jnp.zeros((1, K), dtype=jnp.float32),
        "search_trace": {"best_proxy_per_gen": [0.0], "note": "trivial_zero"},
    }
