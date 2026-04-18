"""modal_app.py — Modal image + app definitions for the eval-v1 pipeline.

Two containers:
  1. search_container  — A100 GPU, network egress BLOCKED (Guard 2).
                         Runs solution.search() + substrate rollout.
  2. judge_container   — CPU only, anthropic-api-key secret mounted.
                         Calls Claude Sonnet 4.6 judge API.

This module is imported by evaluator.py (via _get_modal_app()) and can also
be used directly with `modal run modal_app.py` for debugging.

Guard mapping:
  Guard 1  — search image does NOT bind-mount research/eval/judge/ or
             research/eval/baseline/.  Baseline scalar injected via env var
             ASAL_BASELINE_LCF only.
  Guard 2  — @app.function(..., block_network=True) on search container.
  Guard 7  — No Haiku prefilter; no strip-SHA cache.
"""

from __future__ import annotations

import modal

# ── Pinned package versions ──────────────────────────────────────────────────
_JAX_VER = "0.4.38"
_JAXLIB_VER = "0.4.38"
_NUMPY_VER = "1.26.4"
_PILLOW_VER = "10.4.0"
_TRANSFORMERS_VER = "4.47.1"
_FLAX_VER = "0.10.2"
_SCIPY_VER = "1.13.1"
_ANTHROPIC_VER = "0.96.0"

# ── Modal Volumes (pre-populated at campaign bootstrap) ──────────────────────
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
jax_cache_vol = modal.Volume.from_name("jax-jit-cache", create_if_missing=True)

# ── Search container image — A100, no judge secrets ──────────────────────────
# Guard 1: repo bind-mount intentionally excludes judge/ and baseline/ paths.
# The evaluator injects ASAL_BASELINE_LCF as an env var scalar (not the file).
search_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        f"jax[cuda12]=={_JAX_VER}",
        f"jaxlib=={_JAXLIB_VER}",
        f"numpy=={_NUMPY_VER}",
        f"pillow=={_PILLOW_VER}",
        f"transformers=={_TRANSFORMERS_VER}",
        f"flax=={_FLAX_VER}",
        f"scipy=={_SCIPY_VER}",
        extra_options=(
            "--find-links "
            "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
        ),
    )
    .env({
        # Deterministic GPU ops for reproducibility (C2 audit requirement).
        "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
        # JAX JIT cache on persistent Modal Volume.
        "JAX_COMPILATION_CACHE_DIR": "/cache/jax",
        # HuggingFace CLIP weights on persistent Modal Volume.
        "HF_HOME": "/cache/hf",
        "TRANSFORMERS_CACHE": "/cache/hf",
        # Float32 matmul precision for reproducibility.
        "JAX_DEFAULT_MATMUL_PRECISION": "float32",
    })
)

# ── Judge container image — CPU only, anthropic SDK, no JAX/GPU ──────────────
judge_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        f"anthropic=={_ANTHROPIC_VER}",
        f"pillow=={_PILLOW_VER}",
        f"scipy=={_SCIPY_VER}",
        f"numpy=={_NUMPY_VER}",
    )
)

# Make evaluator.py available inside both containers.
# (Comments above about "repo bind-mount" were aspirational — Modal requires an
# explicit include. Doing this here keeps evaluator.py importable from both
# container functions without duplicating code.)
search_image = search_image.add_local_python_source("evaluator", "search_worker")
judge_image = judge_image.add_local_python_source("evaluator")

# ── App ───────────────────────────────────────────────────────────────────────
app = modal.App("eval-lcf-judge-v1")

# ── Search function — Guard 2: block_network=True ────────────────────────────
@app.function(
    image=search_image,
    gpu="A100",           # 40 GB; enough for Flow-Lenia 128² + CLIP + CMA-ES pop
    cpu=8.0,
    memory=32768,         # 32 GB RAM
    timeout=3600,         # 1-hour Modal hard kill (safety net above 30-min SIGKILL)
    volumes={
        "/cache/hf": hf_cache_vol,    # CLIP weights (read-only at eval time)
        "/cache/jax": jax_cache_vol,  # JIT cache (read-write)
    },
    block_network=True,   # Guard 2: no outbound network from search container
    # NO secrets mounted here — orbits cannot call api.anthropic.com even if
    # they bypass block_network (no key present).  Judge API key is in judge fn.
)
def search_container(
    solution_code: str,
    substrate_name: str,
    seed: int,
    run_id: str,
    baseline_lcf_scalar: float,
    search_budget_s: int = 1800,
) -> dict:
    """Run orbit search() + substrate rollout on A100.

    Imports evaluator module functions (embedded in the Modal image via
    evaluator.py being part of the repo mount, minus judge/ and baseline/).

    Returns:
      {
        "status": "ok" | "search_timeout" | "search_crash" | "non_finite_params"
                  | "rollout_crash" | "no_best_params",
        "strips": [bytes, ...]   # 16 sanitised PNG strips, or [],
        "env_audit": {...},
      }
    """
    # Deferred import: evaluator.py is available in the image because the
    # campaign repo is mounted (minus judge/ and baseline/).
    import sys
    import os
    import signal
    import subprocess
    import tempfile
    from pathlib import Path

    import numpy as np

    os.environ["ASAL_BASELINE_LCF"] = str(baseline_lcf_scalar)
    os.environ["RE_SEARCH_WALL_CLOCK_S"] = str(int(search_budget_s))

    # Import helpers from evaluator (same package, available in image).
    from evaluator import (
        _rollout_and_render,
        _sanitise_strip,
        _env_audit,
        SEARCH_WALL_CLOCK_S,
        N_TEST_SEEDS,
        TEST_SEED_OFFSET,
    )
    import search_worker  # packaged alongside evaluator in the image

    # Write solution to tmp path
    sol_dir = tempfile.mkdtemp(prefix="sol_")
    sol_file = Path(sol_dir) / "solution.py"
    sol_file.write_text(solution_code)

    worker_file = Path(search_worker.__file__)

    fd, out_path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)

    cmd = [
        sys.executable, str(worker_file),
        str(sol_file), out_path, substrate_name, str(seed), run_id,
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    search_status = "ok"
    stderr_tail = ""
    try:
        _stdout, _stderr = proc.communicate(timeout=SEARCH_WALL_CLOCK_S)
        if proc.returncode != 0:
            search_status = "search_crash"
            try: stderr_tail = _stderr.decode("utf-8", errors="replace")[-800:]
            except Exception: stderr_tail = "<stderr decode failed>"
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGKILL)
        proc.communicate()
        search_status = "search_timeout"

    # Load best_params
    best_params = None
    if Path(out_path).exists():
        try:
            arr = np.load(out_path)
            if arr.ndim > 0 and arr.size > 0 and np.all(np.isfinite(arr)):
                best_params = arr
            else:
                search_status = "non_finite_params"
        except Exception:
            search_status = "search_crash"

    # Checkpoint recovery on timeout
    if best_params is None and search_status == "search_timeout":
        ckpt = Path(f"/tmp/ckpt/{run_id}_best.npy")
        if ckpt.exists():
            try:
                arr = np.load(str(ckpt))
                if arr.ndim > 0 and np.all(np.isfinite(arr)):
                    best_params = arr
            except Exception:
                pass

    env = _env_audit()

    if best_params is None:
        return {"status": search_status, "strips": [], "env_audit": env,
                "stderr_tail": stderr_tail}

    # Guard 8: explicit finiteness guard before rollout
    if not np.all(np.isfinite(best_params)):
        return {"status": "non_finite_params", "strips": [], "env_audit": env}

    import jax.numpy as jnp
    test_seeds = jnp.arange(N_TEST_SEEDS) + TEST_SEED_OFFSET
    try:
        raw_strips = _rollout_and_render(
            jnp.asarray(best_params, dtype=jnp.float32),
            substrate_name,
            test_seeds,
        )
    except Exception as exc:
        return {
            "status": "rollout_crash",
            "reason": str(exc)[:300],
            "strips": [],
            "env_audit": env,
        }

    sanitised = [_sanitise_strip(s) for s in raw_strips]
    return {
        "status": search_status,
        "strips": sanitised,
        "env_audit": env,
    }


# ── Judge function — CPU, no GPU, anthropic secret mounted ───────────────────
@app.function(
    image=judge_image,
    cpu=2.0,
    memory=4096,
    timeout=900,
    secrets=[modal.Secret.from_name("anthropic-api-key")],
    # Network unrestricted: needs api.anthropic.com.
)
def judge_container(strips_png: list[bytes]) -> dict:
    """Score 16 strips × 3 runs = 48 Claude Sonnet 4.6 calls.

    Returns:
      {
        "per_strip": [[{tier: float}|None, ...], ...],  # [N_strips][3]
        "audit": {...},
      }
    """
    from evaluator import run_judge_batch
    per_strip, audit = run_judge_batch(strips_png)
    return {"per_strip": per_strip, "audit": audit}


# ── Constants re-exported for evaluator.py import ────────────────────────────
GPU_TYPE = "A100"
CPU_COUNT = 8.0
MEMORY_MB = 32768
TIMEOUT_SECS = 3600
