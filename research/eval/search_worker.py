"""search_worker.py — subprocess worker launched by evaluator.py (Guard 6).

Called as:
  python3 search_worker.py <solution_path> <output_npy> <substrate> <seed> <run_id>

Imports solution.py, calls search(), writes best_params to output_npy,
and also checkpoints to /tmp/ckpt/<run_id>_best.npy for timeout recovery.
Parent evaluator.py SIGKILLs this process at 1800 s if it has not exited.
"""


def _main() -> None:
    import importlib.util
    import os
    import sys
    from pathlib import Path

    import numpy as np

    solution_path = sys.argv[1]
    output_path   = sys.argv[2]
    substrate     = sys.argv[3]
    seed          = int(sys.argv[4])
    run_id        = sys.argv[5]

    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/cache/jax")
    os.environ.setdefault("HF_HOME", "/cache/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/hf")

    spec = importlib.util.spec_from_file_location("solution", solution_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import jax
    import jax.numpy as jnp

    train_seeds = jnp.arange(16) + 1000
    rng = jax.random.PRNGKey(seed)

    budget_s = int(os.environ.get("RE_SEARCH_WALL_CLOCK_S", "1800"))
    result = mod.search(
        substrate_name=substrate,
        seed_pool_train=train_seeds,
        budget={"wall_clock_s": budget_s},
        rng=rng,
    )

    best = result["best_params"]
    arr  = np.asarray(best, dtype=np.float32)
    np.save(output_path, arr)

    ckpt = Path(f"/tmp/ckpt/{run_id}_best.npy")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(ckpt), arr)

    print(f"SEARCH_OK output={output_path}", flush=True)


if __name__ == "__main__":
    _main()
