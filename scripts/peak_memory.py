"""Peak device memory of one forward solve or one gradient, in a fresh process (peak_bytes_in_use is cumulative per process).

Usage: python peak_memory.py OUT.jsonl GRID HERMITE T_MAX {forward,gradient} CHECKPOINTS [TOLERANCE]
Appends one JSON line. Uses the example's setup/solve/objective so the numbers match the paper runs.
"""
import importlib, json, os, sys, time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
root = os.environ["SPECTRAX_ROOT"]; sys.path.insert(0, root); sys.path.insert(0, os.path.join(root, "Examples"))
ex = importlib.import_module("2D_Orszag_Tang_optimization")
out, grid, hermite, t_max, mode, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5], int(sys.argv[6])
tolerance = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-7
theta = ex.initial_controls(8, 0)
loss = lambda th: ex.OBJECTIVES["conversion"](ex.solve(ex.setup(th, grid, hermite, t_max, tolerance=tolerance), grid, hermite, K))
fn = jax.jit(loss if mode == "forward" else jax.value_and_grad(loss))
row = dict(grid=grid, hermite=hermite, t_max=t_max, mode=mode, checkpoints=K, tolerance=tolerance,
           state_MiB=(2 * hermite**3 + 6) * grid * (grid // 2 + 1) * 16 / 2**20, provenance=ex.provenance())
try:
    start = time.perf_counter(); jax.block_until_ready(fn(theta)); row["first_call_s"] = time.perf_counter() - start
    start = time.perf_counter(); jax.block_until_ready(fn(theta)); row["second_call_s"] = time.perf_counter() - start
    row["peak_MiB"] = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**20
    row["steps"] = int(ex.solve(ex.setup(theta, grid, hermite, t_max, tolerance=tolerance), grid, hermite, K)["solver_stats"]["num_accepted_steps"])
except Exception as error:   # record out-of-memory as a result rather than crashing the sweep
    row["error"] = f"{type(error).__name__}: {str(error)[:200]}"
open(out, "a").write(json.dumps(row) + "\n")
print(json.dumps({k: v for k, v in row.items() if k != "provenance"}), flush=True)
