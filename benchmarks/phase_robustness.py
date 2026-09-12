"""Test phase-control benefit across seeds and a terminal-time window.

Reuses the example's objective. Reports absolute gain changes as well as ratios:
ratios are not meaningful when the baseline gain vanishes or changes sign.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import jax
import numpy as np
from scipy.optimize import minimize

spec = importlib.util.spec_from_file_location(
    "phase_control", Path(__file__).parents[1] / "Examples" / "2D_phase_control.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--hermite", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="*", default=[11, 23])
    parser.add_argument("--times", type=float, nargs="+", default=[40., 50., 60.])
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads((Path(__file__).parent / "results/phase_control_cpu.json").read_text())
    objective, _ = example.problem(args.grid, args.hermite, 50., round(50 / args.dt))
    fg = jax.jit(jax.value_and_grad(objective))
    rows = []
    pairs = [(7, np.asarray(source["initial_phase"]), np.asarray(source["optimized_phase"]))]
    for seed in args.seeds:
        initial = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(source["initial_phase"]))
        result = minimize(fg, initial, jac=True, method="L-BFGS-B",
                          options=dict(maxiter=args.iterations, ftol=1e-12, gtol=1e-9))
        rows.append(dict(seed=seed, grid=args.grid, hermite=args.hermite,
                         success=bool(result.success), message=str(result.message),
                         iterations=int(result.nit), initial_phase=initial.tolist(),
                         optimized_phase=result.x.tolist(), objective=float(result.fun)))
        pairs.append((seed, initial, result.x))
        (args.output / "optimizations.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows[-1]), flush=True)
    window = []
    for time in args.times:
        scalar, _ = example.problem(args.grid, args.hermite, time, round(time / args.dt))
        scalar = jax.jit(scalar)
        for seed, initial, optimized in pairs:
            baseline_gain = -float(scalar(initial))
            optimized_gain = -float(scalar(optimized))
            window.append(dict(seed=seed, grid=args.grid, hermite=args.hermite,
                               time=time, baseline_gain=baseline_gain,
                               optimized_gain=optimized_gain,
                               improvement=optimized_gain - baseline_gain,
                               ratio=optimized_gain / baseline_gain if baseline_gain > 1e-12 else None))
        (args.output / "time_window.json").write_text(json.dumps(window, indent=2) + "\n")
        print(json.dumps(window[-len(pairs):]), flush=True)


if __name__ == "__main__":
    main()
