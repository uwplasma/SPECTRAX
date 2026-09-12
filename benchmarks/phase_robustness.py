"""Test phase-control benefit across seeds and a terminal-time window.

Reuses the example's objective. Reports absolute gain changes as well as ratios:
ratios are not meaningful when the baseline gain vanishes or changes sign.
"""
import argparse
import importlib.util
import json
import hashlib
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
from scipy.optimize import minimize

from gradient_scaling import metadata

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
    parser.add_argument("--window", type=float, nargs=2, metavar=("START", "END"))
    parser.add_argument("--shifts", type=float, nargs="+", default=[-10., -5., 0., 5., 10.])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse", type=Path,
                        help="Re-evaluate additional-seed controls from an optimizations.json file")
    args = parser.parse_args()
    if 7 in args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("Seed 7 is the saved reference; additional seeds must be distinct")
    args.output.mkdir(parents=True, exist_ok=True)
    source_path = Path(__file__).parent / "results" / (
        "window_control_cpu.json" if args.window else "phase_control_cpu.json")
    source = json.loads(source_path.read_text())
    if args.dt <= 0 or (args.window and not 0 <= args.window[0] < args.window[1]):
        parser.error("Require dt > 0 and 0 <= window start < window end")
    if args.window and min(args.shifts) + args.window[0] < 0:
        parser.error("Shifted windows must start at nonnegative time")
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.pop("resume")
    manifest = metadata() | dict(settings=settings, source_controls_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                                 study_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.reuse:
        manifest["reused_controls_sha256"] = hashlib.sha256(args.reuse.read_bytes()).hexdigest()
    manifest_path = args.output / "manifest.json"
    if args.resume and (not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest):
        raise ValueError("Resume requires the same settings, source and environment")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    final_time = args.window[1] if args.window else 50.
    objective, _ = example.problem(args.grid, args.hermite, final_time,
                                   round(final_time / args.dt), time_window=args.window)
    fg = jax.jit(jax.value_and_grad(objective))
    results_path = args.output / "optimizations.json"
    rows = json.loads(results_path.read_text()) if args.resume and results_path.exists() else []
    if args.reuse:
        rows = json.loads(args.reuse.read_text())
        if set(args.seeds) - {row["seed"] for row in rows}:
            parser.error("The reused file must contain every requested additional seed")
    pairs = [(7, np.asarray(source["initial_phase"]), np.asarray(source["optimized_phase"]))]
    for seed in args.seeds:
        saved = next((row for row in rows if row["seed"] == seed), None)
        if saved is not None:
            pairs.append((seed, np.asarray(saved["initial_phase"]), np.asarray(saved["optimized_phase"])))
            continue
        initial = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(source["initial_phase"]))
        start = perf_counter()
        result = minimize(fg, initial, jac=True, method="L-BFGS-B",
                          options=dict(maxiter=args.iterations, ftol=1e-12, gtol=1e-9))
        rows.append(dict(seed=seed, grid=args.grid, hermite=args.hermite,
                         success=bool(result.success), message=str(result.message),
                         iterations=int(result.nit), initial_phase=initial.tolist(),
                         optimized_phase=result.x.tolist(), objective=float(result.fun),
                         time_window=args.window, seconds=perf_counter() - start,
                         gradient_norm=float(np.linalg.norm(result.jac))))
        pairs.append((seed, initial, result.x))
        results_path.write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows[-1]), flush=True)
    window_path = args.output / "time_window.json"
    window = json.loads(window_path.read_text()) if args.resume and window_path.exists() else []
    intervals = [[a + shift for a in args.window] for shift in args.shifts] if args.window else [None] * len(args.times)
    times = [interval[1] for interval in intervals] if args.window else args.times
    for time, interval in zip(times, intervals):
        scalar, _ = example.problem(args.grid, args.hermite, time, round(time / args.dt), time_window=interval)
        scalar = jax.jit(scalar)
        for seed, initial, optimized in pairs:
            if any(row["seed"] == seed and row["time"] == time for row in window):
                continue
            baseline_gain = -float(scalar(initial))
            optimized_gain = -float(scalar(optimized))
            if not np.isfinite([baseline_gain, optimized_gain]).all():
                raise ValueError(f"Nonfinite gain at seed {seed}, time {time}")
            window.append(dict(seed=seed, grid=args.grid, hermite=args.hermite,
                               time=time, time_window=interval, baseline_gain=baseline_gain,
                               optimized_gain=optimized_gain,
                               improvement=optimized_gain - baseline_gain,
                               ratio=optimized_gain / baseline_gain if baseline_gain > 1e-12 else None))
        window_path.write_text(json.dumps(window, indent=2) + "\n")
        print(json.dumps(window[-len(pairs):]), flush=True)
        jax.clear_caches()
    if args.window:
        plot_window(window, args.output)


def plot_window(rows, output):
    """Show signed gains so that a sign-changing baseline cannot inflate ratios."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.3), layout="constrained")
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for index, seed in enumerate(sorted({r["seed"] for r in rows})):
        color = colors[index % len(colors)]
        data = sorted([r for r in rows if r["seed"] == seed], key=lambda r: r["time"])
        centers = [sum(r["time_window"]) / 2 for r in data]
        axes[0].plot(centers, [r["baseline_gain"] for r in data], "o--", color=color, alpha=0.65)
        axes[0].plot(centers, [r["optimized_gain"] for r in data], "o-", color=color, label=f"Seed {seed}")
        axes[1].plot(centers, [r["improvement"] for r in data], "o-", color=color, label=f"Seed {seed}")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.6)
        width = rows[0]["time_window"][1] - rows[0]["time_window"][0]
        ax.set_xlabel(f"Window center (width {width:g})")
        ax.legend(frameon=False)
    axes[0].set(title="(a) Dashed: baseline; solid: optimized", ylabel="Normalized averaged electron gain")
    axes[1].set(title="(b) Frozen controls, shifted windows", ylabel="Optimized minus baseline gain")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"window_robustness.{extension}", dpi=300)


if __name__ == "__main__":
    main()
