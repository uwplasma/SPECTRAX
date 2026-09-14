"""Cost and memory of gradients through SPECTRAX, on the Orszag–Tang conversion objective.

(a) Wall time of one reverse-mode gradient against centred finite differences as the number of controls grows.
(b) Compiled reverse-pass workspace against fixed time steps, for a full tape and for 8 or 32 checkpoints.
(c) Forward and gradient workspace against state size. (d) Finite-difference error against step size.

  python 2D_Orszag_Tang_gradient_benchmark.py --grid 16 --hermite 4 --t-max 100
"""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np
from diffrax import Tsit5
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

from orszag_tang_control import COLORS, OBJECTIVES, initial_controls, plt, provenance, save, setup, solve


def timed(fn, x, repeat=3):
    fn(x)
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        jax.block_until_ready(fn(x))
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def workspace(fn, x):
    """Temporary memory of the compiled programme, in MiB."""
    analysis = jax.jit(fn).lower(x).compile().memory_analysis()
    return analysis and analysis.temp_size_in_bytes / 2**20


def benchmark(args):
    def loss(theta, grid=args.grid, hermite=args.hermite, t_max=args.t_max, checkpoints=args.checkpoints, **kwargs):
        return OBJECTIVES["conversion"](solve(setup(theta, grid, hermite, t_max, tolerance=args.tolerance), grid, hermite, checkpoints, **kwargs))

    report = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, provenance=provenance(),
                  cost=[], memory_steps=[], memory_resolution=[], fd_step=[])
    forward, reverse = jax.jit(loss), jax.jit(jax.value_and_grad(loss))
    for M in args.control_counts:
        theta = initial_controls(M)
        t_forward, t_reverse = timed(forward, theta), timed(reverse, theta)
        fd_gradient = lambda h: np.array([float(forward(theta + h * e) - forward(theta - h * e)) / (2 * h) for e in np.eye(theta.size)])
        start = time.perf_counter()
        fd = fd_gradient(1e-4)
        t_fd, gradient = time.perf_counter() - start, np.asarray(reverse(theta)[1])
        error = lambda g: float(np.linalg.norm(g - gradient) / np.linalg.norm(gradient))
        report["cost"].append(dict(controls=theta.size, forward_s=t_forward, reverse_s=t_reverse, fd_s=t_fd, fd_relative_error=error(fd)))
        print(f"P={theta.size:4d}: forward {t_forward:.2f} s, AD gradient {t_reverse:.2f} s ({t_reverse / t_forward:.1f}x), "
              f"FD gradient {t_fd:.1f} s ({t_fd / t_reverse:.1f}x AD), |FD-AD|/|AD| {error(fd):.1e}")
        if M == args.control_counts[0]:
            report["fd_step"] = [dict(step=float(h), relative_error=error(fd_gradient(h))) for h in np.logspace(-9, -1, 9)]
    theta, dt = initial_controls(args.modes), args.t_max / max(args.step_counts)
    for steps in args.step_counts:
        fixed = dict(t_max=steps * dt, dt=dt, adaptive_time_step=False, solver=Tsit5(), max_steps=steps + 1)
        row = dict(steps=steps, forward_MiB=workspace(lambda th: loss(th, checkpoints=1, **fixed), theta))
        for label, K in (("tape", steps), ("K8", 8), ("K32", 32)):
            row[f"reverse_{label}_MiB"] = workspace(jax.value_and_grad(lambda th, K=K: loss(th, checkpoints=K, **fixed)), theta)
        report["memory_steps"].append(row)
        print("memory vs steps:", row)
    for grid, hermite in args.resolutions:
        f = lambda th: loss(th, grid=grid, hermite=hermite)
        report["memory_resolution"].append(dict(grid=grid, hermite=hermite, state_MiB=(2 * hermite**3 + 6) * grid * (grid // 2 + 1) * 16 / 2**20,
                                                forward_MiB=workspace(f, theta), reverse_MiB=workspace(jax.value_and_grad(f), theta)))
        print("memory vs resolution:", report["memory_resolution"][-1])
    return report


def plot(report):
    def data_ticks(ax, values):
        """Label a log x-axis at the measured values only, so minor-tick labels cannot collide."""
        ax.xaxis.set_major_locator(FixedLocator(values)); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_major_formatter(ScalarFormatter())

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.4), layout="constrained")
    c, m, r, f = report["cost"], report["memory_steps"], report["memory_resolution"], report["fd_step"]
    ax, P = axes[0, 0], [x["controls"] for x in c]
    ax.loglog(P, [x["fd_s"] for x in c], "o-", color=COLORS["optimized"], label="centred finite differences (2P solves)")
    ax.loglog(P, [x["reverse_s"] for x in c], "o-", color=COLORS["initial"], label="reverse-mode AD (one solve + adjoint)")
    ax.loglog(P, [x["forward_s"] for x in c], "--", color=COLORS["muted"], label="one forward solve")
    ax.set(title="(a) Gradient wall time vs. number of controls P", xlabel="controls P", ylabel="seconds")
    data_ticks(ax, P)
    ax, N = axes[0, 1], [x["steps"] for x in m]
    for key, color, label in (("tape", "optimized", "full tape"), ("K32", "initial", "32 checkpoints"), ("K8", "third", "8 checkpoints")):
        ax.loglog(N, [x[f"reverse_{key}_MiB"] for x in m], "o-", color=COLORS[color], label=f"reverse, {label}")
    ax.loglog(N, [x["forward_MiB"] for x in m], "--", color=COLORS["muted"], label="forward solve")
    ax.set(title="(b) Compiled workspace vs. time steps N", xlabel="time steps N", ylabel="MiB")
    data_ticks(ax, N)
    ax, S = axes[1, 0], [x["state_MiB"] for x in r]
    ax.loglog(S, [x["reverse_MiB"] for x in r], "o-", color=COLORS["initial"], label="reverse-mode gradient")
    ax.loglog(S, [x["forward_MiB"] for x in r], "o-", color=COLORS["muted"], label="forward solve")
    for i, x in enumerate(r):   # above the gradient curve, at alternating heights, so no label crosses a line or a neighbour
        ax.annotate(f"{x['grid']}²×{x['hermite']}³, ×{x['reverse_MiB'] / x['forward_MiB']:.1f}", (x["state_MiB"], x["reverse_MiB"]),
                    textcoords="offset points", xytext=(-3 if i == 0 else 0, 7 if i % 2 == 0 else 17), ha="left" if i == 0 else "center",
                    va="bottom", fontsize=6)
    ax.set_ylim(min(x["forward_MiB"] for x in r) / 2, max(x["reverse_MiB"] for x in r) * 8)
    ax.set(title="(c) Workspace vs. state size (gradient/forward ratio)", xlabel="state size, MiB", ylabel="MiB")
    axes[1, 1].loglog([x["step"] for x in f], [x["relative_error"] for x in f], "o-", color=COLORS["optimized"])
    axes[1, 1].set(title="(d) Finite-difference error vs. step", xlabel="finite-difference step", ylabel="relative error to AD")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.legend(frameon=False, fontsize=7)
    s = report["settings"]
    fig.suptitle(f"Gradient cost and memory, {s['grid']}² × {s['hermite']}³, {report['provenance']['device']}", fontsize=10)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", type=int, default=4, help="stream-function modes for the memory measurements")
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--hermite", type=int, default=4)
    parser.add_argument("--t-max", type=float, default=200.0)
    parser.add_argument("--checkpoints", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--control-counts", type=int, nargs="+", default=[2, 4, 8, 16, 32], metavar="M")
    parser.add_argument("--step-counts", type=int, nargs="+", default=[25, 50, 100, 200, 400], metavar="N")
    parser.add_argument("--resolutions", nargs="+", default=[(16, 4), (32, 4), (32, 6), (64, 6)], metavar="GRIDxHERMITE",
                        type=lambda r: tuple(map(int, r.split("x"))))
    parser.add_argument("--report", type=Path, help="re-plot an existing report instead of measuring")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    save(json.loads(args.report.read_text()) if args.report else benchmark(args), args.output / "benchmark.json", plot)


if __name__ == "__main__":
    main()
