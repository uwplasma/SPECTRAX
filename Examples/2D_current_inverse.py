"""Recover phase controls from a synthetic final current map with matrix-free SOLVAX.

This is an inverse-problem/API verification example, not an experimental
reconstruction or a new heating claim. Target and fit use the same discretization.
"""
import argparse
import importlib.util
import json
import hashlib
import inspect
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import solvax
from solvax import LeastSquaresConfig, gauss_newton_least_squares

spec = importlib.util.spec_from_file_location("phase_control", Path(__file__).with_name("2D_phase_control.py"))
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def plot_results(output):
    """Render saved native-grid fields without rerunning the plasma solver."""
    report = json.loads((output / "results.json").read_text())
    with np.load(output / "maps.npz") as maps:
        target, before, after = (maps[key] for key in ("target", "initial", "recovered"))
    history = report["cost_history"]
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), layout="constrained")
    limit = max(float(np.max(np.abs(x))) for x in (target, before, after))
    for ax, field, title in zip(axes[:3], (target, before, after),
                                ("(a) Synthetic target", "(b) Initial phases", "(c) Recovered phases")):
        im = ax.imshow(field, origin="lower", extent=(0, 50, 0, 50), cmap="RdBu_r", interpolation="nearest", vmin=-limit, vmax=limit)
        ax.set(xlabel=r"$x/d_e$", ylabel=r"$y/d_e$", title=title)
    fig.colorbar(im, ax=axes[:3].tolist(), label=r"$J_z(T)$", shrink=0.75)
    axes[3].semilogy(np.arange(len(history)), np.maximum(history, 1e-30), "o-", color="#0072B2")
    axes[3].set(xlabel="Iteration", ylabel="Normalized least-squares cost", title="(d) Convergence")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"current_inverse.{extension}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--hermite", type=int, default=4)
    parser.add_argument("--time", type=float, default=20.)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--controls", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("current-inverse"))
    parser.add_argument("--plot-only", action="store_true", help="Render saved results without solving")
    args = parser.parse_args()
    if args.plot_only:
        plot_results(args.output)
        return
    if min(args.steps, args.controls, args.iterations) < 1 or args.time <= 0:
        parser.error("Time, steps, controls and iterations must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    _, terminal = example.problem(args.grid, args.hermite, args.time, args.steps)

    def current(phases):
        parameters = example.setup(phases, args.grid, args.hermite, args.time)
        return example.current_z(terminal(phases), parameters, args.grid, args.hermite)

    reference = np.random.default_rng(19).uniform(-np.pi, np.pi, args.controls)
    initial = reference + 0.6 * np.random.default_rng(11).normal(size=args.controls)
    current = jax.jit(current)
    target = current(reference)
    normalization = jnp.linalg.norm(target)

    def residual(phases):
        return ((current(phases) - target) / normalization).ravel()

    config = LeastSquaresConfig(rtol=1e-9, max_steps=args.iterations,
                                linear_rtol=1e-5, linear_max_steps=max(16, 2 * args.controls))
    # Both actions are supplied by AD of the checkpointed plasma solve; no
    # current-pixel-by-control Jacobian or state Jacobian is assembled.
    solve = jax.jit(lambda x: gauss_newton_least_squares(residual, x, config=config))
    start = perf_counter()
    executable = solve.lower(initial).compile()
    compile_seconds = perf_counter() - start
    start = perf_counter()
    result = jax.block_until_ready(executable(initial))
    solve_seconds = perf_counter() - start
    allocator = jax.devices()[0].memory_stats() or {}
    memory = executable.memory_analysis()
    before, after = current(initial), current(result.x)
    count = int(result.steps)
    history = np.asarray(result.history.cost[:count + 1])

    # Verify the real-linear adjoint action and a directional loss derivative.
    direction = np.random.default_rng(3).normal(size=args.controls)
    direction /= np.linalg.norm(direction)
    tangent = jax.jvp(residual, (initial,), (direction,))[1]
    value, pullback = jax.vjp(residual, initial)
    cotangent = jnp.cos(jnp.arange(value.size)) / jnp.sqrt(value.size)
    left, right = jnp.vdot(tangent, cotangent), jnp.vdot(direction, pullback(cotangent)[0])
    loss = jax.jit(lambda x: 0.5 * jnp.sum(residual(x) ** 2))
    epsilon = 1e-4
    finite_difference = (loss(initial + epsilon * direction) - loss(initial - epsilon * direction)) / (2 * epsilon)
    derivative = jnp.vdot(value, tangent)
    energies = []
    for phases in (initial, result.x):
        parameters = example.setup(phases, args.grid, args.hermite, args.time)
        energies.append(np.asarray(example.quantities((parameters["Ck_0"], parameters["Fk_0"]),
                                                      parameters, args.grid, args.hermite)[:4]))
    report = dict(grid=args.grid, hermite=args.hermite, final_time=args.time, steps=args.steps,
                  controls=args.controls, device=jax.devices()[0].device_kind, jax_version=jax.__version__,
                  solvax_version=solvax.__version__,
                  source_sha256={name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in
                                 dict(example=__file__, phase_control=example.__file__,
                                      plasma_solver=Path(__file__).parents[1] / "spectrax/_autodiff.py",
                                      least_squares=inspect.getsourcefile(gauss_newton_least_squares),
                                      replay=inspect.getsourcefile(solvax.checkpointed_fori_loop)).items()},
                  compile_seconds=compile_seconds, solve_seconds=solve_seconds,
                  xla_buffer_bytes=memory.argument_size_in_bytes + memory.output_size_in_bytes
                                   + memory.temp_size_in_bytes - memory.alias_size_in_bytes,
                  gpu_peak_after_solve_bytes=allocator.get("peak_bytes_in_use"),
                  converged=bool(result.converged), iterations=count,
                  accepted_steps=int(result.accepted_steps), rejected_steps=int(result.rejected_steps),
                  linear_iterations=int(result.linear_iterations), gradient_norm=float(result.gradient_norm),
                  initial_relative_current_error=float(jnp.linalg.norm(before - target) / normalization),
                  final_relative_current_error=float(jnp.linalg.norm(after - target) / normalization),
                  phase_rms_error=float(jnp.sqrt(jnp.mean(jnp.angle(jnp.exp(1j * (result.x - reference))) ** 2))),
                  adjoint_dot_error=float(jnp.abs(left - right)), directional_ad=float(derivative),
                  directional_fd=float(finite_difference), initial_energy_constraint_error=float(np.max(abs(energies[0] - energies[1]))),
                  reference_phase=reference.tolist(), initial_phase=initial.tolist(), optimized_phase=np.asarray(result.x).tolist(),
                  cost_history=history.tolist())
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(args.output / "maps.npz", target=target, initial=before, recovered=after)
    plot_results(args.output)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
