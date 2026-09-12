"""Phase-only control of electron energization in a kinetic Orszag–Tang plasma.

Run: python Examples/2D_phase_control.py --output /tmp/phase-control
Use --grid 64 --hermite 6 --time 100 --steps 2000 for a GPU convergence study.
The small defaults demonstrate the method; they are not resolved turbulence.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from spectrax import compute_C_nmp, plasma_current, simulation_final


def setup(phases, grid=16, hermite=4, final_time=20.0):
    """Fix every mode amplitude; vary only phases of distinct oblique modes."""
    if hermite < 3:
        raise ValueError("At least three Hermite modes are needed for energy.")
    modes = [(i, j) for i in range(1, grid // 3)
             for j in range(-grid // 3 + 1, grid // 3) if j]
    modes.sort(key=lambda k: (k[0] ** 2 + k[1] ** 2, k))
    if len(phases) > len(modes):
        raise ValueError("Too many controls for the dealiased spatial grid.")
    x = 2 * jnp.pi * jnp.arange(grid) / grid
    X, Y = jnp.meshgrid(x, x, indexing="xy")
    k = 2 * jnp.pi / 50.0
    bx, by = -0.2 * jnp.sin(Y), 0.2 * jnp.sin(2 * X)
    curl = 0.2 * k * (jnp.cos(Y) + 2 * jnp.cos(2 * X))
    i, j = np.asarray(modes[:len(phases)], dtype=float).reshape(-1, 2).T[:, :, None, None]
    radius = np.hypot(i, j)
    amplitude = 0.06 / radius
    angle = i * X + j * Y + jnp.asarray(phases)[:, None, None]
    bx = bx - jnp.sum(amplitude * j / radius * jnp.sin(angle), axis=0)
    by = by + jnp.sum(amplitude * i / radius * jnp.sin(angle), axis=0)
    curl = curl + jnp.sum(amplitude * k * radius * jnp.cos(angle), axis=0)
    zeros = jnp.zeros_like(X)
    velocity = jnp.stack((-0.02 * jnp.sin(Y), 0.02 * jnp.sin(X), zeros))
    velocities = jnp.stack((velocity.at[2].set(-0.5 * curl), velocity))[..., None]
    fields = jnp.stack((zeros, zeros, zeros, bx, by, jnp.ones_like(X)))[..., None]
    parameters = dict(
        Lx=50.0, Ly=50.0, Lz=1.0, mi_me=25.0,
        qs=jnp.array([-1.0, 1.0]), Omega_cs=jnp.array([0.5, 0.02]),
        alpha_s=jnp.array([0.25] * 3 + [0.05] * 3), u_s=jnp.zeros(6),
        nu=1.0, D=0.0, t_max=final_time,
    )
    parameters["Ck_0"] = compute_C_nmp(
        velocities, parameters["alpha_s"], parameters["u_s"],
        hermite, hermite, hermite, 2,
    ).reshape(2 * hermite ** 3, grid, grid // 2 + 1, 1)
    parameters["Fk_0"] = jnp.fft.rfftn(fields, axes=(-1, -3, -2), norm="forward")
    return parameters


def quantities(state, parameters, grid, hermite):
    """Return E energy, B energy, electron/ion kinetic energy, and <Jz^4>."""
    C, F = state
    # Parseval weights: the highest retained mode is doubled on odd grids.
    weights = jnp.where(jnp.arange(grid // 2 + 1) == 0, 1.0, 2.0)
    if grid % 2 == 0:
        weights = weights.at[-1].set(1.0)
    field_energy = 0.5 * parameters["Omega_cs"][0] ** 2 * jnp.sum(
        jnp.abs(F) ** 2 * weights[None, None, :, None], axis=(1, 2, 3)
    )
    moments = C.reshape(2, hermite, hermite, hermite, grid, grid // 2 + 1, 1)
    alpha = parameters["alpha_s"].reshape(2, 3)
    density = moments[:, 0, 0, 0, 0, 0, 0]
    second = jnp.stack((moments[:, 0, 0, 2, 0, 0, 0],
                        moments[:, 0, 2, 0, 0, 0, 0],
                        moments[:, 2, 0, 0, 0, 0, 0]), axis=1)
    kinetic = jnp.real(0.5 * jnp.array([1.0, 25.0]) * jnp.prod(alpha, axis=1)
                       * jnp.sum(alpha ** 2 * (density[:, None] / 2
                                               + second / jnp.sqrt(2)), axis=1))
    current = current_z(state, parameters, grid, hermite)
    return jnp.array([field_energy[:3].sum(), field_energy[3:].sum(),
                      kinetic[0], kinetic[1], jnp.mean(current ** 4)])


def current_z(state, parameters, grid, hermite):
    current = plasma_current(parameters["qs"], parameters["alpha_s"],
                             parameters["u_s"], state[0],
                             hermite, hermite, hermite, 2)
    return jnp.fft.irfftn(current[2], s=(1, grid, grid),
                          axes=(-1, -3, -2), norm="forward")[..., 0]


def problem(grid=16, hermite=4, final_time=20.0, steps=400,
            checkpointing=True, checkpoint_size=None, checkpoints=None, time_window=None):
    """Return an ordinary JAX objective and terminal-state function."""
    if time_window is not None:
        start, end = time_window
        if not 0 <= start < end <= final_time:
            raise ValueError("time_window must lie inside [0, final_time]")

    def run(phases):
        parameters = setup(phases, grid, hermite, final_time)
        integrand = None
        if time_window is not None:
            initial = quantities((parameters["Ck_0"], parameters["Fk_0"]), parameters, grid, hermite)

            def integrand(t, C, F):
                # Compact C1 weight with unit integral over [start, end].
                z = (2 * t - start - end) / (end - start)
                weight = jnp.where(jnp.abs(z) < 1, 15 / (8 * (end - start)) * (1 - z ** 2) ** 2, 0)
                energy = quantities((C, F), parameters, grid, hermite)[2]
                return weight * (energy - initial[2]) / (initial[1] - 0.125)
        return simulation_final(parameters, steps=steps, Nx=grid, Ny=grid,
                                Nn=hermite, Nm=hermite, Np=hermite,
                                checkpointing=checkpointing,
                                checkpoint_size=checkpoint_size, checkpoints=checkpoints, integrand=integrand)

    def terminal(phases):
        return run(phases)[:2]

    def objective(phases):
        if time_window is not None:
            return -run(phases)[2]
        parameters = setup(phases, grid, hermite, final_time)
        initial = quantities((parameters["Ck_0"], parameters["Fk_0"]),
                             parameters, grid, hermite)
        final = quantities(terminal(phases), parameters, grid, hermite)
        # Remove the constant guide-field energy from the normalization.
        return -(final[2] - initial[2]) / (initial[1] - 0.5 * 0.5 ** 2)

    return objective, terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--hermite", type=int, default=4)
    parser.add_argument("--time", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--window", type=float, nargs=2, default=None, metavar=("START", "END"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoints", type=int, default=None,
                        help="Fixed checkpoint count for reverse-only memory budgeting")
    parser.add_argument("--controls", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("phase-control"))
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    initial_phase = np.random.default_rng(options.seed).uniform(-np.pi, np.pi, options.controls)
    objective, terminal = problem(options.grid, options.hermite, options.time, options.steps,
                                  checkpoints=options.checkpoints, time_window=options.window)
    value_grad = jax.jit(jax.value_and_grad(objective))
    start = perf_counter()
    value0, gradient = jax.block_until_ready(value_grad(initial_phase))
    compile_and_first = perf_counter() - start
    history = [float(value0)]

    def callback(x):
        value = float(value_grad(x)[0])
        history.append(value)
        print(f"iteration {len(history)-1}: objective gain / initial magnetic fluctuation energy = {-value:.8g}", flush=True)

    start = perf_counter()
    result = minimize(value_grad, initial_phase, jac=True, method="L-BFGS-B",
                      callback=callback, options=dict(maxiter=options.iterations,
                                                       ftol=1e-12, gtol=1e-9))
    optimize_seconds = perf_counter() - start
    states = jax.jit(terminal)
    before, after = states(initial_phase), states(result.x)
    params = setup(initial_phase, options.grid, options.hermite, options.time)
    params_opt = setup(result.x, options.grid, options.hermite, options.time)
    q0 = quantities((params["Ck_0"], params["Fk_0"]), params, options.grid, options.hermite)
    q0_opt = quantities((params_opt["Ck_0"], params_opt["Fk_0"]), params_opt, options.grid, options.hermite)
    q1 = quantities(before, params, options.grid, options.hermite)
    q2 = quantities(after, params_opt, options.grid, options.hermite)
    # Independent time refinement at the optimized controls.
    refined_objective, _ = problem(options.grid, options.hermite, options.time, 2 * options.steps,
                                           checkpoints=options.checkpoints, time_window=options.window)
    refined_value, refined_gradient = jax.block_until_ready(
        jax.jit(jax.value_and_grad(refined_objective))(result.x))
    final_value, final_gradient = value_grad(result.x)
    direction = np.random.default_rng(3).normal(size=options.controls)
    direction /= np.linalg.norm(direction)
    fd = []
    scalar = jax.jit(objective)
    for epsilon in np.logspace(-2, -6, 5):
        finite_difference = float((scalar(initial_phase + epsilon * direction)
                                   - scalar(initial_phase - epsilon * direction)) / (2 * epsilon))
        fd.append([float(epsilon), finite_difference])
    report = dict(
        grid=options.grid, hermite=options.hermite, steps=options.steps,
        time_window=options.window, seed=options.seed,
        final_time=options.time, controls=options.controls, checkpoints=options.checkpoints,
        device=str(jax.devices()[0]), jax_version=jax.__version__,
        compile_and_first_seconds=compile_and_first, optimization_seconds=optimize_seconds,
        success=bool(result.success), message=str(result.message), iterations=int(result.nit),
        initial_objective=float(value0), final_objective=float(final_value),
        initial_energies=np.asarray(q0[:4]).tolist(),
        optimized_initial_energies=np.asarray(q0_opt[:4]).tolist(),
        baseline_final_energies=np.asarray(q1[:4]).tolist(),
        optimized_final_energies=np.asarray(q2[:4]).tolist(),
        initial_energy_constraint_error=float(jnp.max(jnp.abs(q0_opt[:4] - q0[:4]))),
        energy_balance_error=[float((jnp.sum(q[0:4]) - jnp.sum(q0[0:4])) / jnp.sum(q0[0:4]))
                              for q in (q1, q2)],
        refined_objective=float(refined_value),
        refined_gradient_difference=float(jnp.linalg.norm(refined_gradient - final_gradient)),
        directional_ad=float(jnp.dot(gradient, direction)), finite_differences=fd,
    )
    (options.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(options.output / "data.npz", initial_phase=initial_phase, optimized_phase=result.x,
             history=history, initial_energies=q0, baseline_final=q1, optimized_final=q2,
             baseline_current=current_z(before, params, options.grid, options.hermite),
             optimized_current=current_z(after, params_opt, options.grid, options.hermite))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.7), layout="constrained")
    currents = [np.asarray(current_z(s, params, options.grid, options.hermite))
                for s in (before, after)]
    limit = max(np.max(np.abs(j)) for j in currents)
    for ax, current, label in zip(axes[0], currents, ["(a) Baseline", "(b) Optimized phases"]):
        im = ax.imshow(current, origin="lower", extent=(0, 50, 0, 50),
                       cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="none")
        ax.set(xlabel=r"$x/d_e$", ylabel=r"$y/d_e$", title=label)
    fig.colorbar(im, ax=axes[0].tolist(), label=r"$J_z(T)$", shrink=0.85)
    axes[1, 0].plot(np.arange(len(history)), -np.asarray(history), "o-", color="#0072B2", ms=3)
    axes[1, 0].set(xlabel="Optimization iteration", ylabel=r"$[K_e(T)-K_e(0)]/W_{B,\perp}(0)$",
                   title="(c) Fixed initial energy and spectrum")
    if options.window is not None:
        axes[1, 0].set_ylabel(r"$\overline{\Delta K_e}/W_{B,\perp}(0)$")
    normalization = float(q0[1] - 0.125)
    indices = np.arange(4)
    axes[1, 1].bar(indices - 0.18, (np.asarray(q1[:4]) - np.asarray(q0[:4])) / normalization,
                    width=0.36, color="#777777", label="Baseline")
    axes[1, 1].bar(indices + 0.18, (np.asarray(q2[:4]) - np.asarray(q0[:4])) / normalization,
                    width=0.36, color="#D55E00", label="Optimized")
    axes[1, 1].axhline(0, color="black", linewidth=0.5)
    axes[1, 1].set(xticks=indices, xticklabels=[r"$W_E$", r"$W_B$", r"$K_e$", r"$K_i$"],
                   ylabel=r"$[W(T)-W(0)]/W_{B,\perp}(0)$", title="(d) Energy transfer")
    axes[1, 1].legend(frameon=False)
    for extension in ("png", "pdf"):
        fig.savefig(options.output / f"phase_control.{extension}")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
