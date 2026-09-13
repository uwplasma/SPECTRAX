"""Inverse design of the 2D Orszag–Tang vortex through ``jax.grad`` of ``simulation``.

Controls are the amplitudes and phases of the ``M`` lowest Fourier modes of the in-plane
magnetic stream function, normalised to a fixed in-plane magnetic energy; electrons carry
the consistent out-of-plane current, as in ``2D_Orszag_Tang.py``. With ``--fixed-amplitudes``
only the phases vary, so every mode energy is fixed: linear evolution on the uniform background
cannot change the energy objective, and any gain in it comes from nonlinear coupling.
Any real scalar of the ``simulation`` output can be optimised. Two are provided:

  conversion    in-plane magnetic energy at ``t_max`` over its initial value
                (minimised: the field configuration that converts magnetic energy fastest)
  peak_current  smooth maximum (p-norm) of the out-of-plane current density at ``t_max``
                (maximised: the configuration that forms the most intense current sheet)

Modes::

  python 2D_Orszag_Tang_optimization.py optimize  --objective peak_current --modes 8 --grid 32 --hermite 4
  python 2D_Orszag_Tang_optimization.py validate  results/optimize_peak_current.json --resolutions 32x6 64x6
  python 2D_Orszag_Tang_optimization.py benchmark --grid 16 --hermite 4 --t-max 100
  python 2D_Orszag_Tang_optimization.py plot      results/optimize_peak_current.json results/benchmark.json

Every run writes a JSON report, with commit, package versions and device, to ``--output``;
``plot`` draws figures from JSON only. ``validate`` re-evaluates the frozen initial and optimised
controls of an ``optimize`` report at other resolutions and ODE tolerances. The reverse pass uses
Diffrax's binomial checkpointing (``--checkpoints``), so its memory is a fixed multiple of the
forward solve, independent of the number of time steps.
"""

import argparse
import json
import os
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import Dopri8, Tsit5, NoProgressMeter, RecursiveCheckpointAdjoint
from scipy.optimize import minimize

from spectrax import simulation, compute_C_nmp, plasma_current

jax.config.update("jax_enable_x64", True)
Lx = Ly = 50.0
Omega_ce, mi_me, deltaB = 0.5, 25.0, 0.2          # as in input_2D_orszag_tang.toml
alpha_s, u_s = jnp.array([0.25] * 3 + [0.05] * 3), jnp.zeros(6)
COLORS = {"initial": "#2a78d6", "optimized": "#eb6834", "third": "#1baf7a", "muted": "#7a7975"}


def wavevectors(M):
    """The M lowest nonzero wavevectors (one of each +-k pair; the phase covers the sign)."""
    pairs = sorted(((m, n) for m in range(0, 6) for n in range(-5, 6) if (m, n) != (0, 0) and (m > 0 or n > 0)),
                   key=lambda mn: (mn[0] ** 2 + mn[1] ** 2, mn))[:M]
    return jnp.array(pairs, dtype=float) * 2 * jnp.pi / Lx


def setup(theta, grid, hermite, t_max, nu=1.0, fixed_amplitudes=False):
    """Initial condition from controls ``theta = (log amplitudes, phases)``, or phases only, at fixed in-plane magnetic energy."""
    M = theta.size if fixed_amplitudes else theta.size // 2
    k = wavevectors(M)
    a = jnp.ones(M) if fixed_amplitudes else jnp.exp(theta[:M])
    a = a * deltaB / jnp.sqrt(0.5 * jnp.sum(a**2 * jnp.sum(k**2, axis=1)))   # <|B_perp|^2> = deltaB^2
    x = jnp.arange(grid) * Lx / grid
    X, Y = jnp.meshgrid(x, x, indexing="xy")
    phase = k[:, 0, None, None] * X + k[:, 1, None, None] * Y + theta[-M:, None, None]
    Bx = -jnp.sum(a[:, None, None] * k[:, 1, None, None] * jnp.sin(phase), axis=0)
    By = jnp.sum(a[:, None, None] * k[:, 0, None, None] * jnp.sin(phase), axis=0)
    Jz = jnp.sum(a[:, None, None] * jnp.sum(k**2, axis=1)[:, None, None] * jnp.cos(phase), axis=0)
    U0, k0 = deltaB * Omega_ce / jnp.sqrt(mi_me), 2 * jnp.pi / Lx
    flow = jnp.stack([-U0 * jnp.sin(k0 * Y), U0 * jnp.sin(k0 * X), jnp.zeros_like(X)])
    Us = jnp.stack([flow.at[2].set(-Omega_ce * Jz), flow])[..., None]      # electrons carry the current
    F = jnp.concatenate([jnp.zeros((3, grid, grid)), jnp.stack([Bx, By, jnp.ones_like(X)])])[..., None]
    Ck_0 = compute_C_nmp(Us, alpha_s, u_s, hermite, hermite, hermite, 2).reshape(2 * hermite**3, grid, grid // 2 + 1, 1)
    return dict(Lx=Lx, Ly=Ly, Lz=1.0, mi_me=mi_me, qs=jnp.array([-1.0, 1.0]), Omega_cs=jnp.array([Omega_ce, Omega_ce / mi_me]),
                alpha_s=alpha_s, u_s=u_s, nu=nu, D=0.0, t_max=t_max, ode_tolerance=1e-7,
                Ck_0=Ck_0, Fk_0=jnp.fft.rfftn(F, axes=(-1, -3, -2), norm="forward"))


def solve(parameters, grid, hermite, checkpoints=32, timesteps=2, **kwargs):
    kwargs = dict(dt=0.01, solver=Dopri8(), max_steps=100_000) | kwargs
    return simulation(parameters, Nx=grid, Ny=grid, Nz=1, Nn=hermite, Nm=hermite, Np=hermite, Ns=2, timesteps=timesteps,
                      adjoint=RecursiveCheckpointAdjoint(checkpoints=checkpoints), progress_meter=NoProgressMeter(), **kwargs)


def inplane_magnetic_energy(output, i=-1):
    weights = jnp.where(jnp.arange(output["Fk"].shape[-2]) == 0, 1.0, 2.0)    # real-FFT Parseval weights
    return 0.5 * Omega_ce**2 * jnp.sum(jnp.abs(output["Fk"][i, 3:5]) ** 2 * weights[None, None, :, None])


def current_density(output, i=-1):
    """Out-of-plane current density Jz(x, y) at snapshot ``i`` from the Hermite moments."""
    Ck = output["Ck"][i]
    hermite = round((Ck.shape[0] // 2) ** (1 / 3))
    Jk = plasma_current(output["qs"], output["alpha_s"], output["u_s"], Ck, hermite, hermite, hermite, 2)
    return jnp.fft.irfftn(Jk[2], s=(1, *Ck.shape[1:2], 2 * (Ck.shape[2] - 1)), axes=(-1, -3, -2), norm="forward")[:, :, 0]


def peak(J, p=8):
    """Smooth maximum of |J| (the p-norm of the spatial mean)."""
    return jnp.mean(J**p) ** (1 / p)


OBJECTIVES = {
    "conversion": lambda output: inplane_magnetic_energy(output) / inplane_magnetic_energy(output, 0),
    "peak_current": lambda output: -peak(current_density(output)),
}


def make_loss(args):
    objective = OBJECTIVES[args.objective]
    return lambda theta: objective(solve(setup(theta, args.grid, args.hermite, args.t_max, fixed_amplitudes=args.fixed_amplitudes),
                                         args.grid, args.hermite, args.checkpoints))


def initial_controls(M, seed, fixed_amplitudes=False):
    phases = jnp.asarray(np.random.default_rng(seed).uniform(-np.pi, np.pi, M))
    return phases if fixed_amplitudes else jnp.concatenate([jnp.zeros(M), phases])


def provenance():
    """Commit, package versions and device, recorded in every report."""
    def run(*command):
        try:
            return subprocess.run(command, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    def installed(package):
        try:
            return version(package)
        except Exception:
            return None
    root, device = Path(__file__).resolve().parents[1], jax.devices()[0]
    if device.platform == "gpu":
        name = run("nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    else:
        name = run("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor()
    return dict(commit=run("git", "-C", str(root), "rev-parse", "HEAD") or None,
                modified=bool(run("git", "-C", str(root), "status", "--porcelain", "--untracked-files=no")),
                device=f"{device.platform.upper()}, {name}" if name else device.platform.upper(), python=platform.python_version(),
                versions={p: installed(p) for p in ("jax", "jaxlib", "diffrax", "equinox", "numpy", "scipy")})


def timed(fn, *x, repeat=3):
    fn(*x)
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        jax.block_until_ready(fn(*x))
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def optimize(args):
    value_and_grad = jax.jit(jax.value_and_grad(make_loss(args)))
    theta0 = initial_controls(args.modes, args.seed, args.fixed_amplitudes)
    start = time.perf_counter()
    value_and_grad(theta0)
    compile_seconds = time.perf_counter() - start
    history, accepted = [], [0]

    def fun(theta):
        value, gradient = value_and_grad(jnp.asarray(theta))
        history.append(dict(evaluation=len(history), objective=float(value), seconds=time.perf_counter() - start))
        return float(value), np.asarray(gradient, dtype=float)

    result = minimize(fun, np.asarray(theta0), jac=True, method="L-BFGS-B", options=dict(maxiter=args.iterations),
                      callback=lambda *_: accepted.append(len(history) - 1))
    report = dict(mode="optimize", settings=vars(args) | {"output": str(args.output)}, provenance=provenance(),
                  initial=np.asarray(theta0).tolist(), optimized=np.asarray(result.x).tolist(), iterations=int(result.nit),
                  evaluations=int(result.nfev), message=str(result.message), history=history, accepted=accepted,
                  compile_seconds=compile_seconds, seconds=time.perf_counter() - start, runs={})
    for name, theta in (("initial", theta0), ("optimized", jnp.asarray(result.x))):
        output = solve(setup(theta, args.grid, args.hermite, args.t_max, fixed_amplitudes=args.fixed_amplitudes),
                       args.grid, args.hermite, timesteps=41)
        report["runs"][name] = dict(
            time=np.asarray(output["time"]).tolist(),
            inplane_magnetic=[float(inplane_magnetic_energy(output, i)) for i in range(41)],
            electric=[float(0.5 * Omega_ce**2 * jnp.sum(jnp.abs(output["Fk"][i, :3]) ** 2 * jnp.where(
                jnp.arange(args.grid // 2 + 1) == 0, 1.0, 2.0)[None, None, :, None])) for i in range(41)],
            kinetic=np.asarray(output["kinetic_energy_species"]).tolist(),
            jz_peak=[float(peak(current_density(output, i))) for i in range(41)],
            Jz_initial=np.asarray(current_density(output, 0)).tolist(), Jz_final=np.asarray(current_density(output, -1)).tolist(),
            objective=float(OBJECTIVES[args.objective](output)), steps=int(output["solver_stats"]["num_accepted_steps"]))
    path = args.output / f"optimize_{args.objective}{'_phases' if args.fixed_amplitudes else ''}.json"
    path.write_text(json.dumps(report, indent=1))
    print(f"{args.objective}: {history[0]['objective']:.6g} -> {report['runs']['optimized']['objective']:.6g} "
          f"in {result.nit} iterations ({report['seconds']:.0f} s); wrote {path}")
    return path


def validate(args):
    """Re-evaluate the frozen initial and optimised controls of an optimize report at other resolutions and tolerances."""
    source = json.loads(Path(args.files[0]).read_text())
    s, rows = source["settings"], []
    for grid, hermite in args.resolutions:
        for tolerance in args.tolerances:
            values = {}
            for label in ("initial", "optimized"):
                parameters = setup(jnp.asarray(source[label]), grid, hermite, s["t_max"], fixed_amplitudes=s.get("fixed_amplitudes", False))
                output = solve(parameters | dict(ode_tolerance=tolerance), grid, hermite)
                energy = np.asarray(output["total_energy"])
                values[label] = float(OBJECTIVES[s["objective"]](output))
                rows.append(dict(grid=grid, hermite=hermite, tolerance=tolerance, controls=label, objective=values[label],
                                 steps=int(output["solver_stats"]["num_accepted_steps"]),
                                 energy_error=float(abs(energy[-1] - energy[0]) / abs(energy[0]))))
            change = (values["optimized"] - values["initial"]) / abs(values["initial"])
            print(f"{grid}²×{hermite}³, tolerance {tolerance:.0e}: {values['initial']:.5g} -> {values['optimized']:.5g} "
                  f"({change:+.1%}), energy error {rows[-1]['energy_error']:.1e}")
    path = args.output / f"validate_{Path(args.files[0]).stem}.json"
    path.write_text(json.dumps(dict(mode="validate", source=args.files[0], source_settings=s, provenance=provenance(), rows=rows), indent=1))
    print("wrote", path)
    return path


def benchmark(args):
    """Gradient cost versus number of controls, and reverse-pass memory versus steps and resolution."""
    report = dict(mode="benchmark", settings=vars(args) | {"output": str(args.output)}, provenance=provenance(),
                  cost=[], memory_steps=[], memory_resolution=[], fd_step=[])
    loss = make_loss(args)
    forward, reverse = jax.jit(loss), jax.jit(jax.value_and_grad(loss))
    for M in args.control_counts:
        theta = initial_controls(M, args.seed, args.fixed_amplitudes)
        t_forward, t_reverse = timed(forward, theta), timed(reverse, theta)
        eps, basis = 1e-4, np.eye(theta.size)
        start = time.perf_counter()
        fd = np.array([float(forward(theta + eps * e) - forward(theta - eps * e)) / (2 * eps) for e in basis])
        t_fd = time.perf_counter() - start
        gradient = np.asarray(reverse(theta)[1])
        report["cost"].append(dict(controls=theta.size, forward_s=t_forward, reverse_s=t_reverse, fd_s=t_fd,
                                   fd_relative_error=float(np.linalg.norm(fd - gradient) / np.linalg.norm(gradient))))
        print(f"P={theta.size:4d}: forward {t_forward:.2f} s, AD gradient {t_reverse:.2f} s ({t_reverse/t_forward:.1f}x), "
              f"FD gradient {t_fd:.1f} s ({t_fd/t_reverse:.1f}x AD), |FD-AD|/|AD| {report['cost'][-1]['fd_relative_error']:.1e}")
        if M == args.control_counts[0]:
            for h in np.logspace(-9, -1, 9):
                fd = np.array([float(forward(theta + h * e) - forward(theta - h * e)) / (2 * h) for e in basis])
                report["fd_step"].append(dict(step=float(h), relative_error=float(np.linalg.norm(fd - gradient) / np.linalg.norm(gradient))))

    def workspace(fn, theta):
        analysis = jax.jit(fn).lower(theta).compile().memory_analysis()
        return None if analysis is None else analysis.temp_size_in_bytes / 2**20

    theta = initial_controls(args.modes, args.seed, args.fixed_amplitudes)
    dt = args.t_max / max(args.step_counts)
    for steps in args.step_counts:
        fixed = dict(dt=dt, adaptive_time_step=False, solver=Tsit5(), max_steps=steps + 1)
        loss = lambda th, K: OBJECTIVES[args.objective](solve(setup(th, args.grid, args.hermite, steps * dt, fixed_amplitudes=args.fixed_amplitudes),
                                                              args.grid, args.hermite, K, **fixed))
        row = dict(steps=steps, forward_MiB=workspace(lambda th: loss(th, 1), theta))
        for label, K in (("tape", steps), ("K8", 8), ("K32", 32)):
            row[f"reverse_{label}_MiB"] = workspace(jax.value_and_grad(lambda th, K=K: loss(th, K)), theta)
        report["memory_steps"].append(row)
        print("memory vs steps:", row)
    for grid, hermite in args.resolutions:
        loss = lambda th: OBJECTIVES[args.objective](solve(setup(th, grid, hermite, args.t_max, fixed_amplitudes=args.fixed_amplitudes),
                                                           grid, hermite, args.checkpoints))
        row = dict(grid=grid, hermite=hermite, state_MiB=(2 * hermite**3 + 6) * grid * (grid // 2 + 1) * 16 / 2**20,
                   forward_MiB=workspace(loss, theta), reverse_MiB=workspace(jax.value_and_grad(loss), theta))
        report["memory_resolution"].append(row)
        print("memory vs resolution:", row)
    stats = getattr(jax.devices()[0], "memory_stats", lambda: None)()
    report["device_peak_MiB"] = None if not stats else stats.get("peak_bytes_in_use", 0) / 2**20
    path = args.output / "benchmark.json"
    path.write_text(json.dumps(report, indent=1))
    print("wrote", path)
    return path


def plot(paths, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "pdf.fonttype": 42, "savefig.dpi": 300, "lines.linewidth": 1.6})
    for path in paths:
        report = json.loads(Path(path).read_text())
        stem = output / Path(path).stem
        device = report.get("provenance", {}).get("device", report.get("device", ""))
        if report["mode"] == "validate":
            continue
        if report["mode"] == "optimize":
            runs, s = report["runs"], report["settings"]
            fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), layout="constrained")
            maps = [("initial", "Jz_initial", "(a) $J_z$, initial controls, $t=0$"),
                    ("initial", "Jz_final", f"(b) $J_z$, initial controls, $t={s['t_max']:g}$"),
                    ("optimized", "Jz_final", f"(c) $J_z$, optimised controls, $t={s['t_max']:g}$")]
            limit = max(np.abs(runs[r][key]).max() for r, key, _ in maps)
            for ax, (run, key, title) in zip(axes[0], maps):
                im = ax.imshow(runs[run][key], origin="lower", extent=(0, Lx, 0, Ly), cmap="RdBu_r", vmin=-limit, vmax=limit)
                ax.set(title=title, xlabel="$x/d_e$", ylabel="$y/d_e$"); ax.grid(False)
            fig.colorbar(im, ax=axes[0], shrink=0.8, label="$J_z$")
            for run, color in (("initial", COLORS["initial"]), ("optimized", COLORS["optimized"])):
                t, r = np.array(runs[run]["time"]), runs[run]
                axes[1, 0].plot(t, np.array(r["inplane_magnetic"]) / r["inplane_magnetic"][0], color=color, label=f"{run}: magnetic energy")
                if "jz_peak" in r:
                    axes[1, 0].plot(t, np.array(r["jz_peak"]) / r["jz_peak"][0], color=color, ls=":", label=f"{run}: peak $J_z$")
                axes[1, 1].plot(t, np.array(r["kinetic"])[:, 0] - r["kinetic"][0][0], color=color, label=f"{run}, electrons")
                axes[1, 1].plot(t, np.array(r["kinetic"])[:, 1] - r["kinetic"][0][1], color=color, ls="--", label=f"{run}, ions")
            axes[1, 0].set(title="(d) In-plane magnetic energy and peak $J_z$, relative to $t=0$", xlabel="$t\\,\\omega_{pe}$")
            axes[1, 1].set(title="(e) Kinetic energy change", xlabel="$t\\,\\omega_{pe}$")
            h, ax = report["history"], axes[1, 2]
            evaluations = [r.get("evaluation", r.get("iteration")) for r in h]
            values = np.array([r["objective"] for r in h])
            accepted = report.get("accepted") or [i for i in range(len(h)) if values[i] <= values[:i + 1].min()]
            ax.plot(evaluations, values, "o", color=COLORS["muted"], ms=2.5, alpha=0.5, label="line-search trial")
            ax.plot([evaluations[i] for i in accepted], values[accepted], "o-", color=COLORS["third"], ms=3.5, label="accepted iterate")
            ax.set(title=f"(f) Objective '{s['objective']}'", xlabel="gradient evaluations (forward + reverse pass)")
            for a in (axes[1, 0], axes[1, 2]):
                a.legend(frameon=False, fontsize=7)
            axes[1, 1].legend(frameon=False, fontsize=7)
            controls = len(report["initial"])
            fig.suptitle(f"Orszag–Tang inverse design: {controls} {'phase ' if s.get('fixed_amplitudes') else ''}controls, "
                         f"{s['grid']}² × {s['hermite']}³, {runs['optimized']['steps']} Dopri8 steps, {report['seconds']:.0f} s, {device}", fontsize=10)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.4), layout="constrained")
            c = report["cost"]; P = [r["controls"] for r in c]
            ax = axes[0, 0]
            ax.loglog(P, [r["fd_s"] for r in c], "o-", color=COLORS["optimized"], label="centred finite differences (2P solves)")
            ax.loglog(P, [r["reverse_s"] for r in c], "o-", color=COLORS["initial"], label="reverse-mode AD (one solve + adjoint)")
            ax.loglog(P, [r["forward_s"] for r in c], "--", color=COLORS["muted"], label="one forward solve")
            ax.set(title="(a) Gradient wall time vs. number of controls P", xlabel="controls P", ylabel="seconds"); ax.legend(frameon=False, fontsize=7)
            m = report["memory_steps"]; N = [r["steps"] for r in m]; ax = axes[0, 1]
            ax.loglog(N, [r["reverse_tape_MiB"] for r in m], "o-", color=COLORS["optimized"], label="reverse, full tape")
            ax.loglog(N, [r["reverse_K32_MiB"] for r in m], "o-", color=COLORS["initial"], label="reverse, 32 checkpoints")
            ax.loglog(N, [r["reverse_K8_MiB"] for r in m], "o-", color=COLORS["third"], label="reverse, 8 checkpoints")
            ax.loglog(N, [r["forward_MiB"] for r in m], "--", color=COLORS["muted"], label="forward solve")
            ax.set(title="(b) Compiled workspace vs. time steps N", xlabel="time steps N", ylabel="MiB"); ax.legend(frameon=False, fontsize=7)
            r = report["memory_resolution"]; S = [x["state_MiB"] for x in r]; ax = axes[1, 0]
            ax.loglog(S, [x["reverse_MiB"] for x in r], "o-", color=COLORS["initial"], label="reverse-mode gradient")
            ax.loglog(S, [x["forward_MiB"] for x in r], "o-", color=COLORS["muted"], label="forward solve")
            for x in r:
                ax.annotate(f"{x['grid']}²×{x['hermite']}³\n×{x['reverse_MiB']/x['forward_MiB']:.1f}", (x["state_MiB"], x["reverse_MiB"]),
                            textcoords="offset points", xytext=(4, -14), fontsize=6.5)
            ax.set(title="(c) Workspace vs. state size (gradient/forward ratio)", xlabel="state size, MiB", ylabel="MiB"); ax.legend(frameon=False, fontsize=7)
            f = report["fd_step"]; ax = axes[1, 1]
            ax.loglog([x["step"] for x in f], [x["relative_error"] for x in f], "o-", color=COLORS["optimized"])
            ax.set(title="(d) Finite-difference error vs. step", xlabel="finite-difference step", ylabel="relative error to AD")
            fig.suptitle(f"Gradient cost and memory, {report['settings']['grid']}² × {report['settings']['hermite']}³, {device}", fontsize=10)
        for ext in ("png", "pdf"):
            fig.savefig(f"{stem}.{ext}")
        plt.close(fig)
        print("wrote", f"{stem}.png/.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["optimize", "validate", "benchmark", "plot"])
    parser.add_argument("files", nargs="*", help="JSON reports to plot, or the optimize report to validate")
    parser.add_argument("--objective", choices=OBJECTIVES, default="conversion")
    parser.add_argument("--modes", type=int, default=4, help="number of stream-function modes M (2M controls, M with --fixed-amplitudes)")
    parser.add_argument("--fixed-amplitudes", action="store_true", help="optimise phases only; every mode energy stays fixed")
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--hermite", type=int, default=4)
    parser.add_argument("--t-max", type=float, default=200.0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--checkpoints", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerances", type=float, nargs="+", default=[1e-7], help="ODE tolerances for validate")
    parser.add_argument("--control-counts", type=int, nargs="+", default=[2, 4, 8, 16, 32], metavar="M")
    parser.add_argument("--step-counts", type=int, nargs="+", default=[25, 50, 100, 200, 400], metavar="N")
    parser.add_argument("--resolutions", type=lambda s: tuple(map(int, s.split("x"))), nargs="+", default=[(16, 4), (32, 4), (32, 6), (64, 6)],
                        metavar="GRIDxHERMITE")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "plot":
        plot(args.files, args.output)
    elif args.mode == "validate":
        validate(args)
    else:
        plot([optimize(args) if args.mode == "optimize" else benchmark(args)], args.output)


if __name__ == "__main__":
    main()
