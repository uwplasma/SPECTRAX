"""Gradient-based inverse design of the 2D Orszag–Tang vortex.

L-BFGS-B with ``jax.value_and_grad`` through ``simulation`` finds the initial in-plane magnetic field
(``orszag_tang_control.py``) that optimises one objective:

  conversion       in-plane magnetic energy at t_max over its initial value (minimised)
  peak_current     smooth maximum of the out-of-plane current density at t_max (maximised)
  mean_conversion  in-plane magnetic energy averaged over ``--snapshots`` saved states (minimised)

``--validate 64x8 128x6`` re-simulates the initial and optimised controls at other resolutions. Each run writes a
JSON report and a figure; ``--report`` re-plots, and optionally validates, an existing report.

  python 2D_Orszag_Tang_optimization.py --objective conversion --modes 8 --grid 32 --hermite 4 --t-max 200
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from orszag_tang_control import (COLORS, Lx, Ly, OBJECTIVES, STYLE, current_density, initial_controls, inplane_magnetic_energy,
                                 peak, provenance, setup, solve)


def optimize(args):
    """L-BFGS-B; each gradient is one forward solve plus a checkpointed reverse pass."""
    def loss(theta):
        parameters = setup(theta, args.grid, args.hermite, args.t_max, fixed_amplitudes=args.fixed_amplitudes, tolerance=args.tolerance)
        return OBJECTIVES[args.objective](solve(parameters, args.grid, args.hermite, args.checkpoints, timesteps=args.snapshots))

    report = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, provenance=provenance())
    value_and_grad, theta0 = jax.jit(jax.value_and_grad(loss)), initial_controls(args.modes, args.seed, args.fixed_amplitudes)
    start = time.perf_counter()
    value_and_grad(theta0)
    report["compile_seconds"], history, accepted = time.perf_counter() - start, [], [0]

    def fun(theta):
        value, gradient = value_and_grad(jnp.asarray(theta))
        history.append(dict(evaluation=len(history), objective=float(value), seconds=time.perf_counter() - start))
        return float(value), np.asarray(gradient, dtype=float)

    result = minimize(fun, np.asarray(theta0), jac=True, method="L-BFGS-B", options=dict(maxiter=args.iterations),
                      callback=lambda *_: accepted.append(len(history) - 1))
    report.update(initial=np.asarray(theta0).tolist(), optimized=np.asarray(result.x).tolist(), iterations=int(result.nit),
                  evaluations=int(result.nfev), message=str(result.message), history=history, accepted=accepted,
                  seconds=time.perf_counter() - start, runs={})
    for name, theta in (("initial", theta0), ("optimized", jnp.asarray(result.x))):
        output = solve(setup(theta, args.grid, args.hermite, args.t_max, fixed_amplitudes=args.fixed_amplitudes, tolerance=args.tolerance),
                       args.grid, args.hermite, timesteps=41)
        report["runs"][name] = dict(
            time=np.asarray(output["time"]).tolist(), kinetic=np.asarray(output["kinetic_energy_species"]).tolist(),
            inplane_magnetic=[float(inplane_magnetic_energy(output, i)) for i in range(41)],
            jz_peak=[float(peak(current_density(output, i))) for i in range(41)],
            Jz_initial=np.asarray(current_density(output, 0)).tolist(), Jz_final=np.asarray(current_density(output, -1)).tolist(),
            objective=float(result.fun) if name == "optimized" else history[0]["objective"], steps=int(output["solver_stats"]["num_accepted_steps"]))
    print(f"{args.objective}: {history[0]['objective']:.6g} -> {result.fun:.6g} in {result.nit} iterations ({report['seconds']:.0f} s)")
    return report


def validate(report, resolutions):
    """Re-simulate the frozen initial and optimised controls at other resolutions, with the total-energy error."""
    s, rows = report["settings"], []
    tolerance = s.get("tolerance", s.get("tolerances", [1e-7])[0])
    for grid, hermite in resolutions:
        values = {}
        for label in ("initial", "optimized"):
            parameters = setup(jnp.asarray(report[label]), grid, hermite, s["t_max"], fixed_amplitudes=s.get("fixed_amplitudes", False),
                               tolerance=tolerance)
            output = solve(parameters, grid, hermite, timesteps=s.get("snapshots", 2))
            energy, values[label] = np.asarray(output["total_energy"]), float(OBJECTIVES[s["objective"]](output))
            rows.append(dict(grid=grid, hermite=hermite, tolerance=tolerance, controls=label, objective=values[label],
                             steps=int(output["solver_stats"]["num_accepted_steps"]), energy_error=float(abs(energy[-1] - energy[0]) / abs(energy[0]))))
        change = (values["optimized"] - values["initial"]) / abs(values["initial"])
        print(f"{grid}²×{hermite}³: {values['initial']:.5g} -> {values['optimized']:.5g} ({change:+.1%}), energy error {rows[-1]['energy_error']:.1e}")
    report["validation"] = rows


def plot(report, path):
    """Current maps, energy traces and the objective history."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)
    runs, s = report["runs"], report["settings"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), layout="constrained")
    maps = [("initial", "Jz_initial", "(a) $J_z$, initial controls, $t=0$"), ("initial", "Jz_final", f"(b) $J_z$, initial controls, $t={s['t_max']:g}$"),
            ("optimized", "Jz_final", f"(c) $J_z$, optimised controls, $t={s['t_max']:g}$")]
    limit = max(np.abs(runs[r][key]).max() for r, key, _ in maps)
    for ax, (run, key, title) in zip(axes[0], maps):
        im = ax.imshow(runs[run][key], origin="lower", extent=(0, Lx, 0, Ly), cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set(title=title, xlabel="$x/d_e$", ylabel="$y/d_e$"); ax.grid(False)
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="$J_z$")
    for run in ("initial", "optimized"):
        t, r = np.array(runs[run]["time"]), runs[run]
        axes[1, 0].plot(t, np.array(r["inplane_magnetic"]) / r["inplane_magnetic"][0], color=COLORS[run], label=f"{run}: magnetic energy")
        axes[1, 0].plot(t, np.array(r["jz_peak"]) / r["jz_peak"][0], color=COLORS[run], ls=":", label=f"{run}: peak $J_z$")
        axes[1, 1].plot(t, np.array(r["kinetic"])[:, 0] - r["kinetic"][0][0], color=COLORS[run], label=f"{run}, electrons")
        axes[1, 1].plot(t, np.array(r["kinetic"])[:, 1] - r["kinetic"][0][1], color=COLORS[run], ls="--", label=f"{run}, ions")
    axes[1, 0].set(title="(d) Relative to $t=0$", xlabel="$t\\,\\omega_{pe}$")
    axes[1, 1].set(title="(e) Kinetic energy change", xlabel="$t\\,\\omega_{pe}$")
    values, accepted = np.array([h["objective"] for h in report["history"]]), report["accepted"]
    axes[1, 2].plot(values, "o", color=COLORS["muted"], ms=2.5, alpha=0.5, label="line-search trial")
    axes[1, 2].plot(accepted, values[accepted], "o-", color=COLORS["third"], ms=3.5, label="accepted iterate")
    axes[1, 2].set(title=f"(f) Objective '{s['objective']}'", xlabel="gradient evaluations (forward + reverse pass)")
    for ax in axes[1]:
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Orszag–Tang inverse design: {len(report['initial'])} {'phase ' if s.get('fixed_amplitudes') else ''}controls, "
                 f"{s['grid']}² × {s['hermite']}³, {runs['optimized']['steps']} Dopri8 steps, {report['seconds']:.0f} s, "
                 f"{report['provenance']['device']}", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objective", choices=OBJECTIVES, default="conversion")
    parser.add_argument("--modes", type=int, default=4, help="stream-function modes M: 2M controls, or M with --fixed-amplitudes")
    parser.add_argument("--fixed-amplitudes", action="store_true", help="optimise phases only, so every mode energy stays fixed")
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--hermite", type=int, default=4)
    parser.add_argument("--t-max", type=float, default=200.0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--checkpoints", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--snapshots", type=int, default=2, help="saved states per solve, t=0 and t_max included")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate", nargs="+", default=[], metavar="GRIDxHERMITE", type=lambda r: tuple(map(int, r.split("x"))))
    parser.add_argument("--report", type=Path, help="re-plot, and optionally --validate, an existing report instead of optimising")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text()) if args.report else optimize(args)
    if args.validate:
        validate(report, args.validate)
    args.output.mkdir(parents=True, exist_ok=True)
    s = report["settings"]
    path = args.output / f"optimize_{s['objective']}{'_phases' if s.get('fixed_amplitudes') else ''}.json"
    path.write_text(json.dumps(report, indent=1))
    plot(report, path)
    print("wrote", path, "and its figure")


if __name__ == "__main__":
    main()
