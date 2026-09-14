"""Forward-mode sensitivities of the Orszag–Tang objectives to physical parameters.

For the initial and optimised controls of a report from ``2D_Orszag_Tang_optimization.py``, ``jax.jacfwd`` with
``adjoint=ForwardMode()`` gives d ln|objective| / d ln(parameter) for the collision frequency, mass ratio, guide field
and in-plane field amplitude, checked against centred differences.

  python 2D_Orszag_Tang_sensitivity.py results/optimize_conversion.json --snapshots 21
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import ForwardMode
from matplotlib.ticker import StrMethodFormatter

from orszag_tang_control import COLORS, OBJECTIVES, deltaB, mi_me, plt, provenance, save, setup, solve

PARAMETERS = {"nu": (1.0, "collision frequency $\\nu$"), "mass_ratio": (mi_me, "mass ratio $m_i/m_e$"),
              "guide_field": (1.0, "guide field $B_z$"), "amplitude": (deltaB, "in-plane field $\\delta B$")}


def sensitivities(source, snapshots):
    """Log-sensitivities of every objective at the report's initial and optimised controls, with the error against FD."""
    s, base = source["settings"], np.array([value for value, _ in PARAMETERS.values()])

    def objectives(physics, theta):
        parameters = setup(theta, s["grid"], s["hermite"], s["t_max"], fixed_amplitudes=s.get("fixed_amplitudes", False),
                           tolerance=s.get("tolerance", s.get("tolerances", [1e-7])[0]), **dict(zip(PARAMETERS, physics)))
        output = solve(parameters, s["grid"], s["hermite"], timesteps=snapshots, adjoint=ForwardMode())
        return jnp.stack([objective(output) for objective in OBJECTIVES.values()])

    evaluate, jacobian = jax.jit(objectives), jax.jit(jax.jacfwd(objectives))
    report = dict(source_settings=s, snapshots=snapshots, provenance=provenance(), objectives=list(OBJECTIVES), parameters=list(PARAMETERS),
                  base=base.tolist(), runs={})
    for label in ("initial", "optimized"):
        theta = jnp.asarray(source[label])
        values, J = np.asarray(evaluate(jnp.asarray(base), theta)), np.asarray(jacobian(jnp.asarray(base), theta))
        fd = np.stack([(np.asarray(evaluate(jnp.asarray(base + h * e), theta)) - np.asarray(evaluate(jnp.asarray(base - h * e), theta))) / (2 * h)
                       for h, e in zip(1e-4 * base, np.eye(base.size))], axis=1)
        log_sensitivity = J * base / values[:, None]
        error = np.abs(fd - J) * base / np.abs(values)[:, None]   # log-sensitivity units: relative errors mislead near zero
        report["runs"][label] = dict(values=values.tolist(), jacobian=J.tolist(), finite_difference=fd.tolist(),
                                     log_sensitivity=log_sensitivity.tolist(), log_sensitivity_error_vs_fd=error.tolist())
        for name, row, err in zip(OBJECTIVES, log_sensitivity, error):
            print(f"{label:9s} {name:15s} " + "  ".join(f"{p} {v:+.3g}" for p, v in zip(PARAMETERS, row)) + f"  | max |error| vs FD {err.max():.1e}")
    return report


def plot(report):
    names, y = report["objectives"], np.arange(len(PARAMETERS))
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 2.8), layout="constrained", sharey=True)
    for j, (ax, name) in enumerate(zip(axes, names)):
        for k, run in enumerate(("initial", "optimized")):
            ax.barh(y + (k - 0.5) * 0.4, np.array(report["runs"][run]["log_sensitivity"])[j], height=0.36, color=COLORS[run], label=f"{run} controls")
        ax.axvline(0, color=COLORS["muted"], lw=0.8)
        ax.set(title=f"'{name}'", xlabel="d ln|objective| / d ln(parameter)"); ax.grid(axis="y", visible=False)
        ax.locator_params(axis="x", nbins=4); ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2g}"))
    axes[0].set_yticks(y, [label for _, label in PARAMETERS.values()]); axes[0].invert_yaxis(); axes[0].legend(frameon=False, fontsize=7)
    s = report["source_settings"]
    fig.suptitle(f"Sensitivities at the '{s['objective']}' controls, {s['grid']}² × {s['hermite']}³, {report['snapshots']} snapshots, "
                 f"{report['provenance']['device']}", fontsize=10)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path, help="JSON report from 2D_Orszag_Tang_optimization.py")
    parser.add_argument("--snapshots", type=int, default=2, help="saved states per solve, used by mean_conversion")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report = sensitivities(json.loads(args.report.read_text()), args.snapshots) | dict(source=str(args.report))
    save(report, args.output / f"sensitivity_{args.report.stem}.json", plot)


if __name__ == "__main__":
    main()
