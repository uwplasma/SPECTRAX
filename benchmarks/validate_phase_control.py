"""Re-evaluate saved controls at independently refined space, velocity and time."""

import argparse
import importlib.util
import json
from pathlib import Path

import jax
import numpy as np

spec = importlib.util.spec_from_file_location(
    "phase_control", Path(__file__).parents[1] / "Examples" / "2D_phase_control.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def plot_validation(report, rows, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.3), layout="constrained")
    differences = np.asarray(report["finite_differences"])
    reference = report["directional_ad"]
    relative_error = abs((differences[:, 1] - reference) / reference)
    axes[0].loglog(differences[:, 0], relative_error, "o-", color="#0072B2")
    axes[0].set(xlabel="Centered-difference step", ylabel="Relative directional-gradient error",
                title="(a) Discrete-gradient verification")
    axes[0].grid(alpha=0.15)
    positions = np.arange(len(rows))
    for key, color, label in [("initial_phase", "#777777", "Baseline"),
                               ("optimized_phase", "#D55E00", "Optimized controls")]:
        axes[1].plot(positions, [-row[key]["objective"] for row in rows], "o-", color=color, label=label)
    axes[1].set(xticks=positions,
                xticklabels=[f"{r['grid']}² / {r['hermite']}³\n{r['steps']} steps" for r in rows],
                ylabel=r"$[K_e(T)-K_e(0)]/W_{B,\perp}(0)$",
                title="(b) Independent resolution refinement")
    if report.get("time_window") is not None:
        axes[1].set_ylabel(r"$\overline{\Delta K_e}/W_{B,\perp}(0)$")
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"gradient_validation.{extension}", dpi=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reoptimize", type=int, default=0,
                        help="L-BFGS iteration limit on the finest discretization (warm start)")
    args = parser.parse_args()
    report = json.loads((args.directory / "results.json").read_text())
    data = np.load(args.directory / "data.npz")
    grid, hermite, steps = report["grid"], report["hermite"], report["steps"]
    resolutions = [(grid, hermite, steps), (grid, hermite, 2 * steps),
                   (grid * 3 // 2, hermite, 2 * steps),
                   (grid, hermite + 2, 2 * steps),
                   (grid * 3 // 2, hermite + 2, 2 * steps)]
    rows = []
    for nx, nh, nt in resolutions:
        objective, terminal = example.problem(nx, nh, report["final_time"], nt,
                                                time_window=report.get("time_window"))
        fg = jax.jit(jax.value_and_grad(objective))
        row = dict(grid=nx, hermite=nh, steps=nt)
        for label in ("initial_phase", "optimized_phase"):
            phase = data[label]
            value, gradient = jax.block_until_ready(fg(phase))
            parameters = example.setup(phase, nx, nh, report["final_time"])
            state = jax.jit(terminal)(phase)
            C = np.asarray(state[0]).reshape(2, nh, nh, nh, nx, nx // 2 + 1, 1)
            # Spectral-tail indicator, not a bound on truncation or positivity.
            normalized = C * np.prod(np.asarray(parameters["alpha_s"]).reshape(2, 3), axis=1)[:, None, None, None, None, None, None]
            norm = np.sum(abs(normalized) ** 2)
            tail = np.zeros((nh, nh, nh), dtype=bool)
            tail[-1, :, :] = tail[:, -1, :] = tail[:, :, -1] = True
            tail_fraction = np.sum(abs(normalized[:, tail]) ** 2) / norm
            initial = example.quantities((parameters["Ck_0"], parameters["Fk_0"]), parameters, nx, nh)
            final = example.quantities(state, parameters, nx, nh)
            row[label] = dict(objective=float(value), gradient=np.asarray(gradient).tolist(),
                              hermite_tail_fraction=float(tail_fraction),
                              relative_energy_error=float((final[:4].sum() - initial[:4].sum()) / initial[:4].sum()))
        row["gain_ratio"] = row["optimized_phase"]["objective"] / row["initial_phase"]["objective"]
        rows.append(row)
        print(json.dumps(row), flush=True)
        (args.directory / "resolution_validation.json").write_text(json.dumps(rows, indent=2) + "\n")
        jax.clear_caches()
    plot_validation(report, rows, args.directory)
    if args.reoptimize > 0:
        from scipy.optimize import minimize
        result = minimize(fg, data["optimized_phase"], jac=True, method="L-BFGS-B",
                          options=dict(maxiter=args.reoptimize, ftol=1e-12, gtol=1e-9))
        refined = dict(grid=nx, hermite=nh, steps=nt, time_window=report.get("time_window"),
                       initial_objective=rows[-1]["initial_phase"]["objective"],
                       coarse_controls_objective=rows[-1]["optimized_phase"]["objective"],
                       final_objective=float(result.fun), optimized_phase=result.x.tolist(),
                       gradient=np.asarray(result.jac).tolist(), success=bool(result.success),
                       iterations=int(result.nit), message=str(result.message))
        (args.directory / "refined_optimization.json").write_text(json.dumps(refined, indent=2) + "\n")
        print(json.dumps(refined), flush=True)


if __name__ == "__main__":
    main()
