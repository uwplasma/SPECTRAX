"""Frozen-control refinement check for Examples/2D_Orszag_Tang_optimization.py reports.

Re-evaluates the initial and optimised controls of each report at other resolutions and ODE
tolerances (forward solves only) and reports the objective, energy error, and energy channels.
Usage: SPECTRAX_ROOT=<checkout> python validate_refinement.py OUT.json REPORT.json [...] --resolutions 32x4 64x6 --tolerances 1e-7
"""
import argparse, importlib, json, os, sys, time
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
root = os.environ["SPECTRAX_ROOT"]
sys.path.insert(0, root); sys.path.insert(0, os.path.join(root, "Examples"))
ex = importlib.import_module("2D_Orszag_Tang_optimization")

parser = argparse.ArgumentParser()
parser.add_argument("out"); parser.add_argument("reports", nargs="+")
parser.add_argument("--resolutions", nargs="+", default=["32x4"])
parser.add_argument("--tolerances", nargs="+", type=float, default=[1e-7])
args = parser.parse_args()
p8 = lambda J: float(jnp.mean(J ** 8) ** (1 / 8))
rows = json.load(open(args.out)) if os.path.exists(args.out) else []
done = {(r["objective"], r["grid"], r["hermite"], r["tolerance"], r["controls"]) for r in rows}
for path in args.reports:
    report = json.load(open(path)); s = report["settings"]
    for res in args.resolutions:
        grid, hermite = map(int, res.split("x"))
        for tol in args.tolerances:
            for label in ("initial", "optimized"):
                if (s["objective"], grid, hermite, tol, label) in done:
                    continue
                params = ex.setup(jnp.asarray(report[label]), grid, hermite, s["t_max"])
                params["ode_tolerance"] = tol
                start = time.perf_counter()
                out = ex.solve(params, grid, hermite, timesteps=2)
                jax.block_until_ready(out["Fk"])
                E = np.asarray(out["total_energy"]); kin = np.asarray(out["kinetic_energy_species"])
                row = dict(objective=s["objective"], source=os.path.basename(path), trained=f"{s['grid']}x{s['hermite']}",
                           grid=grid, hermite=hermite, tolerance=tol, controls=label, t_max=s["t_max"],
                           value=float(ex.OBJECTIVES[s["objective"]](out)), steps=int(out["solver_stats"]["num_accepted_steps"]),
                           seconds=time.perf_counter() - start, total_energy_rel_error=float(abs(E[-1] - E[0]) / abs(E[0])),
                           inplane_magnetic=[float(ex.inplane_magnetic_energy(out, 0)), float(ex.inplane_magnetic_energy(out, -1))],
                           jz_p8=[p8(ex.current_density(out, 0)), p8(ex.current_density(out, -1))],
                           kinetic_change=(kin[-1] - kin[0]).tolist(), device=str(jax.devices()[0]))
                rows.append(row); json.dump(rows, open(args.out, "w"), indent=1)
                print(json.dumps({k: row[k] for k in ("objective", "grid", "hermite", "tolerance", "controls", "value", "steps",
                                                      "seconds", "total_energy_rel_error")}), flush=True)
for obj in sorted({r["objective"] for r in rows}):
    for g, h, t in sorted({(r["grid"], r["hermite"], r["tolerance"]) for r in rows if r["objective"] == obj}):
        v = {r["controls"]: r["value"] for r in rows if (r["objective"], r["grid"], r["hermite"], r["tolerance"]) == (obj, g, h, t)}
        if len(v) == 2:
            print(f"SUMMARY {obj:13s} {g}x{h} tol={t:.0e}: initial {v['initial']:.5g} optimized {v['optimized']:.5g} "
                  f"relative change {(v['optimized'] - v['initial']) / abs(v['initial']):+.1%}")
