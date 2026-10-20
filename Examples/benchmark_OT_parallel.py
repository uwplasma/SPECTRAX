"""Strong-scaling benchmark of a SHORT Orszag-Tang run: the same GLOBAL problem on several device layouts.

    python Examples/benchmark_OT_parallel.py [--n 64] [--t-max 0.2] [--hermite 5] [--layouts 1,1,1,1 1,2,1,1 ...]

Each layout (ps, px, py, pz) runs in a FRESH subprocess (the device count is fixed per process) built from
``input_OT_parallel.toml`` with the problem below. A child measures, with ``block_until_ready`` and no progress bar, plots
or export: initial-state projection, first call (compile + setup + solve), warmed call (solve + the diagnostics it
computes), diagnostics alone, and one host gather of the saved Fk. Compile/setup is first minus warmed call. Speedup
T1/TD and efficiency T1/(D*TD) are printed only when the D=1 layout was measured. CPU logical devices share one host's
cores, so CPU numbers check the machinery, not GPU scaling.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import tomllib

# Global problem (not part of the 32x32 functionality input): size, short physical interval, layouts to compare.
N = 64                     # Nx = Ny
T_MAX = 0.2
HERMITE = 5                # Nn = Nm = Np
LAYOUTS = ["1,1,1,1", "1,2,1,1", "2,1,1,1", "1,2,2,1", "2,2,1,1"]  # D=1, D=2 (spatial, species), D=4 (spatial, hybrid)
HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_TOML = os.path.join(HERE, "input_OT_parallel.toml")


def child(path):
    """Run one case in this fresh process and print a single ``RESULT {json}`` line."""
    sys.path.append(HERE)
    import jax
    import numpy as np
    from diffrax import NoProgressMeter
    import OT_parallel
    from spectrax import load_parameters, simulation, diagnostics
    ip, sp = load_parameters(path)  # configures the devices before any array exists
    sp["progress_meter"] = NoProgressMeter()
    def timed(f):  # synchronized wall time of f() and its result
        t0 = time.perf_counter()
        result = jax.block_until_ready(f())
        return time.perf_counter() - t0, result

    t_init, _ = timed(lambda: (OT_parallel.initial_state(ip, sp), ip["Ck_0"], ip["Fk_0"]))
    t_first, _ = timed(lambda: simulation(ip, **sp)["Ck"])  # compile + setup + solve
    t_warm, out = timed(lambda: simulation(ip, **sp))  # solve (compiled), including the diagnostics it computes
    t_diag, _ = timed(lambda: (diagnostics(out), out["total_energy"]))  # diagnostics alone, warmed
    t_host, _ = timed(lambda: np.asarray(out["Fk"]))  # one host gather of the saved field history
    shard = lambda a: tuple(a.sharding.shard_shape(a.shape))
    print("RESULT " + json.dumps(dict(
        D=sp["parallel"].num_devices, layout=sp["parallel"].shape, backend=jax.devices()[0].platform, ck=shard(out["Ck"]),
        fk=shard(out["Fk"]), steps=int(out["num_steps"]), init=t_init, setup=t_first - t_warm, solve=t_warm, diag=t_diag, host=t_host,
        energy=float(out["total_energy"][-1]), dtype=str(out["Ck"].dtype))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=N, help="Nx = Ny")
    parser.add_argument("--t-max", type=float, default=T_MAX)
    parser.add_argument("--hermite", type=int, default=HERMITE, help="Nn = Nm = Np")
    parser.add_argument("--layouts", nargs="+", default=LAYOUTS, help="ps,px,py,pz entries")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return child(args.child)

    with open(INPUT_TOML, "rb") as f:
        base = tomllib.load(f)
    base["input_parameters"]["t_max"] = args.t_max
    base["solver_parameters"].update(Nx=args.n, Ny=args.n, Nn=args.hermite, Nm=args.hermite, Np=args.hermite)
    env = {k: v for k, v in os.environ.items() if k not in ("JAX_NUM_CPU_DEVICES", "XLA_FLAGS")}
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for entry in args.layouts:
            ps, px, py, pz = map(int, entry.split(","))
            toml = {**base, "parallel": dict(device="cpu", num_devices=ps * px * py * pz, species_partitions=ps, mesh_shape=[px, py, pz])}
            path = os.path.join(tmp, f"{ps}{px}{py}{pz}.toml")
            with open(path, "w") as f:
                for table, values in toml.items():
                    f.write(f"[{table}]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in values.items()))
            print(f"Running layout (ps,px,py,pz)=({entry}) in a fresh process...", flush=True)
            run = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", path], env=env, capture_output=True, text=True)
            lines = [l for l in run.stdout.splitlines() if l.startswith("RESULT ")]
            if run.returncode or not lines:
                sys.exit(f"layout {entry} failed:\n{run.stdout}\n{run.stderr}")
            rows.append(json.loads(lines[0][7:]))

    print(f"\nOrszag-Tang N={args.n}x{args.n}, Hermite {args.hermite}^3, t_max={args.t_max}, dtype {rows[0]['dtype']}, backend {rows[0]['backend']}")
    t1 = next((r["solve"] for r in rows if r["D"] == 1), None)
    print(f"{'D':>2} {'(ps,px,py,pz)':>14} {'local Ck':>26} {'local Fk':>22} {'steps':>5} {'init s':>7} {'setup s':>8} {'solve s':>8} {'ms/step':>8} {'diag s':>7} {'host s':>7} {'T1/TD':>6} {'T1/(D TD)':>9}")
    for r in rows:
        sp, eff = (f"{t1 / r['solve']:6.2f}", f"{t1 / (r['D'] * r['solve']):9.2f}") if t1 else ("   n/a", "      n/a")
        print(f"{r['D']:>2} {str(tuple(r['layout'])):>14} {str(tuple(r['ck'])):>26} {str(tuple(r['fk'])):>22} {r['steps']:>5} {r['init']:7.2f} "
              f"{r['setup']:8.2f} {r['solve']:8.2f} {1e3 * r['solve'] / r['steps']:8.1f} {r['diag']:7.3f} {r['host']:7.3f} {sp} {eff}")
    print("solve = warmed call, including the diagnostics it computes; speedups compare solve times of the same global problem "
          "(step counts may differ slightly between layouts).")


if __name__ == "__main__":
    main()
