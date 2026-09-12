"""Stream mechanism diagnostics of saved phase controls without a state trajectory."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from spectrax import simulation_final
from spectrax._initialization import initialize_simulation_parameters
from spectrax._energization import species_energization
from spectrax._velocity_observables import preregistered_tail_spec, velocity_diagnostics, spatial_negative_mass_diagnostics

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('phase_control', ROOT / 'Examples/2D_phase_control.py')
phase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase)


def gpu_processes():
    try:
        return subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                        '--format=csv,noheader'], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), layout='constrained')
    for label, color in [('initial_phase', '#777777'), ('optimized_phase', '#0072B2')]:
        rows = report['runs'][label]['samples']
        t = np.array([r['time'] for r in rows])
        for key, style in [('kinetic_energy', '-'), ('bulk_energy', '--'), ('internal_energy', ':')]:
            values = np.array([r[key][0] for r in rows])
            axes[0, 0].plot(t, (values-values[0])/report['magnetic_normalization'], style,
                            color=color, label=f'{label.split("_")[0]} {key.split("_")[0]}')
        values = np.array([r['kinetic_energy'][0] for r in rows])
        work = np.array([r['integrated_power'][0][0] for r in rows])
        collision = np.array([r['integrated_power'][1][0] for r in rows])
        axes[0, 1].plot(t, values-values[0]-work-collision, color=color, label=label.split('_')[0])
        tail = np.array([r['velocity'][str(report['quadratures'][-1])]['objective'] for r in rows])
        axes[1, 0].plot(t, tail, color=color, label=label.split('_')[0])
        neg = np.array([r['velocity'][str(report['quadratures'][-1])]['negative_mass_fraction'] for r in rows])
        axes[1, 1].semilogy(t, np.maximum(neg, 1e-20), color=color)
    axes[0, 0].set(title='(a) Electron energy partition', ylabel=r'$\Delta K/W_{B,\perp}(0)$')
    axes[0, 0].legend(frameon=False, fontsize=6, ncol=2)
    axes[0, 1].set(title='(b) Electron work balance', ylabel=r'$\Delta K_e-\int(P_E+P_{coll})dt$')
    axes[0, 1].legend(frameon=False)
    axes[1, 0].set(title='(c) Fixed-threshold smooth tail', ylabel='Tail energy / initial thermal energy')
    axes[1, 1].set(title='(d) Averaged distribution negativity', ylabel='Negative mass / density')
    for ax in axes.flat:
        ax.set_xlabel('Time')
        ax.axvspan(40, 60, color='#0072B2', alpha=.06)
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'mechanism.{ext}', dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls-file', type=Path, default=ROOT/'benchmarks/results/window_control_cpu.json')
    parser.add_argument('--grid', type=int, default=16)
    parser.add_argument('--hermite', type=int, default=4)
    parser.add_argument('--time', type=float, default=70.)
    parser.add_argument('--steps', type=int, default=700)
    parser.add_argument('--samples', type=int, default=70, help='Equal intervals; steps must be divisible')
    parser.add_argument('--nu', type=float, default=1.)
    parser.add_argument('--quadratures', nargs='+', type=int, default=[24, 48, 96])
    parser.add_argument('--local-quadratures', nargs='*', type=int, default=[24, 48])
    parser.add_argument('--output', type=Path, default=Path('mechanism'))
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if args.plot_only:
        plot(json.loads((args.output/'results.json').read_text()), args.output)
        return
    if args.samples < 1 or args.steps % args.samples or args.time <= 0:
        parser.error('Positive intervals dividing steps and positive time required')
    if (args.output/'results.json').exists():
        parser.error('Use a fresh output directory; existing results are preserved')
    args.output.mkdir(parents=True, exist_ok=True)
    controls = json.loads(args.controls_file.read_text())
    grid, h = args.grid, args.hermite
    dt_sample = args.time / args.samples
    base = phase.setup(np.asarray(controls['initial_phase']), grid, h, args.time)
    base['nu'] = args.nu
    base = initialize_simulation_parameters(base, grid, grid, 1, h, h, h, 2)
    diagnostic = jax.jit(lambda C, F: species_energization(C, F, base, Nx=grid, Nn=h, Nm=h, Np=h))
    first = diagnostic(base['Ck_0'], base['Fk_0'])
    density = float(first['density'][0].mean())
    initial_thermal = float(first['internal_energy'][0])
    tail_spec = preregistered_tail_spec(initial_thermal / density, density)
    velocity = {q: jax.jit(lambda c, q=q: velocity_diagnostics(
        c, base['alpha_s'][:3], base['u_s'][:3], spec=tail_spec, quadrature_order=q)) for q in args.quadratures}

    local_negative = {q: jax.jit(lambda C, q=q: spatial_negative_mass_diagnostics(
        C, base['alpha_s'], base['u_s'], Nx=grid, Nn=h, Nm=h, Np=h,
        Ns=2, quadrature_order=q)) for q in args.local_quadratures}

    def integrand(t, C, F):
        d = species_energization(C, F, base, Nx=grid, Nn=h, Nm=h, Np=h)
        return jnp.stack([d['electric_power'], d['collision_power']])

    @jax.jit
    def advance(C, F):
        params = dict(base, Ck_0=C, Fk_0=F, t_max=dt_sample)
        return simulation_final(params, steps=args.steps//args.samples, Nx=grid, Ny=grid,
                                Nn=h, Nm=h, Np=h, checkpointing=False, integrand=integrand)

    paths = ['spectrax/_autodiff.py', 'spectrax/_model.py', 'spectrax/_simulation.py',
             'spectrax/_energization.py', 'spectrax/_velocity_observables.py',
             'Examples/2D_phase_control.py', 'benchmarks/energization_mechanism.py']
    report = dict(grid=grid, hermite=h, time=args.time, steps=args.steps, samples=args.samples,
                  nu=args.nu, quadratures=args.quadratures, local_quadratures=args.local_quadratures, device=jax.devices()[0].device_kind,
                  jax_version=jax.__version__, controls_sha256=hashlib.sha256(args.controls_file.read_bytes()).hexdigest(),
                  source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                  tail_spec=dict(threshold=float(tail_spec.threshold), width=float(tail_spec.width),
                                 normalization=float(tail_spec.normalization)),
                  magnetic_normalization=float(controls['initial_energies'][1]-.125),
                  gpu_processes_start=gpu_processes(), runs={})
    start = perf_counter()
    executable = advance.lower(base['Ck_0'], base['Fk_0']).compile()
    report['advance_compile_seconds'] = perf_counter()-start
    coefficients = {}
    for label in ['initial_phase', 'optimized_phase']:
        params = phase.setup(np.asarray(controls[label]), grid, h, args.time)
        C, F = params['Ck_0'], params['Fk_0']
        accumulated = jnp.zeros((2, 2))
        rows, saved = [], []
        report['runs'][label] = dict(samples=rows, phase=controls[label])
        for index in range(args.samples+1):
            if index:
                C, F, work = jax.block_until_ready(executable(C, F))
                accumulated += work
            d = diagnostic(C, F)
            c0 = C[:, 0, 0, 0].real.reshape(2, h, h, h)
            saved.append(np.asarray(c0))
            row = dict(time=index*dt_sample, integrated_power=np.asarray(accumulated).tolist(),
                       density_min=float(d['density'].min()), velocity={})
            for key in ['kinetic_energy', 'bulk_energy', 'internal_energy', 'j_dot_e',
                        'electric_power', 'collision_power']:
                row[key] = np.asarray(d[key]).tolist()
            row['field_energies'] = np.asarray(phase.quantities((C, F), base, grid, h)[:2]).tolist()
            for q, fn in velocity.items():
                row['velocity'][str(q)] = {k: float(v) for k, v in fn(c0[0]).items()}
            if index in (0, args.samples):
                row['local_negativity'] = {str(q): {k: float(v) for k, v in fn(C).items()}
                                           for q, fn in local_negative.items()}
            rows.append(row)
            (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
        coefficients[label] = np.asarray(saved)
        print(json.dumps(dict(run=label, final=rows[-1])), flush=True)
    report['gpu_processes_end'] = gpu_processes()
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez(args.output/'mean_coefficients.npz', **coefficients)
    plot(report, args.output)


if __name__ == '__main__':
    main()
