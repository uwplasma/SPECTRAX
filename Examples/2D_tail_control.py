"""Optimize added smooth suprathermal electron energy at fixed initial energies.

This pilot is not an acceleration claim: check distribution positivity and
Hermite/time/space convergence separately before interpreting a tail optimum.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from spectrax import simulation_final
from spectrax._velocity_observables import make_tail_objective, preregistered_tail_spec

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('phase_control', Path(__file__).with_name('2D_phase_control.py'))
phase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase)


def problem(reference, *, grid=24, hermite=8, steps=1200, window=(40., 60.),
            quadrature=96, nu=1., checkpoints=None):
    """Freeze threshold/normalization; subtract each control's own initial tail."""
    start, end = window
    if not 0 <= start < end:
        raise ValueError('Window must have 0 <= start < end')
    params = phase.setup(reference, grid, hermite, end)
    # Initial prescribed electron Maxwellian internal mean energy, independent
    # of controlled spatial drift: m_e sum(alpha_e**2)/4. Density is one.
    thermal = float(jnp.sum(params['alpha_s'][:3]**2)/4)
    tail_spec = preregistered_tail_spec(thermal)
    tail = make_tail_objective(params['alpha_s'], params['u_s'], Nn=hermite,
                               Nm=hermite, Np=hermite, Ns=2, spec=tail_spec,
                               quadrature_order=quadrature)

    def run(theta):
        p = phase.setup(theta, grid, hermite, end)
        p['nu'] = nu
        initial = tail(p['Ck_0'])

        def integrand(t, C, F):
            z = (2*t-start-end)/(end-start)
            weight = jnp.where(jnp.abs(z) < 1, 15/(8*(end-start))*(1-z*z)**2, 0.)
            return weight*(tail(C)-initial)

        return simulation_final(p, steps=steps, Nx=grid, Ny=grid,
                                Nn=hermite, Nm=hermite, Np=hermite,
                                checkpoints=checkpoints, integrand=integrand)

    return lambda theta: -run(theta)[2], run, tail_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=int, default=24)
    parser.add_argument('--hermite', type=int, default=8)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--window', nargs=2, type=float, default=[40., 60.])
    parser.add_argument('--quadrature', type=int, default=96)
    parser.add_argument('--nu', type=float, default=1.)
    parser.add_argument('--controls', type=int, default=8)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--checkpoints', type=int)
    parser.add_argument('--evaluate-only', action='store_true', help='Objective/gradient pilot without optimization')
    parser.add_argument('--output', type=Path, default=Path('tail-control'))
    args = parser.parse_args()
    if min(args.steps, args.controls, args.quadrature, args.iterations) < 1:
        parser.error('steps, controls, quadrature and iterations must be positive')
    if (args.output/'results.json').exists():
        parser.error('Use a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    theta = np.random.default_rng(args.seed).uniform(-np.pi, np.pi, args.controls)
    loss, run, tail_spec = problem(theta, grid=args.grid, hermite=args.hermite, steps=args.steps,
                                   window=args.window, quadrature=args.quadrature, nu=args.nu,
                                   checkpoints=args.checkpoints)
    start = perf_counter()
    compiled = jax.jit(jax.value_and_grad(loss)).lower(theta).compile()
    compile_seconds = perf_counter()-start
    start = perf_counter()
    value, gradient = jax.block_until_ready(compiled(theta))
    first_seconds = perf_counter()-start
    history = [dict(seconds=0., objective=float(value))]
    cache = {}

    def fg(x):
        value, gradient = jax.block_until_ready(compiled(x))
        cache['x'], cache['value'] = np.array(x), float(value)
        return float(value), np.asarray(gradient)

    def callback(x):
        value = cache['value'] if np.array_equal(x, cache.get('x')) else fg(x)[0]
        history.append(dict(seconds=perf_counter()-start, objective=value))
        (args.output/'history.json').write_text(json.dumps(history, indent=2)+'\n')

    report = dict(grid=args.grid, hermite=args.hermite, steps=args.steps, window=args.window,
                  nu=args.nu, quadrature=args.quadrature, seed=args.seed,
                  device=jax.devices()[0].device_kind, jax_version=jax.__version__,
                  tail_spec=dict(threshold=tail_spec.threshold, width=tail_spec.width,
                                 normalization=tail_spec.normalization),
                  compile_seconds=compile_seconds, first_evaluation_seconds=first_seconds,
                  initial_phase=theta.tolist(), initial_objective=float(value),
                  initial_gradient=np.asarray(gradient).tolist(),
                  interpretation='Signed smooth tail GAIN, not final tail population; requires external validity checks')
    start = perf_counter()
    if not args.evaluate_only:
        result = minimize(fg, theta, jac=True, method='L-BFGS-B', callback=callback,
                          options=dict(maxiter=args.iterations, ftol=1e-12, gtol=1e-9))
        report.update(optimized_phase=result.x.tolist(), final_objective=float(result.fun),
                      final_gradient=np.asarray(result.jac).tolist(), success=bool(result.success),
                      iterations=int(result.nit), message=str(result.message),
                      optimization_seconds=perf_counter()-start)
        a = phase.setup(theta, args.grid, args.hermite, args.window[1])
        b = phase.setup(result.x, args.grid, args.hermite, args.window[1])
        e0 = phase.quantities((a['Ck_0'], a['Fk_0']), a, args.grid, args.hermite)[:4]
        e1 = phase.quantities((b['Ck_0'], b['Fk_0']), b, args.grid, args.hermite)[:4]
        report['initial_energy_constraint_error'] = float(jnp.max(jnp.abs(e1-e0)))
    report['history'] = history
    paths = ['Examples/2D_tail_control.py', 'Examples/2D_phase_control.py',
             'spectrax/_velocity_observables.py', 'spectrax/_autodiff.py', 'spectrax/_model.py']
    report['source_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
