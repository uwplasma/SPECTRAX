"""Serial scalar/refinement checks of frozen tail controls; no saved trajectory.

One primary optimized reverse gradient and one velocity-refined directional JVP
are evaluated. Local positivity and held-out-time validation are separate studies.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter
import traceback

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    'benchmarks/validate_tail_control.py', 'Examples/2D_tail_control.py',
    'Examples/2D_phase_control.py', 'spectrax/_velocity_observables.py',
    'spectrax/_autodiff.py', 'spectrax/_model.py', 'spectrax/_simulation.py',
    'spectrax/_initialization.py', 'spectrax/_initialize_maxwellian.py',
)


def source_hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}


def load_tail():
    spec = importlib.util.spec_from_file_location('tail_control', ROOT / 'Examples/2D_tail_control.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def difference(reference, value, atol, rtol):
    absolute = abs(value - reference)
    return dict(reference=reference, value=value, absolute_error=absolute,
                relative_error=absolute / abs(reference) if reference else None,
                tolerance=atol + rtol * abs(reference),
                passed=absolute <= atol + rtol * abs(reference))


def evaluate_case(row, tail, initial, optimized, direction, common, save):
    """Keep executables local to one case; return only host scalars/small arrays."""
    settings = row['settings']
    row['stage'] = 'problem_setup'
    save()
    start = perf_counter()
    loss, _, spec = tail.problem(initial, **settings, **common)
    row['setup_seconds'] = perf_counter() - start
    row['tail_spec'] = dict(threshold=float(spec.threshold), width=float(spec.width),
                            normalization=float(spec.normalization))
    row['compile_seconds'], row['evaluations'] = {}, {}

    def compile_function(fn, name):
        row['stage'] = 'compile_' + name
        save()
        start = perf_counter()
        executable = jax.jit(fn).lower(initial).compile()
        row['compile_seconds'][name] = perf_counter() - start
        save()
        return executable

    def evaluate(executable, controls, label, method):
        row['stage'] = 'evaluate_' + label
        save()
        start = perf_counter()
        result = jax.block_until_ready(executable(controls))
        elapsed = perf_counter() - start
        value = result[0] if method != 'scalar_loss' else result
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f'Nonfinite {label} loss')
        row['evaluations'][label] = dict(loss=value, window_tail_gain=-value,
                                         seconds=elapsed, method=method)
        save()
        return result

    scalar = compile_function(loss, 'scalar_loss')
    evaluate(scalar, initial, 'initial_phase', 'scalar_loss')
    if row['name'] == 'primary':
        del scalar
        executable = compile_function(jax.value_and_grad(loss), 'value_and_grad')
        _, grad = evaluate(executable, optimized, 'optimized_phase', 'value_and_grad')
        grad = np.asarray(grad)
        if not np.all(np.isfinite(grad)):
            raise ValueError('Nonfinite primary gradient')
        row['optimized_loss_gradient'] = grad.tolist()
        row['gradient_norm'] = float(np.linalg.norm(grad))
        row['directional_loss_derivative'] = float(grad @ direction)
    elif row['name'] == 'velocity_refined':
        del scalar
        def directional(theta):
            return jax.jvp(loss, (theta,), (jnp.asarray(direction),))
        executable = compile_function(directional, 'value_and_jvp')
        _, tangent = evaluate(executable, optimized, 'optimized_phase', 'value_and_jvp')
        tangent = float(tangent)
        if not np.isfinite(tangent):
            raise ValueError('Nonfinite velocity-refined directional derivative')
        row['directional_loss_derivative'] = tangent
    else:
        evaluate(scalar, optimized, 'optimized_phase', 'scalar_loss')
    baseline, best = (row['evaluations'][key]['window_tail_gain']
                      for key in ('initial_phase', 'optimized_phase'))
    row['benefit'] = best - baseline
    row['positive_benefit'] = bool(row['benefit'] > 0)
    row['status'], row['stage'] = 'complete', 'complete'
    save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('tail-validation'),
                        help='New directory; an existing directory is never reused')
    parser.add_argument('--grid', type=int, default=24)
    parser.add_argument('--hermite', type=int, default=16)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--spatial-grid', type=int, default=32)
    parser.add_argument('--velocity-hermite', type=int, default=20)
    parser.add_argument('--quadrature', type=int, help='Default: input report quadrature, or 96')
    parser.add_argument('--atol', type=float, default=1e-10)
    parser.add_argument('--rtol', type=float, default=0.05)
    args = parser.parse_args(argv)
    if (min(args.grid, args.steps) < 1 or args.hermite < 3
            or args.spatial_grid <= args.grid or args.velocity_hermite <= args.hermite):
        parser.error('Require positive grid/steps, H>=3, and strictly finer space/velocity grids')
    if not all(np.isfinite(x) and x >= 0 for x in (args.atol, args.rtol)):
        parser.error('Tolerances must be finite and nonnegative')
    raw = args.controls_file.read_bytes()
    controls = json.loads(raw)
    initial = np.asarray(controls['initial_phase'], dtype=float)
    optimized = np.asarray(controls['optimized_phase'], dtype=float)
    if (initial.ndim != 1 or not initial.size or initial.shape != optimized.shape
            or not np.all(np.isfinite(initial)) or not np.all(np.isfinite(optimized))):
        parser.error('Phases must be matching nonempty finite vectors')
    window = tuple(float(x) for x in controls.get('window', (40., 60.)))
    nu = float(controls['nu'])
    energies = np.asarray(controls['initial_energies'], dtype=float)
    quadrature = args.quadrature if args.quadrature is not None else controls.get('quadrature', 96)
    if (len(window) != 2 or not all(np.isfinite(window)) or not 0 <= window[0] < window[1]
            or not np.isfinite(nu) or nu < 0 or energies.shape != (4,)
            or not np.all(np.isfinite(energies)) or not isinstance(quadrature, int) or quadrature < 2):
        parser.error('Invalid window, nu, initial_energies or quadrature')
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error('Use a fresh output directory; existing output is never overwritten')
    direction = np.random.default_rng(3).normal(size=initial.size)
    direction /= np.linalg.norm(direction)
    cases = [dict(name=name, status='pending', settings=dict(grid=g, hermite=h, steps=n))
             for name, g, h, n in (
                 ('primary', args.grid, args.hermite, args.steps),
                 ('dt_halved', args.grid, args.hermite, 2*args.steps),
                 ('spatial_refined', args.spatial_grid, args.hermite, 2*args.steps),
                 ('velocity_refined', args.grid, args.velocity_hermite, args.steps))]
    report = dict(status='running', controls_file=str(args.controls_file.resolve()),
                  controls_sha256=hashlib.sha256(raw).hexdigest(), source_sha256=source_hashes(),
                  initial_phase=initial.tolist(), optimized_phase=optimized.tolist(),
                  initial_energies=energies.tolist(), window=window, nu=nu, quadrature=quadrature,
                  checkpoints=None, direction_seed=3, direction=direction.tolist(),
                  atol=args.atol, rtol=args.rtol,
                  gate_formula='abs(value-reference) <= atol + rtol*abs(reference)',
                  interpretation='loss=-weighted signed tail gain; benefit=optimized gain-baseline gain. '
                  'One directional check does not certify full-gradient convergence, stationarity, '
                  'global optimality, quadrature convergence or local positivity. '
                  'Fixed nonzero nu changes the Hermite filter when H changes.',
                  cases=cases, errors=[])

    def save():
        # Only this run's newly reserved directory is updated; atomic snapshots.
        temporary = args.output / 'results.json.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output / 'results.json')

    save()
    tail = load_tail()
    report.update(jax_version=jax.__version__, jax_enable_x64=bool(jax.config.jax_enable_x64),
                  devices=[str(d) for d in jax.devices()])
    common = dict(window=window, nu=nu, quadrature=quadrature, checkpoints=None)
    for row in cases:
        row['status'] = 'running'
        h = row['settings']['hermite']
        row['dt'] = window[1] / row['settings']['steps']
        row['cubic_damping_coefficient'] = nu/((h-1)*(h-2)*(h-3)) if h > 3 else 0.
        save()
        try:
            evaluate_case(row, tail, initial, optimized, direction, common, save)
        except Exception as error:
            row['status'] = 'error'
            row['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            report['errors'].append(dict(case=row['name'], stage=row.get('stage'), **row['error']))
        finally:
            jax.clear_caches()
            save()
        print(json.dumps(dict(case=row['name'], status=row['status'], benefit=row.get('benefit'))), flush=True)
    primary = cases[0]
    report['comparisons'] = {}
    if primary['status'] == 'complete':
        for row in cases[1:]:
            if row['status'] != 'complete':
                continue
            comparison = dict(benefit=difference(primary['benefit'], row['benefit'], args.atol, args.rtol))
            for label in ('initial_phase', 'optimized_phase'):
                comparison[label+'_gain'] = difference(
                    primary['evaluations'][label]['window_tail_gain'],
                    row['evaluations'][label]['window_tail_gain'], args.atol, args.rtol)
            if row['name'] == 'velocity_refined':
                comparison['directional_derivative'] = difference(
                    primary['directional_loss_derivative'], row['directional_loss_derivative'], args.atol, args.rtol)
            report['comparisons'][row['name']] = comparison
    report['source_sha256_after'] = source_hashes()
    report['source_unchanged'] = report['source_sha256_after'] == report['source_sha256']
    report['status'] = 'complete' if not report['errors'] else 'completed_with_errors'
    save()
    return 1 if report['errors'] or not report['source_unchanged'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
