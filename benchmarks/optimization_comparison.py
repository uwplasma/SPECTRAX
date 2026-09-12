"""Matched phase-control optimization, reverse AD vs serial centered FD baseline.

python benchmarks/optimization_comparison.py --output comparison
python benchmarks/optimization_comparison.py --controls 128 --grid 32 --output p128
python benchmarks/optimization_comparison.py --plot-only --output comparison

Default physics: example's 16^2/4^3, 600 RK4 steps, weighted window [40,60].
Each seed/method runs serially in a fresh process. No automatic GPU selection.
FD step selection uses initial-point FD self-consistency. Diagnostic-only AD
reference compilation/execution are excluded from method-required total_seconds
and cold traces, but retained in pipeline_wall_seconds and explicit exclusions.
FD self-consistency calibration remains charged. Never infer a favorable AD speed
claim from FD's extra AD validation. No optimum-based tuning or callback solves.
Existing evidence is never overwritten. Interrupted worker JSON remains usable
with --plot-only; restarting an interrupted optimizer is deliberately unsupported.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter, time

ROOT = Path(__file__).resolve().parents[1]
FD_STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
FD_TOLERANCE = 1e-3
FD_GTOL_FRACTION = 0.1
DIRECTION_SEED = 3


def atomic_save(path, report):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def wall_times(report, elapsed):
    """Keep the measured pipeline clock and the primary method-required clock."""
    excluded = sum(report.get('diagnostic_only_seconds', {}).values())
    return dict(pipeline_wall_seconds=elapsed, diagnostic_excluded_seconds=excluded,
                total_seconds=elapsed - excluded)


def centered_fd(evaluate, x, step, role='fd_perturbation'):
    import numpy as np
    gradient = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        offset = np.zeros_like(x)
        offset[i] = step
        gradient[i] = (evaluate(x + offset, role) - evaluate(x - offset, role)) / (2 * step)
    return gradient


def calibrate(evaluate, x, reference, report, save, *, gtol=1e-9):
    """Fixed absolute-radian candidates; minimize FD-only baseline consistency error.

    Compare full and directional FD at successive decade steps, also checking
    directional FD against the current full FD projection. Normalize by current
    ||g_FD|| (floor 1e-14). The absolute allowance is one tenth of the declared
    L-BFGS infinity-norm gtol: bounding vector L2 error by this also bounds every
    component. Relative allowance remains 1e-3; the combined allowance avoids
    requiring relative precision below the optimizer's stopping accuracy.
    First candidate supplies a coarse reference only. Minimize discrepancy /
    (0.1*gtol + 1e-3*max(||g_FD||,1e-14)); fail closed above one.
    AD discrepancies are diagnostic only; they do not select or reject a step.
    This rule and all candidates are saved before any evaluations are performed.
    """
    import numpy as np
    direction = np.random.default_rng(DIRECTION_SEED).normal(size=len(x))
    direction /= np.linalg.norm(direction)
    ad_scale = max(float(np.linalg.norm(reference)), 1e-14)
    report['calibration'] = dict(candidates=list(FD_STEPS), tolerance=FD_TOLERANCE,
                                 optimizer_gtol=gtol, absolute_tolerance=FD_GTOL_FRACTION * gtol,
                                 direction=direction.tolist(), reference=reference.tolist(),
                                 rule='min max(successive FD vector change, successive directional FD change, current FD directional projection discrepancy) / (0.1*optimizer_gtol + 1e-3*max(norm(current_fd),1e-14)); accept score<=1; first candidate ineligible; first tie',
                                 rows=[])
    save()
    previous = None
    for step in FD_STEPS:
        gradient = centered_fd(evaluate, x, step, 'calibration_full')
        directional = (evaluate(x + step * direction, 'calibration_direction')
                       - evaluate(x - step * direction, 'calibration_direction')) / (2 * step)
        full_error = float(np.linalg.norm(gradient - reference) / ad_scale)
        direction_error = float(abs(directional - reference @ direction) / ad_scale)
        scale = max(float(np.linalg.norm(gradient)), 1e-14)
        discrepancy = None if previous is None else float(max(
            np.linalg.norm(gradient - previous[0]), abs(directional - previous[1]),
            abs(directional - gradient @ direction)))
        allowance = FD_GTOL_FRACTION * gtol + FD_TOLERANCE * scale
        score = None if discrepancy is None else discrepancy / allowance
        report['calibration']['rows'].append(dict(step=step, gradient=gradient.tolist(),
            directional=float(directional), full_relative_error=full_error,
            directional_scaled_error=direction_error, self_absolute_discrepancy=discrepancy,
            self_relative_discrepancy=None if discrepancy is None else discrepancy / scale,
            allowed_discrepancy=allowance, score=score))
        previous = gradient, directional
        save()
    best = min(report['calibration']['rows'][1:], key=lambda row: row['score'])
    if best['score'] > 1:
        raise ValueError('No preregistered FD step passed baseline consistency')
    report['calibration']['selected_step'] = best['step']
    save()
    return best['step']


class Evaluations:
    """Every executed scalar or value-and-gradient call counts one objective.

    Reverse replay/RK stages are not separate objective invocations. Each serial
    FD perturbation IS separate. Timers synchronize and include host transfer;
    persistence overhead is excluded from evaluation seconds but included in wall.
    """
    def __init__(self, scalar, value_grad, synchronize, report, save, origin):
        self.scalar, self.value_grad, self.synchronize = scalar, value_grad, synchronize
        self.report, self.save, self.origin = report, save, origin
        report['evaluations'] = []
        self.last_center = None

    def __call__(self, x, role, gradient=False):
        import numpy as np
        started = perf_counter()
        output = self.synchronize((self.value_grad if gradient else self.scalar)(x))
        value = float(output[0] if gradient else output)
        grad = np.asarray(output[1], dtype=float) if gradient else None
        seconds = perf_counter() - started
        diagnostic = (self.report.get('method') == 'fd'
                      and role == 'baseline_reference_first_execution')
        if diagnostic:
            self.report.setdefault('diagnostic_only_seconds', {})['ad_reference_execution'] = seconds
        finite = np.isfinite(value) and (grad is None or np.isfinite(grad).all())
        event = dict(index=len(self.report['evaluations']) + 1, role=role,
                     value=value if np.isfinite(value) else None, finite=bool(finite),
                     phase=np.asarray(x).tolist(), gradient=gradient, seconds=seconds,
                     diagnostic_only=diagnostic, **wall_times(self.report, perf_counter() - self.origin))
        self.report['evaluations'].append(event)
        self.report['objective_evaluations'] = len(self.report['evaluations'])
        self.report['synchronized_evaluation_seconds'] = (
            self.report.get('synchronized_evaluation_seconds', 0.0) + seconds)
        self.report.update(wall_times(self.report, event['pipeline_wall_seconds']))
        self.save()
        if not finite:
            raise ValueError('Nonfinite objective/gradient; see last evaluation')
        if role == 'optimization_center':
            self.last_center = (np.array(x, copy=True), value)
        return (value, grad) if gradient else value


def optimize(evaluate, x, method, step, options, report, save, origin):
    import numpy as np
    from scipy.optimize import minimize
    baseline_role = 'scalar_first_execution' if method == 'fd' else 'baseline_reference_first_execution'
    baseline = next(e for e in report['evaluations'] if e['role'] == baseline_role)
    report['accepted_trace'] = [dict(value=report['baseline_objective'], phase=x.tolist(),
        **wall_times(report, baseline['pipeline_wall_seconds']), warm_seconds=0.0)]
    report['gradient_calls'] = []
    started = perf_counter()

    def fg(phase):
        before = len(report['evaluations'])
        tick = perf_counter()
        if method == 'ad':
            value, gradient = evaluate(phase, 'optimization_center', gradient=True)
        else:
            value = evaluate(phase, 'optimization_center')
            gradient = centered_fd(evaluate, phase, step)
        report['gradient_calls'].append(dict(seconds=perf_counter() - tick,
            objective_evaluations=len(report['evaluations']) - before,
            **wall_times(report, perf_counter() - origin)))
        save()
        return value, gradient

    def callback(phase):
        cached, value = evaluate.last_center
        if not np.array_equal(phase, cached):
            raise RuntimeError('Optimizer callback point missing from evaluation cache')
        report['accepted_trace'].append(dict(value=value, phase=phase.tolist(),
            **wall_times(report, perf_counter() - origin), warm_seconds=perf_counter() - started))
        save()

    result = minimize(fg, x.copy(), jac=True, method='L-BFGS-B', bounds=None,
                      callback=callback, options=options.copy())
    report['optimization_seconds'] = perf_counter() - started
    report['result'] = dict(success=bool(result.success), message=str(result.message),
        status=int(result.status), iterations=int(result.nit), scipy_nfev=int(result.nfev),
        scipy_njev=int(result.njev), objective=float(result.fun), phase=result.x.tolist(),
        gradient_norm=float(np.linalg.norm(result.jac)))
    save()


def command_output(command):
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=10)
        return dict(command=command, returncode=result.returncode,
                    stdout=result.stdout.strip(), stderr=result.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=command, unavailable=str(error))


def device_snapshot(jax):
    gpu = any(d.platform == 'gpu' for d in jax.devices())
    result = dict(timestamp_unix=time(), devices=[dict(id=d.id, platform=d.platform,
        kind=d.device_kind, process_index=d.process_index, memory_stats=d.memory_stats())
        for d in jax.devices()],
        occupancy_note='Point samples only; GPU utilization is NOT SM occupancy. No profiler occupancy measurement; contention between snapshots is unknown.')
    if gpu:
        result['gpu_utilization'] = command_output(['nvidia-smi',
            '--query-gpu=uuid,name,driver_version,utilization.gpu,utilization.memory,memory.used,memory.total',
            '--format=csv'])
        result['compute_processes'] = command_output(['nvidia-smi',
            '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv'])
    return result


def provenance(jax):
    import spectrax
    import solvax
    import diffrax
    paths = {Path(__file__).resolve(), ROOT / 'Examples/2D_phase_control.py'}
    for module in (spectrax, solvax, diffrax):
        paths.update(Path(module.__file__).resolve().parent.rglob('*.py'))
    packages = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get('Name', 'unknown')
        packages[name] = dict(version=dist.version, direct_url=dist.read_text('direct_url.json'))
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
        pid=os.getpid(), cwd=str(Path.cwd()), argv=sys.argv, packages=packages,
        git_head=command_output(['git', 'rev-parse', 'HEAD']),
        git_status=command_output(['git', 'status', '--porcelain']),
        sources_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},
        imported_module_paths={name: str(getattr(module, '__file__', ''))
                               for name, module in sorted(sys.modules.items())
                               if name.split('.')[0] in ('spectrax', 'jax', 'jaxlib', 'solvax', 'diffrax', 'scipy', 'numpy')},
        environment={key: value for key, value in os.environ.items()
            if key.startswith(('JAX_', 'XLA_', 'CUDA_', 'NVIDIA_', 'OMP_', 'MKL_', 'OPENBLAS_'))
            or key == 'PYTHONPATH'}, jax_x64=bool(jax.config.jax_enable_x64),
        backend=jax.default_backend(), initial_device_snapshot=device_snapshot(jax))


def compile_function(jax, function, x, name, report, save):
    tick = perf_counter()
    compiled = jax.jit(function).lower(x).compile()
    report['compile_seconds'][name] = perf_counter() - tick
    if report.get('method') == 'fd' and name == 'value_grad':
        report['diagnostic_only_seconds']['ad_reference_compile'] = report['compile_seconds'][name]
    save()
    return compiled


def load_example():
    spec = importlib.util.spec_from_file_location('phase_control_comparison', ROOT / 'Examples/2D_phase_control.py')
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    return example


def worker(args):
    # Parent timestamp includes interpreter launch/import overhead in cold wall time.
    origin = args.started if args.started is not None else perf_counter()
    path = args.output / f'{args.method}-seed{args.seeds[0]}.json'
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite evidence: {path}')
    report = dict(schema_version=2, status='starting', method=args.method, seed=args.seeds[0],
                  config={k: v for k, v in vars(args).items() if k not in ('output', 'worker', 'started', 'plot_only')},
                  compile_seconds={}, diagnostic_only_seconds=dict(ad_reference_compile=0.0, ad_reference_execution=0.0),
                  timing_note='Fresh process; persistent JAX compilation cache disabled. Primary total_seconds = pipeline_wall_seconds - diagnostic_excluded_seconds. Only FD diagnostic AD reference compilation and synchronized execution are excluded; FD scalar compilation, first execution, self-consistency calibration, warmup, optimizer, launch/imports/provenance and all JSON/callback overhead remain charged. Exclusion is accounting, not a separately measured validation-free run; cache/allocator effects may remain. Warm optimization includes I/O and Python overhead. Final JSON write and parent plotting excluded. All objective counts include diagnostic calls.',
                  constraints='Fixed amplitudes/initial spectra and integrated energies via example.setup; unrestricted periodic phases, bounds=None',
                  fd_protocol=dict(steps=list(FD_STEPS), tolerance=FD_TOLERANCE,
                                   absolute_tolerance=FD_GTOL_FRACTION * args.gtol,
                                   direction_seed=DIRECTION_SEED, baseline_only=True),
                  limitations=['Serial centered FD baseline; no batching comparison.',
                               'No favorable AD speed claim may use FD extra AD validation cost: use method-required total_seconds, not pipeline_wall_seconds.',
                               'Warm timings exclude setup/calibration; total timings include them.',
                               'No convergence, acceleration or uncontended-device claim from this runner alone.'])
    save = lambda: atomic_save(path, report)
    save()
    try:
        import jax
        import numpy as np
        jax.config.update('jax_enable_x64', True)
        jax.config.update('jax_enable_compilation_cache', False)
        example = load_example()
        report['provenance'] = provenance(jax)
        x = np.random.default_rng(args.seeds[0]).uniform(-np.pi, np.pi, args.controls)
        report['initial_phase'] = x.tolist()
        objective, _ = example.problem(args.grid, args.hermite, args.window[1], args.steps,
                                      checkpoints=args.checkpoints, time_window=args.window)
        report['setup_seconds'] = perf_counter() - origin
        report['status'] = 'compiling'
        save()
        # Both compiles are measured; diagnostic reference compilation is excluded
        # only from FD's method-required clock, never from the pipeline wall clock.
        scalar = (compile_function(jax, objective, x, 'scalar', report, save)
                  if args.method == 'fd' else None)
        fg = compile_function(jax, jax.value_and_grad(objective), x, 'value_grad', report, save)
        evaluate = Evaluations(scalar, fg, jax.block_until_ready, report, save, origin)
        report['status'] = 'baseline_and_calibration'
        tick = perf_counter()
        value, reference = evaluate(x, 'baseline_reference_first_execution', gradient=True)
        report['baseline_objective'] = value
        step = None
        if args.method == 'fd':
            primal = evaluate(x, 'scalar_first_execution')
            if not np.isclose(primal, value, rtol=1e-10, atol=1e-12):
                raise ValueError('Scalar and AD primal disagree at baseline')
            report['baseline_objective'] = primal
            step = calibrate(evaluate, x, reference, report, save, gtol=args.gtol)
        report['baseline_calibration_pipeline_seconds'] = perf_counter() - tick
        report['baseline_calibration_seconds'] = (report['baseline_calibration_pipeline_seconds']
            - report['diagnostic_only_seconds']['ad_reference_execution'])
        report['warm_gradient_samples'] = []
        for _ in range(args.warm_repeats):
            tick = perf_counter()
            before = len(report['evaluations'])
            if args.method == 'ad':
                evaluate(x, 'warm_center', gradient=True)
            else:
                evaluate(x, 'warm_center')
                centered_fd(evaluate, x, step, 'warm_fd_perturbation')
            report['warm_gradient_samples'].append(dict(seconds=perf_counter() - tick,
                objective_evaluations=len(report['evaluations']) - before))
            save()
        report['pre_optimization_device_snapshot'] = device_snapshot(jax)
        report['pre_optimization_pipeline_seconds'] = perf_counter() - origin
        report['pre_optimization_seconds'] = wall_times(report, report['pre_optimization_pipeline_seconds'])['total_seconds']
        report['status'] = 'optimizing'
        report['optimizer_options'] = dict(maxiter=args.iterations, maxfun=args.maxfun,
            ftol=args.ftol, gtol=args.gtol, maxls=args.maxls, maxcor=args.maxcor)
        save()
        optimize(evaluate, x, args.method, step, report['optimizer_options'], report, save, origin)
        report['final_device_snapshot'] = device_snapshot(jax)
        report['status'] = 'complete'
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        events = report.get('evaluations', [])
        report['objective_evaluations'] = len(events)
        report['diagnostic_objective_evaluations'] = sum(e['diagnostic_only'] for e in events)
        report['method_objective_evaluations'] = len(events) - report['diagnostic_objective_evaluations']
        report['synchronized_evaluation_seconds'] = sum(e['seconds'] for e in events)
        report.update(wall_times(report, perf_counter() - origin))
        save()


def plot(output):
    """Read progressive JSON only: no JAX import, device initialization or solves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reports = [json.loads(p.read_text()) for p in sorted(output.glob('*-seed*.json'))]
    if not reports:
        raise ValueError(f'No worker JSON evidence in {output}')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for report in reports:
        if report['method'] == 'fd' and report.get('schema_version', 1) < 2:
            raise ValueError('Legacy FD JSON includes diagnostic AD cost; use schema_version 2 evidence for the method-required comparison')
        rows = list(report.get('accepted_trace', []))
        if not rows:
            rows = [dict(value=e['value'], total_seconds=e['total_seconds'], warm_seconds=0.)
                    for e in report.get('evaluations', [])
                    if e['role'] == ('scalar_first_execution' if report['method'] == 'fd'
                                     else 'baseline_reference_first_execution') and e['finite']]
        if not rows:
            continue
        if 'total_seconds' in report:
            rows.append(rows[-1] | dict(total_seconds=report['total_seconds'],
                                       warm_seconds=report.get('optimization_seconds', rows[-1]['warm_seconds'])))
        best = float('inf')
        gain = []
        for row in rows:
            best = min(best, row['value'])
            gain.append(-best)
        label = f"{report['method'].upper()} seed {report['seed']} ({report['status']})"
        for ax, key in zip(axes, ('total_seconds', 'warm_seconds')):
            ax.step([r[key] for r in rows], gain, where='post', label=label)
    for ax, title in zip(axes, ('Method-required total (FD diagnostic AD excluded)', 'Warm optimization wall time')):
        ax.set(xlabel='Seconds', ylabel='Best accepted normalized electron gain', title=title)
        if ax.lines:
            ax.legend(fontsize=7)
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'optimization_comparison.{extension}', dpi=180)
    plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--method', choices=('both', 'ad', 'fd'), default='both')
    p.add_argument('--seeds', type=int, nargs='+', default=[7])
    p.add_argument('--controls', type=int, default=8)
    p.add_argument('--grid', type=int, default=16)
    p.add_argument('--hermite', type=int, default=4)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--window', type=float, nargs=2, default=[40., 60.])
    p.add_argument('--checkpoints', type=int)
    p.add_argument('--iterations', type=int, default=30)
    p.add_argument('--maxfun', type=int, default=15000,
                   help='Same SciPy value/gradient request cap; actual FD scalar calls separately counted')
    p.add_argument('--maxls', type=int, default=20)
    p.add_argument('--maxcor', type=int, default=10)
    p.add_argument('--ftol', type=float, default=1e-12)
    p.add_argument('--gtol', type=float, default=1e-9)
    p.add_argument('--warm-repeats', type=int, default=1)
    p.add_argument('--plot-only', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--started', type=float, help=argparse.SUPPRESS)
    return p


def validate(args):
    import math
    if any(getattr(args, k) < 1 for k in ('controls', 'steps', 'iterations', 'maxfun', 'maxls', 'maxcor', 'warm_repeats')):
        raise ValueError('Counts must be positive')
    if args.hermite < 3 or args.grid < 6:
        raise ValueError('Require hermite >= 3 and grid >= 6')
    modes = [(i, j) for i in range(1, args.grid // 3)
             for j in range(-args.grid // 3 + 1, args.grid // 3) if j]
    if args.controls > len(modes):
        raise ValueError('Too many controls; increase --grid (e.g. 32 for 128 controls)')
    if not all(math.isfinite(v) for v in (*args.window, args.ftol, args.gtol)) or not 0 <= args.window[0] < args.window[1]:
        raise ValueError('Require finite 0 <= window start < end and tolerances')
    if args.ftol < 0 or args.gtol < 0 or (args.checkpoints is not None and args.checkpoints < 1):
        raise ValueError('Invalid tolerance/checkpoint count')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError('Seeds must be distinct nonnegative integers')
    if args.worker and (args.method == 'both' or len(args.seeds) != 1):
        raise ValueError('Worker requires one method and seed')


def main():
    args = parser().parse_args()
    if args.plot_only:
        plot(args.output)
        return
    validate(args)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    methods = ('ad', 'fd') if args.method == 'both' else (args.method,)
    # Preflight all paths so a later collision cannot leave a misleading half pair.
    for seed in args.seeds:
        for method in methods:
            path = args.output / f'{method}-seed{seed}.json'
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite evidence: {path}')
    for seed in args.seeds:
        for method in methods:
            command = [sys.executable, str(Path(__file__).resolve()), '--worker']
            settings = vars(args) | dict(method=method, seeds=[seed], started=perf_counter())
            for key, value in settings.items():
                if key in ('worker', 'plot_only') or value is None:
                    continue
                command.append('--' + key.replace('_', '-'))
                command.extend(str(v) for v in (value if isinstance(value, list) else [value]))
            env = os.environ.copy()
            env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
            subprocess.run(command, env=env, check=True)
    plot(args.output)


if __name__ == '__main__':
    main()
