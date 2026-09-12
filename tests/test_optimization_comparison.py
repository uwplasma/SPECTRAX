"""Tiny algebra checks only: no plasma integration or accelerator jobs."""
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'optimization_comparison', Path(__file__).parents[1] / 'benchmarks/optimization_comparison.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_defaults_and_resolved_controls():
    args = benchmark.parser().parse_args(['--output', '/unused'])
    benchmark.validate(args)
    assert args.window == [40., 60.]
    assert args.steps == 600 and args.controls == 8
    assert args.ftol == 1e-12 and args.gtol == 1e-9
    args.controls = 128
    with pytest.raises(ValueError, match='increase --grid'):
        benchmark.validate(args)
    args.grid = 32
    benchmark.validate(args)


@pytest.mark.parametrize('updates', [dict(window=[60., 40.]), dict(steps=0),
    dict(hermite=2), dict(ftol=float('nan')), dict(seeds=[7, 7]), dict(checkpoints=0)])
def test_invalid_plans(updates):
    args = benchmark.parser().parse_args(['--output', '/unused'])
    vars(args).update(updates)
    with pytest.raises(ValueError):
        benchmark.validate(args)


def algebra_evaluations(tmp_path):
    report = {}
    path = tmp_path / 'evidence.json'
    save = lambda: benchmark.atomic_save(path, report)
    center = np.array([0.3, -0.7, 0.1])
    scalar = lambda x: np.sum((x - center) ** 2)
    fg = lambda x: (scalar(x), 2 * (x - center))
    origin = perf_counter()
    evaluations = benchmark.Evaluations(scalar, fg, lambda x: x, report, save, origin)
    return report, save, evaluations, origin, center, path


def test_serial_fd_count_and_gradient(tmp_path):
    report, _, evaluate, _, center, _ = algebra_evaluations(tmp_path)
    x = np.array([0.5, 0.1, -0.2])
    gradient = benchmark.centered_fd(evaluate, x, 1e-4)
    np.testing.assert_allclose(gradient, 2 * (x - center), rtol=1e-10)
    assert len(report['evaluations']) == 2 * len(x)
    assert all(e['role'] == 'fd_perturbation' for e in report['evaluations'])
    assert all(e['seconds'] >= 0 for e in report['evaluations'])


def test_calibration_is_baseline_only_and_preregistered(tmp_path):
    report, save, evaluate, _, center, path = algebra_evaluations(tmp_path)
    x = np.array([0.5, 0.1, -0.2])
    snapshots = []

    def observe():
        save()
        snapshots.append(json.loads(path.read_text()))

    reference = 2 * (x - center)
    step = benchmark.calibrate(evaluate, x, reference, report, observe)
    assert snapshots[0]['calibration']['rows'] == []
    assert snapshots[0]['calibration']['candidates'] == list(benchmark.FD_STEPS)
    assert step in benchmark.FD_STEPS
    assert len(report['evaluations']) == len(benchmark.FD_STEPS) * (2 * len(x) + 2)
    assert step == min(report['calibration']['rows'][1:], key=lambda row: row['score'])['step']
    assert all(np.linalg.norm(np.array(e['phase']) - x) <= 0.010000001
               for e in report['evaluations'])


def test_step_selection_does_not_depend_on_ad_reference(tmp_path):
    report, save, evaluate, _, _, path = algebra_evaluations(tmp_path)
    step = benchmark.calibrate(evaluate, np.zeros(3), np.ones(3) * 100, report, save)
    second = benchmark.calibrate(evaluate, np.zeros(3), np.zeros(3), report, save)
    assert step == second
    saved = json.loads(path.read_text())
    assert len(saved['calibration']['rows']) == len(benchmark.FD_STEPS)


def test_inconsistent_fd_fails_closed(tmp_path):
    report = {}
    path = tmp_path / 'failed.json'
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match='No preregistered'):
        benchmark.calibrate(lambda x, role: rng.normal(), np.zeros(3), np.ones(3),
                            report, lambda: benchmark.atomic_save(path, report))
    assert 'selected_step' not in json.loads(path.read_text())['calibration']


def test_matched_optimizers_and_all_fd_calls_counted(tmp_path):
    options = dict(maxiter=10, maxfun=100, maxls=20, maxcor=10, ftol=1e-12, gtol=1e-9)
    results = []
    for method in ('ad', 'fd'):
        report, save, evaluate, origin, center, path = algebra_evaluations(tmp_path)
        x = np.array([2., -2., 1.])
        value, _ = evaluate(x, 'baseline_reference_first_execution', gradient=True)
        if method == 'fd':
            value = evaluate(x, 'scalar_first_execution')
        report['baseline_objective'] = value
        benchmark.optimize(evaluate, x, method, 1e-4, options, report, save, origin)
        result = report['result']
        assert result['success']
        np.testing.assert_allclose(result['phase'], center, atol=1e-8)
        multiplier = 1 if method == 'ad' else 1 + 2 * len(x)
        baseline_calls = 2 if method == 'fd' else 1
        assert len(report['evaluations']) == baseline_calls + multiplier * result['scipy_nfev']
        assert len(report['gradient_calls']) == result['scipy_nfev']
        assert all(c['objective_evaluations'] == multiplier for c in report['gradient_calls'])
        assert len(report['accepted_trace']) == 1 + result['iterations']
        times = [r['total_seconds'] for r in report['accepted_trace']]
        assert times == sorted(times)
        assert json.loads(path.read_text())['result'] == result
        results.append(result)
    assert abs(results[0]['objective'] - results[1]['objective']) < 1e-16


def test_interrupted_calibration_retains_completed_evaluations(tmp_path):
    report, save, evaluate, _, _, path = algebra_evaluations(tmp_path)

    def interrupted(x, role):
        if len(report['evaluations']) == 3:
            raise KeyboardInterrupt
        return evaluate(x, role)

    with pytest.raises(KeyboardInterrupt):
        benchmark.calibrate(interrupted, np.zeros(3), np.ones(3), report, save)
    assert len(json.loads(path.read_text())['evaluations']) == 3
    assert not path.with_suffix('.tmp').exists()


def test_nonfinite_saved_as_json_null(tmp_path):
    report, _, evaluate, _, _, path = algebra_evaluations(tmp_path)
    evaluate.scalar = lambda x: float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        evaluate(np.zeros(3), 'bad')
    assert json.loads(path.read_text())['evaluations'][0]['value'] is None


def test_tiny_cpu_aot_compilation_and_synchronization(tmp_path):
    import jax
    import jax.numpy as jnp
    # Explicit CPU context even when this unit suite is run on a GPU host.
    with jax.default_device(jax.devices('cpu')[0]):
        report = {'compile_seconds': {}}
        save = lambda: benchmark.atomic_save(tmp_path / 'compile.json', report)
        x = np.array([0.25, -0.5], dtype=np.float32)
        fg = benchmark.compile_function(jax, jax.value_and_grad(lambda y: jnp.sum(y ** 2)),
                                        x, 'fg', report, save)
        value, gradient = jax.block_until_ready(fg(x))
        np.testing.assert_allclose(value, 0.3125)
        np.testing.assert_allclose(gradient, 2 * x)
        assert report['compile_seconds']['fg'] >= 0


def test_plot_partial_json_without_simulation(tmp_path):
    benchmark.atomic_save(tmp_path / 'ad-seed7.json', dict(method='ad', seed=7,
        status='interrupted', accepted_trace=[dict(value=-0.1, total_seconds=5., warm_seconds=0.),
                                             dict(value=-0.2, total_seconds=6., warm_seconds=1.)]))
    benchmark.plot(tmp_path)
    assert (tmp_path / 'optimization_comparison.png').stat().st_size > 0
    assert (tmp_path / 'optimization_comparison.pdf').stat().st_size > 0


def test_complete_workers_on_tiny_algebra_only(tmp_path, monkeypatch):
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace
    observed = []

    def problem(grid, hermite, end, steps, **kwargs):
        observed.append((grid, hermite, end, steps, kwargs))
        return lambda x: jnp.sum((x - 0.2) ** 2), None

    monkeypatch.setattr(benchmark, 'load_example', lambda: SimpleNamespace(problem=problem))
    monkeypatch.setattr(benchmark, 'provenance', lambda jax: {'test': 'algebra'})
    monkeypatch.setattr(benchmark, 'device_snapshot', lambda jax: {'test': 'cpu'})
    previous_x64 = jax.config.jax_enable_x64
    previous_cache = jax.config.jax_enable_compilation_cache
    try:
        with jax.default_device(jax.devices('cpu')[0]):
            for method in ('ad', 'fd'):
                args = benchmark.parser().parse_args(['--output', str(tmp_path), '--worker',
                    '--method', method, '--controls', '2', '--iterations', '3'])
                benchmark.worker(args)
                report = json.loads((tmp_path / f'{method}-seed7.json').read_text())
                assert report['status'] == 'complete'
                assert report['result']['success']
                assert report['objective_evaluations'] == len(report['evaluations'])
                assert report['total_seconds'] >= report['pre_optimization_seconds'] + report['optimization_seconds']
                excluded = report['diagnostic_excluded_seconds']
                assert excluded == sum(report['diagnostic_only_seconds'].values())
                assert report['total_seconds'] == report['pipeline_wall_seconds'] - excluded
                assert report['total_seconds'] >= (sum(report['compile_seconds'].values())
                    - report['diagnostic_only_seconds']['ad_reference_compile'])
                assert (excluded > 0) == (method == 'fd')
                assert report['diagnostic_objective_evaluations'] == (1 if method == 'fd' else 0)
                for row in report['accepted_trace']:
                    assert row['total_seconds'] == row['pipeline_wall_seconds'] - excluded
                assert report['baseline_calibration_seconds'] == (
                    report['baseline_calibration_pipeline_seconds']
                    - report['diagnostic_only_seconds']['ad_reference_execution'])
                assert report['accepted_trace'][0]['total_seconds'] < report['pre_optimization_seconds']
                with pytest.raises(FileExistsError):
                    benchmark.worker(args)
    finally:
        jax.config.update('jax_enable_x64', previous_x64)
        jax.config.update('jax_enable_compilation_cache', previous_cache)
    assert observed[0] == observed[1]


def test_diagnostic_clock_excludes_only_fd_reference_execution(monkeypatch):
    report = dict(method='fd', diagnostic_only_seconds={'ad_reference_compile': 4.0})
    ticks = iter([10., 12., 13., 14., 15., 16.])
    monkeypatch.setattr(benchmark, 'perf_counter', lambda: next(ticks))
    evaluate = benchmark.Evaluations(lambda x: 1., lambda x: (1., np.ones(2)),
                                     lambda x: x, report, lambda: None, 0.)
    evaluate(np.zeros(2), 'baseline_reference_first_execution', gradient=True)
    assert report['pipeline_wall_seconds'] == 13.
    assert report['diagnostic_excluded_seconds'] == 6.
    assert report['total_seconds'] == 7.
    evaluate(np.zeros(2), 'calibration_full')
    assert report['pipeline_wall_seconds'] == 16.
    assert report['diagnostic_excluded_seconds'] == 6.
    assert report['total_seconds'] == 10.
    assert report['objective_evaluations'] == 2


@pytest.mark.parametrize('scale,gtol,passes', [(1e-10, 1e-9, True),
    (1e-10, 1e-14, False), (1., 1e-9, False)])
def test_calibration_absolute_allowance_tracks_optimizer_gtol(scale, gtol, passes):
    # Controlled 1% FD drift: negligible below optimizer gtol, unacceptable when
    # gradients are resolved. No plasma solve or floating-point cancellation needed.
    direction = np.random.default_rng(benchmark.DIRECTION_SEED).normal(size=2)
    direction /= np.linalg.norm(direction)
    values = []
    for index, step in enumerate(benchmark.FD_STEPS):
        gradient = scale * 1.01 ** index * np.array([1., -0.5])
        for component in gradient:
            values.extend([step * component, -step * component])
        values.extend([step * (gradient @ direction), -step * (gradient @ direction)])
    iterator = iter(values)
    report = {}
    run = lambda: benchmark.calibrate(lambda x, role: next(iterator), np.zeros(2),
        np.zeros(2), report, lambda: None, gtol=gtol)
    if passes:
        assert run() in benchmark.FD_STEPS
    else:
        with pytest.raises(ValueError, match='No preregistered'):
            run()
    assert report['calibration']['absolute_tolerance'] == 0.1 * gtol


def test_cold_plot_uses_method_required_trace(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    calls = []
    original = Axes.step

    def capture(self, x, y, *args, **kwargs):
        calls.append(list(x))
        return original(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, 'step', capture)
    benchmark.atomic_save(tmp_path / 'fd-seed7.json', dict(schema_version=2,
        method='fd', seed=7, status='complete', total_seconds=9., pipeline_wall_seconds=19.,
        optimization_seconds=2., accepted_trace=[
            dict(value=1., total_seconds=6., pipeline_wall_seconds=16., warm_seconds=0.),
            dict(value=0., total_seconds=8., pipeline_wall_seconds=18., warm_seconds=2.)]))
    benchmark.plot(tmp_path)
    assert calls[0] == [6., 8., 9.]
    assert calls[1] == [0., 2., 2.]
