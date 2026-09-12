"""CPU-only instantaneous local-tail value/gradient compiler memory and timing.

Run with SPECTRAX's dependencies installed:
    python benchmarks/local_tail_memory.py --output local-tail-memory.json

Defaults: square spatial grids 12/24, Nz=1, H=3 per velocity axis, q=16/32.
There are no physics rollouts. JSON goes to stdout unless --output is supplied.
XLA memory_analysis is a compiler buffer estimate, NOT actual peak device or
process memory. Four points cannot establish asymptotic scaling or causally
isolate rematerialization; no uncheckpointed implementation is benchmarked.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grids', type=int, nargs='+', default=[12, 24],
                        help='Nx=Ny for each square grid; Nz=1')
    parser.add_argument('--quadratures', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--repeats', type=int, default=3,
                        help='Synchronized timed calls after one warmup')
    parser.add_argument('--output', type=Path, help='New JSON file; default stdout')
    args = parser.parse_args()
    if min(args.grids) < 1 or min(args.quadratures) < 3 or args.repeats < 1:
        parser.error('Require positive grids/repeats and quadratures >= 3 for H=3')
    if len(set(args.grids)) != len(args.grids) or len(set(args.quadratures)) != len(args.quadratures):
        parser.error('Grid and quadrature lists must not contain duplicates')
    if args.output is not None and args.output.exists():
        parser.error('Output exists; choose a new JSON path')

    # Set before importing JAX or SPECTRAX, even when the shell selects a GPU.
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import jax
    import jax.numpy as jnp
    import jaxlib
    import numpy as np

    jax.config.update('jax_enable_x64', True)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from spectrax import _local_tail, _velocity_observables

    for module in (_local_tail, _velocity_observables):
        if Path(module.__file__).resolve().parent != root / 'spectrax':
            raise RuntimeError('Benchmark must import this checkout of SPECTRAX')
    if any(device.platform != 'cpu' for device in jax.devices()):
        raise RuntimeError('This benchmark is CPU-only')

    alpha = np.array([.7, 1.1, 1.3])
    shift = np.array([.3, -.4, .2])
    spec = _velocity_observables.TailSpec(1., .5, 1.7)
    rows = []
    for grid in args.grids:
        # Deterministic, spatially varying density and all three first modes:
        # U is state-dependent, so reverse AD exercises both f and the kernel.
        y, x = np.meshgrid(2*np.pi*np.arange(grid)/grid,
                           2*np.pi*np.arange(grid)/grid, indexing='ij')
        local = np.zeros((1, 3, 3, 3, grid, grid, 1))
        local[0, 0, 0, 0, ..., 0] = (1.2 + .1*np.cos(x)*np.cos(y))/np.prod(alpha)
        local[0, 0, 0, 1, ..., 0] = .12 + .05*np.sin(x)
        local[0, 0, 1, 0, ..., 0] = -.08 + .04*np.cos(y)
        local[0, 1, 0, 0, ..., 0] = .06 + .03*np.sin(x+y)
        local[0, 0, 0, 2, ..., 0] = .1*np.cos(x-y)
        ck = jnp.asarray(np.fft.rfftn(local, axes=(-1, -3, -2), norm='forward')
                         .reshape(27, grid, grid//2+1, 1))
        jax.block_until_ready(ck)
        for order in args.quadratures:
            objective = _local_tail.make_local_tail_objective(
                alpha, shift, Nx=grid, Nn=3, Nm=3, Np=3, Ns=1,
                spec=spec, mass=1., quadrature_order=order)
            fn = jax.jit(jax.value_and_grad(objective))
            start = perf_counter()
            lowered = fn.lower(ck)
            lowering_seconds = perf_counter() - start
            start = perf_counter()
            executable = lowered.compile()
            compile_seconds = perf_counter() - start
            analysis = executable.memory_analysis()
            fields = ('argument_size_in_bytes', 'output_size_in_bytes',
                      'temp_size_in_bytes', 'alias_size_in_bytes',
                      'generated_code_size_in_bytes', 'host_argument_size_in_bytes',
                      'host_output_size_in_bytes', 'host_temp_size_in_bytes',
                      'host_alias_size_in_bytes')
            memory = {name: (int(getattr(analysis, name))
                             if analysis is not None and getattr(analysis, name, None) is not None
                             else None) for name in fields}
            buffer_names = fields[:4]
            memory['estimated_total_buffer_bytes'] = (
                memory[buffer_names[0]] + memory[buffer_names[1]]
                + memory[buffer_names[2]] - memory[buffer_names[3]]
                if all(memory[name] is not None for name in buffer_names) else None)
            start = perf_counter()
            result = jax.block_until_ready(executable(ck))
            warmup_seconds = perf_counter() - start
            samples = []
            for _ in range(args.repeats):
                start = perf_counter()
                result = jax.block_until_ready(executable(ck))
                samples.append(perf_counter() - start)
            value, gradient = result
            if not np.isfinite(float(value)) or not np.all(np.isfinite(np.asarray(gradient))):
                raise RuntimeError(f'Nonfinite value/gradient at grid={grid}, q={order}')
            rows.append(dict(
                grid=dict(Nx=grid, Ny=grid, Nz=1, cells=grid**2), quadrature_order=order,
                input_shape=list(ck.shape), input_array_bytes=int(ck.nbytes),
                output_array_bytes=int(value.nbytes + gradient.nbytes),
                compiled_memory=memory, lowering_seconds=lowering_seconds,
                compile_seconds=compile_seconds, warmup_seconds=warmup_seconds,
                runtime_samples_seconds=samples, runtime_median_seconds=float(np.median(samples)),
                objective=float(value), gradient_l2_norm=float(jnp.linalg.norm(gradient)),
                # Size references only: neither is an observed allocation.
                one_cell_velocity_array_bytes=8*order**3,
                all_cells_velocity_array_bytes=8*grid**2*order**3))
            print(f'CPU grid={grid}, q={order}: temp={memory["temp_size_in_bytes"]} B, '
                  f'total={memory["estimated_total_buffer_bytes"]} B, '
                  f'median={np.median(samples):.6g} s', file=sys.stderr, flush=True)
            del result, value, gradient, executable, lowered, fn, objective
            jax.clear_caches()

    def ratio(a, b):
        return b/a if a is not None and b is not None and a != 0 else None

    comparisons = []
    for axis in ('grid', 'quadrature'):
        fixed_values = args.quadratures if axis == 'grid' else args.grids
        for fixed in fixed_values:
            subset = [r for r in rows if (r['quadrature_order'] if axis == 'grid'
                                          else r['grid']['Nx']) == fixed]
            subset.sort(key=lambda r: r['grid']['Nx'] if axis == 'grid' else r['quadrature_order'])
            for a, b in zip(subset, subset[1:]):
                comparisons.append(dict(
                    varied_axis=axis, fixed_value=fixed,
                    endpoints=[dict(grid=r['grid']['Nx'], q=r['quadrature_order']) for r in (a, b)],
                    temp_bytes_ratio=ratio(a['compiled_memory']['temp_size_in_bytes'],
                                           b['compiled_memory']['temp_size_in_bytes']),
                    total_buffer_bytes_ratio=ratio(a['compiled_memory']['estimated_total_buffer_bytes'],
                                                   b['compiled_memory']['estimated_total_buffer_bytes']),
                    runtime_median_ratio=ratio(a['runtime_median_seconds'], b['runtime_median_seconds']),
                    all_cells_velocity_array_bytes_ratio=ratio(a['all_cells_velocity_array_bytes'],
                                                               b['all_cells_velocity_array_bytes'])))
    report = dict(
        benchmark='instantaneous_local_tail_jitted_value_and_grad', backend=jax.default_backend(),
        jax_version=jax.__version__, jaxlib_version=jaxlib.__version__,
        python_version=platform.python_version(), platform=platform.platform(),
        devices=[dict(platform=d.platform, kind=d.device_kind) for d in jax.devices()],
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__).resolve(), root/'spectrax/_local_tail.py',
                                 root/'spectrax/_velocity_observables.py')},
        configuration=dict(hermite_per_axis=3, species=1, mass=1., dtype='complex128',
                           alpha=alpha.tolist(), basis_shift=shift.tolist(),
                           tail_spec=vars(spec), repeats=args.repeats, warmups=1),
        interpretation=[
            'Compiler estimates, not actual peak device memory, allocator usage, or process RSS.',
            'Total buffer estimate = arguments + outputs + temporaries - aliases; excludes code, '
            'compiler/runtime overhead, and other resident executables. Host fields are reported separately.',
            'Timing synchronizes value and full gradient; compilation and first call are separate. '
            'Host scheduling and other CPU workloads can affect these short timings.',
            'The implementation maps a rematerialized cell body. Spatial Hermite states and adjoints '
            'still grow with cell count. A cell velocity array has q^3 elements.',
            'All-cells velocity bytes describe ONE hypothetical float64 cells*q^3 array, not a measured '
            'tape or a prediction of an uncheckpointed implementation.',
            'These finite cases report numerical comparisons only: they do not prove asymptotic '
            'scaling or isolate the causal benefit of rematerialization. No uncheckpointed baseline, '
            'GPU measurement, physics rollout, or quadrature accuracy claim is included.'],
        cases=rows, comparisons=comparisons)
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output is None:
        sys.stdout.write(payload)
    else:
        with args.output.open('x') as stream:
            stream.write(payload)


if __name__ == '__main__':
    main()
