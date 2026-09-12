"""Fixed-energy phase control of a local flow-relative electron tail.

The objective uses a declared sampled time quadrature, separately from RK4.
It removes local bulk shifts, but includes Maxwellian temperature/anisotropy
changes and is not by itself a nonthermal-acceleration diagnostic.
"""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from solvax import checkpointed_fori_loop

from spectrax import simulation_final
from spectrax._local_tail import make_local_tail_objective
from spectrax._velocity_observables import preregistered_tail_spec

ROOT = Path(__file__).resolve().parents[1]
module = importlib.util.spec_from_file_location('phase_control', ROOT/'Examples/2D_phase_control.py')
phase = importlib.util.module_from_spec(module)
module.loader.exec_module(phase)


def problem(reference, *, grid=16, hermite=8, steps=600, window=(40., 60.),
            intervals=20, quadrature=96, nu=0.):
    """Nested replay stores sublinear time states, not a velocity trajectory.

    Simpson weights times the smooth window are normalized to sum one. The
    finite sampled objective is differentiated exactly; refine intervals to
    assess its approximation to a continuous time average. RK4 grid is shared
    by every segment; require all sample times to lie on that grid.
    """
    start, end = window
    if not 0 <= start < end or intervals < 2 or intervals % 2 or steps < 1:
        raise ValueError('Require valid window, positive steps and positive even intervals')
    dt = end/steps
    before, chunk = round(start/dt), round((end-start)/intervals/dt)
    if chunk < 1 or not np.isclose(before*dt, start) or not np.isclose(chunk*dt*intervals, end-start):
        raise ValueError('Time samples must coincide with the RK4 grid')
    base = phase.setup(reference, grid, hermite, end)
    spec = preregistered_tail_spec(float(jnp.sum(base['alpha_s'][:3]**2)/4))
    tail = make_local_tail_objective(base['alpha_s'], base['u_s'], Nx=grid,
        Nn=hermite, Nm=hermite, Np=hermite, Ns=2, spec=spec, quadrature_order=quadrature)
    z = np.linspace(-1, 1, intervals+1)
    weights = (1-z*z)**2 * np.array([1]+[4 if i%2 else 2 for i in range(1, intervals)]+[1])
    weights = jnp.asarray(weights/weights.sum())

    def run(theta):
        params = phase.setup(theta, grid, hermite, end)
        params['nu'] = nu
        initial = tail(params['Ck_0'])

        def advance(C, F, count):
            p = dict(params, Ck_0=C, Fk_0=F, t_max=count*dt)
            return simulation_final(p, steps=count, Nx=grid, Ny=grid,
                                    Nn=hermite, Nm=hermite, Np=hermite)

        C, F = params['Ck_0'], params['Fk_0']
        if before:
            C, F = advance(C, F, before)

        def sample(i, state):
            C, F, total = state
            C, F = advance(C, F, chunk)
            return C, F, total+weights[i+1]*(tail(C)-initial)

        return checkpointed_fori_loop(0, intervals, sample, (C, F, jnp.asarray(0.)))

    return lambda theta: -run(theta)[2], run, spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('grid',16), ('hermite',8), ('steps',600), ('intervals',20),
                          ('quadrature',96), ('iterations',20), ('seed',7)]:
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--window', nargs=2, type=float, default=[40.,60.])
    parser.add_argument('--controls-file', type=Path)
    parser.add_argument('--controls-key', default='initial_phase', choices=['initial_phase','optimized_phase'])
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--fd-check', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    theta = (np.asarray(json.loads(args.controls_file.read_text())[args.controls_key])
             if args.controls_file else np.random.default_rng(args.seed).uniform(-np.pi,np.pi,8))
    settings = {k:getattr(args,k) for k in ['grid','hermite','steps','window','intervals','quadrature']}
    paths = ['Examples/2D_local_tail_control.py','Examples/2D_phase_control.py',
             'spectrax/_local_tail.py','spectrax/_velocity_observables.py',
             'spectrax/_autodiff.py','spectrax/_model.py','spectrax/_initialization.py']
    hashes = lambda: {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    report = dict(settings=settings, source_sha256=hashes(), initial_phase=theta.tolist(),
        versions={k:importlib.metadata.version(k) for k in ['jax','jaxlib','solvax','diffrax','numpy','scipy']},
        controls_key=args.controls_key,
        controls_sha256=hashlib.sha256(args.controls_file.read_bytes()).hexdigest() if args.controls_file else None,
        evaluate_only=args.evaluate_only, optimizer_options=dict(maxiter=args.iterations,ftol=1e-12,gtol=1e-9),
        sample_times=np.linspace(*args.window,args.intervals+1).tolist(), nu=0.,
        status='running', device=jax.devices()[0].device_kind,
        jax_version=jax.__version__, jax_enable_x64=bool(jax.config.jax_enable_x64),
        interpretation='Local flow-relative signed tail gain, sampled normalized Simpson time weights; not nonthermal proof')
    def save():
        temp=args.output/'results.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.output/'results.json')
    save()
    loss, _, spec = problem(theta, **settings)
    report['tail_spec'] = dict(threshold=spec.threshold,width=spec.width,normalization=spec.normalization)
    start=perf_counter(); compiled=jax.jit(jax.value_and_grad(loss)).lower(theta).compile()
    report['compile_seconds']=perf_counter()-start
    start=perf_counter(); value, grad=jax.block_until_ready(compiled(theta))
    report.update(first_seconds=perf_counter()-start, initial_loss=float(value), initial_gradient=np.asarray(grad).tolist())
    if not np.isfinite(value) or not np.all(np.isfinite(grad)):
        report['status']='nonfinite';save();raise ValueError('Nonfinite local-tail value or gradient')
    report['memory_analysis'] = str(compiled.memory_analysis())
    report['memory_note'] = 'Compiler memory estimate, not measured peak allocator/device memory'
    p0=phase.setup(theta,args.grid,args.hermite,args.window[1])
    e0=np.asarray(phase.quantities((p0['Ck_0'],p0['Fk_0']),p0,args.grid,args.hermite)[:4])
    report['initial_energies']=e0.tolist()
    report['history']=[dict(seconds=0.,loss=float(value))]; save()
    if args.fd_check:
        direction=np.random.default_rng(3).normal(size=len(theta));direction/=np.linalg.norm(direction)
        scalar=jax.jit(loss); exact=float(np.asarray(grad)@direction); checks=[]
        for h in [1e-3,1e-4,1e-5]:
            fd=float((scalar(theta+h*direction)-scalar(theta-h*direction))/(2*h))
            checks.append(dict(step=h,fd=fd,ad=exact,absolute_error=abs(fd-exact)))
        report['directional_fd']=checks;save()
    if not args.evaluate_only:
        cache={}; start=perf_counter()
        def fg(x):
            v,g=jax.block_until_ready(compiled(x));cache['x']=np.array(x);cache['v']=float(v)
            if not np.isfinite(v) or not np.all(np.isfinite(g)):
                raise ValueError('Nonfinite optimizer evaluation')
            return float(v),np.asarray(g)
        def callback(x):
            v=cache['v'] if np.array_equal(x,cache.get('x')) else fg(x)[0]
            report['history'].append(dict(seconds=perf_counter()-start,loss=v));save()
        result=minimize(fg,theta,jac=True,method='L-BFGS-B',callback=callback,
                        options=dict(maxiter=args.iterations,ftol=1e-12,gtol=1e-9))
        report.update(optimized_phase=result.x.tolist(),final_loss=float(result.fun),
                      final_gradient=result.jac.tolist(),success=bool(result.success),
                      message=str(result.message),iterations=int(result.nit),optimization_seconds=perf_counter()-start)
        p1=phase.setup(result.x,args.grid,args.hermite,args.window[1])
        e1=np.asarray(phase.quantities((p1['Ck_0'],p1['Fk_0']),p1,args.grid,args.hermite)[:4])
        report['initial_energy_constraint_error']=float(np.max(np.abs(e1-e0)))
    report['source_unchanged']=report['source_sha256']==hashes()
    report['status']='complete';save()
    print(json.dumps({k:report[k] for k in ['status','initial_loss','first_seconds','source_unchanged']}),flush=True)


if __name__ == '__main__':
    main()
