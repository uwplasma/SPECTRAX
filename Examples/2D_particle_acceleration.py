"""Phase control of sampled, local flow-relative electron tail gain (nu=0).

AD and centered FD minimize the SAME loss: -1e4 * normalized physical gain.
Tail includes heating; excess subtracts a full covariance-matched local Gaussian.
Neither proves nonthermal acceleration: candidates need q96+ convergence,
RK4/Hermite/grid refinement, energy balance, density/covariance and negativity
checks. Small runs demonstrate optimization, not resolved turbulence.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from solvax import checkpointed_fori_loop
from spectrax import compute_C_nmp, simulation_final

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Examples._particle_observables import (make_local_tail_objective,
    make_local_tail_diagnostics, preregistered_tail_spec)

LOSS_SCALE = 1e4


def setup(phases, grid=16, hermite=8, final_time=60.0):
    """Fixed mode amplitudes and integrated initial energy; only phases vary."""
    if grid < 6 or hermite < 3 or np.ndim(phases) != 1 or len(phases) < 1:
        raise ValueError('Require grid >= 6, Hermite >= 3 and a nonempty phase vector')
    modes = [(i, j) for i in range(1, grid // 3)
             for j in range(-grid // 3 + 1, grid // 3) if j]
    modes.sort(key=lambda k: (k[0] ** 2 + k[1] ** 2, k))
    if len(phases) > len(modes):
        raise ValueError('Too many controls for the dealiased spatial grid')
    x = 2 * jnp.pi * jnp.arange(grid) / grid
    X, Y = jnp.meshgrid(x, x, indexing='xy')
    k = 2 * jnp.pi / 50.0
    bx, by = -0.2 * jnp.sin(Y), 0.2 * jnp.sin(2 * X)
    curl = 0.2 * k * (jnp.cos(Y) + 2 * jnp.cos(2 * X))
    i, j = np.asarray(modes[:len(phases)], dtype=float).T[:, :, None, None]
    radius = np.hypot(i, j)
    amplitude = 0.06 / radius
    angle = i * X + j * Y + jnp.asarray(phases)[:, None, None]
    bx -= jnp.sum(amplitude * j / radius * jnp.sin(angle), axis=0)
    by += jnp.sum(amplitude * i / radius * jnp.sin(angle), axis=0)
    curl += jnp.sum(amplitude * k * radius * jnp.cos(angle), axis=0)
    zeros = jnp.zeros_like(X)
    velocity = jnp.stack((-0.02 * jnp.sin(Y), 0.02 * jnp.sin(X), zeros))
    velocities = jnp.stack((velocity.at[2].set(-0.5 * curl), velocity))[..., None]
    fields = jnp.stack((zeros, zeros, zeros, bx, by, jnp.ones_like(X)))[..., None]
    p = dict(Lx=50., Ly=50., Lz=1., mi_me=25., qs=jnp.array([-1., 1.]),
             Omega_cs=jnp.array([0.5, 0.02]), alpha_s=jnp.array([0.25]*3+[0.05]*3),
             u_s=jnp.zeros(6), nu=0., D=0., t_max=final_time)
    p['Ck_0'] = compute_C_nmp(velocities, p['alpha_s'], p['u_s'],
        hermite, hermite, hermite, 2).reshape(2*hermite**3, grid, grid//2+1, 1)
    p['Fk_0'] = jnp.fft.rfftn(fields, axes=(-1, -3, -2), norm='forward')
    return p


def quantities(state, parameters, grid, hermite):
    """Integrated E, B, electron and ion kinetic energies (volume means)."""
    C, F = state[:2]
    weights = jnp.where(jnp.arange(grid//2+1) == 0, 1., 2.)
    if grid % 2 == 0:
        weights = weights.at[-1].set(1.)
    field = .5 * parameters['Omega_cs'][0]**2 * jnp.sum(
        jnp.abs(F)**2 * weights[None, None, :, None], axis=(1, 2, 3))
    moments = C.reshape(2, hermite, hermite, hermite, grid, grid//2+1, 1)
    alpha = parameters['alpha_s'].reshape(2, 3)
    density = moments[:, 0, 0, 0, 0, 0, 0]
    second = jnp.stack((moments[:, 0, 0, 2, 0, 0, 0],
        moments[:, 0, 2, 0, 0, 0, 0], moments[:, 2, 0, 0, 0, 0, 0]), axis=1)
    kinetic = jnp.real(.5 * jnp.array([1., parameters['mi_me']]) * jnp.prod(alpha, axis=1)
        * jnp.sum(alpha**2 * (density[:, None]/2 + second/jnp.sqrt(2)), axis=1))
    return jnp.array([field[:3].sum(), field[3:].sum(), kinetic[0], kinetic[1]])


def problem(reference, *, grid=16, hermite=8, steps=600, window=(40., 60.),
            intervals=20, quadrature=96, objective='tail'):
    """Return RAW negative gain, run(theta)->(C,F,gain), and frozen tail spec.

    Normalized Simpson weights times (1-z²)² define the sampled objective.
    Each control's initial observable is subtracted; nested replay avoids a
    stored velocity trajectory. All sample times must share the RK4 grid.
    """
    start, end = window
    if (not np.all(np.isfinite(window)) or not 0 <= start < end or steps < 1
            or intervals < 2 or intervals % 2 or objective not in ('tail', 'excess')):
        raise ValueError('Require valid window, positive steps, even intervals and tail|excess')
    dt = end / steps
    before, chunk = round(start/dt), round((end-start)/intervals/dt)
    if (chunk < 1 or not np.isclose(before*dt, start, rtol=0, atol=1e-10)
            or before + intervals*chunk != steps):
        raise ValueError('Time samples must coincide with the RK4 grid')
    base = setup(reference, grid, hermite, end)
    spec = preregistered_tail_spec(float(jnp.sum(base['alpha_s'][:3]**2)/4))
    factory = make_local_tail_objective if objective == 'tail' else make_local_tail_diagnostics
    observable = factory(base['alpha_s'], base['u_s'], Nx=grid, Nn=hermite,
        Nm=hermite, Np=hermite, Ns=2, spec=spec, quadrature_order=quadrature)
    measure = observable if objective == 'tail' else lambda C: observable(C)['excess']
    z = np.linspace(-1, 1, intervals+1)
    weights = (1-z*z)**2 * np.array([1]+[4 if i%2 else 2 for i in range(1, intervals)]+[1])
    weights = jnp.asarray(weights/weights.sum())

    def run(theta):
        params = setup(theta, grid, hermite, end)
        initial = measure(params['Ck_0'])

        def advance(C, F, count):
            return simulation_final(dict(params, Ck_0=C, Fk_0=F, t_max=count*dt),
                steps=count, Nx=grid, Ny=grid, Nn=hermite, Nm=hermite, Np=hermite)

        C, F = params['Ck_0'], params['Fk_0']
        if before:
            C, F = advance(C, F, before)

        def sample(i, state):
            C, F, total = state
            C, F = advance(C, F, chunk)
            return C, F, total + weights[i+1]*(measure(C)-initial)

        return checkpointed_fori_loop(0, intervals, sample, (C, F, jnp.asarray(0.)))

    return lambda theta: -run(theta)[2], run, spec


def centered_fd(scalar, x, h=1e-3):
    """All coordinates, exactly two scalar solves each; no AD transformations."""
    eye = np.eye(len(x)) * h
    return np.asarray([(float(scalar(x+d))-float(scalar(x-d)))/(2*h) for d in eye])


def validate(args):
    """Stream frozen controls through training/held-out times and audit physics."""
    from Examples._particle_observables import spatial_negative_mass_diagnostics
    controls = json.loads(args.controls_file.read_text())
    g, h = args.grid, args.hermite
    start, end = args.window
    delta, dt = (end-start)/args.intervals, end/args.steps
    count = round(delta/dt)
    if not np.isfinite(args.validation_end) or args.validation_end < end or count < 1 or not np.isclose(count*dt, delta) or not np.isclose(args.validation_end/delta, round(args.validation_end/delta)):
        raise ValueError('Diagnostic times must share the RK4 grid')
    base = setup(np.asarray(controls['initial_phase']), g, h, delta)
    spec = preregistered_tail_spec(float(np.sum(np.asarray(base['alpha_s'][:3])**2)/4))
    keywords = dict(Nx=g, Nn=h, Nm=h, Np=h, Ns=2, quadrature_order=args.quadrature)
    diagnose = jax.jit(make_local_tail_diagnostics(base['alpha_s'],base['u_s'],spec=spec,**keywords))
    negative = jax.jit(lambda C: spatial_negative_mass_diagnostics(C,base['alpha_s'],base['u_s'],**keywords))
    advance = jax.jit(lambda C,F: simulation_final(dict(base,Ck_0=C,Fk_0=F),
        steps=count,Nx=g,Ny=g,Nn=h,Nm=h,Np=h,checkpointing=False))
    energy = jax.jit(lambda C,F: quantities((C,F),base,g,h))
    report = dict(status='running', grid=g, hermite=h, dt=dt, interval=delta,
        quadrature=args.quadrature, window=args.window, validation_end=args.validation_end,
        controls=controls, device=jax.devices()[0].device_kind, runs={},
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),ROOT/'Examples/_particle_observables.py',ROOT/'spectrax/_autodiff.py']})
    args.output.mkdir(parents=True, exist_ok=False)
    def save():
        (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    for label in ['initial_phase','optimized_phase']:
        p=setup(np.asarray(controls[label]),g,h,delta); C,F=p['Ck_0'],p['Fk_0']
        initial_energy=np.asarray(energy(C,F)); rows=[]; drift=0.
        report['runs'][label]=dict(initial_energies=initial_energy.tolist(),samples=rows)
        for i in range(round(args.validation_end/delta)+1):
            if i: C,F=jax.block_until_ready(advance(C,F))
            energies=np.asarray(energy(C,F)); t=i*delta
            drift=max(drift,abs(energies.sum()-initial_energy.sum())/abs(initial_energy.sum()))
            if i and t < start-1e-10: continue
            row=dict(time=t,energies=energies.tolist())
            row.update(
                {k:float(v) for k,v in diagnose(C).items()})
            row.update({k:float(v) for k,v in negative(C).items()})
            rows.append(row)
            report['runs'][label]['max_relative_energy_drift']=drift
            save(); print(label,t,flush=True)
    report['validity_passed']=all(r['max_relative_energy_drift'] < 1e-5 and
        all(s['min_density'] > 0 and s['min_covariance_eigenvalue'] > 0 and
            s['max_cell_fraction'] < 1e-8 for s in r['samples']) for r in report['runs'].values())
    report['status']='complete';save()


def plot(args):
    """Render two figures from generated reports; no solver runs or stored assets."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ad, fd, physics = [json.loads(p.read_text()) for p in args.plot]
    if (ad['method'] != 'ad' or fd['method'] != 'fd' or ad['settings'] != fd['settings']
            or ad['initial_phase'] != fd['initial_phase']
            or physics['controls']['optimized_phase'] != ad['optimized_phase']):
        raise ValueError('Require matched AD/FD runs and validation of these AD controls')
    if not physics.get('validity_passed', False):
        raise ValueError('Physics validity gate failed; inspect diagnostics before publication')
    if any(d['status'] != 'complete' for d in (ad,fd,physics)):
        raise ValueError('Cannot plot incomplete reports')
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'savefig.dpi':300})
    colors=['#0072B2','#D55E00']
    fig, axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    for label,color,name in zip(['initial_phase','optimized_phase'],colors,['Initial controls','Candidate controls']):
        rows=physics['runs'][label]['samples']; t=np.array([r['time'] for r in rows])
        for ax,key in zip(axes,['tail','excess']):
            values=np.array([r[key] for r in rows]); select=t>=physics['window'][0]
            ax.plot(t[select],1e6*(values[select]-values[0]),color=color,label=name)
            ax.axvspan(*physics['window'],color='0.5',alpha=.08)
            ax.axhline(0,color='0.5',lw=.6);ax.set_xlabel('Time')
    axes[0].set(title='(a) Local flow-relative tail',ylabel=r'Gain / initial thermal energy ($10^{-6}$)')
    axes[1].set(title='(b) Excess over matched local Gaussian',ylabel=r'Excess gain / initial thermal energy ($10^{-6}$)')
    axes[0].legend(frameon=False,fontsize=9)
    for ext in ['png','pdf']:fig.savefig(args.output/f'particle_control.{ext}')
    plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    for d,color,name in zip([ad,fd],colors,['Reverse AD','Centered FD']):
        hist=d['history'];axes[0].plot([r['seconds'] for r in hist],
            [1e6*r['raw_gain'] for r in hist],marker='.',color=color,label=name)
    axes[0].set(title='(a) Same objective and initial controls',xlabel='Elapsed time, including compilation (s)',
                ylabel=r'Objective gain ($10^{-6}$)');axes[0].legend(frameon=False)
    axes[1].bar(['Reverse AD','Centered FD'],[ad['total_seconds'],fd['total_seconds']],color=colors)
    axes[1].set(title=f"(b) Full optimization: {fd['total_seconds']/ad['total_seconds']:.2f}× FD/AD",ylabel='Total elapsed time (s)')
    fig.suptitle(f"{ad['settings']['objective']} objective · {len(ad['initial_phase'])} controls · one matched GPU pair",fontsize=11)
    for ext in ['png','pdf']:fig.savefig(args.output/f'autodiff_comparison.{ext}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('grid',16), ('hermite',8), ('steps',600), ('intervals',20),
                          ('quadrature',96), ('iterations',20), ('seed',7)]:
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--window', nargs=2, type=float, default=[40.,60.])
    parser.add_argument('--method', choices=['ad','fd'], default='ad')
    parser.add_argument('--objective', choices=['tail','excess'], default='tail')
    parser.add_argument('--fd-step', type=float, default=1e-3)
    parser.add_argument('--controls-file', type=Path)
    parser.add_argument('--controls-key', default='initial_phase')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--plot', nargs=3, type=Path, metavar=('AD','FD','VALIDATION'))
    parser.add_argument('--validation-end', type=float, default=70.)
    parser.add_argument('--fd-check', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.fd_step) or args.fd_step <= 0 or args.iterations < 0:
        parser.error('Require positive finite fd-step and nonnegative iterations')
    jax.config.update('jax_enable_x64', True)
    if args.plot: return plot(args)
    if args.validate:
        if args.controls_file is None: parser.error('--validate requires --controls-file')
        return validate(args)
    start_total = perf_counter()
    theta = (np.asarray(json.loads(args.controls_file.read_text())[args.controls_key], dtype=float)
             if args.controls_file else np.random.default_rng(args.seed).uniform(-np.pi,np.pi,8))
    if theta.ndim != 1 or not theta.size or not np.all(np.isfinite(theta)):
        parser.error('Controls must be a nonempty finite vector')
    settings = {k:getattr(args,k) for k in ['grid','hermite','steps','window','intervals','quadrature','objective']}
    raw_loss, _, spec = problem(theta, **settings)
    loss = lambda x: LOSS_SCALE * raw_loss(x)
    options = dict(maxiter=args.iterations, ftol=1e-12, gtol=1e-9, maxls=20)
    paths = [Path(__file__), ROOT/'Examples/_particle_observables.py',
             *sorted((ROOT/'spectrax').glob('*.py'))]
    hashes = lambda: {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = dict(settings=settings, method=args.method, nu=0., loss_scale=LOSS_SCALE,
        loss_definition='-loss_scale * raw_gain; raw_gain is normalized by tail_spec.normalization',
        tail_spec={k:float(getattr(spec,k)) for k in ['threshold','width','normalization']},
        initial_phase=theta.tolist(), optimizer_options=options, fd_step=args.fd_step,
        source_sha256=hashes(), versions={k:version(k) for k in ['jax','jaxlib','solvax','diffrax','numpy','scipy']},
        controls_key=args.controls_key, controls_sha256=hashlib.sha256(args.controls_file.read_bytes()).hexdigest()
        if args.controls_file else None, device=str(jax.devices()[0]), jax_enable_x64=True,
        sample_times=np.linspace(*args.window,args.intervals+1).tolist(), evaluate_only=args.evaluate_only,
        interpretation=__doc__, candidate_checks='Run --validate on frozen controls and refine grid/H/time/quadrature', status='running')
    args.output.mkdir(parents=True, exist_ok=False)

    def save():
        temp = args.output/'results.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(args.output/'results.json')

    save()
    start = perf_counter()
    compiled = jax.jit(jax.value_and_grad(loss) if args.method == 'ad' else loss).lower(theta).compile()
    report['compile_seconds'] = perf_counter()-start
    cache = {}
    report.update(fg_calls=0, objective_calls=0, fd_check_objective_calls=0,
        call_count_note='Includes warm and callbacks; objective calls exclude AD reverse replay and compilation')

    def fg(x):
        report['fg_calls'] += 1
        report['objective_calls'] += 1 if args.method == 'ad' else 1 + 2*len(x)
        if args.method == 'ad':
            v, g = jax.block_until_ready(compiled(x))
        else:
            v, g = float(compiled(x)), centered_fd(compiled, x, args.fd_step)
        v, g = float(v), np.asarray(g, dtype=float)
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            report['status'] = 'nonfinite'; save()
            raise ValueError('Nonfinite objective/gradient; candidate rejected without clipping')
        cache.update(x=np.array(x), v=v)
        return v, g

    def entry(v):
        return dict(seconds=perf_counter()-start_total, loss=v, raw_gain=-v/LOSS_SCALE,
                    energy_density_gain=-v/LOSS_SCALE*spec.normalization)

    start = perf_counter()
    value, grad = fg(theta)
    report.update(warm_seconds=perf_counter()-start, initial_loss=value,
                  initial_raw_gain=-value/LOSS_SCALE, initial_gradient=grad.tolist(), history=[entry(value)])
    if args.fd_check:
        start = perf_counter()
        scalar = jax.jit(loss).lower(theta).compile() if args.method == 'ad' else compiled
        report['fd_check'] = [dict(step=h, gradient=(g:=centered_fd(scalar, theta, h)).tolist(),
            reference_method=args.method, max_absolute_error=float(np.max(np.abs(g-grad))))
            for h in [args.fd_step, args.fd_step/2]]
        report['fd_check_seconds'] = perf_counter()-start
        report['fd_check_objective_calls'] = 4*len(theta)
        report['objective_calls'] += 4*len(theta)

    def callback(x):
        v = cache['v'] if np.array_equal(x, cache['x']) else fg(x)[0]
        report['history'].append(entry(v)); save()

    start = perf_counter()
    result = None if args.evaluate_only else minimize(fg, theta, jac=True, method='L-BFGS-B',
                                                     callback=callback, options=options)
    report['optimization_seconds'] = perf_counter()-start
    final, final_value = (theta, value) if result is None else (result.x, float(result.fun))
    energies = []
    for x in (theta, final):
        p = setup(x, args.grid, args.hermite, args.window[1])
        energies.append(np.asarray(quantities((p['Ck_0'], p['Fk_0']), p, args.grid, args.hermite)))
    report.update(initial_energies=energies[0].tolist(), optimized_initial_energies=energies[1].tolist(),
                  initial_energy_constraint_error=float(np.max(np.abs(energies[1]-energies[0]))))
    report.update(optimized_phase=final.tolist(), final_loss=final_value,
        final_raw_gain=-final_value/LOSS_SCALE, raw_gain_improvement=(value-final_value)/LOSS_SCALE,
        initial_energy_density_gain=-value/LOSS_SCALE*spec.normalization,
        final_energy_density_gain=-final_value/LOSS_SCALE*spec.normalization,
        success=None if result is None else bool(result.success),
        message='evaluated only' if result is None else str(result.message),
        iterations=0 if result is None else int(result.nit),
        final_gradient=grad.tolist() if result is None else np.asarray(result.jac).tolist())
    report.update(source_unchanged=report['source_sha256']==hashes(),
                  total_seconds=perf_counter()-start_total, status='complete')
    save()
    print(json.dumps({k:report[k] for k in ['status','initial_raw_gain','final_raw_gain','total_seconds']}), flush=True)


if __name__ == '__main__':
    main()
