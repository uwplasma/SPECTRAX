"""Focused CPU moment, gradient, and constrained-control checks."""
import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import eval_hermite, factorial

_path = Path(__file__).resolve().parents[1] / 'Examples' / '_particle_observables.py'
_spec = importlib.util.spec_from_file_location('_particle_observables', _path)
h = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = h
_spec.loader.exec_module(h)
jax.config.update('jax_enable_x64', True)


@pytest.fixture(autouse=True)
def cpu_only():
    with jax.default_device(jax.devices('cpu')[0]):
        yield


def spectrum(c):
    return jnp.fft.rfftn(jnp.asarray(c), axes=(-1, -3, -2), norm='forward')


def options(**kw):
    return dict(alpha_s=[.7, 1.2, 1.6], u_s=[.3, -.4, .2], Nx=1,
                Nn=3, Nm=3, Np=3, Ns=1, spec=h.preregistered_tail_spec(.4),
                mass=1.4, quadrature_order=16) | kw


def test_exact_mixed_moments():
    rng = np.random.default_rng(73)
    c = rng.normal(scale=.08, size=(3, 4, 5))
    c[0, 0, 0] = 1.3
    a, u = np.array([.7, 1.2, 1.6]), np.array([2., -.7, .9])
    x, w = np.polynomial.hermite.hermgauss(6)
    points = np.stack(np.meshgrid(x, x, x, indexing='ij'), axis=-1).reshape(-1, 3)
    b = [np.array([eval_hermite(k, points[:, i])/np.sqrt(2.**k*factorial(k))
                   for k in range(c.shape[2-i])]) for i in range(3)]
    signed = np.einsum('i,j,k->ijk', w, w, w).ravel()*np.prod(a)/np.pi**1.5
    signed *= np.einsum('pmn,nq,mq,pq->q', c, *b)
    n = signed.sum()
    mean = signed @ (u+a*points)/n
    central = u+a*points-mean
    cov = np.einsum('q,qi,qj->ij', signed, central, central)/n
    for actual, expected in zip(h._cell_moments(jnp.asarray(c), a, u), (n, mean, cov)):
        np.testing.assert_allclose(actual, expected, atol=2e-14)
    assert np.max(np.abs(cov-np.diag(np.diag(cov)))) > .01


def test_gaussian_reference_and_boost_invariance():
    a = np.array([.7, 1.2, 1.6])
    ck = jnp.full((1, 1, 1, 1), 1.3/np.prod(a), dtype=complex)
    reports = [h.make_local_tail_diagnostics(**options(Nn=1, Nm=1, Np=1, u_s=u))(ck)
               for u in ([0, 0, 0], [2.1, -1.7, .9])]
    for report in reports:
        np.testing.assert_allclose(report['excess'], 0., atol=2e-14)
        np.testing.assert_allclose(report['min_density'], 1.3, atol=2e-14)
        np.testing.assert_allclose(report['min_covariance_eigenvalue'], min(a*a/2))
    np.testing.assert_allclose(reports[0]['tail'], reports[1]['tail'], atol=2e-14)


def test_tail_and_excess_ad_through_flow_and_covariance():
    rng = np.random.default_rng(113)
    shape = (1, 3, 3, 3, 1, 2, 1)
    c = rng.normal(scale=.03, size=shape)
    c[0, 0, 0, 0, 0, :, 0] = [1.1, .9]
    c[0, 0, 0, 1] += .35
    c[0, 0, 1, 1] += .09
    ck, direction = spectrum(c), spectrum(rng.normal(scale=.08, size=shape))
    diagnose = h.make_local_tail_diagnostics(**options(Nx=2))
    objective = h.make_local_tail_objective(**options(Nx=2))
    np.testing.assert_allclose(objective(ck), diagnose(ck)['tail'], atol=1e-14)
    assert diagnose(ck)['min_covariance_eigenvalue'] > .1
    for key in ('tail', 'excess'):
        loss = lambda state: diagnose(state)[key]
        value, tangent = jax.jvp(loss, (ck,), (direction,))
        _, pullback = jax.vjp(loss, ck)
        gradient, = pullback(jnp.ones_like(value))
        np.testing.assert_allclose(jnp.real(jnp.sum(gradient*direction)), tangent, atol=1e-12)
        trajectory = lambda t: loss(ck+jnp.sin(t)*direction)
        step = 1e-5
        np.testing.assert_allclose(tangent, (trajectory(step)-trajectory(-step))/(2*step),
                                   rtol=2e-7, atol=1e-10)
        np.testing.assert_allclose(jax.jit(jax.grad(trajectory))(0.), tangent, atol=1e-12)
        assert np.isfinite(gradient).all()
    # Freeze only flow: a smooth tail must have a measurably different derivative.
    moments = h._cell_moments
    def frozen(c, a, u):
        n, flow, cov = moments(c, a, u)
        return n, jax.lax.stop_gradient(flow), cov
    from unittest.mock import patch
    with patch.object(h, '_cell_moments', frozen):
        frozen_objective = h.make_local_tail_objective(**options(Nx=2))
        frozen_tangent = jax.jvp(frozen_objective, (ck,), (direction,))[1]
    assert abs(float(jax.jvp(objective, (ck,), (direction,))[1]-frozen_tangent)) > 1e-4


@pytest.mark.parametrize('nx', [3, 4])
def test_local_negativity_and_layout(nx):
    c = np.zeros((2, 1, 1, 3, 1, nx, 1))
    c[:, 0, 0, 0] = 1.
    amplitude = 2*np.cos(2*np.pi*np.arange(nx)/nx)
    c[1, 0, 0, 2, 0, :, 0] = amplitude
    x, w = np.polynomial.hermite.hermgauss(16)
    expected = np.maximum(-(1+amplitude[:, None]*(2*x*x-1)/np.sqrt(2)), 0) @ w/np.sqrt(np.pi)
    args = dict(Nx=nx, Nn=3, Nm=1, Np=1, Ns=2, species=1, quadrature_order=16)
    evaluate = lambda ck: h.spatial_negative_mass_diagnostics(ck, np.ones(6), np.zeros(6), **args)
    ck = spectrum(c)
    report = jax.jit(evaluate)(ck.reshape(6, 1, nx//2+1, 1))
    np.testing.assert_allclose(report['mean_negative_mass'], expected.mean(), atol=1e-14)
    np.testing.assert_allclose(report['max_cell_fraction'], expected.max(), atol=1e-14)
    assert report['mean_negative_mass'] > .1
    for key, value in evaluate(ck.at[0].multiply(100)).items():
        np.testing.assert_allclose(report[key], value, atol=1e-14)


def test_invalid_density_and_covariance():
    for bad in ('density', 'negative', 'singular', 'nonfinite'):
        c = np.zeros((1, 3, 3, 3, 1, 2, 1))
        c[0, 0, 0, 0] = 1.
        index = (0, 0, 0) if bad == 'density' else (0, 1, 1) if bad == 'singular' else (0, 0, 2)
        c[(0, *index, 0, 1, 0)] = dict(density=-1., negative=-2., singular=1., nonfinite=np.nan)[bad]
        args = options(Nx=2, alpha_s=[1., 1., 1.])
        report = jax.jit(h.make_local_tail_diagnostics(**args))(spectrum(c))
        assert all(np.isnan(report[k]) for k in ('tail', 'gaussian_tail', 'excess'))
        if bad == 'density':
            args.pop('spec'); args.pop('mass')
            negative = h.spatial_negative_mass_diagnostics(spectrum(c), **args)
            assert np.isinf(negative['max_cell_fraction'])
            np.testing.assert_allclose(negative['mean_negative_mass'], .5, atol=1e-14)


def test_fixed_energy_control_and_sampled_trajectory_derivative():
    import importlib.util
    from spectrax import plasma_current
    module = importlib.util.spec_from_file_location('control', Path(__file__).parents[1] / 'Examples/2D_particle_acceleration.py')
    example = importlib.util.module_from_spec(module)
    module.loader.exec_module(example)
    x = np.array([.2, .7, -.3, .6])
    y = x + np.array([.8, -.4, .1, .6])
    p, q = (example.setup(v, 12, 3, .2) for v in (x, y))
    energies = lambda p: example.quantities((p['Ck_0'], p['Fk_0']), p, 12, 3)
    np.testing.assert_allclose(energies(p), energies(q), atol=1e-14)
    np.testing.assert_allclose(abs(p['Fk_0'])**2, abs(q['Fk_0'])**2, atol=1e-16)
    kx = jnp.fft.rfftfreq(12)*12*2*jnp.pi/p['Lx']
    ky = jnp.fft.fftfreq(12)*12*2*jnp.pi/p['Ly']
    F = p['Fk_0'][..., 0]
    np.testing.assert_allclose(kx[None, :]*F[3]+ky[:, None]*F[4], 0, atol=1e-16)
    current = plasma_current(p['qs'],p['alpha_s'],p['u_s'],p['Ck_0'],3,3,3,2)
    np.testing.assert_allclose(current[2,...,0],.5j*(kx[None,:]*F[4]-ky[:,None]*F[3]),atol=1e-16)
    loss, _, _ = example.problem(x, grid=12, hermite=3, steps=4,
        window=(0., .2), intervals=2, quadrature=16, objective='excess')
    scalar = jax.jit(loss)
    grad = jax.jit(jax.grad(loss))(x)
    fd = example.centered_fd(scalar, x, 1e-3)
    np.testing.assert_allclose(grad, fd, atol=1e-12, rtol=1e-3)
