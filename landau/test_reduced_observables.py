"""Independent moments, radial tails, boosts, signed validity and full AD."""
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_hermite, factorial, expit, roots_hermite

_spec = importlib.util.spec_from_file_location('tail', Path(__file__).resolve().parents[1]/'Examples/_particle_observables.py')
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)
jax.config.update('jax_enable_x64', True)


@pytest.fixture(autouse=True)
def cpu_only():
    with jax.default_device(jax.devices('cpu')[0]):
        yield


def spectrum(c):
    return jnp.fft.rfft(jnp.asarray(c), axis=-1, norm='forward')[:, None, :, None]


@pytest.mark.parametrize('H,q', [(3, 128), (3, 256), (512, 320), (768, 512), (1024, 640)])
def test_maxwellian_radial_quadrature_and_high_order(H, q):
    c = np.zeros((2*H, 1)); c[0] = 1.3
    eps = .75
    exact = quad(lambda r: 4/np.sqrt(np.pi)*r**2*np.exp(-r*r)*(.5*r*r/eps)
                 *expit((.5*r*r-5*eps)/(.5*eps)), 0, np.inf, epsabs=1e-12)[0]*1.3
    report = h.local_tail(np.ones(6), Nx=1, hermite=H, quadrature=q)(spectrum(c))
    assert all(np.isfinite(value) for value in report.values())
    np.testing.assert_allclose(report['tail'], exact, rtol=2e-6)
    np.testing.assert_allclose(report['excess'], 0, atol=1e-14)
    np.testing.assert_allclose(report['internal_energy'], 1.3*eps)
    assert report['bulk_energy'] == report['max_negative_fraction'] == 0


def test_shift_invariance():
    H, shift = 32, .45
    c = np.zeros((2*H, 1)); c[0] = 1
    evaluate = h.local_tail(np.ones(6), Nx=1, hermite=H)
    unshifted = evaluate(spectrum(c))
    c[:H, 0] = (np.sqrt(2)*shift)**np.arange(H)/np.sqrt(factorial(np.arange(H)))
    shifted = evaluate(spectrum(c))
    for key in ('tail', 'gaussian_tail', 'internal_energy', 'min_variance'):
        np.testing.assert_allclose(shifted[key], unshifted[key], rtol=3e-6, atol=1e-8)
    np.testing.assert_allclose(shifted['bulk_energy'], .5*shift**2, atol=1e-14)


@pytest.mark.parametrize('nx', [3, 5])
def test_moments_signed_negativity_and_species(nx):
    a = np.array([.8, 1.1, 1.1]); H = 5
    c = np.zeros((2*H, nx)); c[0] = np.linspace(.9, 1.2, nx)
    c[1] = .2; c[2] = .1; c[4] = -.7
    evaluate = h.local_tail(np.r_[a, [2, 3, 4]], Nx=nx, hermite=H)
    report = jax.jit(evaluate)(spectrum(c))
    x, w = roots_hermite(128)
    polynomial = c[:H].T @ np.array([eval_hermite(k, x)/np.sqrt(2.**k*factorial(k)) for k in range(H)])
    weights = polynomial*w/np.sqrt(np.pi)*np.prod(a)
    n = weights.sum(axis=1); u = weights @ (a[0]*x)/n
    var = (weights*(a[0]*x-u[:, None])**2).sum(axis=1)/n
    np.testing.assert_allclose(report['min_density'], n.min(), atol=1e-14)
    np.testing.assert_allclose(report['min_variance'], min(var.min(), a[1]**2/2), atol=1e-14)
    np.testing.assert_allclose(report['bulk_energy'], np.mean(.5*n*u*u), atol=1e-14)
    np.testing.assert_allclose(report['internal_energy'], np.mean(.5*n*(var+a[1]**2)), atol=1e-14)
    np.testing.assert_allclose(report['max_negative_fraction'], (np.maximum(-weights, 0).sum(axis=1)/n).max())
    assert report['max_negative_fraction'] > 0 and np.isfinite(report['tail'])
    c[H:] = 123
    for key, value in evaluate(spectrum(c)).items():
        np.testing.assert_allclose(value, report[key], atol=1e-14)


def test_fd_jvp_vjp_through_flow_variance_and_own_initial():
    H, nx = 5, 5
    c = np.zeros((2*H, nx)); c[0] = 1.1; c[1] = .3; c[2] = .12; c[4] = .08
    evaluate = h.local_tail(np.ones(6), Nx=nx, hermite=H, quadrature=48)
    state = spectrum(c)
    for mode in (0, 1, 2, 4):
        d = np.zeros_like(c); d[mode] = np.linspace(.1, .2, nx)
        direction = spectrum(d)
        for key in ('tail', 'gaussian_tail', 'excess'):
            # Both evolved and own-initial states depend on the same control.
            loss = lambda z: evaluate(z*1.07)[key] - evaluate(z)[key]
            value, tangent = jax.jvp(loss, (state,), (direction,))
            _, pullback = jax.vjp(loss, state)
            gradient, = pullback(jnp.ones_like(value))
            fd = (loss(state+1e-5*direction)-loss(state-1e-5*direction))/2e-5
            np.testing.assert_allclose(tangent, fd, rtol=2e-6, atol=1e-10)
            np.testing.assert_allclose(jnp.real(jnp.sum(gradient*direction)), tangent, atol=1e-12)
            if mode in (1, 2) or (mode == 4 and key != 'gaussian_tail'):
                assert abs(float(tangent)) > 1e-7


@pytest.mark.parametrize('mode,value', [(0, 0.), (0, -1.), (2, -2.), (1, 1.2), (2, np.nan)])
def test_invalid_local_moments(mode, value):
    c = np.zeros((6, 5)); c[0] = 1.; c[mode, 2] = value
    report = h.local_tail(np.ones(6), Nx=5, hermite=3)(spectrum(c))
    assert all(np.isnan(report[k]) for k in ('tail', 'gaussian_tail', 'excess', 'internal_energy'))


def test_contract_validation():
    for args in ({'Nx': 0}, {'hermite': True}, {'quadrature': 1}):
        with pytest.raises(ValueError):
            h.local_tail(np.ones(6), **(dict(Nx=1, hermite=3) | args))
    with pytest.raises(ValueError):
        h.local_tail([1, 1, 2, 1, 1, 1], Nx=1, hermite=3)
    with pytest.raises(ValueError):
        h.local_tail(np.ones(6), Nx=1, hermite=3)(jnp.ones((3, 1, 1, 1)))


def test_anisotropic_tail_against_independent_tensor_hermite():
    a = np.array([.8, 1.1, 1.1]); c = np.zeros((10, 1))
    c[:5, 0] = [1.1, .2, .1, 0., -.2]
    report = h.local_tail(np.r_[a, a], Nx=1, hermite=5)(spectrum(c))
    x, w = roots_hermite(96)
    p = sum(c[k, 0]*eval_hermite(k, x)/np.sqrt(2.**k*factorial(k)) for k in range(5))
    n = np.prod(a)*c[0, 0]; mean = c[1, 0]/(np.sqrt(2)*c[0, 0])
    var = a[0]**2*(.5+c[2, 0]/(np.sqrt(2)*c[0, 0])-mean**2)
    weights = w[:, None, None]*w[None, :, None]*w[None, None, :]/np.pi**1.5
    perp = .5*a[1]**2*(x[None, :, None]**2+x[None, None, :]**2)
    eps = sum(a*a)/4
    for key, longitudinal, factor in [('tail', .5*a[0]**2*(x-mean)**2, np.prod(a)*p),
                                       ('gaussian_tail', var*x*x, np.full_like(x, n))]:
        energy = longitudinal[:, None, None]+perp
        exact = np.sum(weights*factor[:, None, None]*energy/eps*expit((energy-5*eps)/(.5*eps)))
        np.testing.assert_allclose(report[key], exact, rtol=2e-5, atol=1e-8)
