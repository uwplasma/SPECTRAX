"""Small CPU-only checks for the private local Gaussian-excess diagnostic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import eval_hermite, factorial

from spectrax._local_tail import make_local_tail_objective
from spectrax._local_tail_diagnostics import _cell_moments, make_local_tail_diagnostics
from spectrax._velocity_observables import TailSpec

jax.config.update('jax_enable_x64', True)


@pytest.fixture(autouse=True)
def cpu_only():
    with jax.default_device(jax.devices('cpu')[0]):
        yield


def spectrum(c):
    return jnp.fft.rfftn(jnp.asarray(c), axes=(-1, -3, -2), norm='forward')


def options(**kwargs):
    args = dict(alpha_s=[.9, 1.1, 1.3], u_s=[.3, -.4, .2], Nx=1,
                Nn=3, Nm=3, Np=3, Ns=1, spec=TailSpec(2., .8, 1.7),
                mass=1.4, quadrature_order=12)
    args.update(kwargs)
    return args


def gaussian_coefficients(n, mean, covariance, alpha, u, h):
    # Independent projection: C_pmn = n/prod(alpha) * E_G[h_n h_m h_p].
    x, w = np.polynomial.hermite.hermgauss(h+2)
    points = np.stack(np.meshgrid(x, x, x, indexing='ij'), axis=-1).reshape(-1, 3)
    weight = np.einsum('i,j,k->ijk', w, w, w).ravel() / np.pi**1.5
    v = mean + np.sqrt(2) * points @ np.linalg.cholesky(covariance).T
    xi = (v-u)/alpha
    b = [np.array([eval_hermite(k, xi[:, i])/np.sqrt(2.**k*factorial(k))
                   for k in range(h)]) for i in range(3)]
    return n/np.prod(alpha) * np.einsum('q,pq,mq,nq->pmn', weight, b[2], b[1], b[0], optimize=True)


def test_local_drifting_anisotropic_gaussians_refine():
    alpha = np.array([.9, 1.1, 1.3])
    u = np.array([.3, -.4, .2])
    h, nx = 10, 2
    c = np.zeros((1, h, h, h, 1, nx, 1))
    for ix, (n, shift, widths) in enumerate([
        (1.2, [.09, -.05, .06], [1.03, .97, 1.02]),
        (.8, [-.08, .04, -.05], [.98, 1.02, .97]),
    ]):
        cov = np.diag(alpha**2/2 * np.array(widths))
        c[0, ..., 0, ix, 0] = gaussian_coefficients(n, u+shift, cov, alpha, u, h)
    ck = spectrum(c)
    reports = [make_local_tail_diagnostics(**options(Nx=nx, Nn=h, Nm=h, Np=h,
                quadrature_order=q))(ck) for q in (24, 48, 96)]
    assert abs(float(reports[-1]['excess'])) < 1e-8
    assert abs(float(reports[-1]['excess'])) < abs(float(reports[0]['excess']))
    np.testing.assert_allclose(reports[-1]['tail'], reports[-2]['tail'], rtol=1e-3, atol=1e-10)
    np.testing.assert_allclose(reports[-1]['gaussian_tail'], reports[-2]['gaussian_tail'], rtol=1e-3, atol=1e-10)
    np.testing.assert_allclose(reports[-1]['min_density'], .8, atol=1e-13)


def test_mixed_mapping_and_central_covariance_against_direct_integrals():
    rng = np.random.default_rng(73)
    c = rng.normal(scale=.025, size=(3, 4, 5))
    c[0, 0, 0] = 1.3
    c[0, 0, 1], c[0, 1, 0], c[1, 0, 0] = .21, -.17, .12
    c[0, 1, 1], c[1, 0, 1], c[1, 1, 0] = .23, -.15, .11
    a, u = np.array([.7, 1.2, 1.6]), np.array([2., -.7, .9])
    x, w = np.polynomial.hermite.hermgauss(6)
    points = np.stack(np.meshgrid(x, x, x, indexing='ij'), axis=-1).reshape(-1, 3)
    weight = np.einsum('i,j,k->ijk', w, w, w).ravel()*np.prod(a)/np.pi**1.5
    b = [np.array([eval_hermite(k, points[:, i])/np.sqrt(2.**k*factorial(k))
                   for k in range(c.shape[2-i])]) for i in range(3)]
    signed = weight*np.einsum('pmn,nq,mq,pq->q', c, *b)
    v = u+a*points
    n = signed.sum()
    mean = np.einsum('q,qi->i', signed, v)/n
    central = v-mean
    cov = np.einsum('q,qi,qj->ij', signed, central, central)/n
    actual = _cell_moments(jnp.asarray(c), jnp.asarray(a), jnp.asarray(u))
    for got, expected in zip(actual, (n, mean, cov)):
        np.testing.assert_allclose(got, expected, rtol=2e-13, atol=2e-14)
    assert np.all(np.abs(cov[np.triu_indices(3, 1)]) > .02)


def test_rotated_gaussian_reference():
    alpha = np.ones(3)
    cov = np.array([[.52, .025, -.018], [.025, .49, .022], [-.018, .022, .51]])
    mean = np.array([.07, -.05, .04])
    h = 10
    c = gaussian_coefficients(1.3, mean, cov, alpha, np.zeros(3), h)
    ck = spectrum(c.reshape(1, h, h, h, 1, 1, 1))
    args = options(alpha_s=alpha, u_s=np.zeros(3), Nn=h, Nm=h, Np=h, quadrature_order=32)
    report = make_local_tail_diagnostics(**args)(ck)
    np.testing.assert_allclose(report['min_covariance_eigenvalue'], np.linalg.eigvalsh(cov)[0], atol=1e-13)
    assert abs(float(report['excess'])) < 2e-7
    # Independent Gaussian integral in principal-axis coordinates tests the
    # Cholesky transform and product weight, without using the same factor.
    x, w = np.polynomial.hermite.hermgauss(48)
    eigen = np.linalg.eigvalsh(cov)
    energy = args['mass']*(eigen[0]*x[:, None, None]**2
                           + eigen[1]*x[None, :, None]**2 + eigen[2]*x[None, None, :]**2)
    spec = args['spec']
    expected = 1.3/np.pi**1.5/spec.normalization*np.einsum('i,j,k,ijk->', w, w, w,
                energy/(1+np.exp(-(energy-spec.threshold)/spec.width)))
    np.testing.assert_allclose(report['gaussian_tail'], expected, rtol=2e-6)


@pytest.mark.parametrize('nx', [3, 4])
def test_signed_response_species_layouts_and_jit(nx):
    args = options(Nx=nx, Nn=5, Nm=1, Np=1, Ns=2, species=1,
                   alpha_s=[1., 1., 1., .9, 1.1, 1.3], u_s=[0., 0., 0., .3, -.4, .2])
    c = np.zeros((2, 1, 1, 5, 2, nx, 1))
    c[:, 0, 0, 0] = 1.
    baseline = spectrum(c)
    diagnostic = make_local_tail_diagnostics(**args)
    base = diagnostic(baseline)
    results = []
    for sign in (-1, 1):
        changed = c.copy()
        changed[1, 0, 0, 4] = sign*.4  # signed, non-Gaussian; moments unchanged
        ck = spectrum(changed)
        report = jax.jit(diagnostic)(ck.reshape(10, 2, nx//2+1, 1))
        assert set(report) == {'tail', 'gaussian_tail', 'excess', 'min_density', 'min_covariance_eigenvalue'}
        np.testing.assert_allclose(report['tail'], make_local_tail_objective(**args)(ck), atol=1e-13)
        np.testing.assert_allclose(report['gaussian_tail'], base['gaussian_tail'], atol=1e-13)
        results.append(report['excess'])
    assert float(results[0]*results[1]) < 0
    np.testing.assert_allclose(results[0], -results[1], atol=1e-13)
    other = baseline.at[0].multiply(100.)
    np.testing.assert_allclose(diagnostic(other)['tail'], base['tail'], atol=1e-13)


@pytest.mark.parametrize('bad', ['zero_density', 'negative_density', 'negative_covariance', 'singular_covariance', 'nonfinite'])
def test_invalid_cells_propagate_nan_without_floor(bad):
    c = np.zeros((1, 3, 3, 3, 1, 2, 1))
    c[0, 0, 0, 0] = 1.
    if bad == 'zero_density':
        c[0, 0, 0, 0, 0, 1, 0] = 0.
    elif bad == 'negative_density':
        c[0, 0, 0, 0, 0, 1, 0] = -1.
    elif bad == 'negative_covariance':
        c[0, 0, 0, 2, 0, 1, 0] = -2.
    elif bad == 'singular_covariance':
        # Exact zero eigenvalue: diagonal xi moments = xy moment = 1/2.
        c[0, 0, 1, 1, 0, 1, 0] = 1.
    else:
        c[0, 0, 0, 2, 0, 1, 0] = np.nan
    # Unit widths make the singular covariance exactly representable.
    report = jax.jit(make_local_tail_diagnostics(**options(Nx=2, alpha_s=[1., 1., 1.])))(spectrum(c))
    for key in ('tail', 'gaussian_tail', 'excess'):
        assert np.isnan(report[key])
    if bad == 'negative_density':
        assert report['min_density'] < 0
    if bad == 'negative_covariance':
        assert report['min_covariance_eigenvalue'] < 0


@pytest.mark.parametrize('override', [{'Nx': 0}, {'species': 1}, {'mass': -1.},
    {'quadrature_order': 1}, {'alpha_s': [1., 0., 1.]}, {'Nn': True}])
def test_invalid_setup(override):
    with pytest.raises(ValueError):
        make_local_tail_diagnostics(**options(**override))


def test_invalid_state_shape():
    with pytest.raises(ValueError):
        make_local_tail_diagnostics(**options())(jnp.ones((3, 1, 1, 1)))
