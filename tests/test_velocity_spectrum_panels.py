"""Tiny analytic Gaussian checks only; no archive processing or plasma/JAX imports."""
import numpy as np
from scipy.special import gamma

from benchmarks.velocity_spectrum_panels import (
    energy_spectrum, gaussian_distribution, hermite_distribution, moments, spherical_rule,
)


def test_shifted_anisotropic_maxwellian_density_mean_energy():
    alpha = np.array([0.7, 1.2, 1.6])
    u = np.array([0.3, -0.2, 0.5])
    density, mass = 1.7, 2.3
    c = np.full((1, 1, 1), density/np.prod(alpha))
    m = moments(c, alpha, u, mass)
    np.testing.assert_allclose(m['density'], density, rtol=1e-14)
    np.testing.assert_allclose(m['mean'], u, atol=1e-14)
    np.testing.assert_allclose(m['covariance'], np.diag(alpha**2/2), atol=1e-14)
    np.testing.assert_allclose(m['kinetic_energy'],
                               mass*density/2*(u@u+np.sum(alpha**2)/2), rtol=1e-14)
    # Direct Gaussian quadrature verifies reconstruction and its velocity Jacobian.
    node, weight = np.polynomial.hermite.hermgauss(5)
    xi = np.stack(np.meshgrid(node, node, node, indexing='ij'), axis=-1).reshape(-1, 3)
    w = np.einsum('i,j,k->ijk', weight, weight, weight).ravel()
    v = u+alpha*xi
    f = hermite_distribution(c, v, alpha, u)
    np.testing.assert_allclose(f, gaussian_distribution(v, density, u, np.diag(alpha**2/2)),
                               rtol=3e-14)
    weighted = w*np.prod(alpha)*np.exp(np.sum(xi**2, axis=1))*f
    np.testing.assert_allclose(np.sum(weighted), density, rtol=1e-14)
    np.testing.assert_allclose(weighted@v/density, u, atol=1e-14)
    np.testing.assert_allclose(weighted@(0.5*mass*np.sum(v*v, axis=1)),
                               m['kinetic_energy'], rtol=1e-14)


def test_correlated_gaussian_projected_low_moments():
    # Independently project a Gaussian onto the first/second Hermite test functions.
    density, mass = 1.4, 1.8
    alpha = np.array([0.8, 1.1, 1.3])
    u = np.array([0.2, -0.1, 0.3])
    mean = np.array([-0.1, 0.15, 0.4])
    covariance = np.array([[.4, .04, -.03], [.04, .6, .06], [-.03, .06, .8]])
    node, weight = np.polynomial.hermite.hermgauss(4)
    z = np.stack(np.meshgrid(node, node, node, indexing='ij'), axis=-1).reshape(-1, 3)
    w = np.einsum('i,j,k->ijk', weight, weight, weight).ravel()/np.pi**1.5
    v = mean+np.sqrt(2)*z@np.linalg.cholesky(covariance).T
    xi = (v-u)/alpha
    c = np.zeros((3, 3, 3))
    c0 = density/np.prod(alpha)
    c[0, 0, 0] = c0
    for i in range(3):
        index = np.eye(3, dtype=int)[i]
        c[tuple(index[::-1])] = c0*np.sum(w*np.sqrt(2)*xi[:, i])
        c[tuple((2*index)[::-1])] = c0*np.sum(w*(2*xi[:, i]**2-1)/np.sqrt(2))
        for j in range(i):
            ij = index+np.eye(3, dtype=int)[j]
            c[tuple(ij[::-1])] = c0*np.sum(w*2*xi[:, i]*xi[:, j])
    m = moments(c, alpha, u, mass)
    np.testing.assert_allclose(m['mean'], mean, atol=1e-14)
    np.testing.assert_allclose(m['covariance'], covariance, atol=1e-14)
    np.testing.assert_allclose(m['kinetic_energy'],
                               .5*mass*density*(mean@mean+np.trace(covariance)), rtol=1e-14)


def test_isotropic_gaussian_energy_spectrum_analytic():
    density, temperature, mass = 1.3, .7, 2.
    thermal = 1.5*temperature
    x = np.array([0., .05, .2, .8, 2., 5., 12.])
    fn = lambda v: gaussian_distribution(v, density, np.zeros(3),
                                         np.eye(3)*temperature/mass)
    y = energy_spectrum(fn, x, thermal, mass=mass, angular=(6, 12))
    expected = density*(thermal/temperature)**1.5*np.sqrt(x)*np.exp(
        -x*thermal/temperature)/gamma(1.5)
    np.testing.assert_allclose(y, expected, rtol=2e-14, atol=1e-15)
    # The spectrum must preserve sign; this is not a clipped visualization.
    np.testing.assert_allclose(energy_spectrum(lambda v: -fn(v), x, thermal,
                                              mass=mass, angular=(6, 12)), -y, atol=1e-15)
    direction, weight = spherical_rule(6, 12)
    np.testing.assert_allclose(weight.sum(), 4*np.pi, rtol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(direction, axis=1), 1, atol=1e-14)


def test_drifting_anisotropic_gaussian_shell_integrated_moments():
    density, mass, thermal = 1.2, 1.4, 1.5
    mean = np.array([.15, -.1, .2])
    covariance = np.array([[.7, .03, -.02], [.03, .8, .04], [-.02, .04, .9]])
    fn = lambda v: gaussian_distribution(v, density, mean, covariance)
    # Small independent radial Gauss rule in sqrt(E); no production angular sweep.
    node, w = np.polynomial.legendre.leggauss(40)
    s = (node+1)/2
    x = 12*s*s
    wx = w*12*s
    y = energy_spectrum(fn, x, thermal, mass=mass, angular=(12, 24))
    np.testing.assert_allclose(wx@y, density, rtol=5e-6)
    np.testing.assert_allclose(thermal*(wx@(x*y)),
                               .5*mass*density*(mean@mean+np.trace(covariance)), rtol=5e-5)
