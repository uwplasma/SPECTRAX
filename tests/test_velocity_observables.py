"""Algebra-only velocity quadrature and real-linear AD checks; no simulation."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit, gammaincc, gamma

from spectrax._velocity_observables import (
    TailSpec, make_tail_objective, preregistered_tail_spec,
    quadrature_convergence, reconstruct_distribution,
    spatial_average_coefficients, velocity_diagnostics,
)

jax.config.update("jax_enable_x64", True)


def test_preregistration():
    spec = preregistered_tail_spec(1.5, 2.0)
    assert (spec.threshold, spec.width, spec.normalization) == (7.5, 0.75, 3.0)
    for energy, density in [(0, 1), (-1, 1), (1, 0), (np.nan, 1)]:
        with pytest.raises(ValueError):
            preregistered_tail_spec(energy, density)
    with pytest.raises(ValueError):
        TailSpec(1, 0, 1)


def test_anisotropic_shifted_maxwellian_normalization_and_reconstruction():
    alpha = jnp.array([0.7, 1.2, 1.7])
    u = jnp.array([0.3, -0.2, 0.6])
    mass, density = 2.3, 1.7
    c = jnp.ones((1, 1, 1)) * density / jnp.prod(alpha)
    v = jnp.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.6], [1., 2., -1.]])
    expected = density / (jnp.pi**1.5 * jnp.prod(alpha)) * jnp.exp(
        -jnp.sum(((v - u) / alpha)**2, axis=-1))
    np.testing.assert_allclose(reconstruct_distribution(c, v, alpha, u), expected,
                               rtol=2e-14)
    d = velocity_diagnostics(c, alpha, u, mass=mass,
                             spec=preregistered_tail_spec(1), quadrature_order=8)
    np.testing.assert_allclose(d['density'], density, rtol=2e-14)
    np.testing.assert_allclose(d['kinetic_energy'],
                               density * mass / 2 * jnp.sum(u**2 + alpha**2 / 2),
                               rtol=2e-14)
    assert d['negative_mass'] == 0


def test_maxwellian_tail_analytic_energy_density_and_convergence():
    # Isotropic zero-drift Maxwellian: E/T follows Gamma(3/2,1).
    temperature, density = 1.0, 1.4
    alpha = np.full(3, math.sqrt(2 * temperature))
    c = jnp.ones((1, 1, 1)) * density / np.prod(alpha)
    spec = preregistered_tail_spec(1.5 * temperature, density)
    def integrand(energy, power):
        return (density * energy**(0.5 + power) * np.exp(-energy / temperature)
                / (gamma(1.5) * temperature**1.5))
    smooth = [quad(lambda e: integrand(e, p) * expit(
        (e - spec.threshold) / spec.width), 0, np.inf, epsabs=1e-12)[0]
              for p in (0, 1)]
    # Independent analytic hard-tail normalization identities.
    for power, exact in [(0, density * gammaincc(1.5, spec.threshold / temperature)),
                         (1, density * 1.5 * temperature * gammaincc(
                             2.5, spec.threshold / temperature))]:
        np.testing.assert_allclose(quad(lambda e: integrand(e, power),
                                        spec.threshold, np.inf)[0], exact, rtol=1e-10)
    checks = quadrature_convergence(c, alpha, np.zeros(3), spec=spec,
                                    orders=(16, 24, 32))
    errors = [abs(float(d['smooth_tail_energy']) - smooth[1])
              for d in checks['diagnostics']]
    assert errors[-1] < errors[0]
    np.testing.assert_allclose(checks['diagnostics'][-1]['smooth_tail_energy'],
                               smooth[1], rtol=2e-3)
    np.testing.assert_allclose(checks['diagnostics'][-1]['smooth_tail_number'],
                               smooth[0], rtol=2e-3)
    assert len(checks['absolute_changes']) == 2
    # Hard indicator convergence is substantially slower; use a declared loose bound.
    np.testing.assert_allclose(checks['diagnostics'][-1]['hard_tail_energy'],
                               density * 1.5 * temperature * gammaincc(2.5, 7.5),
                               rtol=0.25)


def test_signed_negative_mass_is_not_clipped():
    # f = Gaussian * (1 + a*(2*x**2-1)/sqrt(2)), with negative central region.
    a = 2.0
    c = jnp.array([[[1.0, 0.0, a]]])
    spec = preregistered_tail_spec(0.75)
    d = velocity_diagnostics(c, jnp.ones(3), jnp.zeros(3), spec=spec,
                             quadrature_order=32)
    root = math.sqrt((1 - math.sqrt(2) / a) / 2)
    negative = quad(lambda x: -np.exp(-x*x) / np.sqrt(np.pi)
                    * (1 + a * (2*x*x - 1) / np.sqrt(2)), -root, root)[0]
    assert d['negative_mass'] > 0
    np.testing.assert_allclose(d['negative_mass'], negative, atol=0.025)
    np.testing.assert_allclose(d['density'], 1.0, atol=2e-14)
    np.testing.assert_allclose(d['kinetic_energy'], 0.75 + a / (2*np.sqrt(2)),
                               atol=2e-14)
    # Replacing signed f by max(f,0) would increase this density by negative_mass.
    assert abs(float(d['density'] + d['negative_mass']) - 1) > 0.05


@pytest.mark.parametrize('nx', [5, 6])
def test_zero_mode_layout_and_hidden_local_negativity(nx):
    x = jnp.arange(nx)
    real_c = jnp.ones((2, 1, 1, 1, 2, nx, 1))
    real_c = real_c.at[1, 0, 0, 0].set(
        jnp.broadcast_to((1 + 2*jnp.cos(2*jnp.pi*x/nx))[None, :, None], (2, nx, 1)))
    ck = jnp.fft.rfftn(real_c, axes=(-1, -3, -2), norm='forward')
    args = dict(Nn=1, Nm=1, Np=1, Ns=2, species=1)
    c = spatial_average_coefficients(ck, **args)
    np.testing.assert_allclose(c, jnp.mean(real_c[1], axis=(-3, -2, -1)), atol=1e-14)
    np.testing.assert_allclose(c, spatial_average_coefficients(
        ck.reshape(2, 2, nx//2+1, 1), **args))
    assert jnp.min(real_c[1]) < 0
    assert velocity_diagnostics(c, jnp.ones(3), jnp.zeros(3),
                                spec=preregistered_tail_spec(0.75),
                                quadrature_order=8)['negative_mass'] == 0


def test_objective_jit_ad_fd_and_saved_coefficient_agreement():
    sizes = dict(Nn=3, Nm=2, Np=2, Ns=2, species=1)
    alpha = np.array([1., 1., 1., 0.7, 1.1, 1.3])
    u = np.array([0., 0., 0., 0.3, -0.1, 0.2])
    spec = preregistered_tail_spec(0.8, 1.2)
    objective = make_tail_objective(alpha, u, **sizes, spec=spec, quadrature_order=24)
    rng = np.random.default_rng(7)
    shape = (24, 2, 3, 1)
    base = jnp.asarray(rng.normal(size=shape) + 1j*rng.normal(size=shape))
    direction = jnp.asarray(rng.normal(size=shape) + 1j*rng.normal(size=shape))
    def loss(t):
        return objective(base + jnp.sin(t) * direction)
    t, h = 0.31, 1e-5
    ad = jax.grad(loss)(t)
    fd = (loss(t+h) - loss(t-h)) / (2*h)
    np.testing.assert_allclose(ad, fd, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(jax.jvp(loss, (t,), (1.,))[1], ad, rtol=1e-13)
    np.testing.assert_allclose(jax.jit(objective)(base), objective(base), rtol=1e-13)
    c = spatial_average_coefficients(base, **sizes)
    d = velocity_diagnostics(c, alpha[3:], u[3:], spec=spec, quadrature_order=24)
    np.testing.assert_allclose(objective(base), d['objective'], rtol=1e-13)
    # Nonzero Fourier modes and imaginary parts of k=0 cannot change a global integral.
    perturbation = jnp.ones(shape, dtype=complex).at[:, 0, 0, 0].set(1j)
    np.testing.assert_allclose(objective(base + perturbation), objective(base), rtol=1e-13)


def test_reconstruction_ad_through_alpha_and_shift():
    c = jnp.array([[[1., 0.1, 0.2]]])
    v = jnp.array([[0.2, -0.1, 0.3], [1., 0.5, -0.4]])
    def value(t):
        return jnp.sum(reconstruct_distribution(c, v, jnp.array([t, 1., 1.]),
                                                jnp.array([0.2*t, 0., 0.])))
    h = 1e-5
    np.testing.assert_allclose(jax.grad(value)(0.8),
                               (value(0.8+h) - value(0.8-h))/(2*h), rtol=1e-8)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        spatial_average_coefficients(jnp.ones((1, 1, 1)), Nn=1, Nm=1, Np=1, Ns=1)
    with pytest.raises(ValueError):
        make_tail_objective([-1, 1, 1], [0, 0, 0], Nn=1, Nm=1, Np=1, Ns=1,
                            spec=preregistered_tail_spec(1))
    with pytest.raises(ValueError):
        velocity_diagnostics(jnp.zeros((1, 1, 6)), jnp.ones(3), jnp.zeros(3),
                             spec=preregistered_tail_spec(1), quadrature_order=2)


@pytest.mark.parametrize('nx', [5, 6])
def test_local_negativity_map_matches_independent_cells(nx):
    from spectrax._velocity_observables import spatial_negative_mass_diagnostics
    alpha = jnp.array([0.7, 1.2, 1.3])
    u = jnp.array([0.3, -0.2, 0.1])
    c = jnp.zeros((1, 1, 1, 3, 1, nx, 1))
    c = c.at[0, 0, 0, 0, 0, :, 0].set(1 / jnp.prod(alpha))
    c = c.at[0, 0, 0, 2, 0, :, 0].set(
        2 * jnp.cos(2*jnp.pi*jnp.arange(nx)/nx) / jnp.prod(alpha))
    ck = jnp.fft.rfftn(c, axes=(-1, -3, -2), norm='forward')
    args = dict(Nx=nx, Nn=3, Nm=1, Np=1, Ns=1, quadrature_order=16)
    local = spatial_negative_mass_diagnostics(ck, alpha, u, **args)
    spec = preregistered_tail_spec(1.)
    expected = [velocity_diagnostics(c[0, ..., 0, i, 0], alpha, u,
                                     spec=spec, quadrature_order=16)
                for i in range(nx)]
    np.testing.assert_allclose(local['mean_negative_mass'],
                               np.mean([d['negative_mass'] for d in expected]), atol=1e-14)
    np.testing.assert_allclose(local['max_cell_fraction'],
                               np.max([d['negative_mass_fraction'] for d in expected]),
                               atol=1e-14)
    np.testing.assert_allclose(local['min_density'], 1, atol=1e-14)
    assert local['mean_negative_mass'] > 0.1
    average = spatial_average_coefficients(ck, Nn=3, Nm=1, Np=1, Ns=1)
    assert velocity_diagnostics(average, alpha, u, spec=spec,
                                quadrature_order=16)['negative_mass'] == 0
    # Flat solver layout and JIT agree with the structured input.
    compiled = jax.jit(lambda state: spatial_negative_mass_diagnostics(
        state, alpha, u, **args))(ck.reshape(3, 1, nx//2+1, 1))
    for key in local:
        np.testing.assert_allclose(compiled[key], local[key], atol=1e-14)


def test_local_nonpositive_density_cannot_pass_and_signed_objective():
    from spectrax._velocity_observables import spatial_negative_mass_diagnostics
    ck = -jnp.ones((1, 1, 1, 1), dtype=complex)
    args = dict(Nn=1, Nm=1, Np=1, Ns=1, quadrature_order=8)
    d = spatial_negative_mass_diagnostics(ck, jnp.ones(3), jnp.zeros(3), Nx=1, **args)
    assert d['max_cell_fraction'] == jnp.inf
    np.testing.assert_allclose(d['min_density'], -1.)
    np.testing.assert_allclose(d['mean_negative_mass'], 1.)
    objective = make_tail_objective(jnp.ones(3), jnp.zeros(3),
                                    spec=preregistered_tail_spec(0.75), **args)
    assert objective(ck) < 0  # Never silently replace a signed tail by a clipped one.
