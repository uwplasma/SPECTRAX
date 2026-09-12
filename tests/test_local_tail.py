"""Tiny CPU algebra tests: no solver trajectories or GPU workloads."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit, gamma

from spectrax._local_tail import make_local_tail_objective
from spectrax._velocity_observables import TailSpec, velocity_diagnostics

jax.config.update('jax_enable_x64', True)


def spectrum(c):
    return jnp.fft.rfftn(c, axes=(-1, -3, -2), norm='forward')


def factory(**overrides):
    args = dict(alpha_s=[.7, 1.1, 1.3], u_s=[.3, -.4, .2],
                Nx=1, Nn=1, Nm=1, Np=1, Ns=1,
                spec=TailSpec(1., .5, 1.7), quadrature_order=8)
    args.update(overrides)
    return make_local_tail_objective(**args)


def test_drifting_maxwellian_invariance_and_analytic_tail():
    density, temperature, mass = 1.4, .8, 2.3
    alpha = np.full(3, np.sqrt(2 * temperature / mass))
    spec = TailSpec(2., 1., 1.9)
    ck = jnp.full((1, 1, 1, 1), density / np.prod(alpha), dtype=complex)
    exact = quad(lambda e: density * e**1.5 * np.exp(-e / temperature)
                 / (gamma(1.5) * temperature**1.5)
                 * expit((e-spec.threshold)/spec.width), 0, np.inf,
                 epsabs=1e-12)[0] / spec.normalization
    values = [factory(alpha_s=alpha, u_s=u, mass=mass, spec=spec,
                      quadrature_order=24)(ck)
              for u in ([0., 0., 0.], [2.1, -1.7, .9])]
    np.testing.assert_allclose(values, exact, rtol=2e-6)
    np.testing.assert_allclose(values[0], values[1], rtol=2e-14)


@pytest.mark.parametrize('nx', [3, 4])
def test_local_cells_species_layouts_and_jit(nx):
    rng = np.random.default_rng(19)
    shape = (2, 2, 3, 4, 2, nx, 2)
    c = rng.normal(scale=.06, size=shape)
    c[:, 0, 0, 0] = 1.2 + rng.uniform(0, .3, size=(2, 2, nx, 2))
    alpha = np.array([[1., 1., 1.], [.7, 1.1, 1.3]])
    u = np.array([[0., 0., 0.], [.3, -.4, .2]])
    spec = TailSpec(1., .5, 1.7)
    objective = factory(alpha_s=alpha, u_s=u, Nx=nx, Nn=4, Nm=3, Np=2,
                        Ns=2, species=1, spec=spec, mass=1.8)
    # Independent cell loop and existing signed velocity integral, with the
    # basis shift expressed in the exact local rest frame.
    expected = []
    for iy in range(2):
        for ix in range(nx):
            for iz in range(2):
                cell = c[1, ..., iy, ix, iz]
                first = np.array([cell[0, 0, 1], cell[0, 1, 0], cell[1, 0, 0]])
                flow = u[1] + alpha[1] * first / (np.sqrt(2) * cell[0, 0, 0])
                expected.append(velocity_diagnostics(cell, alpha[1], u[1]-flow,
                                spec=spec, mass=1.8, quadrature_order=8)['objective'])
    ck = spectrum(c)
    flat = ck.reshape(48, 2, nx//2+1, 2)
    np.testing.assert_allclose(objective(ck), np.mean(expected), rtol=2e-14)
    np.testing.assert_allclose(jax.jit(objective)(flat), np.mean(expected), rtol=2e-14)
    changed = ck.at[0].set(999 * ck[0])
    np.testing.assert_allclose(objective(changed), objective(ck), rtol=2e-14)


def test_jvp_vjp_fd_with_density_and_local_flow_derivatives():
    rng = np.random.default_rng(41)
    shape = (1, 2, 2, 3, 1, 3, 1)
    c = rng.normal(scale=.15, size=shape)
    c[0, 0, 0, 0] = np.array([1., 1.3, .8]).reshape(1, 3, 1)
    c[0, 0, 0, 1] += .35
    ck = spectrum(c)
    direction = spectrum(rng.normal(scale=.1, size=shape))
    objective = factory(Nx=3, Nn=3, Nm=2, Np=2)
    value, tangent = jax.jvp(objective, (ck,), (direction,))
    reverse_value, pullback = jax.vjp(objective, ck)
    gradient, = pullback(jnp.ones_like(value))
    # JAX's real-valued complex cotangent pairing has no conjugation here.
    np.testing.assert_allclose(jnp.real(jnp.sum(gradient * direction)), tangent,
                               rtol=2e-12, atol=1e-13)
    np.testing.assert_allclose(reverse_value, value, rtol=2e-14)
    h = 1e-5
    fd = (objective(ck+h*direction) - objective(ck-h*direction)) / (2*h)
    np.testing.assert_allclose(tangent, fd, rtol=2e-8, atol=1e-10)
    np.testing.assert_allclose(jax.jit(jax.grad(objective))(ck), gradient,
                               rtol=2e-12, atol=1e-13)


def test_zero_threshold_energy_and_exact_mean_derivative():
    # Tiny positive threshold/width approximate the all-energy kernel at every
    # node. The exact internal energy is nonlinear in c0 AND all first modes.
    alpha = jnp.array([.7, 1.1, 1.3])
    mass, scale = 1.8, 1.7
    c = jnp.zeros((1, 3, 3, 3, 1, 2, 1))
    c = c.at[0, 0, 0, 0, 0, :, 0].set(jnp.array([1.1, .9]))
    for index, values in [((0, 0, 1), [.3, -.2]), ((0, 1, 0), [.2, .1]),
                          ((1, 0, 0), [-.1, .4]), ((0, 0, 2), [.1, .2]),
                          ((0, 2, 0), [.2, .1]), ((2, 0, 0), [.1, -.1])]:
        c = c.at[(0, *index, 0, slice(None), 0)].set(jnp.array(values))
    ck = spectrum(c)
    objective = factory(alpha_s=alpha, Nx=2, Nn=3, Nm=3, Np=3, mass=mass,
                        spec=TailSpec(1e-10, 1e-12, scale))

    def exact(state):
        local = jnp.fft.irfftn(state, s=(1, 1, 2), axes=(-1, -3, -2), norm='forward')
        cell = local[0, ..., 0, :, 0]
        c0 = cell[0, 0, 0]
        first = jnp.stack([cell[0, 0, 1], cell[0, 1, 0], cell[1, 0, 0]])
        second = jnp.stack([cell[0, 0, 2], cell[0, 2, 0], cell[2, 0, 0]])
        return jnp.mean(mass * jnp.prod(alpha) / scale * jnp.sum(
            alpha[:, None]**2 * (c0/4 + second/jnp.sqrt(8.) - first**2/(4*c0)), axis=0))

    np.testing.assert_allclose(objective(ck), exact(ck), rtol=2e-14)
    actual, expected = jax.grad(objective)(ck), jax.grad(exact)(ck)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-14)
    # In an all-energy kernel, freezing U would remove the entire kernel's
    # first derivative (integral (v-U) f = 0). A genuine smooth tail below
    # separately exercises that term instead of relying only on energy AD.


def test_smooth_tail_mean_derivative_is_not_frozen():
    objective = factory(Nn=3)
    def state(a):
        return jnp.array([1.1, a, .2], dtype=complex).reshape(3, 1, 1, 1)
    a, h = .4, 1e-5
    alpha = jnp.array([.7, 1.1, 1.3])
    fixed_flow_offset = alpha * jnp.array([a, 0., 0.]) / (np.sqrt(2)*1.1)
    def frozen(a):
        return velocity_diagnostics(state(a).real.reshape(1, 1, 3), alpha,
                                    -fixed_flow_offset, spec=TailSpec(1., .5, 1.7),
                                    quadrature_order=8)['objective']
    derivative = jax.grad(lambda a: objective(state(a)))(a)
    fd = (objective(state(a+h))-objective(state(a-h))) / (2*h)
    np.testing.assert_allclose(derivative, fd, rtol=1e-8, atol=1e-11)
    assert abs(float(derivative - jax.grad(frozen)(a))) > 1e-3


def test_signed_tail_and_nonpositive_cell_density():
    objective = factory(Nn=3, spec=TailSpec(1e-10, 1e-12, 1.))
    # Positive density, negative signed energy; clipping f would fail this.
    ck = jnp.array([1., 0., -6.], dtype=complex).reshape(3, 1, 1, 1)
    assert objective(ck) < 0
    objective = factory(Nx=2)
    for bad in (0., -1.):
        c = jnp.array([1., bad]).reshape(1, 1, 2, 1)
        assert jnp.isnan(jax.jit(objective)(spectrum(c)))


@pytest.mark.parametrize('overrides', [
    {'Nx': 0}, {'Nx': 1.5}, {'Nn': 0}, {'Nm': True}, {'Np': -1}, {'Ns': 0},
    {'species': 1}, {'species': -.5}, {'alpha_s': [-1., 1., 1.]},
    {'alpha_s': [1., np.nan, 1.]}, {'alpha_s': [1., 1.]},
    {'u_s': [0., np.inf, 0.]}, {'mass': 0.}, {'mass': np.nan},
    {'quadrature_order': 1}, {'quadrature_order': 2.5},
    {'quadrature_order': 2, 'Nn': 6},
    {'spec': SimpleNamespace(threshold=1., width=0., normalization=1.)},
    {'spec': SimpleNamespace(threshold=1., width=1., normalization=np.inf)},
])
def test_invalid_setup(overrides):
    with pytest.raises(ValueError):
        factory(**overrides)


@pytest.mark.parametrize('shape', [(1, 1, 1), (2, 1, 1, 1), (1, 1, 2, 1),
                                  (1, 0, 1, 1), (1, 1, 1, 2, 1, 1, 1)])
def test_invalid_state(shape):
    with pytest.raises(ValueError):
        factory()(jnp.ones(shape, dtype=complex))
