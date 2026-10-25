"""Field-scaled closure rate: exact zero at c = 0, analytic value, per-species nu in the kinetic RHS, low moments."""
import numpy as np
import jax
import jax.numpy as jnp
from spectrax import field_scaled_closure_rate, hypercollision_spectrum
from spectrax._remap import low_moments
from tests.test_moving_basis import ORDERS, random_state, ode_args, rhs

jax.config.update("jax_enable_x64", True)


def _fields(Nx=16, E1=3e-3, E0=0.7):
    x = np.arange(Nx) * 2 * np.pi / Nx
    F = np.zeros((6, 1, Nx, 1))
    F[0, 0, :, 0] = E0 + E1 * np.cos(x)
    return jnp.asarray(F)


def test_zero_strength_is_exactly_zero():
    a = jnp.full(6, 0.05)
    r = field_scaled_closure_rate(_fields(), a, jnp.array([-1.0, 1.0]), jnp.array([1.0, 1 / 1836]), 64, 1, 1, 2, 0.0)
    assert r.shape == (2, 1, 1, 1, 1, 1, 1) and float(jnp.abs(r).max()) == 0.0


def test_analytic_value_excludes_uniform_field_and_short_axes():
    a = jnp.array([0.05, 0.02, 0.02, 0.001, 0.001, 0.001])
    qs, om = jnp.array([-1.0, 1.0]), jnp.array([1.0, 1 / 1836])
    r = np.asarray(field_scaled_closure_rate(_fields(), a, qs, om, 64, 1, 1, 2, 0.5)).ravel()
    expect = 0.5 * np.array([1.0, 1 / 1836]) * np.sqrt(128) * 3e-3 / np.array([0.05, 0.001])
    np.testing.assert_allclose(r, expect, rtol=1e-14)  # E0 = 0.7 does not enter; Nm = Np = 1 axes are inactive


def test_jit_and_per_species_nu_in_rhs_keeps_low_moments():
    """Passing the rate as nu gives dC - dC(nu=0) = -nu_s col C per species, with zero rate of n, M_i, M_ii."""
    Ns = 2
    Nn, Nm, Np = ORDERS
    Ck, Fk, u, a = random_state(Ns, seed=3)
    nu_s = jnp.array([0.3, 1.7]).reshape(Ns, 1, 1, 1, 1, 1, 1)
    base = list(ode_args(Ns, u, a, nu=0.0))
    col = hypercollision_spectrum(Nn, Nm, Np, 2)
    base[7 + 14] = col
    with_nu = list(base)
    with_nu[7 + 1] = nu_s
    dC0, _ = rhs(Ns, (Ck, Fk), tuple(base))
    dC1, _ = rhs(Ns, (Ck, Fk), tuple(with_nu))
    np.testing.assert_allclose(dC1 - dC0, -nu_s * col[None, :, :, :, None, None, None] * Ck, rtol=1e-12, atol=1e-15)
    B = jnp.stack([u, a])
    assert float(jnp.abs(low_moments(dC1 - dC0, B, Nn, Nm, Np, Ns)).max()) < 1e-14 * float(jnp.abs(Ck).max())
    f = jax.jit(field_scaled_closure_rate, static_argnames=("Nn", "Nm", "Np", "Ns"))
    r = f(_fields(), jnp.full(6, 0.05), jnp.array([-1.0, 1.0]), jnp.array([1.0, 0.1]), Nn=64, Nm=1, Np=1, Ns=2, c=1.0)
    assert np.all(np.isfinite(np.asarray(r)))
