"""Hou-Li-type exponential filter that skips n <= 2: exposed spectrum, low moments untouched."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from spectrax import filter_spectrum, exponential_filter
from spectrax._remap import low_moments
from tests.test_moving_basis import ORDERS, random_state, ode_args, rhs

jax.config.update("jax_enable_x64", True)


def test_spectrum_shape_and_values():
    Nn, Nm, Np, order = 12, 9, 2, 8
    s = np.asarray(filter_spectrum(Nn, Nm, Np, order))
    ax = lambda N: np.array([(n / (N - 1)) ** order if n > 2 else 0.0 for n in range(N)])
    np.testing.assert_allclose(s, ax(Np)[:, None, None] + ax(Nm)[None, :, None] + ax(Nn)[None, None, :], rtol=1e-14)
    assert s.shape == (Np, Nm, Nn) and s[0, 0, Nn - 1] == 1.0 and np.all(s[:, :3, :3] == 0)


@pytest.mark.parametrize("bad", [dict(order=0), dict(keep=1)])
def test_rejects_bad_parameters(bad):
    with pytest.raises(ValueError):
        filter_spectrum(8, 1, 1, **bad)


def test_zero_strength_is_bitwise_identity():
    Ck, *_ = random_state(2, seed=3)
    out = exponential_filter(Ck, 0.0, filter_spectrum(*ORDERS))
    assert out.dtype == Ck.dtype and bool(jnp.all(out == Ck))


@pytest.mark.parametrize("strength", [1.0, 36.0])
def test_low_moments_untouched_any_basis(strength):
    Ns = 2
    Nn, Nm, Np = ORDERS
    Ck, _, u, a = random_state(Ns, seed=4)
    B = jnp.stack([u, a])
    out = exponential_filter(Ck, strength, filter_spectrum(Nn, Nm, Np, order=4))
    m0, m1 = low_moments(Ck, B, Nn, Nm, Np, Ns), low_moments(out, B, Nn, Nm, Np, Ns)
    assert float(jnp.max(jnp.abs(m1 - m0)) / jnp.max(jnp.abs(m0))) < 1e-14
    assert float(jnp.max(jnp.abs(out - Ck))) > 0  # the tail is damped


def test_rhs_form_equals_filter_over_dt():
    """As collision_matrix with nu = rate, the k = 0, field-free RHS is -rate*s*C: exp(-rate dt s) over dt."""
    Ns, rate = 2, 0.9
    Nn, Nm, Np = ORDERS
    Ck, Fk, u, a = random_state(Ns, seed=5)
    s = filter_spectrum(Nn, Nm, Np, order=6)
    args = list(ode_args(Ns, u, a, nu=rate))
    args[7 + 14] = s
    args[7 + 9:7 + 13] = [0 * x for x in args[7 + 9:7 + 13]]
    dC, _ = rhs(Ns, (Ck, 0 * Fk), tuple(args))
    np.testing.assert_allclose(dC, -rate * s[None, :, :, :, None, None, None] * Ck, rtol=1e-15, atol=0)
    dt = 1e-3
    np.testing.assert_allclose(exponential_filter(Ck, rate * dt, s), Ck * jnp.exp(-rate * dt * s)[None, :, :, :, None, None, None],
                               rtol=1e-15)
