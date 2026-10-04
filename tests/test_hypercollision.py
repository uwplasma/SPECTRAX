"""Hypercollision closure of order >= 2: exposed spectrum, low moments untouched in any basis."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from spectrax import hypercollision_spectrum
from spectrax._remap import low_moments
from tests.test_moving_basis import ORDERS, random_state, ode_args, rhs

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("order", [2, 3])
def test_spectrum_shape_and_values(order):
    Nn, Nm, Np = 12, 9, 3
    col = np.asarray(hypercollision_spectrum(Nn, Nm, Np, order))
    from math import factorial
    ax = lambda N: np.array([factorial(n) / factorial(n - 2 * order + 1) if n >= 2 * order - 1 else 0.0
                             for n in range(N)]) / (factorial(N - 1) / factorial(N - 2 * order))
    expect = ax(Nn)[None, None, :] + ax(Nm)[None, :, None] + (ax(Np) if Np > 2 * order - 1 else np.zeros(Np))[:, None, None]
    np.testing.assert_allclose(col, expect, rtol=1e-14)
    assert col[0, 0, Nn - 1] == 1.0 and np.all(col[:, :, : 2 * order - 1][:, : 2 * order - 1] == 0)


def test_order_one_is_rejected():
    with pytest.raises(ValueError, match="order"):
        hypercollision_spectrum(8, 1, 1, order=1)


@pytest.mark.parametrize("order", [2, 3])
def test_collision_term_is_diagonal_and_keeps_low_moments_for_shifted_bases(order):
    """With fields and streaming removed, dC = -nu col C exactly, and n, M_i, M_ii have zero rate."""
    Ns, nu = 2, 0.7
    Nn, Nm, Np = ORDERS
    Ck, Fk, u, a = random_state(Ns, seed=2)
    args = list(ode_args(Ns, u, a, nu=nu))
    col = hypercollision_spectrum(Nn, Nm, Np, order) if order == 2 else hypercollision_spectrum(8, 7, 6, 3)[:Np, :Nm, :Nn]
    args[7 + 14] = col
    args[7 + 9:7 + 13] = [0 * x for x in args[7 + 9:7 + 13]]  # k = 0: no streaming
    dC, _ = rhs(Ns, (Ck, 0 * Fk), tuple(args))
    np.testing.assert_allclose(dC, -nu * col[None, :, :, :, None, None, None] * Ck, rtol=1e-15, atol=0)
    B = jnp.stack([u, a])
    assert float(jnp.abs(low_moments(dC, B, Nn, Nm, Np, Ns)).max()) == 0.0
