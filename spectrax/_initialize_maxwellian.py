"""Initialization of Hermite coefficients for Maxwellian equilibria."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit

__all__ = ['compute_C_nmp']


def compute_C_nmp(Us, alpha_s, u_s, Nn, Nm, Np, Ns):
    """
    Build the Hermite coefficients for a drifting Maxwellian distribution function.

    Parameters
    ----------
    Us : callable
        Drift velocity field `Us(x, y, z)` returning an array of shape `(Ns, 3, *S)`.
    alpha_s : array-like
        Sequence of thermal scaling parameters flattened as `(3 * Ns,)`, grouped as
        `(alpha_x, alpha_y, alpha_z)` per species.
    u_s : array-like
        Sequence of drift velocities flattened as `(3 * Ns,)`, grouped as
        `(u_x, u_y, u_z)` per species.
    Nn, Nm, Np : int
        Number of Hermite modes retained along the x, y, and z velocity axes.
    Ns : int
        Number of species.

    Returns
    -------
    callable
        Jitted function `C(x, y, z)` returning the Hermite coefficients with shape
        `(Ns, Np, Nm, Nn, *S)` corresponding to a Maxwellian with velocity shifts.
    """

    # Same coefficients built by successive multiplication, so no intermediate overflows unless the result does:
    #   C_nmp = a_n(d_x) * a_m(d_y) * a_p(d_z) / (alpha_x * alpha_y * alpha_z),  d = (U - u) / alpha,
    #   a_k(d) = (sqrt(2) d)^k / sqrt(k!) = a_{k-1}(d) * sqrt(2 / k) * d,  a_0 = 1.
    alpha = jnp.array(alpha_s).reshape(Ns, 3)
    u = jnp.array(u_s).reshape(Ns, 3)

    def hermite_powers(d, N):
        """Return a_k(d) for k = 0..N-1 with shape (Ns, N, *d.shape[1:])."""
        k = jnp.arange(N).reshape((1, N) + (1,) * (d.ndim - 1))
        steps = jnp.where(k == 0, 1.0, jnp.sqrt(2.0 / jnp.maximum(k, 1)) * d[:, None])
        return jnp.cumprod(steps, axis=1)

    @jit
    def C(x, y, z):
        U = Us(x, y, z)                                     # (Ns, 3, *S)
        bshape = (Ns, 3) + (1,) * (U.ndim - 2)
        d = (U - u.reshape(bshape)) / alpha.reshape(bshape) # normalized drift, (Ns, 3, *S)
        a_n = hermite_powers(d[:, 0], Nn)                   # (Ns, Nn, *S)
        a_m = hermite_powers(d[:, 1], Nm)                   # (Ns, Nm, *S)
        a_p = hermite_powers(d[:, 2], Np)                   # (Ns, Np, *S)
        norm = jnp.prod(alpha, axis=1).reshape((Ns, 1, 1, 1) + (1,) * (U.ndim - 2))
        return a_p[:, :, None, None] * a_m[:, None, :, None] * a_n[:, None, None, :] / norm  # (Ns, Np, Nm, Nn, *S)

    return C
