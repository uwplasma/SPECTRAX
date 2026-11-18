"""Initialization of Hermite–Fourier coefficients for Maxwellian equilibria."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax
from jax.numpy.fft import rfftn
from functools import partial

__all__ = ['compute_C_nmp']


def _maxwellian_modes(d, c2, N):
    """Normalised Hermite coefficients g_n = c^n H_n(d / c) / sqrt(2^n n!), n < N, of a 1D Maxwellian.

    d is the drift in units of the Hermite width and c2 = 1 - 2 vth^2 / alpha^2 (negative for
    vth > alpha / sqrt(2); the series decays iff |c2| < 1, i.e. vth < alpha); c2 = 0 when the
    Maxwellian is matched to the basis (then g_n = (sqrt(2) d)^n / sqrt(n!)). The recurrence
    g_{n+1} = (sqrt(2) d g_n - sqrt(n) c2 g_{n-1}) / sqrt(n + 1) avoids factorials.
    """
    def step(g, n):  # g = (g_{n-1}, g_n) -> (g_n, g_{n+1}); a scan keeps compile time independent of N
        g_next = (jnp.sqrt(2.0) * d * g[1] - jnp.sqrt(n) * c2 * g[0]) / jnp.sqrt(n + 1.0)
        return (g[1], g_next), g_next
    g0, g1 = jnp.ones_like(d), jnp.sqrt(2.0) * d
    _, rest = lax.scan(step, (g0, g1), jnp.arange(1.0, max(N - 1, 1)))
    return jnp.concatenate([g0[None], g1[None], rest])[:N]


@partial(jit, static_argnames=['Nn', 'Nm', 'Np', 'Ns'])
def compute_C_nmp(Us_grid, alpha_s, u_s, Nn, Nm, Np, Ns, vth_s=None):
    """
    Build the Hermite-Fourier coefficients for a drifting Maxwellian distribution function.

    Parameters
    ----------
    Us_grid : jnp.ndarray
        Velocity grid for each species with shape `(Ns, 3, Ny, Nx, Nz)`, containing the
        phase-space coordinates at which the Maxwellian is sampled.
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
    vth_s : array-like, optional
        Thermal speeds (standard deviations) of the Maxwellians, flattened as `(3 * Ns,)`.
        Default: `alpha_s / sqrt(2)`, the Maxwellian matched to the basis weight
        `exp(-(v - u)^2 / alpha^2)`. The expansion converges in the basis' weighted norm
        (finite `int f^2 exp((v - u)^2 / alpha^2) dv`) iff `vth < alpha`: the normalised
        coefficients decay like `|1 - 2 vth^2 / alpha^2|^(n/2)`. So a thin beam can share a wide
        basis, and Maxwellians up to (not including) `vth = alpha`, i.e. broader than matched,
        are admissible too. This concerns the initial expansion in this fixed norm only; it does
        not show that a truncated nonlinear run stays resolved.

    Returns
    -------
    jnp.ndarray
        Complex Hermite-Fourier coefficients with shape `(Ns, Np, Nm, Nn, Ny, Nx//2+1, Nz)`
        corresponding to the Maxwellian evaluated on the supplied grid.
    """
    alpha = jnp.array(alpha_s).reshape(Ns, 3)[:, :, None, None, None]  # (Ns, 3, 1, 1, 1)
    u = jnp.array(u_s).reshape(Ns, 3)[:, :, None, None, None]
    vth = alpha / jnp.sqrt(2.0) if vth_s is None else jnp.array(vth_s).reshape(Ns, 3)[:, :, None, None, None]
    c2 = 1 - 2 * (vth / alpha) ** 2  # 0 when the Maxwellian has the basis width
    g = [_maxwellian_modes((Us_grid[:, i] - u[:, i]) / alpha[:, i], c2[:, i], N) / alpha[:, i]
         for i, N in enumerate((Nn, Nm, Np))]  # each (N, Ns, Ny, Nx, Nz)
    C = jnp.einsum('psyxz,msyxz,nsyxz->spmnyxz', g[2], g[1], g[0])  # (Ns, Np, Nm, Nn, Ny, Nx, Nz)
    Ck_0 = rfftn(C, axes=(-1, -3, -2), norm="forward")  # shape (Ns, Np, Nm, Nn, Ny, Nx//2+1, Nz)
  
    return Ck_0
