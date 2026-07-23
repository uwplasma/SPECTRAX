"""Initialization of Hermite–Fourier coefficients for Maxwellian equilibria."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from jax.numpy.fft import rfftn
from functools import partial

__all__ = ['compute_C_nmp']


@partial(jit, static_argnames=['Nn', 'Nm', 'Np', 'Ns'])
def compute_C_nmp(Us_grid, alpha_s, u_s, Nn, Nm, Np, Ns):
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

    Returns
    -------
    jnp.ndarray
        Complex Hermite-Fourier coefficients with shape `(Ns, Np, Nm, Nn, Ny, Nx//2+1, Nz)`
        corresponding to the Maxwellian evaluated on the supplied grid.
    """
    
    alpha = jnp.array(alpha_s).reshape(Ns, 3)
    u = jnp.array(u_s).reshape(Ns, 3)

    def coefficients(U, scale, shift, modes):
        normalized = ((U - shift[:, None, None, None])
                      / scale[:, None, None, None])
        factors = (normalized[:, None]
                   * jnp.sqrt(2 / jnp.arange(1, modes))[None, :, None, None, None])
        powers = jnp.concatenate(
            (jnp.ones_like(normalized[:, None]), jnp.cumprod(factors, axis=1)),
            axis=1,
        )[:, :modes]
        return powers / scale[:, None, None, None, None]

    Cn = coefficients(Us_grid[:, 0], alpha[:, 0], u[:, 0], Nn)
    Cm = coefficients(Us_grid[:, 1], alpha[:, 1], u[:, 1], Nm)
    Cp = coefficients(Us_grid[:, 2], alpha[:, 2], u[:, 2], Np)
    C = Cp[:, :, None, None] * Cm[:, None, :, None] * Cn[:, None, None]
    
    Ck_0 = rfftn(C, axes=(-1, -3, -2), norm="forward")  # shape (Ns, Np, Nm, Nn, Ny, Nx//2+1, Nz)
  
    return Ck_0
