"""Initialization of Hermite–Fourier coefficients for Maxwellian equilibria."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from jax.numpy.fft import fftn, fftshift
from jax.scipy.special import factorial
from functools import partial

__all__ = ['compute_C_nmp']


@partial(jit, static_argnames=['Nn', 'Nm', 'Np', 'Ns'])
def compute_C_nmp(Us_grid, alpha_s, u_s, Nn, Nm, Np, Ns):
    """
    Build the Hermite-Fourier coefficients for a drifting Maxwellian reference state.

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
        Complex Hermite-Fourier coefficients with shape `(Ns, Np, Nm, Nn, Ny, Nx, Nz)`
        corresponding to the Maxwellian evaluated on the supplied grid.
    """
    
    U_x = Us_grid[:, 0, None, None, None, :, :, :] # shape (Ns, 1, 1, 1, Ny, Nx, Nz)
    U_y = Us_grid[:, 1, None, None, None, :, :, :] # shape (Ns, 1, 1, 1, Ny, Nx, Nz)
    U_z = Us_grid[:, 2, None, None, None, :, :, :] # shape (Ns, 1, 1, 1, Ny, Nx, Nz)

    alpha = jnp.array(alpha_s).reshape(Ns, 3)
    alpha_x = alpha[:, 0, None, None, None, None, None, None] # shape (Ns, 1, 1, 1, 1, 1, 1)
    alpha_y = alpha[:, 1, None, None, None, None, None, None] # shape (Ns, 1, 1, 1, 1, 1, 1)
    alpha_z = alpha[:, 2, None, None, None, None, None, None] # shape (Ns, 1, 1, 1, 1, 1, 1)

    u = jnp.array(u_s).reshape(Ns, 3)
    u_x = u[:, 0, None, None, None, None, None, None] # shape (Ns, 1, 1, 1, 1, 1, 1)
    u_y = u[:, 1, None, None, None, None, None, None] # shape (Ns, 1, 1, 1, 1, 1, 1)
    u_z = u[:, 2, None, None, None, None, None, None] # shape (Ns, 1, 1, 1, 1, 1, 1)

    p = jnp.arange(Np)[None, :, None, None, None, None, None] # shape (1, Np, 1, 1, 1, 1, 1)
    m = jnp.arange(Nm)[None, None, :, None, None, None, None] # shape (1, 1, Nm, 1, 1, 1, 1)
    n = jnp.arange(Nn)[None, None, None, :, None, None, None] # shape (1, 1, 1, Nn, 1, 1, 1)

    C = (jnp.sqrt(2 ** (n + m + p) / (factorial(n) * factorial(m) * factorial(p))) 
        * (1 / (alpha_x ** (n + 1) * alpha_y ** (m + 1) * alpha_z ** (p + 1)))
        * (U_x - u_x) ** n * (U_y - u_y) ** m * (U_z - u_z) ** p)
    
    Ck_0 = fftshift(fftn(C, axes=(-3, -2, -1), norm="forward"), axes=(-3, -2, -1))  # shape (Ns, Np, Nm, Nn, Ny, Nx, Nz)
  
    return Ck_0
