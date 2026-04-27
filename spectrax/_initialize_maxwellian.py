"""Initialization of Hermite–Galerkin coefficients for Maxwellian equilibria."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from jax.scipy.special import factorial
from functools import partial
from spectrax import legT

__all__ = ['compute_C_nmp']


@partial(jit, static_argnames=['Nn', 'Nm', 'Np', 'Ns'])
def compute_C_nmp(Us, alpha_s, u_s, Nn, Nm, Np, Ns, Nx=1, Ny=1, Nz=1, N_DG=1):
    """
    Build the Hermite-Galerkin coefficients for a drifting Maxwellian distribution function.

    Parameters
    ----------
    Us : jnp.ndarray
        Velocity function for each species with shape `(Ns, 3)`.
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
        Hermite-DG coefficients with shape `(Ns, Np, Nm, Nn, Ny, Nx, Nz, Nl)`
        corresponding to a Maxwellian with velocity shifts.
    """
    
    U_x = lambda x, y, z: Us(x, y, z)[:, 0, None, None, None]
    U_y = lambda x, y, z: Us(x, y, z)[:, 1, None, None, None]
    U_z = lambda x, y, z: Us(x, y, z)[:, 2, None, None, None]

    alpha = jnp.array(alpha_s).reshape(Ns, 3)
    alpha_x = alpha[:, 0, None, None, None, None, None, None, None, None, None] # shape (Ns, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    alpha_y = alpha[:, 1, None, None, None, None, None, None, None, None, None] # shape (Ns, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    alpha_z = alpha[:, 2, None, None, None, None, None, None, None, None, None] # shape (Ns, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    u = jnp.array(u_s).reshape(Ns, 3)
    u_x = u[:, 0, None, None, None, None, None, None, None, None, None] # shape (Ns, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    u_y = u[:, 1, None, None, None, None, None, None, None, None, None] # shape (Ns, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    u_z = u[:, 2, None, None, None, None, None, None, None, None, None] # shape (Ns, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    p = jnp.arange(Np)[None, :, None, None, None, None, None, None, None, None] # shape (1, Np, 1, 1, 1, 1, 1, 1, 1, 1)
    m = jnp.arange(Nm)[None, None, :, None, None, None, None, None, None, None] # shape (1, 1, Nm, 1, 1, 1, 1, 1, 1, 1)
    n = jnp.arange(Nn)[None, None, None, :, None, None, None, None, None, None] # shape (1, 1, 1, Nn, 1, 1, 1, 1, 1, 1)

    C = lambda x, y, z: (jnp.sqrt(2 ** (n + m + p) / (factorial(n) * factorial(m) * factorial(p))) 
        * (1 / (alpha_x ** (n + 1) * alpha_y ** (m + 1) * alpha_z ** (p + 1)))
        * (U_x(x, y, z) - u_x) ** n * (U_y(x, y, z) - u_y) ** m * (U_z(x, y, z) - u_z) ** p)
    
    Ck_0 = legT(C, basis_idx, N_DG, Lx, Nx, Ly, Ny, Lz, Nz)  # shape (Ns, Np, Nm, Nn, Ny, Nx, Nz, Nl)
  
    return Ck_0
