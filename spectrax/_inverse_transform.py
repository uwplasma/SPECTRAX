"""Inverse Hermite–Fourier transform utilities.

This module reconstructs the real-space distribution function ``f(x, v)`` from
Hermite–Fourier coefficients. The implementation is written for JAX and uses
`orthax.hermite.hermval` to evaluate Hermite polynomials efficiently.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy.fft import ifftn, ifftshift
from jax.scipy.special import factorial
from orthax.hermite import hermval
from jax import jit, vmap
from functools import partial

__all__ = ['generate_Hermite_basis', 'generate_Hermite_term_vmap', 'inverse_HF_transform']

# Generate Hermite polynomial values for all orders from 0 to N-1
def generate_Hermite_basis(N, x):
    """Evaluate Hermite polynomials of orders ``0..N-1`` on a grid.

    Parameters
    ----------
    N : int
        Number of Hermite modes to evaluate.
    x : jnp.ndarray
        Points at which to evaluate the polynomials.

    Returns
    -------
    jnp.ndarray
        Array of shape ``(N, *x.shape)`` containing ``H_n(x)`` for ``n=0..N-1``.
    """
    def compute_herm(n):
        c = jnp.zeros((N,))
        c = c.at[n].set(1.0)
        return hermval(x, c)
    return vmap(compute_herm)(jnp.arange(N))  # Shape: (N, *x.shape)

@partial(jit, static_argnames=["Nn", "Nm", "Np"])
def generate_Hermite_term(C, Herm_x, Herm_y, Herm_z, Nn, Nm, Np, xi_x, xi_y, xi_z):
    """Compute the inverse Hermite contribution via array contractions.

    This helper forms the weighted Hermite basis products and contracts them with
    the Hermite coefficients ``C``. It is intended to be used only from 
    :func:`inverse_HF_transform`.

    Parameters
    ----------
    C : jnp.ndarray
        Real-space Hermite coefficients after inverse FFT in space.
    Herm_x, Herm_y, Herm_z : jnp.ndarray
        Precomputed Hermite polynomial values for each velocity-space axis.
    Nn, Nm, Np : int
        Number of Hermite modes along the x/y/z velocity axes.
    xi_x, xi_y, xi_z : jnp.ndarray
        Normalized velocity-space coordinates.

    Returns
    -------
    jnp.ndarray
        Reconstructed distribution function contribution with the same leading
        dimensions as ``C`` and trailing dimensions matching ``xi_*``.
    """
    N_modes = Nn * Nm * Np
    index = jnp.arange(N_modes)

    # Precompute all combinations (n, m, p)
    p = index // (Nn * Nm)
    m = (index % (Nn * Nm)) // Nn
    n = index % Nn

    # Gaussian factor is common to all Hermite modes; apply once after contraction.
    gauss = jnp.exp(-(xi_x**2 + xi_y**2 + xi_z**2))

    # Gather Hermite basis values per (n, m, p) mode into arrays of shape:
    #   (N_modes, Nvy, Nvx, Nvz)
    Hx = Herm_x[n]
    Hy = Herm_y[m]
    Hz = Herm_z[p]

    # Normalization for the 3D Hermite basis. Shape: (N_modes,).
    norm = (
        jnp.sqrt(jnp.pi**3)
        * (2.0 ** ((n + m + p) / 2.0))
        * jnp.sqrt(factorial(n) * factorial(m) * factorial(p))
    )
    basis = (Hx * Hy * Hz) / norm[:, None, None, None]

    # Contract over Hermite modes:
    #   C:     (Nt, N_modes, Ny, Nx, Nz)
    #   basis: (N_modes, Nvy, Nvx, Nvz)
    # -> f:    (Nt, Ny, Nx, Nz, Nvy, Nvx, Nvz)
    return jnp.tensordot(C, basis, axes=([1], [0])) * gauss[None, None, None, None, :, :, :]

@partial(jit, static_argnames=['Nn', 'Nm', 'Np'])
def inverse_HF_transform(Ck, Nn, Nm, Np, xi_x, xi_y, xi_z):
    """Reconstruct ``f(x, v)`` from Hermite–Fourier coefficients.

    Parameters
    ----------
    Ck : jnp.ndarray
        Fourier-space Hermite coefficients for (one or more) species/moments.
        The inverse FFT is applied along the last three axes.
    Nn, Nm, Np : int
        Number of Hermite modes retained along each velocity-space axis.
    xi_x, xi_y, xi_z : jnp.ndarray
        Normalized velocity-space coordinates (typically ``(v - u)/alpha``) used
        to evaluate Hermite polynomials and the Gaussian weight.

    Returns
    -------
    jnp.ndarray
        The reconstructed distribution function evaluated on ``(x, xi_x, xi_y, xi_z)``.
    """
    C = ifftn(ifftshift(Ck, axes=(-3, -2, -1)), axes=(-3, -2, -1)).real

    # Precompute Hermite functions up to desired order
    Herm_x = generate_Hermite_basis(Nn, xi_x)  # (Nn, Nvy, Nvx, Nvz)
    Herm_y = generate_Hermite_basis(Nm, xi_y)  # (Nm, Nvy, Nvx, Nvz)
    Herm_z = generate_Hermite_basis(Np, xi_z)  # (Np, Nvy, Nvx, Nvz)

    f = generate_Hermite_term(C, Herm_x, Herm_y, Herm_z, Nn, Nm, Np, xi_x, xi_y, xi_z)
    return f
