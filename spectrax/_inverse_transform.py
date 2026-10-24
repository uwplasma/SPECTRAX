"""Inverse Hermite–DG transform utilities.

This module reconstructs the phase-space distribution function ``f(x, v)`` from
Hermite–DG coefficients. The implementation is written for JAX and evaluates
the asymmetrically weighted Hermite functions with a normalized three-term
recurrence, which stays bounded at high order (no large factorials).
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax
from functools import partial

__all__ = ["inverse_HDG_transform"]

# Generate asymmetrically weighted Hermite function values for all orders from 0 to N-1
def generate_Hermite_basis(N, x):
    """Evaluate asymmetrically weighted Hermite functions of orders ``0..N-1`` on a grid.

    Computes ``psi_n(x) = H_n(x) exp(-x^2) / sqrt(pi 2^n n!)`` directly via the
    normalized three-term recurrence, avoiding large polynomials and factorials.

    Parameters
    ----------
    N : int
        Number of Hermite modes to evaluate.
    x : jnp.ndarray
        Points at which to evaluate the functions.

    Returns
    -------
    jnp.ndarray
        Array of shape ``(N, *x.shape)`` containing the asymmetrically weighted, normalized
        Hermite functions for ``n=0..N-1``.
    """
    psi0 = jnp.exp(-x**2) / jnp.sqrt(jnp.pi)
    psi1 = jnp.sqrt(2.0) * x * psi0

    def step(carry, n):
        psi_nm1, psi_n = carry
        psi_np1 = jnp.sqrt(2.0 / (n + 1)) * x * psi_n - jnp.sqrt(n / (n + 1)) * psi_nm1
        return (psi_n, psi_np1), psi_np1

    _, rest = lax.scan(step, (psi0, psi1), jnp.arange(1, N - 1))
    return jnp.concatenate([psi0[None], psi1[None], rest])[:N]  # Shape: (N, *x.shape)

@partial(jit, static_argnames=["Nn", "Nm", "Np"])
def generate_Hermite_term(C, Herm_x, Herm_y, Herm_z, Nn, Nm, Np):
    """Compute the inverse Hermite contribution via array contractions.

    This helper forms the weighted Hermite basis products and contracts them with
    the Hermite coefficients ``C``. It is intended to be used only from 
    :func:`inverse_HDG_transform`.

    Parameters
    ----------
    C : jnp.ndarray
        Real-space Hermite coefficients after inverse FFT in space.
    Herm_x, Herm_y, Herm_z : jnp.ndarray
        Precomputed weighted, normalized Hermite function values for each velocity-space axis.
    Nn, Nm, Np : int
        Number of Hermite modes along the x/y/z velocity axes.

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

    # Gather Hermite basis values per (n, m, p) mode into arrays of shape:
    #   (N_modes, Nvy, Nvx, Nvz)
    Hx = Herm_x[n]
    Hy = Herm_y[m]
    Hz = Herm_z[p]

    # Gaussian weight and normalization are already included in each 1D Hermite function.
    basis = Hx * Hy * Hz

    # Contract over Hermite modes:
    #   C:     (Nt, N_modes, Ny, Nx, Nz)
    #   basis: (N_modes, Nvy, Nvx, Nvz)
    # -> f:    (Nt, Ny, Nx, Nz, Nvy, Nvx, Nvz)
    return jnp.tensordot(C, basis, axes=([1], [0]))

@partial(jit, static_argnames=['Nn', 'Nm', 'Np'])
def inverse_HDG_transform(Ck, Nn, Nm, Np, xi_x, xi_y, xi_z):
    """Reconstruct ``f(x, v)`` from Hermite–DG coefficients.

    Parameters
    ----------
    Ck : jnp.ndarray
        Hermite-DG coefficients for (one or more) species/moments.
        Value for element is given by the mean (zeroth) mode.
    Nn, Nm, Np : int
        Number of Hermite modes retained along each velocity-space axis.
    xi_x, xi_y, xi_z : jnp.ndarray
        Normalized velocity-space coordinates (typically ``(v - u)/alpha``) used
        to evaluate Hermite polynomials and the Gaussian weight.

    Returns
    -------
    jnp.ndarray
        The reconstructed distribution function evaluated on ``(t, x, y, z, xi_x, xi_y, xi_z)``.
    """
    C = Ck[..., 0] # Pick out average value on element.

    # Precompute Hermite functions up to desired order
    Herm_x = generate_Hermite_basis(Nn, xi_x)  # (Nn, Nvy, Nvx, Nvz)
    Herm_y = generate_Hermite_basis(Nm, xi_y)  # (Nm, Nvy, Nvx, Nvz)
    Herm_z = generate_Hermite_basis(Np, xi_z)  # (Np, Nvy, Nvx, Nvz)

    f = generate_Hermite_term(C, Herm_x, Herm_y, Herm_z, Nn, Nm, Np)
    return f
