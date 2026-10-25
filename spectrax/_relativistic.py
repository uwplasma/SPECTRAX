"""Relativistic streaming, current and kinetic energy in the Hermite basis (opt-in, ``inv_c2 > 0``).

SPECTRAX expands each species in momentum-like variables ``u = p / m`` (velocity units of ``c``) with asymmetrically
weighted Hermite functions of ``xi = (u - u_s) / alpha_s``. The Vlasov equation written in ``u`` is

    d_t f + v(u) . grad_x f + (q/m) (E + v x B) . grad_u f = 0,    v = u / gamma,   gamma = sqrt(1 + inv_c2 |u|^2).

``inv_c2 = 1 / c^2`` in the code units (``inv_c2 = 1`` is physical when velocities are measured in units of ``c``, the
SPECTRAX default; ``inv_c2 = 0`` is the nonrelativistic ladder and is the default everywhere). The electric force term is
unchanged (it differentiates polynomial test functions). Everything that contains ``v(u)`` is a Galerkin *multiplication*
operator: with the Hermite recurrence matrix ``T = P diag(lam) P^T`` per axis (``lam`` are the Gauss-Hermite nodes),

    M[g] = (P (x) P (x) P) diag( g(u_s + alpha_s lam) ) (P (x) P (x) P)^T ,

which equals the existing ladder for ``g(u) = u_i`` (``u_s + alpha T``) and is the Gauss-Hermite quadrature of the product
otherwise. It is applied as nodal transforms along the three Hermite axes (cost ``O(N^4)`` per Fourier mode in 3V, ``O(N^2)`` in 1V).
An axis with a single Hermite mode contributes a single node at its centre ``u_s``: that velocity component is treated as cold.

Conservation (see ``docs`` of the PR): charge and momentum exact; streaming energy flux exact; the work identity
``dK/dt = <J.E>`` holds to the spectral projection error of ``v`` (zero for the nonrelativistic ladder with N >= 3).
"""
from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np

__all__ = ["hermite_nodes", "speed_apply", "gamma_minus_one_apply", "relativistic_kinetic_energy"]


@lru_cache(maxsize=None)
def hermite_nodes(N):
    """``(lam, P)``: eigenvalues (Gauss-Hermite nodes) and orthogonal eigenvectors of the Hermite recurrence matrix of size ``N``."""
    off = np.sqrt(np.arange(1, N) / 2.0)
    T = np.diag(off, 1) + np.diag(off, -1) if N > 1 else np.zeros((1, 1))
    lam, P = np.linalg.eigh(T)
    return lam, P


def _node_grid(alpha, u, Nn, Nm, Np, Ns):
    """Momentum components and ``gamma^2 - 1`` over the nodal grid, each of shape ``(Ns, Np, Nm, Nn)`` (before ``inv_c2``)."""
    a = alpha.reshape(Ns, 3)
    uu = u.reshape(Ns, 3)
    ux = uu[:, 0, None] + a[:, 0, None] * jnp.asarray(hermite_nodes(Nn)[0])[None]
    uy = uu[:, 1, None] + a[:, 1, None] * jnp.asarray(hermite_nodes(Nm)[0])[None]
    uz = uu[:, 2, None] + a[:, 2, None] * jnp.asarray(hermite_nodes(Np)[0])[None]
    UX = ux[:, None, None, :]
    UY = uy[:, None, :, None]
    UZ = uz[:, :, None, None]
    return UX, UY, UZ, UX ** 2 + UY ** 2 + UZ ** 2


def _apply_nodal(Ck, g, Nn, Nm, Np, Ns):
    """``M[g] Ck`` for ``Ck`` of shape ``(Ns, Np, Nm, Nn, ...)`` and nodal values ``g`` of shape ``(Ns, Np, Nm, Nn)``."""
    Pn, Pm, Pp = (jnp.asarray(hermite_nodes(N)[1]) for N in (Nn, Nm, Np))
    F = jnp.einsum("pa,mb,nc,spmn...->sabc...", Pp, Pm, Pn, Ck)
    F = F * g.reshape(g.shape + (1,) * (Ck.ndim - 4))
    F = jnp.einsum("pa,mb,nc,sabc...->spmn...", Pp, Pm, Pn, F)
    return F


def speed_apply(Ck, axis, alpha_s, u_s, inv_c2, Nn, Nm, Np, Ns):
    """Apply the velocity ``v_axis = u_axis / gamma`` (axis 0, 1, 2 for x, y, z) in the Hermite basis to ``Ck`` (shape ``(Ns, Np, Nm, Nn, ...)``)."""
    UX, UY, UZ, u2 = _node_grid(alpha_s, u_s, Nn, Nm, Np, Ns)
    comp = (UX, UY, UZ)[axis]
    return _apply_nodal(Ck, comp / jnp.sqrt(1.0 + inv_c2 * u2), Nn, Nm, Np, Ns)


def gamma_minus_one_apply(Ck, alpha_s, u_s, inv_c2, Nn, Nm, Np, Ns):
    """Apply ``(gamma - 1) / inv_c2`` (``-> |u|^2 / 2`` as ``inv_c2 -> 0``, evaluated without cancellation)."""
    _, _, _, u2 = _node_grid(alpha_s, u_s, Nn, Nm, Np, Ns)
    return _apply_nodal(Ck, u2 / (1.0 + jnp.sqrt(1.0 + inv_c2 * u2)), Nn, Nm, Np, Ns)


def relativistic_kinetic_energy(Ck, alpha_s, u_s, masses, inv_c2, Nn, Nm, Np, Ns):
    """Box-averaged kinetic energy density per species, ``m (gamma - 1) / inv_c2`` integrated over ``f`` (k = 0 component), shape ``(Ns,)``.

    ``Ck`` has shape ``(Ns * Np * Nm * Nn, Ny, Nx//2+1, Nz)``. For ``inv_c2 = 0`` the limit ``m <|u|^2 / 2>`` is returned.
    """
    C = Ck.reshape(Ns, Np, Nm, Nn, *Ck.shape[-3:])
    G = gamma_minus_one_apply(C, alpha_s, u_s, inv_c2, Nn, Nm, Np, Ns)
    a = alpha_s.reshape(Ns, 3)
    return masses * a[:, 0] * a[:, 1] * a[:, 2] * jnp.real(G[:, 0, 0, 0, 0, 0, 0])
