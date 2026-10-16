"""Hermite–Fourier model operators for the Vlasov–Maxwell system.

This module contains the spectral Ampère–Maxwell current operator
(:func:`plasma_current`) and the right-hand side of the Hermite–Fourier moment
equations (:func:`Hermite_Fourier_system`).
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

__all__ = ['plasma_current', 'Hermite_Fourier_system', 'basis_rate_terms', 'uniform_acceleration',
           'field_scaled_closure_rate', 'dougherty_moments', 'dougherty_rate_real',
           'dougherty_cross_moments', 'dougherty_collisions']


@partial(jit, static_argnames=['Nn', 'Nm', 'Np', 'Ns'])
def plasma_current(qs, alpha_s, u_s, Ck, Nn, Nm, Np, Ns):
    """
    Compute the spectral Ampère-Maxwell current from Hermite-Fourier coefficients.

    Parameters
    ----------
    qs : jnp.ndarray, shape (Ns,)
        Charges of the species.
    alpha_s : jnp.ndarray, shape (3 * Ns,)
        Velocity scaling factors for each species.
    u_s : jnp.ndarray, shape (3 * Ns,)
        Velocity shift for each species.
    Ck : jnp.ndarray, shape (Ns * Np * Nm * Nn, Ny, Nx//2+1, Nz)
        Hermite-Fourier coefficients for all species stacked along the first axis.
    Nn, Nm, Np : int
        Number of Hermite modes in x, y, and z respectively.
    Ns : int
        Number of species.

    Returns
    -------
    jnp.ndarray, shape (3, Ny, Nx, Nz)
        The total Ampère-Maxwell current components `(Jx, Jy, Jz)`.
    """
    # Reshape Ck into structured Hermite-Fourier coefficients: (Ns, Np, Nm, Nn, Ny, Nx//2+1, Nz)
    Ck = Ck.reshape(Ns, Np, Nm, Nn, *Ck.shape[-3:])
    
    # Reshape alpha and velocity
    alpha = alpha_s.reshape(Ns, 3)
    u = u_s.reshape(Ns, 3)

    # Grab the modes we need (0,1,1,1) for jx, jy, jz contributions
    C0 = Ck[:, 0, 0, 0]  # shape: (Ns, Ny, Nx//2+1, Nz)
    C100 = Ck[:, 0, 0, 1] if Nn > 1 else jnp.zeros_like(C0)
    C010 = Ck[:, 0, 1, 0] if Nm > 1 else jnp.zeros_like(C0)
    C001 = Ck[:, 1, 0, 0] if Np > 1 else jnp.zeros_like(C0)

    # Pull out alpha and u components
    a0, a1, a2 = alpha[:, 0], alpha[:, 1], alpha[:, 2]
    u0, u1, u2 = u[:, 0], u[:, 1], u[:, 2]
    q = qs

    # Compute terms
    pre = q * a0 * a1 * a2  # shape: (Ns,)

    term1 = (1.0 / jnp.sqrt(2.0)) * jnp.stack([a0[:, None, None, None] * C100,
                                               a1[:, None, None, None] * C010,
                                               a2[:, None, None, None] * C001], axis=0)
    term2 = jnp.stack([u0[:, None, None, None] * C0,
                       u1[:, None, None, None] * C0,
                       u2[:, None, None, None] * C0], axis=0)

    # Final current per species: shape (3, Ns, Ny, Nx//2+1, Nz)
    J_species = (term1 + term2) * pre[None, :, None, None, None]

    # Sum over species → shape: (3, Ny, Nx//2+1, Nz)
    return jnp.sum(J_species, axis=1)

def _pad_hermite_axes(Ck):
    """Pad Hermite axes (p, m, n) by one cell on both sides.

    This padding enables safe slicing for the zero-padded shift operator used in
    :func:`shift_multi`.
    """
    return jnp.pad(
        Ck,
        ((0,0), (1,1), (1,1), (1,1), (0,0), (0,0), (0,0))
    )

def shift_multi(Ck, dn=0, dm=0, dp=0):
    """
    Zero-padded shift along Hermite axes (n,m,p) simultaneously.
    dn=+1 means 'use source at n-1', dn=-1 means 'use source at n+1', dn=0 is identity.
    Same for dm, dp. Works for values in {-1,0,+1}.
    """
    P = _pad_hermite_axes(Ck)
    _, Np, Nm, Nn, _, _, _ = Ck.shape
    # Start indices in the padded array
    n0 = 1 + dn   # dn=+1 -> 0 ; dn=0 -> 1 ; dn=-1 -> 2
    m0 = 1 + dm
    p0 = 1 + dp
    return P[:, p0:p0+Np, m0:m0+Nm, n0:n0+Nn, :, :, :]

@partial(jit, static_argnames=['Nn', 'Nm', 'Np', 'Ns'])
def Hermite_Fourier_system(Ck, C, F, kx_grid, ky_grid, kz_grid, k2_grid, col, 
                           sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus, 
                           Lx, Ly, Lz, nu, D, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, Ns, mask23, F0=None):
    """
    Evaluate the right-hand side of the coupled Hermite-Fourier moment equations.

    Parameters
    ----------
    Ck : jnp.ndarray
        Hermite-Fourier coefficients with shape `(Ns * Np * Nm * Nn, Ny, Nx, Nz)`.
    C : jnp.ndarray
        Configuration-space Hermite coefficients (i.e., inverse FFT of `Ck`), with
        shape `(Ns * Np * Nm * Nn, Ny, Nx, Nz)`. The array is reshaped internally
        to separate the species and Hermite indices.
    F : jnp.ndarray
        Configuration-space electromagnetic fields with shape `(6, Ny, Nx, Nz)` ordered
        as `(Ex, Ey, Ez, Bx, By, Bz)`.
    kx_grid, ky_grid, kz_grid : jnp.ndarray
        Fourier wave-number grids scaled to the physical domain length.
    k2_grid : jnp.ndarray
        Squared magnitude of the wave number.
    col : jnp.ndarray
        Precomputed collision coefficients.
    sqrt_* : jnp.ndarray
        Square-root ladder coefficients for the Hermite recurrences along each axis.
    Lx, Ly, Lz : float
        Domain lengths in each spatial direction.
    nu : float or jnp.ndarray
        Collision frequency: a scalar, or one rate per species with shape ``(Ns, 1, 1, 1, 1, 1, 1)``
        (see :func:`field_scaled_closure_rate`).
    D : float
        Hyper-diffusion coefficient.
    alpha_s, u_s : jnp.ndarray
        Thermal scaling parameters and drift velocities per species.
    qs : jnp.ndarray
        Species charges.
    Omega_cs : jnp.ndarray
        Cyclotron frequencies per species.
    Nn, Nm, Np, Ns : int
        Number of Hermite modes and species.
    mask23 : jnp.ndarray
        Boolean mask implementing the 2/3 de-aliasing rule in Fourier space.
    F0 : jnp.ndarray, shape (6,), optional
        Uniform field already carried by a moving basis (pump frame, see
        :func:`uniform_acceleration`). The force then uses ``E - E0`` and ``u x (B - B0)``;
        the uniform acceleration ``(q/m)(E0 + u x B0)`` is the basis-centre rate instead.
        Default None: the full force (bitwise the previous behaviour).

    Returns
    -------
    jnp.ndarray
        Time derivative `dCk/dt` with shape `(Ns, Np, Nm, Nn, Ny, Nx//2+1, Nz)`.
    """

    Ck = Ck.reshape(Ns, Np, Nm, Nn, *Ck.shape[-3:])
    C = C.reshape(Ns, Np, Nm, Nn, *C.shape[-3:])
    F = F[:, None, None, None, None, :, :, :]  # (6,1,1,1,Nx,Ny,Nz) for broadcasting  
    
    # Define u, alpha, charge, and gyrofrequency depending on species.
    alpha = alpha_s.reshape(Ns, 3)
    u = u_s.reshape(Ns, 3)
    a0 = alpha[:, 0][:, None, None, None, None, None, None]
    a1 = alpha[:, 1][:, None, None, None, None, None, None]
    a2 = alpha[:, 2][:, None, None, None, None, None, None]
    u0 = u[:, 0][:, None, None, None, None, None, None]
    u1 = u[:, 1][:, None, None, None, None, None, None]
    u2 = u[:, 2][:, None, None, None, None, None, None]
    q = qs[:, None, None, None, None, None, None]
    Omega_c = Omega_cs[:, None, None, None, None, None, None]

    
    
    # Define terms to be used in ODEs below.
    C_aux_x = (sqrt_m_minus * sqrt_p_minus * (a2 / a1 - a1 / a2) * shift_multi(C, dn=0, dm=-1, dp=-1) + 
        sqrt_m_minus * sqrt_p_plus * (a2 / a1) * shift_multi(C, dn=0, dm=-1, dp=1) - 
        sqrt_m_plus * sqrt_p_minus * (a1 / a2) * shift_multi(C, dn=0, dm=1, dp=-1) + 
        jnp.sqrt(2) * sqrt_m_minus * (u2 / a1) * shift_multi(C, dn=0, dm=-1, dp=0) - 
        jnp.sqrt(2) * sqrt_p_minus * (u1 / a2) * shift_multi(C, dn=0, dm=0, dp=-1)) 

    C_aux_y = (sqrt_n_minus * sqrt_p_minus * (a0 / a2 - a2 / a0) * shift_multi(C, dn=-1, dm=0, dp=-1) + 
        sqrt_n_plus * sqrt_p_minus * (a0 / a2) * shift_multi(C, dn=1, dm=0, dp=-1) - 
        sqrt_n_minus * sqrt_p_plus * (a2 / a0) * shift_multi(C, dn=-1, dm=0, dp=1) + 
        jnp.sqrt(2) * sqrt_p_minus * (u0 / a2) * shift_multi(C, dn=0, dm=0, dp=-1) - 
        jnp.sqrt(2) * sqrt_n_minus * (u2 / a0) * shift_multi(C, dn=-1, dm=0, dp=0))
    
    C_aux_z = (sqrt_n_minus * sqrt_m_minus * (a1 / a0 - a0 / a1) * shift_multi(C, dn=-1, dm=-1, dp=0) + 
        sqrt_n_minus * sqrt_m_plus * (a1 / a0) * shift_multi(C, dn=-1, dm=1, dp=0) - 
        sqrt_n_plus * sqrt_m_minus * (a0 / a1) * shift_multi(C, dn=1, dm=-1, dp=0) + 
        jnp.sqrt(2) * sqrt_n_minus * (u1 / a0) * shift_multi(C, dn=-1, dm=0, dp=0) - 
        jnp.sqrt(2) * sqrt_m_minus * (u0 / a1) * shift_multi(C, dn=0, dm=-1, dp=0))


    E = F[:3] if F0 is None else F[:3] - F0[:3, None, None, None, None, None, None, None]
    force = ((sqrt_n_minus * jnp.sqrt(2) / a0) * E[0] * shift_multi(C, dn=-1, dm=0, dp=0) +
             (sqrt_m_minus * jnp.sqrt(2) / a1) * E[1] * shift_multi(C, dn=0, dm=-1, dp=0) +
             (sqrt_p_minus * jnp.sqrt(2) / a2) * E[2] * shift_multi(C, dn=0, dm=0, dp=-1) +
             F[3] * C_aux_x + F[4] * C_aux_y + F[5] * C_aux_z)
    if F0 is not None:  # remove (q/m) u x B0: the u-terms of C_aux_{x,y,z}
        r2 = jnp.sqrt(2)
        force = force - (
            F0[3] * r2 * (sqrt_m_minus * (u2 / a1) * shift_multi(C, dn=0, dm=-1, dp=0)
                          - sqrt_p_minus * (u1 / a2) * shift_multi(C, dn=0, dm=0, dp=-1))
            + F0[4] * r2 * (sqrt_p_minus * (u0 / a2) * shift_multi(C, dn=0, dm=0, dp=-1)
                            - sqrt_n_minus * (u2 / a0) * shift_multi(C, dn=-1, dm=0, dp=0))
            + F0[5] * r2 * (sqrt_n_minus * (u1 / a0) * shift_multi(C, dn=-1, dm=0, dp=0)
                            - sqrt_m_minus * (u0 / a1) * shift_multi(C, dn=0, dm=-1, dp=0)))

    Col  = -nu * col[None, :, :, :, None, None, None] * Ck
    
    Diff = -D * k2_grid * Ck
        
    # ODEs for Hermite-Fourier coefficients.
    # Closure is achieved by setting to zero coefficients with index out of range.
    dCk_s_dt = (-(kx_grid * (1j / Lx)) * a0 * (
        sqrt_n_plus / jnp.sqrt(2) * shift_multi(Ck, dn=1, dm=0, dp=0) +
        sqrt_n_minus / jnp.sqrt(2) * shift_multi(Ck, dn=-1, dm=0, dp=0) +
        (u0 / a0) * Ck
    ) - (ky_grid * (1j / Ly)) * a1 * (
        sqrt_m_plus / jnp.sqrt(2) * shift_multi(Ck, dn=0, dm=1, dp=0) +
        sqrt_m_minus / jnp.sqrt(2) * shift_multi(Ck, dn=0, dm=-1, dp=0) +
        (u1 / a1) * Ck
    ) - (kz_grid * 1j / Lz) * a2 * (
        sqrt_p_plus / jnp.sqrt(2) * shift_multi(Ck, dn=0, dm=0, dp=1) +
        sqrt_p_minus / jnp.sqrt(2) * shift_multi(Ck, dn=0, dm=0, dp=-1) +
        (u2 / a2) * Ck
    ) + q * Omega_c * (
        jnp.fft.rfftn(force, axes=(-1, -3, -2), norm="forward") * mask23
    ) + Col + Diff)
    
    return dCk_s_dt


def _lower(C, axis, k):
    """C shifted up the Hermite index along `axis` by k: result[n] = C[n - k], zero for n < k."""
    pad = [(0, 0)] * C.ndim
    pad[axis] = (k, 0)
    return jax.lax.slice_in_dim(jnp.pad(C, pad), 0, C.shape[axis], axis=axis)


@partial(jit, static_argnames=['Nn', 'Nm', 'Np', 'Ns'])
def basis_rate_terms(Ck, u_dot, a_dot, alpha_s, Nn, Nm, Np, Ns):
    """
    Rate of change of the coefficients caused by moving the basis, at fixed distribution.

    With ``xi = (v - u(t)) / a(t)`` per species and velocity axis, and SPECTRAX coefficients
    ``C = n* g / (a_x a_y a_z)``, the exact (closure-free) extra terms are, per axis ``i``,

        dC_n/dt += -(u_dot_i / a_i) sqrt(2 n_i) C_{n - e_i}
                   -(a_dot_i / a_i) [(n_i + 1) C_n + sqrt(n_i (n_i - 1)) C_{n - 2 e_i}].

    In ``g`` the width term reads ``-(a_dot/a)[n g_n + sqrt(n(n-1)) g_{n-2}]``; the extra ``C_n``
    comes from the ``1/a`` normalization. Both terms only lower the index, so no closure is needed;
    density is unchanged and the momentum and energy rows keep the physical moments invariant.

    Parameters
    ----------
    Ck : jnp.ndarray, shape (Ns * Np * Nm * Nn, ...) or (Ns, Np, Nm, Nn, ...)
    u_dot, a_dot, alpha_s : jnp.ndarray, shape (3 * Ns,)

    Returns
    -------
    jnp.ndarray with the shape of ``Ck``.
    """
    shape = Ck.shape
    C = Ck.reshape(Ns, Np, Nm, Nn, -1)
    ud, ad, a = (x.reshape(Ns, 3) for x in (u_dot, a_dot, alpha_s))
    out = jnp.zeros_like(C)
    for i, (axis, N) in enumerate(((3, Nn), (2, Nm), (1, Np))):  # storage axes of n_x, n_y, n_z
        idx = [1] * 5
        idx[axis] = N
        n = jnp.arange(N, dtype=float).reshape(idx)
        r_u = (ud[:, i] / a[:, i])[:, None, None, None, None]
        r_a = (ad[:, i] / a[:, i])[:, None, None, None, None]
        out = out - r_u * jnp.sqrt(2 * n) * _lower(C, axis, 1) \
                  - r_a * ((n + 1) * C + jnp.sqrt(n * (n - 1)) * _lower(C, axis, 2))
    return out.reshape(shape)


def uniform_acceleration(F, u_s, qs, Omega_cs, Ns):
    """
    Pump-frame basis-centre rate ``u_dot_s = (q/m)_s (E0 + u_s x B0)`` from the uniform fields.

    ``F`` holds the real-space fields ``(6, Ny, Nx, Nz)``; ``F0`` is their box average. Pass the
    returned ``F0`` to :func:`Hermite_Fourier_system` so the same uniform force is removed from
    the kinetic operator: the two terms then cancel identically instead of in floating point.
    ``(q/m)_s`` is ``qs * Omega_cs`` in SPECTRAX units.

    Returns
    -------
    u_dot : jnp.ndarray, shape (3 * Ns,)
    F0 : jnp.ndarray, shape (6,)
    """
    F0 = jnp.mean(F, axis=(-3, -2, -1))
    u = u_s.reshape(Ns, 3)
    accel = (qs * Omega_cs)[:, None] * (F0[None, :3] + jnp.cross(u, F0[None, 3:]))
    return accel.reshape(3 * Ns), F0


def field_scaled_closure_rate(F, alpha_s, qs, Omega_cs, Nn, Nm, Np, Ns, c, u_s=None, magnetic=False):
    """
    Per-species closure rate that follows the field, for the asymmetric-Hermite top-mode instability.

    The truncated asymmetrically weighted (AW) Fourier-Hermite operator ``-v d/dx + (q/m) E d/dv`` is not
    anti-self-adjoint: in a non-uniform field it has spurious growing modes that live in the top third of the
    Hermite ladder at the largest retained ``|k|``, with growth rate of order ``0.3 sqrt(2N) |q/m| |E| / a``
    (rising with N and with the Fourier cut-off; the symmetrically weighted basis gives exactly zero). A fixed
    ``nu`` is overtaken once the self-consistent field grows. This returns

        nu_s = c |q_s/m_s| max_i sqrt(2 N_i) max_x |E_i - <E_i>| / a_{s,i},

    over the velocity axes with ``N_i > 3``, shaped ``(Ns, 1, 1, 1, 1, 1, 1)`` so it can be passed as ``nu`` to
    :func:`Hermite_Fourier_system` together with ``hypercollision_spectrum(order>=2)``, which leaves density,
    momentum and energy untouched. The uniform part ``<E>`` is excluded: it only shifts the distribution.
    ``c = 0`` gives exactly zero.

    Parameters
    ----------
    F : jnp.ndarray, shape (6, Ny, Nx, Nz)
        Real-space fields (de-aliased), as used by the kinetic RHS.
    alpha_s : jnp.ndarray, shape (3 * Ns,)
    qs, Omega_cs : jnp.ndarray, shape (Ns,)
        ``|q/m|_s = |qs * Omega_cs|`` in SPECTRAX units.
    c : float
        Dimensionless strength.
    u_s : jnp.ndarray, shape (3 * Ns,), optional
        Basis centres; only used with ``magnetic=True`` (default zero).
    magnetic : bool
        Also scale with the non-uniform magnetic force. On axis ``i`` the field scale
        ``max_x |E_i - <E_i>|`` becomes ``max_x |E_i - <E_i>| + sum_{j != i} V_j max_x |B_l - <B_l>|``
        (``i, j, l`` a permutation), ``V_j = |u_j| + a_j sqrt(2 N_j)`` the largest velocity the kept
        ladder on axis ``j`` represents (turning point of its top Hermite function). The ``v x B`` term
        gives the same non-normal top-mode growth as ``E`` once ``B`` is non-uniform (e.g. Weibel past
        U_B ~ 1e-3, where the electric-only rate leaves it unchanged). Default ``False`` is the
        electric-only rate above, unchanged.
    """
    N = jnp.array([Nn, Nm, Np], dtype=float)
    a = alpha_s.reshape(Ns, 3)
    if magnetic:
        dev = jnp.max(jnp.abs(F - jnp.mean(F, axis=(-3, -2, -1), keepdims=True)), axis=(-3, -2, -1))  # (6,)
        dE, dB = dev[:3], dev[3:6]
        u = jnp.zeros_like(a) if u_s is None else u_s.reshape(Ns, 3)
        V = jnp.abs(u) + a * jnp.sqrt(2 * N)[None, :]
        scale = dE[None, :] + jnp.stack([V[:, 1] * dB[2] + V[:, 2] * dB[1],
                                         V[:, 2] * dB[0] + V[:, 0] * dB[2],
                                         V[:, 0] * dB[1] + V[:, 1] * dB[0]], axis=1)
    else:
        E = F[:3]
        dE = jnp.max(jnp.abs(E - jnp.mean(E, axis=(-3, -2, -1), keepdims=True)), axis=(-3, -2, -1))  # (3,)
        scale = jnp.broadcast_to(dE[None, :], a.shape)
    active = (N > 3)[None, :]
    per_axis = jnp.where(active, jnp.sqrt(2 * N)[None, :] * scale / a, 0.0)
    rate = c * jnp.abs(qs * Omega_cs) * jnp.max(per_axis, axis=1)
    return rate.reshape(Ns, 1, 1, 1, 1, 1, 1)


def _dougherty_axes(Nn, Nm, Np):
    """Velocity axes that carry the Dougherty operator: ``(i, storage axis, N_i)`` with ``N_i >= 3``.

    An axis with ``N_i <= 2`` cannot hold the second moment; on it the projected operator vanishes
    identically (rows 0 and 1 are zero), so it is skipped and not counted in the velocity dimension.
    """
    return tuple((i, ax, N) for i, (ax, N) in enumerate(((3, Nn), (2, Nm), (1, Np))) if N >= 3)


def dougherty_moments(C, alpha_s, u_s, Nn, Nm, Np, Ns):
    """
    Spatially local density, mean velocity and temperature of each species from its Hermite coefficients.

    With ``f = sum_n C_n psi_n(xi)``, ``xi = (v - u)/a`` per axis, only ``n <= 2`` per axis enter:

        n_s   = a_x a_y a_z C_0,
        U_i   = u_i + a_i C_{e_i} / (sqrt(2) C_0),
        P_ii / (m n) = a_i^2 [1/2 + C_{2 e_i} / (sqrt(2) C_0) - (C_{e_i} / (sqrt(2) C_0))^2],
        theta = (1/d) sum_i P_ii / (m n)   over the d axes with N_i >= 3 (Gkeyll's v_t^2 = tr P / (d m n)).

    Parameters
    ----------
    C : jnp.ndarray, shape (Ns, Np, Nm, Nn, Ny, Nx, Nz)
        Real-space coefficients.

    Returns
    -------
    dens : (Ns, Ny, Nx, Nz); U : (Ns, 3, Ny, Nx, Nz) (``U_i = u_i`` on axes without the operator);
    theta : (Ns, Ny, Nx, Nz).
    """
    axes = _dougherty_axes(Nn, Nm, Np)
    if not axes:
        raise ValueError("the Dougherty operator needs at least one velocity axis with N >= 3 Hermite modes")
    a, u = alpha_s.reshape(Ns, 3), u_s.reshape(Ns, 3)
    sp = (slice(None),) + (None,) * (C.ndim - 4)
    C0 = C[:, 0, 0, 0]
    dens = jnp.prod(a, axis=1)[sp] * C0
    U, theta = [u[:, i][sp] * jnp.ones_like(C0) for i in range(3)], 0.0
    for i, ax, _ in axes:
        idx1, idx2 = [0, 0, 0, 0], [0, 0, 0, 0]
        idx1[ax], idx2[ax] = 1, 2
        d = C[(slice(None), *idx1[1:])] / (jnp.sqrt(2.0) * C0)
        U[i] = U[i] + a[:, i][sp] * d
        theta = theta + a[:, i][sp] ** 2 * (0.5 + C[(slice(None), *idx2[1:])] / (jnp.sqrt(2.0) * C0) - d ** 2)
    return dens, jnp.stack(U, axis=1), theta / len(axes)


def dougherty_rate_real(C, U, theta, nu, alpha_s, u_s, Nn, Nm, Np, Ns):
    """
    Exact Hermite projection of ``nu d_v . [(v - U) f + theta d_v f]`` in real space, any basis ``(u_s, a_s)``.

    With ``delta_i = (U_i - u_i)/a_i`` and ``tau_i = theta / a_i^2`` the operator in ``xi`` is
    ``nu sum_i d_xi_i [(xi_i - delta_i) f + tau_i d_xi_i f]``. Using ``d psi_n = -sqrt(2(n+1)) psi_{n+1}``
    and ``xi psi_n = sqrt((n+1)/2) psi_{n+1} + sqrt(n/2) psi_{n-1}``, the stencils compose to

        (C f)_n = nu sum_i [ -n_i C_n + sqrt(2 n_i) delta_i C_{n - e_i} + (2 tau_i - 1) sqrt(n_i (n_i - 1)) C_{n - 2 e_i} ].

    Every source has index ``<= n``: the projection onto the retained modes is exact (no truncation of an
    intermediate). Rows 0, 1 vanish and the sum of the row-2 energy terms vanishes by the definition of
    ``U`` and ``theta`` (:func:`dougherty_moments`); any Maxwellian with matching ``U, theta`` is in the
    kernel, truncated or not. ``U``, ``theta`` and ``nu`` may vary in space (shapes as returned by
    :func:`dougherty_moments`; ``nu`` broadcastable to ``(Ns, Ny, Nx, Nz)``).
    """
    a, u = alpha_s.reshape(Ns, 3), u_s.reshape(Ns, 3)
    sp = (slice(None),) + (None,) * (C.ndim - 4)
    out = jnp.zeros_like(C)
    for i, ax, N in _dougherty_axes(Nn, Nm, Np):
        shape = [1] * C.ndim
        shape[ax] = N
        n = jnp.arange(N, dtype=float).reshape(shape)
        delta = ((U[:, i] - u[:, i][sp]) / a[:, i][sp])[:, None, None, None]
        tau = (theta / a[:, i][sp] ** 2)[:, None, None, None]
        out = out - n * C + jnp.sqrt(2 * n) * delta * _lower(C, ax, 1) \
                  + (2 * tau - 1) * jnp.sqrt(n * (n - 1)) * _lower(C, ax, 2)
    return jnp.asarray(nu)[:, None, None, None] * out if jnp.ndim(nu) else nu * out


def dougherty_cross_moments(dens, U, theta, ms, s, r, active):
    """
    Pair moments ``(u_sr, theta_sr, theta_rs)`` that make ``C_sr + C_rs`` conserve momentum and energy.

    With ``m_s n_s nu_sr = m_r n_r nu_rs`` and a common ``u_sr = u_rs = (U_s + U_r)/2`` the friction is
    antisymmetric, ``R_sr = -R_rs``. A common temperature ``T_sr = T_rs``,

        T_sr = (m_r T_s + m_s T_r)/(m_s + m_r) + m_s m_r |U_s - U_r|^2 / (2 d (m_s + m_r)),

    makes the exchanges cancel, ``Q_sr + Q_rs = 0`` (``T = m theta``; ``d`` and the drift difference run over
    the ``active`` velocity axes, those with ``N_i >= 3``, the only ones on which the operator acts).
    This is Greene's form with ``beta = 0``.
    """
    usr = 0.5 * (U[s] + U[r])
    d = len(active)
    du2 = sum((U[s, i] - U[r, i]) ** 2 for i in active)
    T = ms[s] * ms[r] * (theta[s] + theta[r]) / (ms[s] + ms[r]) \
        + ms[s] * ms[r] * du2 / (2 * d * (ms[s] + ms[r]))
    return usr, T / ms[s], T / ms[r]


def dougherty_collisions(C, nu_matrix, alpha_s, u_s, Omega_cs, Nn, Nm, Np, Ns, mask23):
    """
    Spectral Dougherty collision rate ``dCk/dt`` of all species (self and pair collisions), de-aliased.

    ``nu_matrix`` (Ns, Ns): ``nu_ss`` on the diagonal; for ``s < r`` the entry ``[s, r]`` is ``nu_sr`` and
    ``nu_rs = nu_sr m_s n_s / (m_r n_r)`` is set pointwise (momentum conservation), the lower triangle is
    ignored. Masses are ``m_s = 1 / Omega_cs`` (SPECTRAX units). Moments and products are formed in real
    space from the masked state and the result is transformed and masked with the same strict 2/3 mask.
    """
    dens, U, theta = dougherty_moments(C, alpha_s, u_s, Nn, Nm, Np, Ns)
    nu_matrix = jnp.asarray(nu_matrix, dtype=float)
    nu_self = jnp.diagonal(nu_matrix)[:, None, None, None]
    rate = dougherty_rate_real(C, U, theta, nu_self, alpha_s, u_s, Nn, Nm, Np, Ns)
    ms = 1.0 / Omega_cs
    active = tuple(i for i, _, _ in _dougherty_axes(Nn, Nm, Np))
    for s in range(Ns):
        for r in range(s + 1, Ns):
            usr, th_sr, th_rs = dougherty_cross_moments(dens, U, theta, ms, s, r, active)
            nu_sr = nu_matrix[s, r]
            nu_rs = nu_sr * ms[s] * dens[s] / (ms[r] * dens[r])
            for a_, b_, th, nu in ((s, r, th_sr, nu_sr), (r, s, th_rs, nu_rs)):
                one = dougherty_rate_real(C[a_:a_ + 1], usr[None], th[None], jnp.broadcast_to(nu, th.shape)[None],
                                          alpha_s.reshape(Ns, 3)[a_], u_s.reshape(Ns, 3)[a_], Nn, Nm, Np, 1)
                rate = rate.at[a_].add(one[0])
    return jnp.fft.rfftn(rate, axes=(-1, -3, -2), norm="forward") * mask23
