"""Hermite–Disontinuous_Galerkin model operators for the Vlasov–Maxwell system.

This module contains the right-hand side of the Hermite–DG moment
equations (:func:`Hermite_DG_system`).
"""

import jax.numpy as jnp
from jax import vmap, jit, debug
from functools import partial

__all__ = ['Hermite_DG_system']

def _pad_hermite_axes(Ck):
    """Pad Hermite axes (p, m, n) by one cell on both sides.

    This padding enables safe slicing for the zero-padded shift operator used in
    :func:`shift_multi`.
    """
    return jnp.pad(
        Ck,
        ((0,0), (1,1), (1,1), (1,1), (0,0), (0,0), (0,0), (0,0))
    )

def shift_multi(Ck, dn=0, dm=0, dp=0):
    """
    Zero-padded shift along Hermite axes (n,m,p) simultaneously.
    dn=+1 means 'use source at n-1', dn=-1 means 'use source at n+1', dn=0 is identity.
    Same for dm, dp. Works for values in {-1,0,+1}.
    """
    P = _pad_hermite_axes(Ck)
    _, Np, Nm, Nn, _, _, _, _ = Ck.shape
    # Start indices in the padded array
    n0 = 1 + dn   # dn=+1 -> 0 ; dn=0 -> 1 ; dn=-1 -> 2
    m0 = 1 + dm
    p0 = 1 + dp
    return P[:, p0:p0+Np, m0:m0+Nm, n0:n0+Nn, :, :, :, :]

def shift_element(C, dx, dy, dz):

    return jnp.roll(C, (-dy, -dx, -dz), axis=(-4, -3, -2))

# Unit element offsets (dx, dy, dz) and the matching Hermite axis of Ck (Ns, Np, Nm, Nn, ...) for each direction.
_UNIT_OFFSETS = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_HERMITE_AXIS = (3, 2, 1)

# Below this many modes XLA fuses the broadcast form of the flux into a single pass, which is faster than a
# matrix multiply; above it the broadcast product gets materialized and the matmul form is far cheaper (CPU timings).
_MATMUL_MIN_MODES = 32

def _hermite_matmul(A, X, axis):
    """Apply per-species matrices A (Ns, N, N) along Hermite axis `axis` of X, as one matmul per species."""
    Xt = jnp.moveaxis(X, axis, 1)
    out = jnp.matmul(A, Xt.reshape(Xt.shape[:2] + (-1,))).reshape(Xt.shape)
    return jnp.moveaxis(out, 1, axis)

def _upwind_flux(A_p, A_m, C, C_plus, C_minus, inner_mm, inner_pm, inner_mp, inner_pp, axis):
    """Net upwind flux through the two faces of every element along one direction.

    Computes A_p (C·mm - C_minus·mp) + A_m (C_plus·pm - C·pp), where A_p, A_m (Ns, N, N) act on Hermite
    axis `axis` of the (Ns, Np, Nm, Nn, Ny, Nx, Nz, Nl) arrays C, C_plus, C_minus (the element and its
    neighbours above and below), and each inner_* (Nl, Nl) acts on the last (DG mode) axis.
    """
    N = A_p.shape[-1]
    if N >= _MATMUL_MIN_MODES:
        face = lambda X, inner: jnp.tensordot(X, inner, axes=(-1, -1))
        return (_hermite_matmul(A_p, face(C, inner_mm) - face(C_minus, inner_mp), axis)
                + _hermite_matmul(A_m, face(C_plus, inner_pm) - face(C, inner_pp), axis))

    # Broadcast form: insert the output Hermite axis before `axis` and the output DG axis before the last axis,
    # then reduce over the input Hermite and DG axes.
    trailing = (1,) * (C.ndim - axis)
    to_bcast = lambda A: A.reshape((A.shape[0],) + (1,) * (axis - 1) + A.shape[1:] + trailing)
    expand = lambda X: jnp.expand_dims(X, (axis, X.ndim))
    C, C_plus, C_minus = expand(C), expand(C_plus), expand(C_minus)
    return jnp.sum(to_bcast(A_p) * (C * inner_mm - C_minus * inner_mp)
                   + to_bcast(A_m) * (C_plus * inner_pm - C * inner_pp), axis=(axis + 1, -1))

@jit
def cross_product(k_vec, F_vec):
    """
    Computes the cross product of two 3D vectors.
    Args:
        k_vec (array-like): A 3-element array representing the first vector (k).
        F_vec (array-like): A 3-element array representing the second vector (F).
    Returns:
        jnp.ndarray: A 3-element array representing the cross product k x F.
    """
    kx, ky, kz = k_vec
    Fx, Fy, Fz = F_vec
    return jnp.array([ky * Fz - kz * Fy, kz * Fx - kx * Fz, kx * Fy - ky * Fx])

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'dims', 'shift_element'])
def Hermite_DG_system(Ck, Fk, col, sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus, basis_idx,
                        inner_mm, inner_pm, inner_mp, inner_pp, di_inner_product, triple_product, Ax_p, Ax_m, Ay_p, Ay_m, Az_p, Az_m, R_p, R_m,
                           Lx, Ly, Lz, nu, D, alpha_s, u_s, ms, qs, Omega_ce, Nx, Ny, Nz, Nn, Nm, Np, Ns, dims, shift_element=shift_element):
    """
    Evaluate the right-hand side of the coupled Hermite-DG moment equations.

    Parameters
    ----------
    Ck : jnp.ndarray
        Hermite-DG coefficients with shape `(Ns, Np, Nm, Nn, Ny, Nx, Nz, Nl)` (or with the Hermite and species
        axes flattened into one).
    Fk : jnp.ndarray
        DG coefficient-space electromagnetic fields with shape `(6, Ny, Nx, Nz, Nl)` ordered
        as `(Ex, Ey, Ez, Bx, By, Bz)`.
    col : jnp.ndarray
        Precomputed collision coefficients.
    sqrt_* : jnp.ndarray
        Square-root ladder coefficients for the Hermite recurrences along each axis.
    basis_idx : jnp.ndarray
        Index array of each direction's mode for each basis index
    inner_* : jnp.ndarray
        Array of basis function evaluations at element boundary for each direction and basis index. 
        m="minus": using value approaching element boundary from below.
        p="plus": using value from above.
    di_inner_product : jnp.ndarray
        Array of inner products between one basis element and the derivative in one of the directions of another basis element.
    triple_product : jnp.ndarray
        Array of integrals of three multiplied basis elements.
    A* : jnp.ndarray
        Velocity-space Hermite-coefficient matrix for coefficient time derivative, shape `(Ns, N, N)` with N the
        number of Hermite modes along that direction. (x,y,z) = derivative in (x,y,z) direction.
        (p,m) = matrix decomposition into positive and negative eigenvalue parts, respectively, for upwinding scheme.
    R_* : jnp.ndarray
        EM-field derivative coupling matrix (through curl E / curl B), shape `(3, 6, 6)` indexed by direction.
        (p,m) = matrix decomposition into positive and negative eigenvalue parts, respectively.
    Lx, Ly, Lz : float
        Domain lengths in each spatial direction.
    nu : float
        Collision frequency.
    D : float
        Hyper-diffusion coefficient.
    alpha_s, u_s : jnp.ndarray
        Thermal scaling parameters and drift velocities per species.
    ms : jnp.ndarray
        Species masses.
    qs : jnp.ndarray
        Species charges.
    Omega_ce : jnp.ndarray
        Electron cyclotron frequency.
    Nx, Ny, Nz : int
        Number of elements per spatial dimension in the whole domain (which may differ from the array
        shapes when only part of the domain is passed in).
    Nn, Nm, Np, Ns : int
        Number of Hermite modes and species.
    dims : int
        Number of spatial dimensions in which the DG basis varies.
    shift_element : callable, optional
        `shift_element(C, dx, dy, dz)` returning the coefficients of the neighbouring element at offset
        `(dx, dy, dz)`. Defaults to a periodic shift of the arrays passed in.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Time derivatives `(dCk/dt, dFk/dt)` with shapes `(Ns, Np, Nm, Nn, Ny, Nx, Nz, Nl)` and `(6, Ny, Nx, Nz, Nl)`.
    """

    Ck = Ck.reshape(Ns, Np, Nm, Nn, *Ck.shape[-4:])
    dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz

    # A direction with a single element and no basis variation has no internal derivative or net face flux,
    # so its terms vanish identically and are skipped.
    Nl = basis_idx.shape[0]
    active_dirs = [d for d, N_elements in enumerate((Nx, Ny, Nz)) if N_elements > 1 or (Nl > 1 and d < dims)]

    # Define u, alpha, charge, and gyrofrequency depending on species.
    alpha = alpha_s.reshape(Ns, 3)
    u = u_s.reshape(Ns, 3)
    a0 = alpha[:, 0][:, None, None, None, None, None, None, None]
    a1 = alpha[:, 1][:, None, None, None, None, None, None, None]
    a2 = alpha[:, 2][:, None, None, None, None, None, None, None]
    u0 = u[:, 0][:, None, None, None, None, None, None, None]
    u1 = u[:, 1][:, None, None, None, None, None, None, None]
    u2 = u[:, 2][:, None, None, None, None, None, None, None]
    q = qs[:, None, None, None, None, None, None, None]
    ms = ms[:, None, None, None, None, None, None, None]
    
    # ODEs for Hermite-DG coefficients.
    # Closure is achieved by setting to zero coefficients with index out of range.

    ################### Distribution function linear integral term ###################
    
    a_dir, u_dir = (a0, a1, a2), (u0, u1, u2)
    sqrt_plus, sqrt_minus = (sqrt_n_plus, sqrt_m_plus, sqrt_p_plus), (sqrt_n_minus, sqrt_m_minus, sqrt_p_minus)
    hermite_shift = ('dn', 'dm', 'dp')
    integral = 0
    for d in active_dirs:
        streaming = (sqrt_plus[d] / jnp.sqrt(2) * shift_multi(Ck, **{hermite_shift[d]: 1}) +
                     sqrt_minus[d] / jnp.sqrt(2) * shift_multi(Ck, **{hermite_shift[d]: -1}) +
                     (u_dir[d] / a_dir[d]) * Ck)
        integral = integral - a_dir[d] * jnp.tensordot(streaming, di_inner_product[d], axes=(-1, -1))

    ################### Boundary Terms ###################

    A_p, A_m = (Ax_p, Ay_p, Az_p), (Ax_m, Ay_m, Az_m)
    bdy = 0
    for d in active_dirs:
        offset = _UNIT_OFFSETS[d]
        bdy = bdy + _upwind_flux(A_p[d], A_m[d], Ck, shift_element(Ck, *offset), shift_element(Ck, *(-o for o in offset)),
                                 inner_mm[d], inner_pm[d], inner_mp[d], inner_pp[d], _HERMITE_AXIS[d])

    ###################### Nonlinear terms #########################

    C_aux_x = (sqrt_m_minus * sqrt_p_minus * (a2 / a1 - a1 / a2) * shift_multi(Ck, dn=0, dm=-1, dp=-1) + 
        sqrt_m_minus * sqrt_p_plus * (a2 / a1) * shift_multi(Ck, dn=0, dm=-1, dp=1) - 
        sqrt_m_plus * sqrt_p_minus * (a1 / a2) * shift_multi(Ck, dn=0, dm=1, dp=-1) + 
        jnp.sqrt(2) * sqrt_m_minus * (u2 / a1) * shift_multi(Ck, dn=0, dm=-1, dp=0) - 
        jnp.sqrt(2) * sqrt_p_minus * (u1 / a2) * shift_multi(Ck, dn=0, dm=0, dp=-1)) 

    C_aux_y = (sqrt_n_minus * sqrt_p_minus * (a0 / a2 - a2 / a0) * shift_multi(Ck, dn=-1, dm=0, dp=-1) + 
        sqrt_n_plus * sqrt_p_minus * (a0 / a2) * shift_multi(Ck, dn=1, dm=0, dp=-1) - 
        sqrt_n_minus * sqrt_p_plus * (a2 / a0) * shift_multi(Ck, dn=-1, dm=0, dp=1) + 
        jnp.sqrt(2) * sqrt_p_minus * (u0 / a2) * shift_multi(Ck, dn=0, dm=0, dp=-1) - 
        jnp.sqrt(2) * sqrt_n_minus * (u2 / a0) * shift_multi(Ck, dn=-1, dm=0, dp=0))
    
    C_aux_z = (sqrt_n_minus * sqrt_m_minus * (a1 / a0 - a0 / a1) * shift_multi(Ck, dn=-1, dm=-1, dp=0) + 
        sqrt_n_minus * sqrt_m_plus * (a1 / a0) * shift_multi(Ck, dn=-1, dm=1, dp=0) - 
        sqrt_n_plus * sqrt_m_minus * (a0 / a1) * shift_multi(Ck, dn=1, dm=-1, dp=0) + 
        jnp.sqrt(2) * sqrt_n_minus * (u1 / a0) * shift_multi(Ck, dn=-1, dm=0, dp=0) - 
        jnp.sqrt(2) * sqrt_m_minus * (u0 / a1) * shift_multi(Ck, dn=0, dm=-1, dp=0))

    # Hermite-space factor multiplying each field component (Ex, Ey, Ez, Bx, By, Bz) in the Lorentz force term.
    hermite_factors = (sqrt_n_minus * jnp.sqrt(2) / a0 * shift_multi(Ck, dn=-1, dm=0, dp=0),
                       sqrt_m_minus * jnp.sqrt(2) / a1 * shift_multi(Ck, dn=0, dm=-1, dp=0),
                       sqrt_p_minus * jnp.sqrt(2) / a2 * shift_multi(Ck, dn=0, dm=0, dp=-1),
                       C_aux_x, C_aux_y, C_aux_z)

    # Per-element field matrices M[f, y, x, z, j, k] = sum_i Fk[f, y, x, z, i] * triple_product[i, j, k].
    M = jnp.einsum('fyxzi,ijk->fyxzjk', Fk, triple_product)
    lorentz = sum(jnp.sum(H[..., :, None] * M[f], axis=-2) for f, H in enumerate(hermite_factors))

    dNL = -(q * Omega_ce / ms) * lorentz

    ################## Inverse mass matrix and final expression #################
    
    def compute_collisions(Ck, nu):
        p = jnp.arange(Np)[None, :, None, None, None, None, None, None]
        m = jnp.arange(Nm)[None, None, :, None, None, None, None, None]
        n = jnp.arange(Nn)[None, None, None, :, None, None, None, None]

        C000 = Ck[:, 0, 0, 0, None, None, None, :, :, :, 0, None]
        C100 = Ck[:, 1, 0, 0, None, None, None, :, :, :, 0, None]
        C010 = Ck[:, 0, 1, 0, None, None, None, :, :, :, 0, None]
        C001 = Ck[:, 0, 0, 1, None, None, None, :, :, :, 0, None]
        C200 = Ck[:, 2, 0, 0, None, None, None, :, :, :, 0, None]
        C020 = Ck[:, 0, 2, 0, None, None, None, :, :, :, 0, None]
        C002 = Ck[:, 0, 0, 2, None, None, None, :, :, :, 0, None]
        
        Col000 = - (n + m + p)
        Col100 = jnp.sqrt(p) * C100 / C000
        Col010 = jnp.sqrt(m) * C010 / C000
        Col001 = jnp.sqrt(n) * C001 / C000
        Col200 = jnp.sqrt(p * (p - 1)) * (jnp.sqrt(2) * C200 / C000 - (C100 / C000)**2)
        Col020 = jnp.sqrt(m * (m - 1)) * (jnp.sqrt(2) * C020 / C000 - (C010 / C000)**2)
        Col002 = jnp.sqrt(n * (n - 1)) * (jnp.sqrt(2) * C002 / C000 - (C001 / C000)**2)
        
        terms0 = Col000 * Ck
        terms1 = Col100 * shift_multi(Ck, dn=0, dm=0, dp=-1) + Col010 * shift_multi(Ck, dn=0, dm=-1, dp=0) + Col001 * shift_multi(Ck, dn=-1, dm=0, dp=0)
        terms2 = (Col200 * shift_multi(shift_multi(Ck, dn=0, dm=0, dp=-1), dn=0, dm=0, dp=-1)
                    + Col020 * shift_multi(shift_multi(Ck, dn=0, dm=-1, dp=0), dn=0, dm=-1, dp=0)
                      + Col002 * shift_multi(shift_multi(Ck, dn=-1, dm=0, dp=0), dn=-1, dm=0, dp=0))

        return nu * (terms0 + terms1 + terms2)

    Col = -nu * col[None, :, :, :, None, None, None, None] * Ck
    # Col = compute_collisions(Ck, nu) - 0 * col[None, :, :, :, None, None, None, None] * Ck

    Diff = -D * 0 * Ck # Diffusion disabled for now. Adding it is low priority.

    inv_m = (2 * basis_idx[:, 0] + 1) * (2 * basis_idx[:, 1] + 1) * (2 * basis_idx[:, 2] + 1) / (dx * dy * dz)
    dCk_s_dt = -inv_m * (integral + bdy + dNL) + Col + Diff

    ####################### Field evolution equations (Maxwell) #####################
    
    no_mode = jnp.zeros_like(Ck[:, 0, 0, 0])
    Cn1 = Ck[:, 0, 0, 1] if Nn > 1 else no_mode # Only use first order mode for current if it exists
    Cm1 = Ck[:, 0, 1, 0] if Nm > 1 else no_mode
    Cp1 = Ck[:, 1, 0, 0] if Np > 1 else no_mode

    source_x = jnp.sum((q * a0 * a1 * a2)[:, 0, 0, 0] * (Ck[:, 0, 0, 0] * u0[:, 0, 0, 0] + a0[:, 0, 0, 0] * Cn1 / jnp.sqrt(2)), axis=0)
    source_y = jnp.sum((q * a0 * a1 * a2)[:, 0, 0, 0] * (Ck[:, 0, 0, 0] * u1[:, 0, 0, 0] + a1[:, 0, 0, 0] * Cm1 / jnp.sqrt(2)), axis=0)
    source_z = jnp.sum((q * a0 * a1 * a2)[:, 0, 0, 0] * (Ck[:, 0, 0, 0] * u2[:, 0, 0, 0] + a2[:, 0, 0, 0] * Cp1 / jnp.sqrt(2)), axis=0)
    source = -(1 / Omega_ce) * jnp.stack((source_x, source_y, source_z, jnp.zeros_like(source_x), jnp.zeros_like(source_y), jnp.zeros_like(source_z)))

    dF_integral = 0
    for d in active_dirs:
        e = jnp.array(_UNIT_OFFSETS[d])
        cross_field = jnp.concatenate((cross_product(-e, Fk[3:]), cross_product(e, Fk[:3])), axis=0)
        dF_integral = dF_integral + jnp.tensordot(cross_field, di_inner_product[d], axes=(-1, -1))

    ####################### Boundary ########################

    # The 6 field components take the place of a Hermite axis, so the same flux kernel applies.
    F = Fk[None, None, None]
    dF_bdy = 0
    for d in active_dirs:
        offset = _UNIT_OFFSETS[d]
        dF_bdy = dF_bdy + _upwind_flux(R_p[d][None], R_m[d][None], F, shift_element(F, *offset), shift_element(F, *(-o for o in offset)),
                                       inner_mm[d], inner_pm[d], inner_mp[d], inner_pp[d], axis=3)[0, 0, 0]

    dFk_dt = inv_m * (dF_integral - dF_bdy) + source
    return dCk_s_dt, dFk_dt