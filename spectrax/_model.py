import jax.numpy as jnp
from jax import vmap, jit, debug
from functools import partial
from jax.lax import dynamic_slice, cond
from jax.scipy.signal import convolve

__all__ = ['Hermite_DG_system']

def _pad_hermite_axes(Ck):
    # pad +1 on both sides for n,m,p only
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

@partial(jit, static_argnames=['Nn', 'Nm', 'Np', 'Ns'])
def Hermite_DG_system(Ck, Fk, col, sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus, basis_idx, 
                        inner_mm, inner_pm, inner_mp, inner_pp, di_inner_product, tripple_product, Ax_p, Ax_m, Ay_p, Ay_m, Az_p, Az_m, R_p, R_m,
                           Lx, Ly, Lz, nu, D, alpha_s, u_s, ms, qs, Omega_ce, Nn, Nm, Np, Ns):
    """
    Computes the time derivative of a single Hermite-DG coefficient Ck[n, m, p] for species s
    in a Vlasov-Maxwell spectral solver using a Hermite velocity, Galerkin space decomposition

    Returns
    -------
    dCk_s_dt : jax.Array, shape (Ny, Nx, Nz, Nl)
        Time derivative of the Hermite-Galerkin coefficient Ck[p, m, n] for species s.
    """

    Ck = Ck.reshape(Ns, Np, Nm, Nn, *Ck.shape[-4:])
    Ny, Nx, Nz = Ck.shape[-4], Ck.shape[-3], Ck.shape[-2]
    dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz 

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
    
    integral = - (a0 * jnp.tensordot(sqrt_n_plus / jnp.sqrt(2) * shift_multi(Ck, dn=1, dm=0, dp=0) +
        sqrt_n_minus / jnp.sqrt(2) * shift_multi(Ck, dn=-1, dm=0, dp=0) +
        (u0 / a0) * Ck, di_inner_product[0], axes=(-1, -1))
    ) - (a1 * jnp.tensordot(sqrt_m_plus / jnp.sqrt(2) * shift_multi(Ck, dn=0, dm=1, dp=0) +
        sqrt_m_minus / jnp.sqrt(2) * shift_multi(Ck, dn=0, dm=-1, dp=0) +
        (u1 / a1) * Ck, di_inner_product[1], axes=(-1, -1))
    ) - (a2 * jnp.tensordot(sqrt_p_plus / jnp.sqrt(2) * shift_multi(Ck, dn=0, dm=0, dp=1) +
        sqrt_p_minus / jnp.sqrt(2) * shift_multi(Ck, dn=0, dm=0, dp=-1) +
        (u2 / a2) * Ck, di_inner_product[2], axes=(-1, -1))
    )

    ################### Boundary Terms ###################

    # Ax_p/m should have shape (Ns, 1, 1, Nn, Nn 1, 1, 1, 1, 1) then multiplied by Ck[:, :, :, None, :, :, :, :, None, :] then sum over axis 4
    bdy_xp = jnp.sum(Ax_p * Ck[:, :, :, None, :, :, :, :, None, :] * inner_mm[0] 
                        + Ax_m * shift_element(Ck, dx=+1, dy=0, dz=0)[:, :, :, None, :, :, :, :, None, :] * inner_pm[0], axis=(4, -1))
    bdy_xm = jnp.sum(Ax_m * Ck[:, :, :, None, :, :, :, :, None, :] * inner_pp[0] 
                        + Ax_p * shift_element(Ck, dx=-1, dy=0, dz=0)[:, :, :, None, :, :, :, :, None, :] * inner_mp[0], axis=(4, -1))
    bdy_yp = jnp.sum(Ay_p * Ck[:, :, None, :, :, :, :, :, None, :] * inner_mm[1]
                        + Ay_m * shift_element(Ck, dx=0, dy=+1, dz=0)[:, :, None, :, :, :, :, :, None, :] * inner_pm[1], axis=(3, -1))
    bdy_ym = jnp.sum(Ay_m * Ck[:, :, None, :, :, :, :, :, None, :] * inner_pp[1] 
                        + Ay_p * shift_element(Ck, dx=0, dy=-1, dz=0)[:, :, None, :, :, :, :, :, None, :] * inner_mp[1], axis=(3, -1))
    bdy_zp = jnp.sum(Az_p * Ck[:, None, :, :, :, :, :, :, None, :] * inner_mm[2] 
                        + Az_m * shift_element(Ck, dx=0, dy=0, dz=+1)[:, None, :, :, :, :, :, :, None, :] * inner_pm[2], axis=(2, -1))
    bdy_zm = jnp.sum(Az_m * Ck[:, None, :, :, :, :, :, :, None, :] * inner_pp[2] 
                        + Az_p * shift_element(Ck, dx=0, dy=0, dz=-1)[:, None, :, :, :, :, :, :, None, :] * inner_mp[2], axis=(2, -1))

    bdy = bdy_xp - bdy_xm + bdy_yp - bdy_ym + bdy_zp - bdy_zm

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

    ECx = Fk[0, None, None, None, None, :, :, :, :, None] * shift_multi(Ck, dn=-1, dm=0, dp=0)[:, :, :, :, :, :, :, None, :]    
    Ex_term = (sqrt_n_minus * jnp.sqrt(2) / a0) * jnp.tensordot(ECx, tripple_product, axes=([-2, -1], [0, 1]))

    BC_aux_x = Fk[3, None, None, None, None, :, :, :, :, None] * C_aux_x[:, :, :, :, :, :, :, None, :]
    Bx_term = jnp.tensordot(BC_aux_x, tripple_product, axes=([-2, -1], [0, 1]))

    ECy = Fk[1, None, None, None, None, :, :, :, :, None] * shift_multi(Ck, dn=0, dm=-1, dp=0)[:, :, :, :, :, :, :, None, :]
    Ey_term = (sqrt_m_minus * jnp.sqrt(2) / a1) * jnp.tensordot(ECy, tripple_product, axes=([-2, -1], [0, 1]))

    BC_aux_y = Fk[4, None, None, None, None, :, :, :, :, None] * C_aux_y[:, :, :, :, :, :, :, None, :]
    By_term = jnp.tensordot(BC_aux_y, tripple_product, axes=([-2, -1], [0, 1]))

    ECz = Fk[2, None, None, None, None, :, :, :, :, None] * shift_multi(Ck, dn=0, dm=0, dp=-1)[:, :, :, :, :, :, :, None, :]
    Ez_term = (sqrt_p_minus * jnp.sqrt(2) / a2) * jnp.tensordot(ECz, tripple_product, axes=([-2, -1], [0, 1]))

    BC_aux_z = Fk[5, None, None, None, None, :, :, :, :, None] * C_aux_z[:, :, :, :, :, :, :, None, :]
    Bz_term = jnp.tensordot(BC_aux_z, tripple_product, axes=([-2, -1], [0, 1]))

    dNL = -(q * Omega_ce / ms) * (Ex_term + Bx_term + Ey_term + By_term + Ez_term + Bz_term)

    ################## Inverse mass matrix and final expression #################
    
    Col  = -nu * col[None, :, :, :, None, None, None, None] * Ck
    Diff = -D * 0 * Ck # Diffusion disabled for now. Adding it is very low priority.

    inv_m = (2 * basis_idx[:, 0] + 1) * (2 * basis_idx[:, 1] + 1) * (2 * basis_idx[:, 2] + 1) / (dx * dy * dz)
    dCk_s_dt = -inv_m * (integral + bdy + dNL) + Col + Diff

    ####################### Field evolution equations (Maxwell) #####################
    
    Cn1 = cond(Nn > 1, lambda x: Ck[:, 0, 0, 1], lambda x: jnp.zeros_like(Ck[:, 0, 0, 0]), operand=None) # Only use first order mode for current if it exists
    Cm1 = cond(Nm > 1, lambda x: Ck[:, 0, 1, 0], lambda x: jnp.zeros_like(Ck[:, 0, 0, 0]), operand=None)
    Cp1 = cond(Np > 1, lambda x: Ck[:, 1, 0, 0], lambda x: jnp.zeros_like(Ck[:, 0, 0, 0]), operand=None)

    source_x = jnp.sum((q * a0 * a1 * a2)[:, 0, 0, 0] * (Ck[:, 0, 0, 0] * u0[:, 0, 0, 0] + a0[:, 0, 0, 0] * Cn1 / jnp.sqrt(2)), axis=0)
    source_y = jnp.sum((q * a0 * a1 * a2)[:, 0, 0, 0] * (Ck[:, 0, 0, 0] * u1[:, 0, 0, 0] + a1[:, 0, 0, 0] * Cm1 / jnp.sqrt(2)), axis=0)
    source_z = jnp.sum((q * a0 * a1 * a2)[:, 0, 0, 0] * (Ck[:, 0, 0, 0] * u2[:, 0, 0, 0] + a2[:, 0, 0, 0] * Cp1 / jnp.sqrt(2)), axis=0)
    source = -(1 / Omega_ce) * jnp.stack((source_x, source_y, source_z, jnp.zeros_like(source_x), jnp.zeros_like(source_y), jnp.zeros_like(source_z)))

    ex = jnp.array([1, 0, 0])
    ey = jnp.array([0, 1, 0])
    ez = jnp.array([0, 0, 1])

    cross_field_x = jnp.concatenate((cross_product(-ex, Fk[3:]), cross_product(ex, Fk[:3])), axis=0)
    cross_field_y = jnp.concatenate((cross_product(-ey, Fk[3:]), cross_product(ey, Fk[:3])), axis=0)
    cross_field_z = jnp.concatenate((cross_product(-ez, Fk[3:]), cross_product(ez, Fk[:3])), axis=0)

    dF_integral =  (jnp.tensordot(cross_field_x, di_inner_product[0], axes=(-1, -1)) 
                    + jnp.tensordot(cross_field_y, di_inner_product[1], axes=(-1, -1)) 
                    + jnp.tensordot(cross_field_z, di_inner_product[2], axes=(-1, -1)))

    ####################### Boundary ########################
    
    dF_bdy_xp = jnp.sum(R_p[0] * Fk[None, :, :, :, :, None, :] * inner_mm[0] 
                        + R_m[0] * shift_element(Fk, dx=+1, dy=0, dz=0)[None, :, :, :, :, None, :] * inner_pm[0], axis=(1, -1))
    dF_bdy_xm = jnp.sum(R_m[0] * Fk[None, :, :, :, :, None, :] * inner_pp[0] 
                        + R_p[0] * shift_element(Fk, dx=-1, dy=0, dz=0)[None, :, :, :, :, None, :] * inner_mp[0], axis=(1, -1))
    dF_bdy_yp = jnp.sum(R_p[1] * Fk[None, :, :, :, :, None, :] * inner_mm[1] 
                        + R_m[1] * shift_element(Fk, dx=0, dy=+1, dz=0)[None, :, :, :, :, None, :] * inner_pm[1], axis=(1, -1))
    dF_bdy_ym = jnp.sum(R_m[1] * Fk[None, :, :, :, :, None, :] * inner_pp[1] 
                        + R_p[1] * shift_element(Fk, dx=0, dy=-1, dz=0)[None, :, :, :, :, None, :] * inner_mp[1], axis=(1, -1))
    dF_bdy_zp = jnp.sum(R_p[2] * Fk[None, :, :, :, :, None, :] * inner_mm[2] 
                        + R_m[2] * shift_element(Fk, dx=0, dy=0, dz=+1)[None, :, :, :, :, None, :] * inner_pm[2], axis=(1, -1))
    dF_bdy_zm = jnp.sum(R_m[2] * Fk[None, :, :, :, :, None, :] * inner_pp[2] 
                        + R_p[2] * shift_element(Fk, dx=0, dy=0, dz=-1)[None, :, :, :, :, None, :] * inner_mp[2], axis=(1, -1))
    
    dF_bdy = dF_bdy_xp + dF_bdy_yp + dF_bdy_zp - (dF_bdy_xm + dF_bdy_ym + dF_bdy_zm)

    dFk_dt = inv_m * (dF_integral - dF_bdy) + source
    return jnp.concatenate([dCk_s_dt.reshape(-1), dFk_dt.reshape(-1)])