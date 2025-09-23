import jax.numpy as jnp
from jax import vmap, jit
from functools import partial
from jax.lax import dynamic_slice
from jax.scipy.signal import convolve

__all__ = ['plasma_current', 'Hermite_Fourier_system']

@partial(jit, static_argnames=['Nn', 'Nm', 'Np', 'Ns'])
def plasma_current(qs, alpha_s, u_s, Ck, Nn, Nm, Np, Ns):
    """
    Computes the Ampère-Maxwell current from spectral coefficients for multiple species.
    
    Parameters
    ----------
    qs : jnp.ndarray, shape (Ns,)
        Charges of the species.
    alpha_s : jnp.ndarray, shape (3 * Ns,)
        Scaling factors for each species
    u_s : jnp.ndarray, shape (3 * Ns,)
        Velocity components for each species, packed like alpha_s.
    Ck : jnp.ndarray, shape (Ns * Nn * Nm * Np, Nx, Ny, Nz)
        Spectral coefficients.
    Nn, Nm, Np : int
        Number of Hermite modes in x, y, and z respectively.
    Ns : int
        Number of species.

    Returns
    -------
    jnp.ndarray, shape (3, Nx, Ny, Nz)
        The total Ampère-Maxwell current (Jx, Jy, Jz).
    """
    # Reshape Ck into structured Hermite-Fourier coefficients: (Ns, Nn, Nm, Np, Ny, Nx, Nz)
    Ck = Ck.reshape(Ns, Np, Nm, Nn, *Ck.shape[-3:])
    
    # Reshape alpha and velocity
    alpha = alpha_s.reshape(Ns, 3)
    u = u_s.reshape(Ns, 3)

    # Grab the modes we need (0,1,1,1) for jx, jy, jz contributions
    C0 = Ck[:, 0, 0, 0]  # shape: (Ns, Nx, Ny, Nz)
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

    # Final current per species: shape (3, Ns, Nx, Ny, Nz)
    J_species = (term1 + term2) * pre[None, :, None, None, None]

    # Sum over species → shape: (3, Nx, Ny, Nz)
    return jnp.sum(J_species, axis=1)

# @jit
# def collision(Nn, Nm, Np, n, m, p):
#     """
#     Computes the scalar collision term for the Vlasov-Maxwell system using 
#     a symmetric form over the Hermite-Fourier indices (n, m, p). 
#     The contribution from each index is included only if the corresponding 
#     mode count (Nn, Nm, Np) exceeds 3. The result is a differentiable sum 
#     over valid contributions.
#     Args:
#         Nn (int): Number of grid points in the n direction.
#         Nm (int): Number of grid points in the m direction.
#         Np (int): Number of grid points in the p direction.
#         n (int): Index in the n direction.
#         m (int): Index in the m direction.
#         p (int): Index in the p direction.
#     Returns:
#         jnp.ndarray: Scalar collision term.
#     """
#     N = jnp.array([Nn, Nm, Np], dtype=jnp.float32)
#     idx = jnp.array([n, m, p], dtype=jnp.float32)
#     # Avoid division by zero by masking invalid terms
#     def safe_term(Nj, nj):
#         term = nj * (nj - 1) * (nj - 2)
#         denom = (Nj - 1) * (Nj - 2) * (Nj - 3)
#         return jnp.where(Nj > 3, term / denom, 0.0)
#     terms = vmap(safe_term)(N, idx)
#     return jnp.sum(terms)

# @jit
# def diffusion(kx, ky, kz, D, exponent=2):
#     """
#     Computes the diffusion term for the Vlasov-Maxwell system.

#     Parameters
#     ----------
#     kx, ky, kz : jnp.ndarray
#         Wavevector components in the x, y, and z directions.
#     D : float
#         Diffusion coefficient.

#     Returns
#     -------
#     jnp.ndarray
#         Diffusion term proportional to kx**2 + ky**2 + kz**2.
#     """
#     k_squared = kx**exponent + ky**exponent + kz**exponent
#     return -D * k_squared
    
# @partial(jit, static_argnames=['Nn', 'Nm', 'Np'])
# def Hermite_Fourier_system(Ck, Ck_hat, Fk_hat, kx_grid, ky_grid, kz_grid, k2_grid, col, Lx, Ly, Lz, nu, D, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, index):
#     """
#     Computes the time derivative of a single Hermite-Fourier coefficient Ck[n, m, p] for species s
#     in a Vlasov-Maxwell spectral solver using a Hermite-Fourier basis.

#     Returns
#     -------
#     dCk_s_dt : jax.Array, shape (Nx, Ny, Nz)
#         Time derivative of the Hermite-Fourier coefficient Ck[n, m, p] for species s.
#     """
#     Ny, Nx, Nz = Ck.shape[-3], Ck.shape[-2], Ck.shape[-1]

#     # Species. s = 0 corresponds to electrons and s = 1 corresponds to ions.
#     s = jnp.floor(index / (Nn * Nm * Np)).astype(jnp.int32)
    
#     # Indices below represent order of Hermite polynomials (they identify the Hermite-Fourier coefficients Ck[n, p, m]).
#     p = jnp.floor((index - s * Nn * Nm * Np) / (Nn * Nm)).astype(jnp.int32)
#     m = jnp.floor((index - s * Nn * Nm * Np - p * Nn * Nm) / Nn).astype(jnp.int32)
#     n = (index - s * Nn * Nm * Np - p * Nn * Nm - m * Nn).astype(jnp.int32)
    
#     # Define u, alpha, charge, and gyrofrequency depending on species.
#     u = dynamic_slice(u_s, (s * 3,), (3,))
#     alpha = dynamic_slice(alpha_s, (s * 3,), (3,))
#     q, Omega_c = qs[s], Omega_cs[s]
    
#     # Define terms to be used in ODEs below.
#     Ck_hat_aux_x = (jnp.sqrt(m * p) * (alpha[2] / alpha[1] - alpha[1] / alpha[2]) * Ck_hat[n + (m-1) * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) * jnp.sign(p) + 
#         jnp.sqrt(m * (p + 1)) * (alpha[2] / alpha[1]) * Ck_hat[n + (m-1) * Nn + (p+1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) * jnp.sign(Np - p - 1) - 
#         jnp.sqrt((m + 1) * p) * (alpha[1] / alpha[2]) * Ck_hat[n + (m+1) * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p) * jnp.sign(Nm - m - 1) + 
#         jnp.sqrt(2 * m) * (u[2] / alpha[1]) * Ck_hat[n + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) - 
#         jnp.sqrt(2 * p) * (u[1] / alpha[2]) * Ck_hat[n + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p)) 

#     Ck_hat_aux_y = (jnp.sqrt(n * p) * (alpha[0] / alpha[2] - alpha[2] / alpha[0]) * Ck_hat[n-1 + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) * jnp.sign(p) + 
#         jnp.sqrt((n + 1) * p) * (alpha[0] / alpha[2]) * Ck_hat[n+1 + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p) * jnp.sign(Nn - n - 1) - 
#         jnp.sqrt(n * (p + 1)) * (alpha[2] / alpha[0]) * Ck_hat[n-1 + m * Nn + (p+1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) * jnp.sign(Np - p - 1) + 
#         jnp.sqrt(2 * p) * (u[0] / alpha[2]) * Ck_hat[n + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p) - 
#         jnp.sqrt(2 * n) * (u[2] / alpha[0]) * Ck_hat[n-1 + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n))
    
#     Ck_hat_aux_z = (jnp.sqrt(n * m) * (alpha[1] / alpha[0] - alpha[0] / alpha[1]) * Ck_hat[n-1 + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) * jnp.sign(m) + 
#         jnp.sqrt(n * (m + 1)) * (alpha[1] / alpha[0]) * Ck_hat[n-1 + (m+1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) * jnp.sign(Nm - m - 1) - 
#         jnp.sqrt((n + 1) * m) * (alpha[0] / alpha[1]) * Ck_hat[n+1 + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) * jnp.sign(Nn - n - 1) + 
#         jnp.sqrt(2 * n) * (u[1] / alpha[0]) * Ck_hat[n-1 + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) - 
#         jnp.sqrt(2 * m) * (u[0] / alpha[1]) * Ck_hat[n + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m))

#     Col  = -nu * col[n, m, p] * Ck[n + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...]
    
#     Diff = -D * k2_grid * Ck[n + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...]
        
#     # ODEs for Hermite-Fourier coefficients.
#     # Closure is achieved by setting to zero coefficients with index out of range.
#     dCk_s_dt = (- (kx_grid * 1j / Lx) * alpha[0] * (
#         jnp.sqrt((n + 1) / 2) * Ck[n+1 + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(Nn - n - 1) +
#         jnp.sqrt(n / 2) * Ck[n-1 + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) +
#         (u[0] / alpha[0]) * Ck[n + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...]
#     ) - (ky_grid * 1j / Ly) * alpha[1] * (
#         jnp.sqrt((m + 1) / 2) * Ck[n + (m+1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(Nm - m - 1) +
#         jnp.sqrt(m / 2) * Ck[n + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) +
#         (u[1] / alpha[1]) * Ck[n + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...]
#     ) - (kz_grid * 1j / Lz) * alpha[2] * (
#         jnp.sqrt((p + 1) / 2) * Ck[n + m * Nn + (p+1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(Np - p - 1) +
#         jnp.sqrt(p / 2) * Ck[n + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p) +
#         (u[2] / alpha[2]) * Ck[n + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...]
#     ) + q * Omega_c * Nx * Ny * Nz * (
#         jnp.fft.fftshift(jnp.fft.fftn((jnp.sqrt(2 * n) / alpha[0]) * Fk_hat[0] * Ck_hat[n-1 + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) +
#                       (jnp.sqrt(2 * m) / alpha[1]) * Fk_hat[1] * Ck_hat[n + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) + 
#                       (jnp.sqrt(2 * p) / alpha[2]) * Fk_hat[2] * Ck_hat[n + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p) +
#                       Fk_hat[3] * Ck_hat_aux_x + Fk_hat[4] * Ck_hat_aux_y + Fk_hat[5] * Ck_hat_aux_z, axes=(-3, -2, -1)), axes=(-3, -2, -1)) 
#     ) + Col + Diff)
    
#     return dCk_s_dt


def _pad_hermite_axes(Ck):
    # pad +1 on both sides for n,m,p only
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
                           Lx, Ly, Lz, nu, D, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, Ns):
    """
    Computes the time derivative of a single Hermite-Fourier coefficient Ck[n, m, p] for species s
    in a Vlasov-Maxwell spectral solver using a Hermite-Fourier basis.

    Returns
    -------
    dCk_s_dt : jax.Array, shape (Nx, Ny, Nz)
        Time derivative of the Hermite-Fourier coefficient Ck[n, m, p] for species s.
    """

    Ck = Ck.reshape(Ns, Np, Nm, Nn, *Ck.shape[-3:])
    C = C.reshape(Ns, Np, Nm, Nn, *C.shape[-3:])
    F = F[:, None, None, None, :, :, :]  # (6,1,1,1,Nx,Ny,Nz) for broadcasting  
    Ny, Nx, Nz = Ck.shape[-3], Ck.shape[-2], Ck.shape[-1]
    
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
    ) + q * Omega_c * Nx * Ny * Nz * (
        jnp.fft.fftshift(jnp.fft.fftn((sqrt_n_minus * jnp.sqrt(2) / a0) * F[0] * shift_multi(C, dn=-1, dm=0, dp=0) +
                      (sqrt_m_minus * jnp.sqrt(2) / a1) * F[1] * shift_multi(C, dn=0, dm=-1, dp=0) + 
                      (sqrt_p_minus * jnp.sqrt(2) / a2) * F[2] * shift_multi(C, dn=0, dm=0, dp=-1) +
                      F[3] * C_aux_x + F[4] * C_aux_y + F[5] * C_aux_z, axes=(-3, -2, -1)), axes=(-3, -2, -1)) 
    ) + Col + Diff)
    
    return dCk_s_dt