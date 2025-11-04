import jax.numpy as jnp
from jax import vmap, jit
from functools import partial
from jax.lax import dynamic_slice
from jax.scipy.signal import convolve

__all__ = ['plasma_current', 'Hermite_Fourier_system']


@partial(jit, static_argnames=['Nn', 'Nm', 'Np', 'Ns'])
def plasma_current(qs, alpha_s, u_s, Ck, Nn, Nm, Np, Ns):
    """
    Compute the spectral Ampère-Maxwell current from Hermite-Fourier coefficients.

    Parameters
    ----------
    qs : jnp.ndarray, shape (Ns,)
        Charges of the species.
    alpha_s : jnp.ndarray, shape (3 * Ns,)
        Scaling factors for each species.
    u_s : jnp.ndarray, shape (3 * Ns,)
        Velocity components for each species, packed like alpha_s.
    Ck : jnp.ndarray, shape (Ns * Np * Nm * Nn, Ny, Nx, Nz)
        Fourier-space Hermite coefficients for all species stacked along the first axis.
    Nn, Nm, Np : int
        Number of Hermite modes in x, y, and z respectively.
    Ns : int
        Number of species.

    Returns
    -------
    jnp.ndarray, shape (3, Ny, Nx, Nz)
        The total Ampère-Maxwell current components `(Jx, Jy, Jz)`.
    """
    # Reshape Ck into structured Hermite-Fourier coefficients: (Ns, Np, Nm, Nn, Ny, Nx, Nz)
    Ck = Ck.reshape(Ns, Np, Nm, Nn, *Ck.shape[-3:])
    
    # Reshape alpha and velocity
    alpha = alpha_s.reshape(Ns, 3)
    u = u_s.reshape(Ns, 3)

    # Grab the modes we need (0,1,1,1) for jx, jy, jz contributions
    C0 = Ck[:, 0, 0, 0]  # shape: (Ns, Ny, Nx, Nz)
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

    # Final current per species: shape (3, Ns, Ny, Nx, Nz)
    J_species = (term1 + term2) * pre[None, :, None, None, None]

    # Sum over species → shape: (3, Ny, Nx, Nz)
    return jnp.sum(J_species, axis=1)

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
                           Lx, Ly, Lz, nu, D, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, Ns, mask23):
    """
    Evaluate the right-hand side of the coupled Hermite-Fourier moment equations.

    Parameters
    ----------
    Ck : jnp.ndarray
        Spectral Hermite coefficients with shape `(Ns * Np * Nm * Nn, Ny, Nx, Nz)`.
    C : jnp.ndarray
        Configuration-space coefficients, typically the inverse FFT of `Ck`, with
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
    nu : float
        Collision frequency.
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

    Returns
    -------
    jnp.ndarray
        Time derivative `dCk/dt` with shape `(Ns, Np, Nm, Nn, Ny, Nx, Nz)`.
    """

    Ck = Ck.reshape(Ns, Np, Nm, Nn, *Ck.shape[-3:])
    C = C.reshape(Ns, Np, Nm, Nn, *C.shape[-3:])
    F = F[:, None, None, None, None, :, :, :]  # (6,1,1,1,Nx,Ny,Nz) for broadcasting  
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
    ) + q * Omega_c * (
        jnp.fft.fftshift(jnp.fft.fftn((sqrt_n_minus * jnp.sqrt(2) / a0) * F[0] * shift_multi(C, dn=-1, dm=0, dp=0) +
                      (sqrt_m_minus * jnp.sqrt(2) / a1) * F[1] * shift_multi(C, dn=0, dm=-1, dp=0) + 
                      (sqrt_p_minus * jnp.sqrt(2) / a2) * F[2] * shift_multi(C, dn=0, dm=0, dp=-1) +
                      F[3] * C_aux_x + F[4] * C_aux_y + F[5] * C_aux_z, axes=(-3, -2, -1), norm="forward"), axes=(-3, -2, -1)) * mask23
    ) + Col + Diff)
    
    return dCk_s_dt
