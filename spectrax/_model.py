import jax.numpy as jnp
from jax import vmap, jit
from functools import partial
from jax.lax import dynamic_slice
from jax.scipy.signal import convolve

__all__ = ['plasma_current', 'collision', 'Hermite_Fourier_system', 'diffusion']

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
    # Reshape alpha and velocity
    alpha = alpha_s.reshape(Ns, 3)
    u = u_s.reshape(Ns, 3)

    ncps = jnp.array([Nn[s] * Nm[s] * Np[s] for s in range(Ns)])
    offsets = jnp.cumsum(jnp.concatenate([jnp.array([0]), ncps[:-1]]))

    zero_C_slice = jnp.zeros(Ck.shape[1:], dtype=Ck.dtype)

    def calculate_j_s(Nn_s, Nm_s, Np_s, offset_s,
                       alpha_vals_s, u_vals_s, q_s,
                       Ck_full, zero_slice_template):

        # C0: (n=0,m=0,p=0)
        idx_C0 = offset_s
        valid_C0 = (Nn_s > 0) & (Nm_s > 0) & (Np_s > 0)
        safe_idx_C0 = jnp.where(valid_C0, idx_C0, 0)
        val_C0 = Ck_full[safe_idx_C0, ...]
        C0 = jnp.where(valid_C0, val_C0, zero_slice_template)

        # C1: (n=1,m=0,p=0) - for jx contribution
        idx_C1 = offset_s + 1
        valid_C1 = (Nn_s > 1) & (Nm_s > 0) & (Np_s > 0)
        safe_idx_C1 = jnp.where(valid_C1, idx_C1, 0)
        val_C1 = Ck_full[safe_idx_C1, ...]
        C1 = jnp.where(valid_C1, val_C1, zero_slice_template)

        # Cn: (n=0,m=1,p=0) - for jy contribution
        idx_Cn = offset_s + Nn_s
        valid_Cn = (Nn_s > 0) & (Nm_s > 1) & (Np_s > 0)
        safe_idx_Cn = jnp.where(valid_Cn, idx_Cn, 0)
        val_Cn = Ck_full[safe_idx_Cn, ...]
        Cn = jnp.where(valid_Cn, val_Cn, zero_slice_template)

        # Cnm: (n=0,m=0,p=1) - for jz contribution
        idx_Cnm = offset_s + (Nn_s * Nm_s)
        valid_Cnm = (Nn_s > 0) & (Nm_s > 0) & (Np_s > 1)
        safe_idx_Cnm = jnp.where(valid_Cnm, idx_Cnm, 0)
        val_Cnm = Ck_full[safe_idx_Cnm, ...]
        Cnm = jnp.where(valid_Cnm, val_Cnm, zero_slice_template)

        a0_s, a1_s, a2_s = alpha_vals_s[0], alpha_vals_s[1], alpha_vals_s[2]
        u0_s, u1_s, u2_s = u_vals_s[0], u_vals_s[1], u_vals_s[2]

        # Base terms
        jx_base = u0_s * C0 + (a0_s / jnp.sqrt(2.0)) * C1
        jy_base = u1_s * C0 + (a1_s / jnp.sqrt(2.0)) * Cn
        jz_base = u2_s * C0 + (a2_s / jnp.sqrt(2.0)) * Cnm

        # 'pre' factor for this species (q * a0 * a1 * a2)
        pre = q_s * a0_s * a1_s * a2_s

        # Apply 'pre' factor
        jx = jx_base * pre
        jy = jy_base * pre
        jz = jz_base * pre

        return jnp.stack([jx, jy, jz], axis=0)

    J_species_stacked = vmap(
        calculate_j_s,
        in_axes=(0, 0, 0, 0, 0, 0, 0, None, None),
        out_axes=0
    )(jnp.asarray(Nn), jnp.asarray(Nm), jnp.asarray(Np), offsets,
      alpha, u, qs,
      Ck, zero_C_slice)

    total_J = jnp.sum(J_species_stacked, axis=0)

    return total_J

@jit
def collision(Nn, Nm, Np, n, m, p):
    """
    Computes the scalar collision term for the Vlasov-Maxwell system using 
    a symmetric form over the Hermite-Fourier indices (n, m, p). 
    The contribution from each index is included only if the corresponding 
    mode count (Nn, Nm, Np) exceeds 3. The result is a differentiable sum 
    over valid contributions.
    Args:
        Nn (int): Number of grid points in the n direction.
        Nm (int): Number of grid points in the m direction.
        Np (int): Number of grid points in the p direction.
        n (int): Index in the n direction.
        m (int): Index in the m direction.
        p (int): Index in the p direction.
    Returns:
        jnp.ndarray: Scalar collision term.
    """
    N = jnp.array([Nn, Nm, Np], dtype=jnp.float32)
    idx = jnp.array([n, m, p], dtype=jnp.float32)
    # Avoid division by zero by masking invalid terms
    def safe_term(Nj, nj):
        term = nj * (nj - 1) * (nj - 2)
        denom = (Nj - 1) * (Nj - 2) * (Nj - 3)
        return jnp.where(Nj > 3, term / denom, 0.0)
    terms = vmap(safe_term)(N, idx)
    return jnp.sum(terms)

@jit
def diffusion(kx, ky, kz, D, exponent=2):
    """
    Computes the diffusion term for the Vlasov-Maxwell system.

    Parameters
    ----------
    kx, ky, kz : jnp.ndarray
        Wavevector components in the x, y, and z directions.
    D : float
        Diffusion coefficient.

    Returns
    -------
    jnp.ndarray
        Diffusion term proportional to kx**2 + ky**2 + kz**2.
    """
    k_squared = kx**exponent + ky**exponent + kz**exponent
    return -D * k_squared

@partial(jit, static_argnames=['Nn', 'Nm', 'Np'])
def Hermite_Fourier_system(Ck, Fk, kx_grid, ky_grid, kz_grid, Lx, Ly, Lz, nu, D, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, index):
    """
    Computes the time derivative of a single Hermite-Fourier coefficient Ck[n, m, p] for species s
    in a Vlasov-Maxwell spectral solver using a Hermite-Fourier basis.

    Returns
    -------
    dCk_s_dt : jax.Array, shape (Nx, Ny, Nz)
        Time derivative of the Hermite-Fourier coefficient Ck[n, m, p] for species s.
    """

    Nn_jnp = jnp.asarray(Nn)
    Nm_jnp = jnp.asarray(Nm)
    Np_jnp = jnp.asarray(Np)
    
    num_coeffs_per_species = jnp.array([Nn[s_idx] * Nm[s_idx] * Np[s_idx] for s_idx in range(len(Nn))])

    cum_ends = jnp.cumsum(num_coeffs_per_species)

    s = jnp.sum(index >= cum_ends).astype(jnp.int32)
    offsets = jnp.cumsum(jnp.concatenate([jnp.array([0]), num_coeffs_per_species[:-1]]))
    index_within_species = index - offsets[s]

    n = index_within_species % Nn_jnp[s] 
    m_temp = index_within_species // Nn_jnp[s] 
    m = m_temp % Nm_jnp[s] 
    p = m_temp // Np_jnp[s] 

    u = dynamic_slice(u_s, (s * 3,), (3,))
    alpha = dynamic_slice(alpha_s, (s * 3,), (3,))
    q = qs[s]
    Omega_c = Omega_cs[s]

    Ck_aux_x = (jnp.sqrt(m * p) * (alpha[2] / alpha[1] - alpha[1] / alpha[2]) * Ck[n + (m-1) * Nn_jnp[s] + (p-1) * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(m) * jnp.sign(p) +
        jnp.sqrt(m * (p + 1)) * (alpha[2] / alpha[1]) * Ck[n + (m-1) * Nn_jnp[s] + (p+1) * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(m) * jnp.sign(Np_jnp[s] - p - 1) -
        jnp.sqrt((m + 1) * p) * (alpha[1] / alpha[2]) * Ck[n + (m+1) * Nn_jnp[s] + (p-1) * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(p) * jnp.sign(Nm_jnp[s] - m - 1) +
        jnp.sqrt(2 * m) * (u[2] / alpha[1]) * Ck[n + (m-1) * Nn_jnp[s] + p * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(m) -
        jnp.sqrt(2 * p) * (u[1] / alpha[2]) * Ck[n + m * Nn_jnp[s] + (p-1) * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(p))

    Ck_aux_y = (jnp.sqrt(n * p) * (alpha[0] / alpha[2] - alpha[2] / alpha[0]) * Ck[n-1 + m * Nn_jnp[s] + (p-1) * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(n) * jnp.sign(p) +
        jnp.sqrt((n + 1) * p) * (alpha[0] / alpha[2]) * Ck[n+1 + m * Nn_jnp[s] + (p-1) * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(p) * jnp.sign(Nn_jnp[s] - n - 1) -
        jnp.sqrt(n * (p + 1)) * (alpha[2] / alpha[0]) * Ck[n-1 + m * Nn_jnp[s] + (p+1) * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(n) * jnp.sign(Np_jnp[s] - p - 1) +
        jnp.sqrt(2 * p) * (u[0] / alpha[2]) * Ck[n + m * Nn_jnp[s] + (p-1) * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(p) -
        jnp.sqrt(2 * n) * (u[2] / alpha[0]) * Ck[n-1 + m * Nn_jnp[s] + p * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(n))

    Ck_aux_z = (jnp.sqrt(n * m) * (alpha[1] / alpha[0] - alpha[0] / alpha[1]) * Ck[n-1 + (m-1) * Nn_jnp[s] + p * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(n) * jnp.sign(m) +
        jnp.sqrt(n * (m + 1)) * (alpha[1] / alpha[0]) * Ck[n-1 + (m+1) * Nn_jnp[s] + p * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(n) * jnp.sign(Nm_jnp[s] - m - 1) -
        jnp.sqrt((n + 1) * m) * (alpha[0] / alpha[1]) * Ck[n+1 + (m-1) * Nn_jnp[s] + p * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(m) * jnp.sign(Nn_jnp[s] - n - 1) +
        jnp.sqrt(2 * n) * (u[1] / alpha[0]) * Ck[n-1 + m * Nn_jnp[s] + p * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(n) -
        jnp.sqrt(2 * m) * (u[0] / alpha[1]) * Ck[n + (m-1) * Nn_jnp[s] + p * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(m))

    Col  = -nu * collision(Nn_jnp[s], Nm_jnp[s], Np_jnp[s], n, m, p)    * Ck[n + m * Nn_jnp[s] + p * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...]

    Diff = diffusion(kx_grid, ky_grid, kz_grid, D) * Ck[n + m * Nn_jnp[s] + p * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...]

    # ODEs for Hermite-Fourier coefficients.
    # Closure is achieved by setting to zero coefficients with index out of range.
    dCk_s_dt = (- (kx_grid * 1j / Lx) * alpha[0] * (
        jnp.sqrt((n + 1) / 2) * Ck[n+1 + m * Nn_jnp[s] + p * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(Nn_jnp[s] - n - 1) +
        jnp.sqrt(n / 2) * Ck[n-1 + m * Nn_jnp[s] + p * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(n) +
        (u[0] / alpha[0]) * Ck[n + m * Nn_jnp[s] + p * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...]
    ) - (ky_grid * 1j / Ly) * alpha[1] * (
        jnp.sqrt((m + 1) / 2) * Ck[n + (m+1) * Nn_jnp[s] + p * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(Nm_jnp[s] - m - 1) +
        jnp.sqrt(m / 2) * Ck[n + (m-1) * Nn_jnp[s] + p * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(m) +
        (u[1] / alpha[1]) * Ck[n + m * Nn_jnp[s] + p * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...]
    ) - (kz_grid * 1j / Lz) * alpha[2] * (
        jnp.sqrt((p + 1) / 2) * Ck[n + m * Nn_jnp[s] + (p+1) * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(Np_jnp[s] - p - 1) +
        jnp.sqrt(p / 2) * Ck[n + m * Nn_jnp[s] + (p-1) * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(p) +
        (u[2] / alpha[2]) * Ck[n + m * Nn_jnp[s] + p * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...]
    ) + q * Omega_c * (
        (jnp.sqrt(2 * n) / alpha[0]) * convolve(Fk[0, ...], Ck[n-1 + m * Nn_jnp[s] + p * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(n), mode='same') +
        (jnp.sqrt(2 * m) / alpha[1]) * convolve(Fk[1, ...], Ck[n + (m-1) * Nn_jnp[s] + p * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(m), mode='same') +
        (jnp.sqrt(2 * p) / alpha[2]) * convolve(Fk[2, ...], Ck[n + m * Nn_jnp[s] + (p-1) * Nn_jnp[s] * Nm_jnp[s] + s * Nn_jnp[s] * Nm_jnp[s] * Np_jnp[s], ...] * jnp.sign(p), mode='same')
    ) + q * Omega_c * (
        convolve(Fk[3, ...], Ck_aux_x, mode='same') +
        convolve(Fk[4, ...], Ck_aux_y, mode='same') +
        convolve(Fk[5, ...], Ck_aux_z, mode='same')
    ) + Col + Diff)

    return dCk_s_dt
