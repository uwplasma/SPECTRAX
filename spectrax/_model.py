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
    alpha_per_species = alpha_s.reshape(Ns, 3)
    u_per_species = u_s.reshape(Ns, 3)

    # Calculate the number of coefficients and offsets for each species
    c_per_species = jnp.array([Nn[s] * Nm[s] * Np[s] for s in range(Ns)])
    offsets = jnp.cumsum(jnp.concatenate([jnp.array([0]), c_per_species[:-1]]))

    # Reshape Ck into structured Hermite-Fourier coefficients: (Ns, Nn, Nm, Np, Nx, Ny, Nz)
    Ck = Ck.reshape(Ns, Nn, Nm, Np, *Ck.shape[-3:])

    total_J = jnp.zeros((3, *Ck.shape[1:]), dtype=Ck.dtype)

    # Loop over species to calculate current contribution from each
    for s_idx in range(Ns):
        Nn_s, Nm_s, Np_s = Nn[s_idx], Nm[s_idx], Np[s_idx]
        offset_s = offsets[s_idx]

        # Helper function to get a specific coefficient for the current species 's_idx'
        # Local Hermite indices (n,m,p) are relative to the start of that species' modes.
        # Assumes local indexing: n_local is fastest, then m_local, then p_local.
        def get_c_s(n_local, m_local, p_local):
            # Check if the requested local Hermite indices are valid for this species
            is_valid_n = (0 <= n_local) & (n_local < Nn_s)
            is_valid_m = (0 <= m_local) & (m_local < Nm_s)
            is_valid_p = (0 <= p_local) & (p_local < Np_s)
            is_valid_mode = is_valid_n & is_valid_m & is_valid_p

            # Calculate the 1D local index within this species' block of coefficients
            local_idx_1d = n_local + m_local * Nn_s + p_local * Nn_s * Nm_s
            # Calculate the global 1D index into the flat Ck_flat array
            global_idx_1d = offset_s + local_idx_1d

            # Return the coefficient if valid, otherwise return zeros
            return jnp.where(is_valid_mode,
                             Ck[global_idx_1d, ...],
                             jnp.zeros_like(Ck[0, ...]))

        C0_s = get_c_s(0, 0, 0)
        C1_s = get_c_s(1, 0, 0)
        Cm_s = get_c_s(0, 1, 0)
        Cp_s = get_c_s(0, 0, 1)

        alpha_vals_s = alpha_per_species[s_idx]
        u_s = u_per_species[s_idx]
        q_s = qs[s_idx]

        jx_s = q_s * (u_s[0] * C0_s + (alpha_vals_s[0] / jnp.sqrt(2.0)) * C1_s)
        jy_s = q_s * (u_s[1] * C0_s + (alpha_vals_s[1] / jnp.sqrt(2.0)) * Cm_s)
        jz_s = q_s * (u_s[2] * C0_s + (alpha_vals_s[2] / jnp.sqrt(2.0)) * Cp_s)

        J_species_s = jnp.stack([jx_s, jy_s, jz_s], axis=0)  # Shape (3, Ny, Nx, Nz)
        total_J += J_species_s

    return total_J

@jit
def collision(Nn_s, Nm_s, Np_s, n, m, p):
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
    N = jnp.array([Nn_s, Nm_s, Np_s], dtype=jnp.float32)
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

    num_coeffs_per_species = jnp.array([Nn[s_idx] * Nm[s_idx] * Np[s_idx] for s_idx in range(len(Nn))])

    cum_ends = jnp.cumsum(num_coeffs_per_species)

    s = jnp.sum(index >= cum_ends).astype(jnp.int32)
    offsets = jnp.cumsum(jnp.concatenate([jnp.array([0]), num_coeffs_per_species[:-1]]))

    index_within_species = index - offsets[s]
    n = index_within_species % Nn[s]
    m_temp = index_within_species // Nn[s] 
    m = m_temp % Nm[s] 
    p = m_temp // Nm[s]

    u = dynamic_slice(u_s, (s * 3,), (3,))
    alpha = dynamic_slice(alpha_s, (s * 3,), (3,)) 
    q = qs[s]
    Omega_c = Omega_cs[s]

    # Define terms to be used in ODEs below.
    Ck_aux_x = (jnp.sqrt(m * p) * (alpha[2] / alpha[1] - alpha[1] / alpha[2]) * Ck[n + (m-1) * Nn[s] + (p-1) * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(m) * jnp.sign(p) +
        jnp.sqrt(m * (p + 1)) * (alpha[2] / alpha[1]) * Ck[n + (m-1) * Nn[s] + (p+1) * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(m) * jnp.sign(Np[s] - p - 1) -
        jnp.sqrt((m + 1) * p) * (alpha[1] / alpha[2]) * Ck[n + (m+1) * Nn[s] + (p-1) * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(p) * jnp.sign(Nm[s] - m - 1) +
        jnp.sqrt(2 * m) * (u[2] / alpha[1]) * Ck[n + (m-1) * Nn[s] + p * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(m) -
        jnp.sqrt(2 * p) * (u[1] / alpha[2]) * Ck[n + m * Nn[s] + (p-1) * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(p))

    Ck_aux_y = (jnp.sqrt(n * p) * (alpha[0] / alpha[2] - alpha[2] / alpha[0]) * Ck[n-1 + m * Nn[s] + (p-1) * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(n) * jnp.sign(p) +
        jnp.sqrt((n + 1) * p) * (alpha[0] / alpha[2]) * Ck[n+1 + m * Nn[s] + (p-1) * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(p) * jnp.sign(Nn[s] - n - 1) -
        jnp.sqrt(n * (p + 1)) * (alpha[2] / alpha[0]) * Ck[n-1 + m * Nn[s] + (p+1) * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(n) * jnp.sign(Np[s] - p - 1) +
        jnp.sqrt(2 * p) * (u[0] / alpha[2]) * Ck[n + m * Nn[s] + (p-1) * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(p) -
        jnp.sqrt(2 * n) * (u[2] / alpha[0]) * Ck[n-1 + m * Nn[s] + p * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(n))

    Ck_aux_z = (jnp.sqrt(n * m) * (alpha[1] / alpha[0] - alpha[0] / alpha[1]) * Ck[n-1 + (m-1) * Nn[s] + p * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(n) * jnp.sign(m) +
        jnp.sqrt(n * (m + 1)) * (alpha[1] / alpha[0]) * Ck[n-1 + (m+1) * Nn[s] + p * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(n) * jnp.sign(Nm[s] - m - 1) -
        jnp.sqrt((n + 1) * m) * (alpha[0] / alpha[1]) * Ck[n+1 + (m-1) * Nn[s] + p * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(m) * jnp.sign(Nn[s] - n - 1) +
        jnp.sqrt(2 * n) * (u[1] / alpha[0]) * Ck[n-1 + m * Nn[s] + p * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(n) -
        jnp.sqrt(2 * m) * (u[0] / alpha[1]) * Ck[n + (m-1) * Nn[s] + p * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(m))

    Col  = -nu * collision(Nn[s], Nm[s], Np[s], n, m, p)    * Ck[n + m * Nn[s] + p * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...]

    Diff = diffusion(kx_grid, ky_grid, kz_grid, D) * Ck[n + m * Nn[s] + p * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...]

    # ODEs for Hermite-Fourier coefficients.
    # Closure is achieved by setting to zero coefficients with index out of range.
    dCk_s_dt = (- (kx_grid * 1j / Lx) * alpha[0] * (
        jnp.sqrt((n + 1) / 2) * Ck[n+1 + m * Nn[s] + p * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(Nn[s] - n - 1) +
        jnp.sqrt(n / 2) * Ck[n-1 + m * Nn[s] + p * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(n) +
        (u[0] / alpha[0]) * Ck[n + m * Nn[s] + p * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...]
    ) - (ky_grid * 1j / Ly) * alpha[1] * (
        jnp.sqrt((m + 1) / 2) * Ck[n + (m+1) * Nn[s] + p * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(Nm[s] - m - 1) +
        jnp.sqrt(m / 2) * Ck[n + (m-1) * Nn[s] + p * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(m) +
        (u[1] / alpha[1]) * Ck[n + m * Nn[s] + p * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...]
    ) - (kz_grid * 1j / Lz) * alpha[2] * (
        jnp.sqrt((p + 1) / 2) * Ck[n + m * Nn[s] + (p+1) * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(Np[s] - p - 1) +
        jnp.sqrt(p / 2) * Ck[n + m * Nn[s] + (p-1) * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(p) +
        (u[2] / alpha[2]) * Ck[n + m * Nn[s] + p * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...]
    ) + q * Omega_c * (
        (jnp.sqrt(2 * n) / alpha[0]) * convolve(Fk[0, ...], Ck[n-1 + m * Nn[s] + p * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(n), mode='same') +
        (jnp.sqrt(2 * m) / alpha[1]) * convolve(Fk[1, ...], Ck[n + (m-1) * Nn[s] + p * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(m), mode='same') +
        (jnp.sqrt(2 * p) / alpha[2]) * convolve(Fk[2, ...], Ck[n + m * Nn[s] + (p-1) * Nn[s] * Nm[s] + s * Nn[s] * Nm[s] * Np[s], ...] * jnp.sign(p), mode='same')
    ) + q * Omega_c * (
        convolve(Fk[3, ...], Ck_aux_x, mode='same') +
        convolve(Fk[4, ...], Ck_aux_y, mode='same') +
        convolve(Fk[5, ...], Ck_aux_z, mode='same')
    ) + Col + Diff)

    return dCk_s_dt
