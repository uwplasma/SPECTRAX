import jax.numpy as jnp

__all__ = ['diagnostics']

def diagnostics(output):
    alpha_s = output["alpha_s"]
    mi_me = output["mi_me"]
    u_s = output["u_s"]
    Fk = output["Fk"]
    Ck = output["Ck"]
    Lx = output["Lx"]
    Nx = output["Nx"]
    Ny = output["Nx"]
    Nz = output["Nx"]
    Omega_cs = output["Omega_cs"]
    Nn = output["Nn"]
    Nm = output["Nm"]
    Np = output["Np"]
    Ns = output["Ns"]

    ncps = jnp.array([Nn[s] * Nm[s] * Np[s] for s in range(Ns)])
    offsets = jnp.cumsum(jnp.concatenate([jnp.array([0]), ncps[:-1]]))
    
    lambda_D = jnp.sqrt(1 / (2 * (1 / alpha_s[0] ** 2 + 1 / (mi_me * alpha_s[3] ** 2)))) # Debye length.
    k_norm = jnp.sqrt(2) * jnp.pi * alpha_s[0] / Lx # Perturbation wavenumber normalized to the inverse of the Debye length.
    
    half_nx = jnp.array((Nx-1)/2, int)
    half_ny = jnp.array((Ny-1)/2, int)
    half_nz = jnp.array((Nz-1)/2, int)

    def S_time(s, n, m, p):

        Nn_s = Nn[s]
        Nm_s = Nm[s]
        Np_s = Np[s]

        current_offset_s = offsets[s]

        is_valid_mode = (0 <= n < Nn_s) and \
                        (0 <= m < Nm_s) and \
                        (0 <= p < Np_s)

        idx_f = current_offset_s + n + m * Nn_s + p * Nn_s * Nm_s

        time_0 = jnp.zeros(Ck.shape[0], dtype=Ck.dtype)

        return jnp.where(is_valid_mode,Ck[:, idx_f, half_ny, half_nx, half_nz], time_0)

    # --- Kinetic Energy Species 1 (s=0) ---
    s0 = 0
    # Coefficients for species 0 
    C000_s0 = S_time(s0, 0, 0, 0)
    C100_s0 = S_time(s0, 1, 0, 0)  # n=1 mode
    C010_s0 = S_time(s0, 0, 1, 0)  # m=1 mode 
    C001_s0 = S_time(s0, 0, 0, 1)  # p=1 mode 
    C200_s0 = S_time(s0, 2, 0, 0)  # n=2 mode
    C020_s0 = S_time(s0, 0, 2, 0)  # m=2 mode
    C002_s0 = S_time(s0, 0, 0, 2)  # p=2 mode

    # Parameters for species 0
    alpha_s0_vals = alpha_s[s0 * 3: (s0 + 1) * 3] 
    u_s0_vals = u_s[s0 * 3: (s0 + 1) * 3] 

    kinetic_energy_species1 = (0.5 * alpha_s0_vals[0] * alpha_s0_vals[1] * alpha_s0_vals[2]) * (
            (0.5 * (alpha_s0_vals[0] ** 2 + alpha_s0_vals[1] ** 2 + alpha_s0_vals[2] ** 2) +
             (u_s0_vals[0] ** 2 + u_s0_vals[1] ** 2 + u_s0_vals[2] ** 2)) * C000_s0 +
            jnp.sqrt(2.0) * (alpha_s0_vals[0] * u_s0_vals[0] * C100_s0 +
                             alpha_s0_vals[1] * u_s0_vals[1] * C010_s0 +
                             alpha_s0_vals[2] * u_s0_vals[2] * C001_s0) +
            (1.0 / jnp.sqrt(2.0)) * (alpha_s0_vals[0] ** 2 * C200_s0 +
                                     alpha_s0_vals[1] ** 2 * C020_s0 +
                                     alpha_s0_vals[2] ** 2 * C002_s0)
    )

    # --- Kinetic Energy Species 2 (s=1) ---
    s1 = 1
    # Coefficients for species 1
    C000_s1 = S_time(s1, 0, 0, 0)
    C100_s1 = S_time(s1, 1, 0, 0)
    C010_s1 = S_time(s1, 0, 1, 0)
    C001_s1 = S_time(s1, 0, 0, 1)
    C200_s1 = S_time(s1, 2, 0, 0)
    C020_s1 = S_time(s1, 0, 2, 0)
    C002_s1 = S_time(s1, 0, 0, 2)

    # Parameters for species 1
    alpha_s1_vals = alpha_s[s1 * 3: (s1 + 1) * 3]
    u_s1_vals = u_s[s1 * 3: (s1 + 1) * 3]

    kinetic_energy_species2 = (0.5 * mi_me * alpha_s1_vals[0] * alpha_s1_vals[1] * alpha_s1_vals[2]) * (
            (0.5 * (alpha_s1_vals[0] ** 2 + alpha_s1_vals[1] ** 2 + alpha_s1_vals[2] ** 2) +
             (u_s1_vals[0] ** 2 + u_s1_vals[1] ** 2 + u_s1_vals[2] ** 2)) * C000_s1 +
            jnp.sqrt(2.0) * (alpha_s1_vals[0] * u_s1_vals[0] * C100_s1 +
                             alpha_s1_vals[1] * u_s1_vals[1] * C010_s1 +
                             alpha_s1_vals[2] * u_s1_vals[2] * C001_s1) +
            (1.0 / jnp.sqrt(2.0)) * (alpha_s1_vals[0] ** 2 * C200_s1 +
                                     alpha_s1_vals[1] ** 2 * C020_s1 +
                                     alpha_s1_vals[2] ** 2 * C002_s1)
    )

    electric_field_energy = 0.5 * jnp.sum(jnp.abs(Fk) ** 2, axis=(-4, -3, -2, -1)) * Omega_cs[0] ** 2
    total_energy = kinetic_energy_species1 + kinetic_energy_species2 + electric_field_energy

    output.update({
        'lambda_D': lambda_D,
        'k_norm': k_norm,
        'kinetic_energy_species1': kinetic_energy_species1,
        'kinetic_energy_species2': kinetic_energy_species2,
        'kinetic_energy': kinetic_energy_species1 + kinetic_energy_species2,
        'electric_field_energy': electric_field_energy,
        'total_energy': total_energy,
    })
