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
    Ny = output["Ny"]
    Nz = output["Nz"]
    Omega_cs = output["Omega_cs"]
    Nn = output["Nn"]
    Nm = output["Nm"]
    Np = output["Np"]
    
    lambda_D = jnp.sqrt(1 / (2 * (1 / alpha_s[0] ** 2 + 1 / (mi_me * alpha_s[3] ** 2)))) # Debye length.
    k_norm = jnp.sqrt(2) * jnp.pi * alpha_s[0] / Lx # Perturbation wavenumber normalized to the inverse of the Debye length.
    
    k0x = 0
    k0y = 0
    k0z = 0

    kinetic_energy_species1 = (0.5 * alpha_s[0] * alpha_s[1] * alpha_s[2]) * ((0.5 * (alpha_s[0] ** 2 + alpha_s[1] ** 2 + alpha_s[2] ** 2) + 
                                                (u_s[0] ** 2 + u_s[1] ** 2 + u_s[2] ** 2)) * Ck[:, 0, k0y, k0x, k0z] + 
                                                jnp.sqrt(2) * (alpha_s[0] * u_s[0] * Ck[:, 1, k0y, k0x, k0z] * jnp.sign(Nn - 1) + 
                                                               alpha_s[1] * u_s[1] * Ck[:, Nn, k0y, k0x, k0z] * jnp.sign(Nm - 1) +
                                                               alpha_s[2] * u_s[2] * Ck[:, Nn * Nm, k0y, k0x, k0z] * jnp.sign(Np - 1)) +
                                          (1 / jnp.sqrt(2)) * (alpha_s[0] ** 2 * Ck[:, 2, k0y, k0x, k0z] * jnp.sign(Nn - 1) * jnp.sign(Nn - 2) +
                                                               alpha_s[1] ** 2 * Ck[:, 2 * Nn, k0y, k0x, k0z] * jnp.sign(Nm - 1) * jnp.sign(Nm - 2) + 
                                                               alpha_s[2] ** 2 * Ck[:, 2 * Nn * Nm, k0y, k0x, k0z] * jnp.sign(Np - 1) * jnp.sign(Np - 2)))

    kinetic_energy_species2 = (0.5 * mi_me *  alpha_s[3] * alpha_s[4] * alpha_s[5]) * ((0.5 * (alpha_s[3] ** 2 + alpha_s[4] ** 2 + alpha_s[5] ** 2) + 
                                                (u_s[3] ** 2 + u_s[4] ** 2 + u_s[5] ** 2)) * Ck[:, Nn * Nm * Np, k0y, k0x, k0z] + 
                                                jnp.sqrt(2) * (alpha_s[3] * u_s[3] * Ck[:, Nn * Nm * Np + 1, k0y, k0x, k0z] * jnp.sign(Nn - 1) + 
                                                               alpha_s[4] * u_s[4] * Ck[:, Nn * Nm * Np + Nn, k0y, k0x, k0z] * jnp.sign(Nm - 1) +
                                                               alpha_s[5] * u_s[5] * Ck[:, Nn * Nm * Np + Nn * Nm, k0y, k0x, k0z] * jnp.sign(Np - 1)) +
                                          (1 / jnp.sqrt(2)) * (alpha_s[3] ** 2 * Ck[:, Nn * Nm * Np + 2, k0y, k0x, k0z] * jnp.sign(Nn - 1) * jnp.sign(Nn - 2) +
                                                               alpha_s[4] ** 2 * Ck[:, Nn * Nm * Np + 2 * Nn, k0y, k0x, k0z] * jnp.sign(Nm - 1) * jnp.sign(Nm - 2) + 
                                                               alpha_s[5] ** 2 * Ck[:, Nn * Nm * Np + 2 * Nn * Nm, k0y, k0x, k0z] * jnp.sign(Np - 1) * jnp.sign(Np - 2)))
                                                    
    EM_energy = 0.5 * jnp.sum(jnp.abs(Fk) ** 2, axis=(-4, -3, -2, -1)) * Omega_cs[0] ** 2
    
    total_energy = kinetic_energy_species1 + kinetic_energy_species2 + EM_energy

    output.update({
        'lambda_D': lambda_D,
        'k_norm': k_norm,
        'kinetic_energy_species1': kinetic_energy_species1,
        'kinetic_energy_species2': kinetic_energy_species2,
        'kinetic_energy': kinetic_energy_species1 + kinetic_energy_species2,
        'EM_energy': EM_energy,
        'total_energy': total_energy,
    })
    