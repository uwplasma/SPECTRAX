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
    Omega_cs = output["Omega_cs"]
    Nn = output["Nn"]
    Nm = output["Nm"]
    Np = output["Np"]
    
    lambda_D = jnp.sqrt(1 / (2 * (1 / alpha_s[0] ** 2 + 1 / (mi_me * alpha_s[3] ** 2)))) # Debye length.
    k_norm = jnp.sqrt(2) * jnp.pi * alpha_s[0] / Lx # Perturbation wavenumber normalized to the inverse of the Debye length.

    kinetic_energy_species1 = (0.5 * alpha_s[0] * alpha_s[1] * alpha_s[2]) * ((0.5 * (alpha_s[0] ** 2 + alpha_s[1] ** 2 + alpha_s[2] ** 2) + 
                                                (u_s[0] ** 2 + u_s[1] ** 2 + u_s[2] ** 2)) * Ck[:, 0, 0, int((Nx-1)/2), 0] + 
                                        jnp.sqrt(2) * alpha_s[0] * u_s[0] * Ck[:, 1, 0, int((Nx-1)/2), 0] + 
                                        (1 / jnp.sqrt(2)) * (alpha_s[0] ** 2) * Ck[:, 2, 0, int((Nx-1)/2), 0])

    kinetic_energy_species2 = (0.5 * alpha_s[3] * alpha_s[4] * alpha_s[5]) * ((0.5 * (alpha_s[3] ** 2 + alpha_s[4] ** 2 + alpha_s[5] ** 2) + 
                                                (u_s[3] ** 2 + u_s[4] ** 2 + u_s[5] ** 2)) * Ck[:, Nn, 0, int((Nx-1)/2), 0] + 
                                        jnp.sqrt(2) * alpha_s[3] * u_s[3] * Ck[:, Nn + 1, 0, int((Nx-1)/2), 0] + 
                                        (1 / jnp.sqrt(2)) * (alpha_s[3] ** 2) * Ck[:, Nn + 2, 0, int((Nx-1)/2), 0])
                                                    
    electric_field_energy = 0.5 * jnp.sum(jnp.abs(Fk[:, 0, 0, :, 0]) ** 2, axis=-1) * Omega_cs[0] ** 2
    
    total_energy = (kinetic_energy_species1 + kinetic_energy_species2) + electric_field_energy

    output.update({
        'lambda_D': lambda_D,
        'k_norm': k_norm,
        'kinetic_energy_species1': kinetic_energy_species1,
        'kinetic_energy_species2': kinetic_energy_species2,
        'kinetic_energy': kinetic_energy_species1 + kinetic_energy_species2,
        'electric_field_energy': electric_field_energy,
        'total_energy': total_energy,
    })
    