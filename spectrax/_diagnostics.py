import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from ._inverse_transform import inverse_HF_transform

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
    Lx = output["Lx"]
    Ly = output["Ly"]
    Lz = output["Lz"]
    
    lambda_D = jnp.sqrt(1 / (2 * (1 / alpha_s[0] ** 2 + 1 / (mi_me * alpha_s[3] ** 2)))) # Debye length.
    k_norm = jnp.sqrt(2) * jnp.pi * alpha_s[0] / Lx # Perturbation wavenumber normalized to the inverse of the Debye length.
    
    kinetic_energy_species1 = (0.5 * alpha_s[0] * alpha_s[1] * alpha_s[2]) * ((0.5 * (alpha_s[0] ** 2 + alpha_s[1] ** 2 + alpha_s[2] ** 2) + 
                                                (u_s[0] ** 2 + u_s[1] ** 2 + u_s[2] ** 2)) * Ck[:, 0, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] + 
                                                jnp.sqrt(2) * (alpha_s[0] * u_s[0] * Ck[:, 1, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Nn - 1) + 
                                                               alpha_s[1] * u_s[1] * Ck[:, Nn, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Nm - 1) +
                                                               alpha_s[2] * u_s[2] * Ck[:, Nn * Nm, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Np - 1)) +
                                          (1 / jnp.sqrt(2)) * (alpha_s[0] ** 2 * Ck[:, 2, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Nn - 1) * jnp.sign(Nn - 2) +
                                                               alpha_s[1] ** 2 * Ck[:, 2 * Nn, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Nm - 1) * jnp.sign(Nm - 2) + 
                                                               alpha_s[2] ** 2 * Ck[:, 2 * Nn * Nm, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Np - 1) * jnp.sign(Np - 2)))

    kinetic_energy_species2 = (0.5 * mi_me *  alpha_s[3] * alpha_s[4] * alpha_s[5]) * ((0.5 * (alpha_s[3] ** 2 + alpha_s[4] ** 2 + alpha_s[5] ** 2) + 
                                                (u_s[3] ** 2 + u_s[4] ** 2 + u_s[5] ** 2)) * Ck[:, Nn * Nm * Np, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] + 
                                                jnp.sqrt(2) * (alpha_s[3] * u_s[3] * Ck[:, 1, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Nn - 1) + 
                                                               alpha_s[4] * u_s[4] * Ck[:, Nn * Nm * Np + Nn, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Nm - 1) +
                                                               alpha_s[5] * u_s[5] * Ck[:, Nn * Nm * Np + Nn * Nm, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Np - 1)) +
                                          (1 / jnp.sqrt(2)) * ((alpha_s[3] ** 2) * Ck[:, Nn * Nm * Np + 2, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Nn - 1) * jnp.sign(Nn - 2) +
                                                               (alpha_s[4] ** 2) * Ck[:, Nn * Nm * Np + 2 * Nn, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Nm - 1) * jnp.sign(Nm - 2) + 
                                                               (alpha_s[5] ** 2) * Ck[:, Nn * Nm * Np + 2 * Nn * Nm, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Np - 1) * jnp.sign(Np - 2)))
                                                    
    electric_field_energy = 0.5 * jnp.sum(jnp.abs(Fk) ** 2, axis=(-4, -3, -2, -1)) * Omega_cs[0] ** 2
    
    total_energy = kinetic_energy_species1 + kinetic_energy_species2 + electric_field_energy
    
    # Nv = 201
    # vx = jnp.linspace(-4 * alpha_s[0], 4 * alpha_s[0], Nv)
    # Vx, Vy, Vz = jnp.meshgrid(vx, jnp.array([0.]), jnp.array([0.]), indexing='xy')
    # i=0
    # f1 = inverse_HF_transform(Ck[:600, i*Nn:(i+1)*Nn, ...], Nn, Nm, Np, 
    #                               (Vx - u_s[3*i]) / alpha_s[3*i], 
    #                               (Vy - u_s[3*i+1]) / alpha_s[3*i+1], 
    #                               (Vz - u_s[3*i+2]) / alpha_s[3*i+2])
    # i=1
    # f2 = inverse_HF_transform(Ck[:600, i*Nn:(i+1)*Nn, ...], Nn, Nm, Np, 
    #                               (Vx - u_s[3*i]) / alpha_s[3*i], 
    #                               (Vy - u_s[3*i+1]) / alpha_s[3*i+1], 
    #                               (Vz - u_s[3*i+2]) / alpha_s[3*i+2])
    
    # x = jnp.linspace(0, Lx, Nx)
    # y = jnp.linspace(0, Ly, Ny)
    # z = jnp.linspace(0, Lz, Nz)
    # vx = jnp.linspace(-4 * alpha_s[0], 4 * alpha_s[0], Nv)
    # vy = jnp.linspace(-4 * alpha_s[1], 4 * alpha_s[1], Nv)
    # vz = jnp.linspace(-4 * alpha_s[2], 4 * alpha_s[2], Nv)
        
    # entropy_1 = (trapezoid(trapezoid(trapezoid(trapezoid(trapezoid(trapezoid(
    #           (jnp.abs(f1) * jnp.log(jnp.abs(f1))), y, axis=-6) , x, axis=-5), z, axis=-4), vy, axis=-3), vx, axis=-2), vz, axis=-1))
    
    # vx = jnp.linspace(-4 * alpha_s[3], 4 * alpha_s[3], Nv)
    # vy = jnp.linspace(-4 * alpha_s[4], 4 * alpha_s[4], Nv)
    # vz = jnp.linspace(-4 * alpha_s[5], 4 * alpha_s[5], Nv)
    # entropy_2 = (trapezoid(trapezoid(trapezoid(trapezoid(trapezoid(trapezoid(
    #           (jnp.abs(f2) * jnp.log(jnp.abs(f2))), y, axis=-6) , x, axis=-5), z, axis=-4), vy, axis=-3), vx, axis=-2), vz, axis=-1))
    

    output.update({
        'lambda_D': lambda_D,
        'k_norm': k_norm,
        'kinetic_energy_species1': kinetic_energy_species1,
        'kinetic_energy_species2': kinetic_energy_species2,
        'kinetic_energy': kinetic_energy_species1 + kinetic_energy_species2,
        'electric_field_energy': electric_field_energy,
        'total_energy': total_energy,
        # 'entropy_1': entropy_1,
        # 'entropy_2': entropy_2,
        # 'entropy': entropy_1 + entropy_2,
    })
    