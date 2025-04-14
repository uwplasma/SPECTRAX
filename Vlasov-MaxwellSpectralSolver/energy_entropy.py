import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid



def compute_energy(Ck, Fk, Omega_ce, mi_me, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nvx, Nvy, Nvz, Nn, Nm, Np):
    
    plasma_energy_1 = (0.5 * alpha_s[0] * alpha_s[1] * alpha_s[2]) * ((0.5 * (alpha_s[0] ** 2 + alpha_s[1] ** 2 + alpha_s[2] ** 2) + 
                                                (u_s[0] ** 2 + u_s[1] ** 2 + u_s[2] ** 2)) * Ck[:, 0, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] + 
                                                jnp.sqrt(2) * (alpha_s[0] * u_s[0] * Ck[:, 1, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Nn - 1) + 
                                                               alpha_s[1] * u_s[1] * Ck[:, Nn, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Nm - 1) +
                                                               alpha_s[2] * u_s[2] * Ck[:, Nn * Nm, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Np - 1)) +
                                          (1 / jnp.sqrt(2)) * (alpha_s[0] ** 2 * Ck[:, 2, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Nn - 1) * jnp.sign(Nn - 2) +
                                                               alpha_s[1] ** 2 * Ck[:, 2 * Nn, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Nm - 1) * jnp.sign(Nm - 2) + 
                                                               alpha_s[2] ** 2 * Ck[:, 2 * Nn * Nm, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Np - 1) * jnp.sign(Np - 2)))
    
    plasma_energy_2 = (0.5 * mi_me *  alpha_s[3] * alpha_s[4] * alpha_s[5]) * ((0.5 * (alpha_s[3] ** 2 + alpha_s[4] ** 2 + alpha_s[5] ** 2) + 
                                                (u_s[3] ** 2 + u_s[4] ** 2 + u_s[5] ** 2)) * Ck[:, Nn * Nm * Np, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] + 
                                                jnp.sqrt(2) * (alpha_s[3] * u_s[3] * Ck[:, 1, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Nn - 1) + 
                                                               alpha_s[4] * u_s[4] * Ck[:, Nn * Nm * Np + Nn, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Nm - 1) +
                                                               alpha_s[5] * u_s[5] * Ck[:, Nn * Nm * Np + Nn * Nm, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Np - 1)) +
                                          (1 / jnp.sqrt(2)) * ((alpha_s[3] ** 2) * Ck[:, Nn * Nm * Np + 2, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Nn - 1) * jnp.sign(Nn - 2) +
                                                               (alpha_s[4] ** 2) * Ck[:, Nn * Nm * Np + 2 * Nn, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Nm - 1) * jnp.sign(Nm - 2) + 
                                                               (alpha_s[5] ** 2) * Ck[:, Nn * Nm * Np + 2 * Nn * Nm, (Ny-1)//2, (Nx-1)//2, (Nz-1)//2] * jnp.sign(Np - 1) * jnp.sign(Np - 2)))
  
        
    plasma_energy = plasma_energy_1 + plasma_energy_2

    EM_energy = 0.5 * jnp.sum(jnp.abs(Fk) ** 2, axis=(-4, -3, -2, -1)) * Omega_ce[0] ** 2
    
    
    return plasma_energy, EM_energy


def compute_entropy(f1, f2, alpha_s, Lx, Ly, Lz, Nx, Ny, Nz, Nvx, Nvy, Nvz):

    x = jnp.linspace(0, Lx, Nx)
    y = jnp.linspace(0, Ly, Ny)
    z = jnp.linspace(0, Lz, Nz)
    vx = jnp.linspace(-4 * alpha_s[0], 4 * alpha_s[0], Nvx)
    vy = jnp.linspace(-4 * alpha_s[1], 4 * alpha_s[1], Nvy)
    vz = jnp.linspace(-4 * alpha_s[2], 4 * alpha_s[2], Nvz)
    
    entropy_1 = (trapezoid(trapezoid(trapezoid(trapezoid(trapezoid(trapezoid(
              (jnp.abs(f1) * jnp.log(jnp.abs(f1))), y, axis=-6) , x, axis=-5), z, axis=-4), vy, axis=-3), vx, axis=-2), vz, axis=-1))
    
    vx = jnp.linspace(-4 * alpha_s[3], 4 * alpha_s[3], Nvx)
    vy = jnp.linspace(-4 * alpha_s[4], 4 * alpha_s[4], Nvy)
    vz = jnp.linspace(-4 * alpha_s[5], 4 * alpha_s[5], Nvz)
    
    entropy_2 = (trapezoid(trapezoid(trapezoid(trapezoid(trapezoid(trapezoid(
              (jnp.abs(f2) * jnp.log(jnp.abs(f2))), y, axis=-6) , x, axis=-5), z, axis=-4), vy, axis=-3), vx, axis=-2), vz, axis=-1))
    

    return entropy_1 + entropy_2