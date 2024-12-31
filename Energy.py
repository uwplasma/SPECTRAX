import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy.fft import ifftn, ifftshift


def compute_energy(Ck, Fk, Omega_ce, mi_me, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nvx, Nvy, Nvz, Nn, Nm, Np):
    
    F = ifftn(ifftshift(Fk, axes=(-3, -2, -1)), axes=(-3, -2, -1))
    E, B = F[:, :3, ...].real, F[:, 3:, ...].real
        
    C = ifftn(ifftshift(Ck, axes=(-3, -2, -1)), axes=(-3, -2, -1))
    
    Ce = C[:, :(Nn * Nm * Np), ...].real
    Ci = C[:, (Nn * Nm * Np):, ...].real
    
    # x = jnp.linspace(0, Lx, Nx)
    # y = jnp.linspace(0, Ly, Ny)
    # z = jnp.linspace(0, Lz, Nz)
    # vx = jnp.linspace(-10*alpha_s[0], 10*alpha_s[0], Nvx)
    # vy = jnp.linspace(-10*alpha_s[1], 10*alpha_s[1], Nvy)
    # vz = jnp.linspace(-10*alpha_s[2], 10*alpha_s[2], Nvz)
    # X, Y, Z, Vx, Vy, Vz = jnp.meshgrid(x, y, z, vx, vy, vz, indexing='ij')
    
    # xi_x = (Vx - u_s[0]) / alpha_s[0]
    # xi_y = (Vy - u_s[1]) / alpha_s[1]
    # xi_z = (Vz - u_s[2]) / alpha_s[2]
    
    # full_Hermite_basis_e = jax.vmap(generate_Hermite_basis, in_axes=(None, None, None, None, None, None, 0))(xi_x, xi_y, xi_z, Nn, Nm, Np, jnp.arange(Nn * Nm * Np))
    
# 	vx = jnp.linspace(-10*alpha_s[3], 10*alpha_s[3], Nvx)
# 	vy = jnp.linspace(-10*alpha_s[4], 10*alpha_s[4], Nvy)
# 	vz = jnp.linspace(-10*alpha_s[5], 10*alpha_s[5], Nvz)
# 	X, Y, Z, Vx, Vy, Vz = jnp.meshgrid(x, y, z, vx, vy, vz, indexing='ij')
    
# 	xi_x = (Vx - u_s[3]) / alpha_s[3]
# 	xi_y = (Vy - u_s[4]) / alpha_s[4]
# 	xi_z = (Vz - u_s[5]) / alpha_s[5]
# 	
# 	full_Hermite_basis_i = jax.vmap(generate_Hermite_basis, in_axes=(None, None, None, None, None, None, 0))(xi_x, xi_y, xi_z, Nn, Nm, Np, jnp.arange(Nn * Nm * Np))
    
    # shape_C_expanded = Ce.shape + (Nvx, Nvy, Nvz)
    
    # Ce_expanded = jnp.expand_dims(Ce, (4, 5, 6))
# 	Ci_expanded = jnp.expand_dims(Ci, (4, 5, 6))
    
    # Ce_expanded = jnp.broadcast_to(Ce_expanded, shape_C_expanded)
# 	Ci_expanded = jnp.broadcast_to(Ci_expanded, shape_C_expanded)
    
    # fe = jnp.array([jnp.sum(Ce_expanded[i, ...] * full_Hermite_basis_e, axis=0) for i in jnp.arange(Ce.shape[0])])
# 	fi = jnp.array([jnp.sum(Ci_expanded[i, ...] * full_Hermite_basis_i, axis=0) for i in jnp.arange(Ce.shape[0])])


        
    # entropy_e = (trapezoid(trapezoid(trapezoid(trapezoid(
    #         (fe[:, :, 1, 1, ...] * jnp.log(fe[:, :, 1, 1, ...])),
    #         x, axis=-4), (vx - u_s[0]) / alpha_s[0], axis=-3),
    #         (vy - u_s[1]) / alpha_s[1], axis=-2), (vz - u_s[2]) / alpha_s[2], axis=-1))
    
    # entropy_e = 0
  
        
    electron_energy_dens = 0.5 * ((0.5 * (alpha_s[0] ** 2 + alpha_s[1] ** 2 + alpha_s[2] ** 2) + 
                                         (u_s[0] ** 2 + u_s[1] ** 2 + u_s[2] ** 2)) * 
                                         alpha_s[0] * alpha_s[1] * alpha_s[2] * Ce[:, 0, ...] + 
                                  jnp.sqrt(2) * (alpha_s[0] * u_s[0] * Ce[:, 1, ...] * jnp.sign(Nn - 1) + 
                                                 alpha_s[1] * u_s[1] * Ce[:, Nn, ...] * jnp.sign(Nm - 1) + 
                                                 alpha_s[2] * u_s[2] * Ce[:, Nn * Nm, ...] * jnp.sign(Np - 1)) *
                                                alpha_s[0] * alpha_s[1] * alpha_s[2] + 
                                  (1 / jnp.sqrt(2)) * ((alpha_s[0] ** 2) * Ce[:, 2, ...] * jnp.sign(Nn - 1) * jnp.sign(Nn - 2) + 
                                                       (alpha_s[1] ** 2) * Ce[:, 2 * Nn, ...] * jnp.sign(Nm - 1) * jnp.sign(Nm - 2) + 
                                                       (alpha_s[2] ** 2) * Ce[:, 2 * Nn * Nm, ...] * jnp.sign(Np - 1) * jnp.sign(Np - 2)) * 
                                                      alpha_s[0] * alpha_s[1] * alpha_s[2])
    
    ion_energy_dens = 0.5 * mi_me * ((0.5 * (alpha_s[3] ** 2 + alpha_s[4] ** 2 + alpha_s[5] ** 2) + 
                                         (u_s[3] ** 2 + u_s[4] ** 2 + u_s[5] ** 2)) * 
                                         alpha_s[3] * alpha_s[4] * alpha_s[5] * Ci[:, 0, ...] + 
                                  jnp.sqrt(2) * (alpha_s[3] * u_s[3] * Ci[:, 1, ...] * jnp.sign(Nn - 1) + 
                                                 alpha_s[4] * u_s[4] * Ci[:, Nn, ...] * jnp.sign(Nm - 1) + 
                                                 alpha_s[5] * u_s[5] * Ci[:, Nn * Nm, ...] * jnp.sign(Np - 1)) *
                                                alpha_s[3] * alpha_s[4] * alpha_s[5] + 
                                  (1 / jnp.sqrt(2)) * ((alpha_s[3] ** 2) * Ci[:, 2, ...] * jnp.sign(Nn - 1) * jnp.sign(Nn - 2) + 
                                                       (alpha_s[4] ** 2) * Ci[:, 2 * Nn, ...] * jnp.sign(Nm - 1) * jnp.sign(Nm - 2) + 
                                                       (alpha_s[5] ** 2) * Ci[:, 2 * Nn * Nm, ...] * jnp.sign(Np - 1) * jnp.sign(Np - 2)) * 
                                                      alpha_s[3] * alpha_s[4] * alpha_s[5])
                                
                                
    
    plasma_energy = jnp.mean(electron_energy_dens[:, ...], axis=(-3, -2, -1)) + jnp.mean(ion_energy_dens[:, ...], axis=(-3, -2, -1))
    
    EM_energy = (jnp.mean((E[:, 0, ...] ** 2 + E[:, 1, ...] ** 2 + E[:, 2, ...] ** 2 + 
                           B[:, 0, ...] ** 2 + B[:, 1, ...] ** 2 + B[:, 2, ...] ** 2), axis=(-3, -2, -1)) * Omega_ce ** 2 / 2)
    
    
    return plasma_energy, EM_energy