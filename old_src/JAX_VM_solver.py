# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 10:55:45 2024

@author: cristian
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.signal import convolve
from jax.scipy.special import factorial
from jax.numpy.fft import fftn, ifftn, fftshift, ifftshift
from jax.experimental.ode import odeint
from diffrax import diffeqsolve, Dopri5, Tsit5, Kvaerno3, ImplicitEuler, ODETerm, SaveAt, ConstantStepSize, PIDController
from functools import partial


def cross_product(k_vec, F_vec):
    """
    I have to add docstrings!
    """
    
    # Separate vectors into x, y, z components.
    kx, ky, kz = k_vec
    Fx, Fy, Fz = F_vec
    
    # Compute the cross product k x F.
    result_x = ky * Fz - kz * Fy
    result_y = kz * Fx - kx * Fz
    result_z = kx * Fy - ky * Fx

    return jnp.array([result_x, result_y, result_z])


def collision(Nn, Nm, Np, n, m, p):
    # Pack inputs into arrays
    Nnmp = jnp.array([Nn, Nm, Np])
    nmp = jnp.array([n, m, p])
    
    # Sort by descending N values
    indices = jnp.argsort(-Nnmp)
    sorted_Nnmp = Nnmp[indices]
    sorted_nmp = nmp[indices]
    
    # Helper function to calculate collision term for a single component
    def collision_term(Nj, nj):
        return nj * (nj - 1) * (nj - 2) / ((Nj - 1) * (Nj - 2) * (Nj - 3))
    
    # Calculate all possible terms in a vectorized way
    valid_indices = sorted_Nnmp > 3
    
    # Calculate the sum for valid indices
    result = jnp.sum(jnp.where(valid_indices, collision_term(sorted_Nnmp, sorted_nmp), 0.0))
    
    # Return 0 if no valid indices
    return jax.lax.cond(
        jnp.any(valid_indices),
        lambda _: result,
        lambda _: jnp.array(0.0),
        operand=None
    )

def compute_dCk_s_dt(Ck, Fk, kx_grid, ky_grid, kz_grid, Lx, Ly, Lz, nu, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, index):
    """
    I have to add docstrings!
    """
    
    # Species. s = 0 corresponds to electrons and s = 1 corresponds to ions.
    s = jnp.floor(index / (Nn * Nm * Np)).astype(int)
    
    # Indices below represent order of Hermite polynomials (they identify the Hermite-Fourier coefficients Ck[n, p, m]).
    p = jnp.floor((index - s * Nn * Nm * Np) / (Nn * Nm)).astype(int)
    m = jnp.floor((index - s * Nn * Nm * Np - p * Nn * Nm) / Nn).astype(int)
    n = (index - s * Nn * Nm * Np - p * Nn * Nm - m * Nn).astype(int)
    
    # Define u, alpha, charge, and gyrofrequency depending on species.
    u = jax.lax.dynamic_slice(u_s, (s * 3,), (3,))
    alpha = jax.lax.dynamic_slice(alpha_s, (s * 3,), (3,))
    q, Omega_c = qs[s], Omega_cs[s]
    
    # Define terms to be used in ODEs below.
    Ck_aux_x = (jnp.sqrt(m * p) * (alpha[2] / alpha[1] - alpha[1] / alpha[2]) * Ck[n + (m-1) * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) * jnp.sign(p) + 
        jnp.sqrt(m * (p + 1)) * (alpha[2] / alpha[1]) * Ck[n + (m-1) * Nn + (p+1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) * jnp.sign(Np - p - 1) - 
        jnp.sqrt((m + 1) * p) * (alpha[1] / alpha[2]) * Ck[n + (m+1) * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p) * jnp.sign(Nm - m - 1) + 
        jnp.sqrt(2 * m) * (u[2] / alpha[1]) * Ck[n + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) - 
        jnp.sqrt(2 * p) * (u[1] / alpha[2]) * Ck[n + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p)) 

    Ck_aux_y = (jnp.sqrt(n * p) * (alpha[0] / alpha[2] - alpha[2] / alpha[0]) * Ck[n-1 + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) * jnp.sign(p) + 
        jnp.sqrt((n + 1) * p) * (alpha[0] / alpha[2]) * Ck[n+1 + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p) * jnp.sign(Nn - n - 1) - 
        jnp.sqrt(n * (p + 1)) * (alpha[2] / alpha[0]) * Ck[n-1 + m * Nn + (p+1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) * jnp.sign(Np - p - 1) + 
        jnp.sqrt(2 * p) * (u[0] / alpha[2]) * Ck[n + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p) - 
        jnp.sqrt(2 * n) * (u[2] / alpha[0]) * Ck[n-1 + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n))
    
    Ck_aux_z = (jnp.sqrt(n * m) * (alpha[1] / alpha[0] - alpha[0] / alpha[1]) * Ck[n-1 + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) * jnp.sign(m) + 
        jnp.sqrt(n * (m + 1)) * (alpha[1] / alpha[0]) * Ck[n-1 + (m+1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) * jnp.sign(Nm - m - 1) - 
        jnp.sqrt((n + 1) * m) * (alpha[0] / alpha[1]) * Ck[n+1 + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) * jnp.sign(Nn - n - 1) + 
        jnp.sqrt(2 * n) * (u[1] / alpha[0]) * Ck[n-1 + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) - 
        jnp.sqrt(2 * m) * (u[0] / alpha[1]) * Ck[n + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m))
    
    # Define "unphysical" collision operator to eliminate recurrence.

    Col = -nu * collision(Nn,Nm,Np,n,m,p) * Ck[n + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...]
        
    # Define ODEs for Hermite-Fourier coefficients.
    # Clossure is achieved by setting to zero coefficients with index out of range.
    dCk_s_dt = (- (kx_grid * 1j / Lx) * alpha[0] * (
        jnp.sqrt((n + 1) / 2) * Ck[n+1 + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(Nn - n - 1) +
        jnp.sqrt(n / 2) * Ck[n-1 + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) +
        (u[0] / alpha[0]) * Ck[n + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...]
    ) - (ky_grid * 1j / Ly) * alpha[1] * (
        jnp.sqrt((m + 1) / 2) * Ck[n + (m+1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(Nm - m - 1) +
        jnp.sqrt(m / 2) * Ck[n + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) +
        (u[1] / alpha[1]) * Ck[n + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...]
    ) - (kz_grid * 1j / Lz) * alpha[2] * (
        jnp.sqrt((p + 1) / 2) * Ck[n + m * Nn + (p+1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(Np - p - 1) +
        jnp.sqrt(p / 2) * Ck[n + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p) +
        (u[2] / alpha[2]) * Ck[n + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...]
    ) + q * Omega_c * (
        (jnp.sqrt(2 * n) / alpha[0]) * convolve(Fk[0, ...], Ck[n-1 + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n), mode='same') +
        (jnp.sqrt(2 * m) / alpha[1]) * convolve(Fk[1, ...], Ck[n + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m), mode='same') +
        (jnp.sqrt(2 * p) / alpha[2]) * convolve(Fk[2, ...], Ck[n + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p), mode='same')
    ) + q * Omega_c * (
        convolve(Fk[3, ...], Ck_aux_x, mode='same') + 
        convolve(Fk[4, ...], Ck_aux_y, mode='same') + 
        convolve(Fk[5, ...], Ck_aux_z, mode='same')
    ) + Col)
    
    return dCk_s_dt


def ampere_maxwell_current(qs, alpha_s, u_s, Ck, Nn, Nm, Np, Ns):
    """
    I have to add docstrings!
    """
      
    # Add next term in the current. Body function of fori_loop below.
    def add_current_term(s, partial_sum):
        return partial_sum + qs[s] * alpha_s[s * 3] * alpha_s[s * 3 + 1] * alpha_s[s * 3 + 2] * (
            (1 / jnp.sqrt(2)) * jnp.array([alpha_s[s * 3] * Ck[s * Nn * Nm * Np + 1, ...] * jnp.sign(Nn - 1),
                                           alpha_s[s * 3 + 1] * Ck[s * Nn * Nm * Np + Nn, ...] * jnp.sign(Nm - 1),
                                           alpha_s[s * 3 + 2] * Ck[s * Nn * Nm * Np + Nn * Nm, ...] * jnp.sign(Np - 1)]) + 
                                jnp.array([u_s[s * 3] * Ck[s * Nn * Nm * Np, ...],
                                           u_s[s * 3 + 1] * Ck[s * Nn * Nm * Np, ...],
                                           u_s[s * 3 + 2] * Ck[s * Nn * Nm * Np, ...]]))
    
    # Return full current for Ns particle species.
    return jax.lax.fori_loop(0, Ns, add_current_term, jnp.zeros_like(Ck[:3, ...]))


# def ode_system(t, Ck_Fk, qs, nu, Omega_cs, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns):
def ode_system(t, Ck_Fk, args):

    qs, nu, Omega_cs, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns = args
    
    # Define wave vectors.
    kx = (jnp.arange(-Nx//2, Nx//2) + 1) * 2 * jnp.pi
    ky = (jnp.arange(-Ny//2, Ny//2) + 1) * 2 * jnp.pi
    kz = (jnp.arange(-Nz//2, Nz//2) + 1) * 2 * jnp.pi  

    # Create 3D grids of kx, ky, kz.
    kx_grid, ky_grid, kz_grid = jnp.meshgrid(kx, ky, kz, indexing='xy')
    
    # Separate between initial conditions for distribution functions (coefficients Ck)
    # and electric and magnetic fields (coefficients Fk).
    Ck = Ck_Fk[:(-6 * Nx * Ny * Nz)].reshape(Ns * Nn * Nm * Np, Ny, Nx, Nz)
    Fk = Ck_Fk[(-6 * Nx * Ny * Nz):].reshape(6, Ny, Nx, Nz)
    
    # Vectorize over n, m, p, and s to generate ODEs for all coefficients Ck.
    dCk_s_dt = (jax.vmap(
        compute_dCk_s_dt, 
        in_axes=(None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0))
        (Ck, Fk, kx_grid, ky_grid, kz_grid, Lx, Ly, Lz, nu, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, jnp.arange(Nn * Nm * Np * Ns)))
    
    
    current = ampere_maxwell_current(qs, alpha_s, u_s, Ck, Nn, Nm, Np, Ns)
        
    # Generate ODEs for Bk and Ek.
    dBk_dt = - 1j * cross_product(jnp.array([kx_grid/Lx, ky_grid/Ly, kz_grid/Lz]), Fk[:3, ...])
    dEk_dt = 1j * cross_product(jnp.array([kx_grid/Lx, ky_grid/Ly, kz_grid/Lz]), Fk[3:, ...]) - (1 / Omega_cs[0]) * current
            
            # (qs[0] * alpha_s[0] * alpha_s[1] * alpha_s[2] * (
            # (1 / jnp.sqrt(2)) * jnp.array([alpha_s[0] * Ck[1, ...] * jnp.sign(Nn - 1),
            #                                alpha_s[1] * Ck[Nn + 1, ...] * jnp.sign(Nm - 1),
            #                                alpha_s[2] * Ck[Nn * Nm + 1, ...] * jnp.sign(Np - 1)]) + 
            #                     jnp.array([u_s[0] * Ck[0, ...],
            #                                u_s[1] * Ck[0, ...],
            #                                u_s[2] * Ck[0, ...]])) + \
            #                     qs[1] * alpha_s[3] * alpha_s[4] * alpha_s[5] * (
            # (1 / jnp.sqrt(2)) * jnp.array([alpha_s[3] * Ck[Nn * Nm * Np + 1, ...] * jnp.sign(Nn - 1),
            #                                alpha_s[4] * Ck[Nn + Nn * Nm * Np + 1, ...] * jnp.sign(Nm - 1),
            #                                alpha_s[5] * Ck[Nn * Nm + Nn * Nm * Np + 1, ...] * jnp.sign(Np - 1)]) + 
            #                     jnp.array([u_s[3] * Ck[Nn * Nm * Np, ...],
            #                                u_s[4] * Ck[Nn * Nm * Np, ...],
            #                                u_s[5] * Ck[Nn * Nm * Np, ...]])))

    # Combine dC/dt and dF/dt into a single array and flatten it into a 1D array for an ODE solver.
    dFk_dt = jnp.concatenate([dEk_dt, dBk_dt])
    dy_dt = jnp.concatenate([dCk_s_dt.flatten(), dFk_dt.flatten()])

    
    return dy_dt


@partial(jax.jit, static_argnums=[11, 12, 13, 14, 15, 16, 17, 19])
def VM_simulation(Ck_0, Fk_0, qs, nu, Omega_cs, alpha_s, mi_me, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns, t_max, t_steps, dt):
    

    # Combine initial conditions.
    initial_conditions = jnp.concatenate([Ck_0.flatten(), Fk_0.flatten()])

    # Define the time array for data output.
    t = jnp.linspace(0, t_max, t_steps)

    args = (qs, nu, Omega_cs, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns)
    
###################################################################################################    
    
    # Solve ODE system using diffrax.
    saveat = SaveAt(ts=t)
    term = ODETerm(ode_system)
    solver = Dopri5()
    # stepsize_controller = ConstantStepSize()
    # stepsize_controller = PIDController(rtol=1e-6, atol=1e-6)
    result = diffeqsolve(term, solver, t0=0, t1=t_max, dt0=dt, y0=initial_conditions, args=args, saveat=saveat, max_steps=100000)
    
    Ck = result.ys[:,:(-6 * Nx * Ny * Nz)].reshape(len(result.ts), Ns * Nn * Nm * Np, Ny, Nx, Nz)
    Fk = result.ys[:,(-6 * Nx * Ny * Nz):].reshape(len(result.ts), 6, Ny, Nx, Nz)
    
    # jnp.save('Ck.npy', np.array(Ck))
    # jnp.save('Fk.npy', np.array(Fk))
    # jnp.save('t.npy', np.array(t))

    return Ck, Fk, result.ts
    
###################################################################################################
    # # Solve the ODE system using odeint.
    
    # dy_dt = partial(ode_system, args=args)
    # result = odeint(dy_dt, initial_conditions, t)
    
    # Ck = result[:,:(-6 * Nx * Ny * Nz)].reshape(len(t), Ns * Nn * Nm * Np, Ny, Nx, Nz)
    # Fk = result[:,(-6 * Nx * Ny * Nz):].reshape(len(t), 6, Ny, Nx, Nz)
    
    # return Ck, Fk, t 
