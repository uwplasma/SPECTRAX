# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 10:55:45 2024

@author: cristian
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.signal import convolve
from jax.numpy.fft import fftn, fftshift
from jax.scipy.special import factorial
from jax.scipy.integrate import trapezoid
from jax.experimental.ode import odeint
# from quadax import quadgk
from diffrax import diffeqsolve, Dopri5, Tsit5, Dopri8, ODETerm, SaveAt, ConstantStepSize, PIDController
from functools import partial
from Examples_1D import density_perturbation_1D, density_perturbation_solution, Landau_damping_1D, Landau_damping_HF_1D
from Examples_2D import Kelvin_Helmholtz_2D


def Hermite(n, x):
    """
    I have to add docstrings!
    """
    
    n = n.astype(int) # Ensure that n is an integer.
    
    # Add next term in Hermite polynomial. Body function of fori_loop below.
    def add_Hermite_term(m, partial_sum):
        return partial_sum + ((-1)**m / (factorial(m) * factorial(n - 2*m))) * (2*x)**(n - 2*m)
    
    # Return Hermite polynomial of order n.
    return factorial(n) * jax.lax.fori_loop(0, (n // 2) + 1, add_Hermite_term, jnp.zeros_like(x))


def generate_Hermite_basis(xi_x, xi_y, xi_z, Nn, Nm, Np, indices):
    """
    I have to add docstrings!
    """
    
    # Indices below represent order of Hermite polynomials.
    p = jnp.floor(indices / (Nn * Nm)).astype(int)
    m = jnp.floor((indices - p * Nn * Nm) / Nn).astype(int)
    n = (indices - p * Nn * Nm - m * Nn).astype(int)
    
    # Generate element of AW Hermite basis in 3D space.
    Hermite_basis = (Hermite(n, xi_x) * Hermite(m, xi_y) * Hermite(p, xi_z) * 
                jnp.exp(-(xi_x**2 + xi_y**2 + xi_z**2)) / 
                jnp.sqrt((jnp.pi)**3 * 2**(n + m + p) * factorial(n) * factorial(m) * factorial(p)))
    
    return Hermite_basis


def moving_average(data, window_size):
    """
    I have to add docstrings!
    """

    data_array = jnp.array(data)
    kernel = jnp.ones(window_size) / window_size
    return jnp.convolve(data_array, kernel, mode='valid')


def compute_C_nmp(f, alpha, u, Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, indices):
    """
    I have to add docstrings!
    """
    
    # Indices below represent order of Hermite polynomials.
    p = jnp.floor(indices / (Nn * Nm)).astype(int)
    m = jnp.floor((indices - p * Nn * Nm) / Nn).astype(int)
    n = (indices - p * Nn * Nm - m * Nn).astype(int)

    # Generate 6D space for particle distribution function f.
    x = jnp.linspace(0, Lx, Nx)
    y = jnp.linspace(0, Ly, Ny)
    z = jnp.linspace(0, Lz, Nz)
    vx = jnp.linspace(-5 * alpha[0] + u[0], 5 * alpha[0] + u[0], 40)
    vy = jnp.linspace(-5 * alpha[1] + u[1], 5 * alpha[1] + u[1], 40)
    vz = jnp.linspace(-5 * alpha[2] + u[2], 5 * alpha[2] + u[2], 40)
      
    def add_C_nmp(i, C_nmp):
        ivx = jnp.floor(i / (5 ** 2)).astype(int)
        ivy = jnp.floor((i - ivx * 5 ** 2) / 5).astype(int)
        ivz = (i - ivx * 5 ** 2 - ivy * 5).astype(int)
        
        vx_slice = jax.lax.dynamic_slice(vx, (ivx * 8,), (8,))
        vy_slice = jax.lax.dynamic_slice(vy, (ivy * 8,), (8,))
        vz_slice = jax.lax.dynamic_slice(vz, (ivz * 8,), (8,))
        
        X, Y, Z, Vx, Vy, Vz = jnp.meshgrid(x, y, z, vx_slice, vy_slice, vz_slice, indexing='xy')

        # Define variables for Hermite polynomials.
        xi_x = (Vx - u[0]) / alpha[0]
        xi_y = (Vy - u[1]) / alpha[1]
        xi_z = (Vz - u[2]) / alpha[2]

        # Compute coefficients of Hermite decomposition of 3D velocity space.
        return C_nmp + (trapezoid(trapezoid(trapezoid(
                (f(X, Y, Z, Vx, Vy, Vz) * Hermite(n, xi_x) * Hermite(m, xi_y) * Hermite(p, xi_z)) /
                jnp.sqrt(factorial(n) * factorial(m) * factorial(p) * 2 ** (n + m + p)),
                (vx_slice - u[0]) / alpha[0], axis=-3), (vy_slice - u[1]) / alpha[1], axis=-2), (vz_slice - u[2]) / alpha[2], axis=-1))
                
    Nv = 125
        
    return jax.lax.fori_loop(0, Nv, add_C_nmp, jnp.zeros((Ny, Nx, Nz)))


def initialize_system_xp(Omega_ce, mi_me, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns):
    """
    I have to add docstrings!
    """
    
    # # Initialize fields and distributions.
    B, E, fe, fi = Kelvin_Helmholtz_2D(Lx, Ly, Omega_ce, alpha_s[0], alpha_s[1])
        
    # Hermite decomposition of dsitribution funcitons.
    Ce_0 = (jax.vmap(
        compute_C_nmp, in_axes=(
            None, None, None, None, None, None, None, None, None, None, None, None, 0))
        (fe, alpha_s[:3], u_s[:3], Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, jnp.arange(Nn * Nm * Np)))
    Ci_0 = (jax.vmap(
        compute_C_nmp, in_axes=(
            None, None, None, None, None, None, None, None, None, None, None, None, 0))
        (fi, alpha_s[3:], u_s[3:], Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, jnp.arange(Nn * Nm * Np)))

    # Combine Ce_0 and Ci_0 into single array and compute the fast Fourier transform.
    
    Cek_0 = fftshift(fftn(Ce_0, axes=(-3, -2, -1)), axes=(-3, -2, -1))
    Cik_0 = fftshift(fftn(Ci_0, axes=(-3, -2, -1)), axes=(-3, -2, -1))

    
    Ck_0 = jnp.concatenate([Cek_0, Cik_0])
    # Ck_0 = fftshift(fftn(C_0, axes=(-3, -2, -1)), axes=(-3, -2, -1))
    
    ############################################################################################################   
    # Attempt to generalize to more than two species (work in progress).
    
    # B, E, f = density_perturbation_1D(Lx, Omega_ce, mi_me)
    
    # C_0 = jnp.zeros((Ns * Nn * Nm * Np))
    # for s in jnp.arange(Ns):
    #     C_s = (jax.vmap(compute_C_nmp, in_axes=(
    #         None, None, None, None, None, None, None, None, None, None, None, None, 0))
    #     (f[s], alpha_s[(s * 3):((s + 1) * 3)], u_s[(s * 3):((s + 1) * 3)], Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, jnp.arange(Nn * Nm * Np)))
        
    #     C_0.at[(s * Ns * Nn * Nm * Np):((s + 1) * Ns * Nn * Nm * Np)].set(C_s)
    
    ############################################################################################################
    
    # Define 3D grid for functions E(x, y, z) and B(x, y, z).
    x = jnp.linspace(0, Lx, Nx)
    y = jnp.linspace(0, Ly, Ny)
    z = jnp.linspace(0, Lz, Nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='xy')
    
    # Combine E and B into single array and compute the fast Fourier transform.
    
    Ek_0 = fftshift(fftn(E(X, Y, Z), axes=(-3, -2, -1)), axes=(-3, -2, -1))
    Bk_0 = fftshift(fftn(B(X, Y, Z), axes=(-3, -2, -1)), axes=(-3, -2, -1))

    Fk_0 = jnp.concatenate([Ek_0, Bk_0])
    # Fk_0 = fftshift(fftn(F_0, axes=(-3, -2, -1)), axes=(-3, -2, -1))
    
    return Ck_0, Fk_0


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


def compute_dCk_s_dt(Ck, Fk, kx_grid, ky_grid, kz_grid, Lx, Ly, Lz, nu, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, indices):
    """
    I have to add docstrings!
    """
    
    # Species. s = 0 corresponds to electrons and s = 1 corresponds to ions.
    s = jnp.floor(indices / (Nn * Nm * Np)).astype(int)
    
    # Indices below represent order of Hermite polynomials (they identify the Hermite-Fourier coefficients Ck[n, p, m]).
    p = jnp.floor((indices - s * Nn * Nm * Np) / (Nn * Nm)).astype(int)
    m = jnp.floor((indices - s * Nn * Nm * Np - p * Nn * Nm) / Nn).astype(int)
    n = (indices - s * Nn * Nm * Np - p * Nn * Nm - m * Nn).astype(int)
    
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
    
    # Collision operator for Nn, Nm, Np > 3.
    # Col = -nu * ((n * (n - 1) * (n - 2)) / ((Nn - 1) * (Nn - 2) * (Nn - 3)) + 
    #              (m * (m - 1) * (m - 2)) / ((Nm - 1) * (Nm - 2) * (Nm - 3)) +
    #              (p * (p - 1) * (p - 2)) / ((Np - 1) * (Np - 2) * (Np - 3))) * Ck[n + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...]
    
    # Collision operator for Np < 4.
    # Col = -nu * ((n * (n - 1) * (n - 2)) / ((Nn - 1) * (Nn - 2) * (Nn - 3)) + 
    #              (m * (m - 1) * (m - 2)) / ((Nm - 1) * (Nm - 2) * (Nm - 3))) * Ck[n + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...]
    
    # Collision operator for Nm, Np < 4.
    # Col = -nu * (n * (n - 1) * (n - 2)) / ((Nn - 1) * (Nn - 2) * (Nn - 3)) * Ck[n + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...]
    
    
    Col = 0
        
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


@partial(jax.jit, static_argnums=[9, 10, 11, 12, 13, 14, 15, 17])
def VM_simulation(qs, nu, Omega_cs, alpha_s, mi_me, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns, t_max, t_steps):
    
   
    # Load initial conditions.
    Ck_0, Fk_0 = initialize_system_xp(Omega_cs[0], mi_me, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns)

    # Load initial conditions in Hermite-Fourier space.
    # Ck_0, Fk_0 = Landau_damping_HF_1D(Lx, Ly, Lz, Omega_cs[0], alpha_s[0], alpha_s[3], Nn)
    
    # Combine initial conditions.
    initial_conditions = jnp.concatenate([Ck_0.flatten(), Fk_0.flatten()])

    # Define the time array for data output.
    t = jnp.linspace(0, t_max, t_steps)

    args = (qs, nu, Omega_cs, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns)
    
    # Solve ODE system using diffrax.
    saveat = SaveAt(ts=t)
    term = ODETerm(ode_system)
    solver = Dopri5()
    # stepsize_controller = ConstantStepSize()
    # stepsize_controller = PIDController(rtol=1e-12, atol=1e-12)
    result = diffeqsolve(term, solver, t0=0, t1=t_max, dt0=0.05, y0=initial_conditions, args=args, saveat=saveat)
    
###################################################################################################
    # # Solve the ODE system using odeint.
    
    # dy_dt = partial(ode_system, args=args)
    # result = odeint(dy_dt, initial_conditions, t)
    
###################################################################################################
    
    Ck = result.ys[:,:(-6 * Nx * Ny * Nz)].reshape(len(result.ts), Ns * Nn * Nm * Np, Ny, Nx, Nz)
    Fk = result.ys[:,(-6 * Nx * Ny * Nz):].reshape(len(result.ts), 6, Ny, Nx, Nz)
    
    # jnp.save('Ck.npy', np.array(Ck))
    # jnp.save('Fk.npy', np.array(Fk))
    # jnp.save('t.npy', np.array(t))

    return Ck, Fk, result.ts
