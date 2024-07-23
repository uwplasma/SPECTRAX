# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 10:55:45 2024

@author: cristian
"""

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve
from jax.numpy.fft import fftn, fftshift
from jax.scipy.special import factorial
from jax.scipy.integrate import trapezoid
from jax.experimental.ode import odeint
from functools import partial
import json


# Perhaps move examples to separate file.
def Orszag_Tang(Lx, Ly, Omega_ce, mi_me):
    """
    I have to add docstrings!
    """
    
    vte = jnp.sqrt(0.25 / 2) * Omega_ce # Electron thermal velocity.
    vti = vte * jnp.sqrt(1 / mi_me) # Ion thermal velocity.
    deltaB = 0.2 # In-plane magnetic field amplitude. 
    U0 = deltaB * Omega_ce / jnp.sqrt(mi_me) # Fluid velocity amplitude.
    
    # Wavenumbers.
    kx = 2 * jnp.pi / Lx
    ky = 2 * jnp.pi / Ly
    
    # Electron and ion fluid velocities.
    Ue = lambda x, y, z: U0 * jnp.array([-jnp.sin(ky * y), jnp.sin(kx * x), -deltaB * Omega_ce * (2 * kx * jnp.cos(2 * kx * x) + ky * jnp.cos(ky * y))])
    Ui = lambda x, y, z: U0 * jnp.array([-jnp.sin(ky * y), jnp.sin(kx * x), jnp.zeros_like(x)])
    
    # Magnetic and electric fields.
    B = lambda x, y, z: jnp.array([-deltaB * jnp.sin(ky * y), deltaB * jnp.sin(2 * kx * x), jnp.ones_like(x)])
    E = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)])
    
    # Electron and ion distribution functions.
    fe = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * vte ** 3) * 
                                        jnp.exp(-((vx - Ue(x, y, z)[0])**2 + (vy - Ue(x, y, z)[1])**2 + (vz - Ue(x, y, z)[2])**2) / (2 * vte ** 2))))
    fi = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * vti ** 3) * 
                                        jnp.exp(-((vx - Ui(x, y, z)[0])**2 + (vy - Ui(x, y, z)[1])**2 + (vz - Ui(x, y, z)[2])**2) / (2 * vti ** 2))))
    
    return B, E, fe, fi


def simple_example(Lx, Ly):
    """
    I have to add docstrings!
    """
    
    vte = 0.4 # Electron thermal velocity.
    vti = 0.4 # Ion thermal velocity.
    deltaB = 0.2 # In-plane magnetic field amplitude.
    
    # Wavenumbers.
    kx = 2 * jnp.pi / Lx
    ky = 2 * jnp.pi / Ly
    
    # Define elements of 3D Hermite basis.
    Hermite_000 = lambda xi_x, xi_y, xi_z: generate_Hermite_basis(xi_x, xi_y, xi_z, 1, 1, 1, 0)
    Hermite_100 = lambda xi_x, xi_y, xi_z: generate_Hermite_basis(xi_x, xi_y, xi_z, 1, 1, 1, 1)
    
    # Magnetic and electric fields.
    B = lambda x, y, z: jnp.array([-deltaB * jnp.sin(ky * y), deltaB * jnp.sin(2 * kx * x), jnp.ones_like(x)])
    E = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)])
    
    # Electron and ion distribution functions.
    fe = (lambda x, y, z, vx, vy, vz: 3 * jnp.sin(kx * x) * Hermite_000(vx/vte, vy/vte, vz/vte) + 
          2 * jnp.sin(2 * ky * y) * Hermite_100(vx/vte, vy/vte, vz/vte))
    fi = (lambda x, y, z, vx, vy, vz: 3 * jnp.sin(kx * x) * Hermite_000(vx/vti, vy/vti, vz/vti) + 
          2 * jnp.sin(2 * ky * y) * Hermite_100(vx/vti, vy/vti, vz/vti))
    
    return B, E, fe, fi


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
    p = jnp.floor(indices / (Nn * Nm))
    m = jnp.floor((indices - p * Nn * Nm) / Nn)
    n = indices - p * Nn * Nm - m * Nn
    
    # Generate element of AW Hermite basis in 3D space.
    Hermite_basis = (Hermite(n, xi_x) * Hermite(m, xi_y) * Hermite(p, xi_z) * 
                  jnp.exp(-(xi_x**2 + xi_y**2 + xi_z**2)) / 
                  jnp.sqrt((jnp.pi)**3 * 2**(n + m + p) * factorial(n) * factorial(m) * factorial(p)))
    
    return Hermite_basis


def compute_C_nmp(f, alpha, u, Nx, Ny, Nz, Lx, Ly, Lz, Nvx, Nvy, Nvz, Nn, Nm, Np, indices):
    """
    I have to add docstrings!
    """
    
    # Indices below represent order of Hermite polynomials.
    p = jnp.floor(indices / (Nn * Nm))
    m = jnp.floor((indices - p * Nn * Nm) / Nn)
    n = indices - p * Nn * Nm - m * Nn
    
    # Generate 6D space for particle distribution function f.
    x = jnp.linspace(0, Lx, Nx)
    y = jnp.linspace(0, Ly, Ny)
    z = jnp.linspace(0, Lz, Nz)
    vx = jnp.linspace(-5, 5, 50) # Possibly define limits in terms of thermal velocity or alpha.
    vy = jnp.linspace(-5, 5, 50)
    vz = jnp.linspace(-5, 5, 50)
    X, Y, Z, Vx, Vy, Vz = jnp.meshgrid(x, y, z, vx, vy, vz, indexing='ij')
    
    # Define variables for Hermite polynomials.
    xi_x = (Vx - u[0]) / alpha[0]
    xi_y = (Vy - u[1]) / alpha[1]
    xi_z = (Vz - u[2]) / alpha[2]
    
    # Compute coefficients of Hermite decomposition of 3D velocity space.
    # Possible improvement: integrate using quadax.quadgk.
    C_nmp = (trapezoid(trapezoid(trapezoid(
                (f(X, Y, Z, Vx, Vy, Vz) * Hermite(n, xi_x) * Hermite(m, xi_y) * Hermite(p, xi_z)) /
                jnp.sqrt((factorial(n) * factorial(m) * factorial(p)) * 2 ** (n + m + p)),
                (vx - u[0]) / alpha[0], axis=-3), (vx - u[0]) / alpha[0], axis=-2), (vx - u[0]) / alpha[0], axis=-1))
    
    return C_nmp


@partial(jax.jit, static_argnums=[7,8,9,10,11,12,13,14,15])
def initialize_system(Omega_ce, mi_me, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nvx, Nvy, Nvz, Nn, Nm, Np):
    """
    I have to add docstrings!
    """
    
    # Initialize fields and distributions.
    B, E, fe, fi = Orszag_Tang(Lx, Ly, Omega_ce, mi_me)
        
    # Hermite decomposition of dsitribution funcitons.
    Ce_0 = (jax.vmap(
        compute_C_nmp, in_axes=(
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0))
        (fe, alpha_s[:3], u_s[:3], Nx, Ny, Nz, Lx, Ly, Lz, Nvx, Nvy, Nvz, Nn, Nm, Np, jnp.arange(Nn * Nm * Np)))
    Ci_0 = (jax.vmap(
        compute_C_nmp, in_axes=(
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0))
        (fi, alpha_s[3:], u_s[3:], Nx, Ny, Nz, Lx, Ly, Lz, Nvx, Nvy, Nvz, Nn, Nm, Np, jnp.arange(Nn * Nm * Np)))

    # Combine Ce_0 and Ci_0 into single array and compute the fast Fourier transform.
    C_0 = jnp.concatenate([Ce_0, Ci_0])
    Ck_0 = fftshift(fftn(C_0, axes=(-3, -2, -1)), axes=(-3, -2, -1))
    
    # Define 3D grid for functions E(x, y, z) and B(x, y, z).
    x = jnp.linspace(0, Lx, Nx)
    y = jnp.linspace(0, Ly, Ny)
    z = jnp.linspace(0, Lz, Nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    
    # Combine E and B into single array and compute the fast Fourier transform.
    F_0 = jnp.concatenate([E(X, Y, Z), B(X, Y, Z)])
    Fk_0 = fftshift(fftn(F_0, axes=(-3, -2, -1)), axes=(-3, -2, -1))
    
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


def periodic_convolve(H, C):
    """
    I have to add docstrings!
    """
    
    # Extend the input array by wrapping around at the boundaries to account for periodic boundary conditions.
    pad_width = [(k//2, k//2) for k in H.shape]
    extended_H = jnp.pad(H, pad_width, mode='wrap')
    
    # Perform convolution on the extended array.
    # Possible improvement: write my own convolve() function with built in periodic boundary conditions so that
    # the padding above is not required or even let the user choose berween different boundary conditions.
    result = convolve(extended_H, C, mode='valid')
        
    return result


def compute_dCk_s_dt(Ck, Fk, kx_grid, ky_grid, kz_grid, Lx, Ly, Lz, nu, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, indices):
    """
    I have to add docstrings!
    """
    
    # Species. s = 0 corresponds to electrons and s = 1 corresponds to ions.
    s = jnp.floor(indices / (Nn * Nm * Np))
    
    # Indices below represent order of Hermite polynomials (they identify the Hermite-Fourier coefficients Ck[n, p, m]).
    p = jnp.floor((indices - s * Nn * Nm * Np) / (Nn * Nm))
    m = jnp.floor((indices - s * Nn * Nm * Np - p * Nn * Nm) / Nn)
    n = indices - s * Nn * Nm * Np - p * Nn * Nm - m * Nn
    
    # Define u, alpha, charge, and gyrofrequency depending on species.
    u, alpha, q, Omega_c = u_s[(s * 3):(s * 3 + 2)], alpha_s[(s * 3):(s * 3 + 2)], qs[s], Omega_cs[s]
    
    # Define terms to be used in ODEs below.
    Ck_aux_x = (jnp.sqrt(m * p) * (alpha[2]/alpha[1] - alpha[1]/alpha[2]) * Ck[n + (m-1) * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) * jnp.sign(p) + 
        jnp.sqrt(m * (p + 1)) * (alpha[2] / alpha[1]) * Ck[n + (m-1) * Nn + (p+1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) * jnp.sign(Np - p - 1) - 
        jnp.sqrt((m + 1) * p) * (alpha[1] / alpha[2]) * Ck[n + (m+1) * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p) * jnp.sign(Nm - m - 1)+ 
        jnp.sqrt(2 * m) * (u[2] / alpha[1]) * Ck[n + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) - 
        jnp.sqrt(2 * p) * (u[1] / alpha[2]) * Ck[n + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p)) 

    Ck_aux_y = (jnp.sqrt(n * p) * (alpha[0]/alpha[2] - alpha[2]/alpha[0]) * Ck[n-1 + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) * jnp.sign(p) + 
        jnp.sqrt((n + 1) * p) * (alpha[0] / alpha[2]) * Ck[n+1 + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p) * jnp.sign(Nn - n - 1) - 
        jnp.sqrt(n * (p + 1)) * (alpha[2] / alpha[0]) * Ck[n-1 + m * Nn + (p+1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) * jnp.sign(Np - p - 1) + 
        jnp.sqrt(2 * p) * (u[0] / alpha[2]) * Ck[n + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p) - 
        jnp.sqrt(2 * n) * (u[2] / alpha[0]) * Ck[n-1 + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n))
    
    Ck_aux_z = (jnp.sqrt(n * m) * (alpha[1]/alpha[0] - alpha[0]/alpha[1]) * Ck[n-1 + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) * jnp.sign(m) + 
        jnp.sqrt(n * (m + 1)) * (alpha[1] / alpha[0]) * Ck[n-1 + (m+1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) * jnp.sign(Nm - m - 1) - 
        jnp.sqrt((n + 1) * m) * (alpha[0] / alpha[1]) * Ck[n+1 + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) * jnp.sign(Nn - n - 1) + 
        jnp.sqrt(2 * n) * (u[1] / alpha[0]) * Ck[n-1 + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) - 
        jnp.sqrt(2 * m) * (u[0] / alpha[1]) * Ck[n + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m))
    
    # Define "unphysical" collision operator to eliminate recurrence.
    Col = -nu * ((n * (n - 1) * (n - 2)) / ((Nn - 1) * (Nn - 2) * (Nn - 3)) + 
                 (m * (m - 1) * (m - 2)) / ((Nm - 1) * (Nm - 2) * (Nm - 3)) +
                 (p * (p - 1) * (p - 2)) / ((Np - 1) * (Np - 2) * (Np - 3))) * Ck[n + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...]
    
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
        (jnp.sqrt(2 * n) / alpha[0]) * periodic_convolve(Fk[0, ...], Ck[n-1 + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n)) +
        (jnp.sqrt(2 * m) / alpha[1]) * periodic_convolve(Fk[1, ...], Ck[n + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m)) +
        (jnp.sqrt(2 * p) / alpha[2]) * periodic_convolve(Fk[2, ...], Ck[n + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p))
    ) + q * Omega_c * (
        periodic_convolve(Fk[3, ...], Ck_aux_x) + 
        periodic_convolve(Fk[4, ...], Ck_aux_y) + 
        periodic_convolve(Fk[5, ...], Ck_aux_z)
    ) + Col)
    
    return dCk_s_dt


def ode_system(Ck_Fk, t, qs, nu, Omega_cs, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns):     
    
    # Define wave vectors.
    kx = jnp.arange(-Nx//2, Nx//2 + 1) * 2 * jnp.pi
    ky = jnp.arange(-Ny//2, Nz//2 + 1) * 2 * jnp.pi
    kz = jnp.arange(-Ny//2, Nz//2 + 1) * 2 * jnp.pi

    # Create 3D grids of kx, ky, kz.
    kx_grid, ky_grid, kz_grid = jnp.meshgrid(kx, ky, kz, indexing='ij')
    
    # Separate between initial conditions for distribution functions (coefficients Ck)
    # and electric and magnetic fields (coefficients Fk).
    Ck = Ck_Fk[:(-6*Nx*Ny*Nz)]
    Fk = Ck_Fk[(-6*Nx*Ny*Nz):]
      
    # Initialize dCk_s_dt with the same shape as Ck.
    dCk_s_dt = jnp.zeros_like(Ck)
    
    # Vectorize over n, m, p, and s to generate ODEs for all coefficients Ck.
    dCk_s_dt = (jax.vmap(
        compute_dCk_s_dt, 
        in_axes=(None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0))
        (Ck, Fk, kx_grid, ky_grid, kz_grid, Lx, Ly, Lz, nu, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, jnp.arange(Nn * Nm * Np * Ns)))
    
    # Generate ODEs for Bk and Ek.
    dBk_dt = - 1j * cross_product(jnp.array([kx_grid/Lx, ky_grid/Ly, kz_grid/Lz]), Fk[:3, ...])
    dEk_dt = 1j * cross_product(jnp.array([kx_grid/Lx, ky_grid/Ly, kz_grid/Lz]), Fk[3:6, ...]) - \
             (1 / Omega_cs[0]) * (qs[0] * alpha_s[0] * alpha_s[1] * alpha_s[2] * (
             (1 / jnp.sqrt(2)) * jnp.array([alpha_s[0] * Ck[1, ...],
                                            alpha_s[1] * Ck[Nn, ...],
                                            alpha_s[2] * Ck[Nn * Nm, ...]]) + 
                                 u_s[:3] * Ck[0, ...]) + \
                                  qs[1] * alpha_s[3] * alpha_s[4] * alpha_s[5] * (
             (1 / jnp.sqrt(2)) * jnp.array([alpha_s[3] * Ck[1 + Nn * Nm * Np, ...],
                                            alpha_s[4] * Ck[Nn + Nn * Nm * Np, ...],
                                            alpha_s[5] * Ck[Nn * Nm * + Nn * Nm * Np, ...]]) + 
                                 u_s[3:] * Ck[Nn * Nm * Np, ...]))

    # Combine dC/dt and dF/dt into a single array and flatten it into a 1D array for an ODE solver.
    dy_dt = jnp.concatenate([dCk_s_dt.flatten(), dBk_dt.flatten(), dEk_dt.flatten()])
    
    return dy_dt

def main():
    # Load simulation parameters.
    with open('plasma_parameters.json', 'r') as file:
        parameters = json.load(file)
    
    # Unpack parameters.
    Nx, Ny, Nz = parameters['Nx'], parameters['Ny'], parameters['Nz']
    Nvx, Nvy, Nvz = parameters['Nvx'], parameters['Nvy'], parameters['Nvz']
    Lx, Ly, Lz = parameters['Lx'], parameters['Ly'], parameters['Lz']
    Nn, Nm, Np, Ns = parameters['Nn'], parameters['Nm'], parameters['Np'], parameters['Ns']
    mi_me = parameters['mi_me']
    Omega_cs = parameters['Omega_ce'] * jnp.array([1.0, 1.0 / mi_me])
    qs = jnp.array(parameters['qs'])
    alpha_s = jnp.array(parameters['alpha_s'])
    u_s = jnp.array(parameters['u_s'])
    nu = parameters['nu']

    # Load initial conditions.
    Ck_0, Fk_0 = initialize_system(Omega_cs[0], mi_me, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nvx, Nvy, Nvz, Nn, Nm, Np)

    # Combine initial conditions.
    initial_conditions = jnp.concatenate([Ck_0.flatten(), Fk_0.flatten()])

    # Define the time array.
    t = jnp.linspace(0, 10, 100)  # Example time array from 0 to 10 with 100 points

    # Solve the ODE system (I have to rewrite this part of the code).
    result = odeint(
        ode_system, 
        initial_conditions, 
        t, 
        args=(qs, nu, Omega_cs, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns)
        )

if __name__ == "__main__":
    main()
