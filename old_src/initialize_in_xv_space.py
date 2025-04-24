import sys

sys.path.append(r'/Users/csvega/Desktop/Madison/Code/Simulations')
sys.path.append(r'/Users/csvega/Desktop/Madison/Code/Vlasov-Maxwell_Spectral_Solver/Vlasov-MaxwellSpectralSolver')

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy.fft import fftn, fftshift
from jax.scipy.special import factorial
from jax.scipy.integrate import trapezoid
from orthax.hermite import hermval3d

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


def generate_Hermite_function(xi_x, xi_y, xi_z, Nn, Nm, indices):
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
                (vy_slice - u[0]) / alpha[0], axis=-3), (vx_slice - u[1]) / alpha[1], axis=-2), (vz_slice - u[2]) / alpha[2], axis=-1))
                
    Nv = 125
        
    return jax.lax.fori_loop(0, Nv, add_C_nmp, jnp.zeros((Ny, Nx, Nz)))


def initialize_system_xv(B, E, f1, f2, Omega_ce, mi_me, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns):
    """
    I have to add docstrings!
    """
        
    # Hermite decomposition of dsitribution funcitons.
    C1_0 = (jax.vmap(
        compute_C_nmp, in_axes=(
            None, None, None, None, None, None, None, None, None, None, None, None, 0))
        (f1, alpha_s[:3], u_s[:3], Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, jnp.arange(Nn * Nm * Np)))
    C2_0 = (jax.vmap(
        compute_C_nmp, in_axes=(
            None, None, None, None, None, None, None, None, None, None, None, None, 0))
        (f2, alpha_s[3:], u_s[3:], Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, jnp.arange(Nn * Nm * Np)))

    # Combine Ce_0 and Ci_0 into single array and compute the fast Fourier transform.
    
    C1k_0 = fftshift(fftn(C1_0, axes=(-3, -2, -1)), axes=(-3, -2, -1))
    C2k_0 = fftshift(fftn(C2_0, axes=(-3, -2, -1)), axes=(-3, -2, -1))

    
    Ck_0 = jnp.concatenate([C1k_0, C2k_0])
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