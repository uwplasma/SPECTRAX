# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 10:55:45 2024

@author: cristian
"""

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve
from jax.numpy.fft import fftn, ifftn, fftshift, ifftshift
from jax.scipy.special import factorial
from jax.scipy.integrate import trapezoid
from jax.experimental.ode import odeint
# from quadax import quadgk
from functools import partial
from Examples import density_perturbation, density_perturbation_solution, Landau_damping_1D
import json
import matplotlib.pyplot as plt


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
	vx = jnp.linspace(-4, 4, 40) # Possibly define limits in terms of thermal velocity or alpha.
	vy = jnp.linspace(-4, 4, 40)
	vz = jnp.linspace(-4, 4, 40)
	X, Y, Z, Vx, Vy, Vz = jnp.meshgrid(x, y, z, vx, vy, vz, indexing='ij')

	# Define variables for Hermite polynomials.
	xi_x = (Vx - u[0]) / alpha[0]
	xi_y = (Vy - u[1]) / alpha[1]
	xi_z = (Vz - u[2]) / alpha[2]

	# Compute coefficients of Hermite decomposition of 3D velocity space.
	# Possible improvement: integrate using quadax.quadgk.
	C_nmp = (trapezoid(trapezoid(trapezoid(
			(f(X, Y, Z, Vx, Vy, Vz) * Hermite(n, xi_x) * Hermite(m, xi_y) * Hermite(p, xi_z)) /
			jnp.sqrt(factorial(n) * factorial(m) * factorial(p) * 2 ** (n + m + p)),
			(vx - u[0]) / alpha[0], axis=-3), (vy - u[1]) / alpha[1], axis=-2), (vz - u[2]) / alpha[2], axis=-1))

	# def integral_vz(x, y, z, vx, vy):
	#     interval = jnp.array([-jnp.inf, jnp.inf])
	#     integral = quadgk(lambda vz: f(x, y, z, vx, vy, vz) * Hermite(n, (vx - u[0]) / alpha[0]) * 
	#                   Hermite(m, (vy - u[1]) / alpha[1]) * Hermite(p, (vz - u[2]) / alpha[2]) /
	#             jnp.sqrt(factorial(n) * factorial(m) * factorial(p) * 2 ** (n + m + p)), interval)[0]
	#     return integral

	# def integral_vy(x, y, z, vx):
	#     interval = jnp.array([-jnp.inf, jnp.inf])
	#     return quadgk(lambda vy: integral_vz(x, y, z, vx, vy), interval)[0]

	# def C(x, y, z):
	#     interval = jnp.array([-jnp.inf, jnp.inf])
	#     return quadgk(lambda vx: integral_vy(x, y, z, vx), interval)[0]


	# x = jnp.linspace(0, Lx, Nx)
	# y = jnp.linspace(0, Ly, Ny)
	# z = jnp.linspace(0, Lz, Nz)
	# X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')

	# C_nmp = C(X, Y, Z)

	return C_nmp


# @partial(jax.jit, static_argnums=[7, 8, 9, 10, 11, 12])
def initialize_system(Omega_ce, mi_me, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np):
	"""
	I have to add docstrings!
	"""
	
	# Initialize fields and distributions.
	B, E, fe, fi = density_perturbation(Lx, Omega_ce, mi_me)
		
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
	Ck_aux_x = (jnp.sqrt(m * p) * (alpha[2]/alpha[1] - alpha[1]/alpha[2]) * Ck[n + (m-1) * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) * jnp.sign(p) + 
		jnp.sqrt(m * (p + 1)) * (alpha[2] / alpha[1]) * Ck[n + (m-1) * Nn + (p+1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) * jnp.sign(Np - p - 1) - 
		jnp.sqrt((m + 1) * p) * (alpha[1] / alpha[2]) * Ck[n + (m+1) * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p) * jnp.sign(Nm - m - 1) + 
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
	# Col = -nu * ((n * (n - 1) * (n - 2)) / ((Nn - 1) * (Nn - 2) * (Nn - 3)) + 
	#              (m * (m - 1) * (m - 2)) / ((Nm - 1) * (Nm - 2) * (Nm - 3)) +
	#              (p * (p - 1) * (p - 2)) / ((Np - 1) * (Np - 2) * (Np - 3))) * Ck[n + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...]
	
	Col = -nu * (n * (n - 1) * (n - 2)) / ((Nn - 1) * (Nn - 2) * (Nn - 3)) * Ck[n + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...]
	
	# Col = 0
	
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

# @partial(jax.jit, static_argnums=[10, 11, 12, 13, 14, 15, 16])
def ode_system(Ck_Fk, t, qs, nu, Omega_cs, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns):     
	
	# Define wave vectors.
	kx = (jnp.arange(-Nx//2, Nx//2) + 1) * 2 * jnp.pi
	ky = (jnp.arange(-Ny//2, Ny//2) + 1) * 2 * jnp.pi
	kz = (jnp.arange(-Nz//2, Nz//2) + 1) * 2 * jnp.pi

	# Create 3D grids of kx, ky, kz.
	kx_grid, ky_grid, kz_grid = jnp.meshgrid(kx, ky, kz, indexing='ij')
	
	# Separate between initial conditions for distribution functions (coefficients Ck)
	# and electric and magnetic fields (coefficients Fk).
	Ck = Ck_Fk[:(-6 * Nx * Ny * Nz)].reshape(Ns * Nn * Nm * Np, Nx, Ny, Nz)
	Fk = Ck_Fk[(-6 * Nx * Ny * Nz):].reshape(6, Nx, Ny, Nz)
	
	# Vectorize over n, m, p, and s to generate ODEs for all coefficients Ck.
	dCk_s_dt = (jax.vmap(
		compute_dCk_s_dt, 
		in_axes=(None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0))
		(Ck, Fk, kx_grid, ky_grid, kz_grid, Lx, Ly, Lz, nu, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, jnp.arange(Nn * Nm * Np * Ns)))
		
	# Generate ODEs for Bk and Ek.
	dBk_dt = - 1j * cross_product(jnp.array([kx_grid/Lx, ky_grid/Ly, kz_grid/Lz]), Fk[:3, ...])
	dEk_dt = 1j * cross_product(jnp.array([kx_grid/Lx, ky_grid/Ly, kz_grid/Lz]), Fk[3:, ...]) - \
			(1 / Omega_cs[0]) * (qs[0] * alpha_s[0] * alpha_s[1] * alpha_s[2] * (
			(1 / jnp.sqrt(2)) * jnp.array([alpha_s[0] * Ck[1, ...] * jnp.sign(Nn - 1),
											alpha_s[1] * Ck[Nn + 1, ...] * jnp.sign(Nm - 1),
											alpha_s[2] * Ck[Nn * Nm + 1, ...] * jnp.sign(Np - 1)]) + 
								jnp.array([u_s[0] * Ck[0, ...],
											u_s[1] * Ck[0, ...],
											u_s[2] * Ck[0, ...]])) + \
								qs[1] * alpha_s[3] * alpha_s[4] * alpha_s[5] * (
			(1 / jnp.sqrt(2)) * jnp.array([alpha_s[3] * Ck[Nn * Nm * Np + 1, ...] * jnp.sign(Nn - 1),
											alpha_s[4] * Ck[Nn + Nn * Nm * Np + 1, ...] * jnp.sign(Nm - 1),
											alpha_s[5] * Ck[Nn * Nm + Nn * Nm * Np + 1, ...] * jnp.sign(Np - 1)]) + 
								jnp.array([u_s[3] * Ck[Nn * Nm * Np, ...],
											u_s[4] * Ck[Nn * Nm * Np, ...],
											u_s[5] * Ck[Nn * Nm * Np, ...]])))

	# Combine dC/dt and dF/dt into a single array and flatten it into a 1D array for an ODE solver.
	dFk_dt = jnp.concatenate([dEk_dt, dBk_dt])
	dy_dt = jnp.concatenate([dCk_s_dt.flatten(), dFk_dt.flatten()])
	
	return dy_dt

# @partial(jax.jit, static_argnums=[7, 8, 9, 10, 11, 12, 13, 14, 15])
def anti_transform(Ck, Fk, mi_me, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nvx, Nvy, Nvz, Nn, Nm, Np):
	
	F = ifftn(ifftshift(Fk, axes=(-3, -2, -1)), axes=(-3, -2, -1))
	E, B = F[:, :3, ...], F[:, 3:, ...]
		
	C = ifftn(ifftshift(Ck, axes=(-3, -2, -1)), axes=(-3, -2, -1))
	
	Ce = C[:, :(Nn * Nm * Np), ...]
	Ci = C[:, (Nn * Nm * Np):, ...]
	
	# x = jnp.linspace(0, Lx, Nx)
	# y = jnp.linspace(0, Ly, Ny)
	# z = jnp.linspace(0, Lz, Nz)
	# vx = jnp.linspace(-5, 5, Nvx)
	# vy = jnp.linspace(-5, 5, Nvy)
	# vz = jnp.linspace(-5, 5, Nvz)
	# X, Y, Z, Vx, Vy, Vz = jnp.meshgrid(x, y, z, vx, vy, vz, indexing='ij')
	
	# xi_x = (Vx - u_s[0]) / alpha_s[0]
	# xi_y = (Vy - u_s[1]) / alpha_s[1]
	# xi_z = (Vz - u_s[2]) / alpha_s[2]
	
	# full_Hermite_basis_e = jax.vmap(generate_Hermite_basis, in_axes=(None, None, None, None, None, None, 0))(xi_x, xi_y, xi_z, Nn, Nm, Np, jnp.arange(Nn * Nm * Np))
	
	# xi_x = (Vx - u_s[3]) / alpha_s[3]
	# xi_y = (Vy - u_s[4]) / alpha_s[4]
	# xi_z = (Vz - u_s[5]) / alpha_s[5]
	
	# full_Hermite_basis_i = jax.vmap(generate_Hermite_basis, in_axes=(None, None, None, None, None, None, 0))(xi_x, xi_y, xi_z, Nn, Nm, Np, jnp.arange(Nn * Nm * Np))
	
	# shape_C_expanded = Ce.shape + (Nvx, Nvy, Nvz)
	
	# Ce_expanded = jnp.expand_dims(Ce, (4, 5, 6))
	# Ci_expanded = jnp.expand_dims(Ci, (4, 5, 6))
	
	# Ce_expanded = jnp.broadcast_to(Ce_expanded, shape_C_expanded)
	# Ci_expanded = jnp.broadcast_to(Ci_expanded, shape_C_expanded)
	
	# fe = jnp.array([jnp.sum(Ce_expanded[i, ...] * full_Hermite_basis_e, axis=0) for i in jnp.arange(Ce.shape[0])])
	# fi = jnp.array([jnp.sum(Ci_expanded[i, ...] * full_Hermite_basis_i, axis=0) for i in jnp.arange(Ce.shape[0])])
	
	# The electron and ion energy formulas below assume that us = 0. Generalize them.
	electron_energy_dens = 0.5 * ((alpha_s[0] * (u_s[0] ** 2 + 0.5 * alpha_s[0] ** 2) + 
								alpha_s[1] * (u_s[1] ** 2 + 0.5 * alpha_s[1] ** 2) +
								alpha_s[2] * (u_s[2] ** 2 + 0.5 * alpha_s[2] ** 2)) * Ce[:, 0, ...] + 
								(jnp.sqrt(2) * alpha_s[0] ** 2) * u_s[0] * Ce[:, 1, ...] * jnp.sign(Nn - 1) +
								(alpha_s[0] ** 3 / jnp.sqrt(2)) * Ce[:, 2, ...] * jnp.sign(Nn - 1) * jnp.sign(Nn - 2) + 
								(jnp.sqrt(2) * alpha_s[1] ** 2) * u_s[1] * Ce[:, Nn + 1, ...] * jnp.sign(Nm - 1) +
								(alpha_s[1] ** 3 / jnp.sqrt(2)) * Ce[:, Nn + 2, ...] * jnp.sign(Nm - 1) * jnp.sign(Nm - 2) +
								(jnp.sqrt(2) * alpha_s[2] ** 2) * u_s[2] * Ce[:, Nn * Nm + 1, ...] * jnp.sign(Np - 1)+
								(alpha_s[2] ** 3 / jnp.sqrt(2)) * Ce[:, Nn * Nm + 2, ...] * jnp.sign(Np - 1) * jnp.sign(Np - 2))
								
	ion_energy_dens = 0.5 * mi_me * ((alpha_s[3] * (u_s[3] ** 2 + 0.5 * alpha_s[3] ** 2) + 
									alpha_s[4] * (u_s[4] ** 2 + 0.5 * alpha_s[4] ** 2) +
									alpha_s[5] * (u_s[5] ** 2 + 0.5 * alpha_s[5] ** 2)) * Ci[:, 0, ...] + 
									(jnp.sqrt(2) * alpha_s[3] ** 2) * u_s[3] * Ci[:, 1, ...] * jnp.sign(Nn - 1) +
									(alpha_s[3] ** 3 / jnp.sqrt(2)) * Ci[:, 2, ...] * jnp.sign(Nn - 1) * jnp.sign(Nn - 2) + 
									(jnp.sqrt(2) * alpha_s[4] ** 2) * u_s[4] * Ci[:, Nn + 1, ...] * jnp.sign(Nm - 1) +
									(alpha_s[4] ** 3 / jnp.sqrt(2)) * Ci[:, Nn + 2, ...] * jnp.sign(Nm - 1) * jnp.sign(Nm - 2) +
									(jnp.sqrt(2) * alpha_s[5] ** 2) * u_s[5] * Ci[:, Nn * Nm + 1, ...] * jnp.sign(Np - 1) +
									(alpha_s[5] ** 3 / jnp.sqrt(2)) * Ci[:, Nn * Nm + 2, ...] * jnp.sign(Np - 1) * jnp.sign(Np - 2))
	
	plasma_energy = jnp.mean(electron_energy_dens[:, :, 1, 1], axis=1) + jnp.mean(ion_energy_dens[:, :, 1, 1], axis=1)
	
	EM_energy = (jnp.mean((E[:, 0, :, 1, 1] ** 2 + E[:, 1, :, 1, 1] ** 2 + E[:, 2, :, 1, 1] ** 2 + 
						B[:, 0, :, 1, 1] ** 2 + B[:, 1, :, 1, 1] ** 2 + B[:, 2, :, 1, 1] ** 2) / 2, axis=1))
	
	return B, E, Ce, Ci, plasma_energy, EM_energy

def main():
	# Load simulation parameters.
	with open('plasma_parameters_density_perturbation.json', 'r') as file:
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
	t_steps, t_max = parameters['t_steps'], parameters['t_max']
	
	# Save parameters into txt.
	with open('C:\Cristian\Postdoc\Madison\Code\Simulations\Density_perturbation\density_perturbation_S9.txt', 'w') as file:
		file.write(f"Nx, Ny, Nz: {Nx}, {Ny}, {Nz}\n")
		file.write(f"Nvx, Nvy, Nvz: {Nvx}, {Nvy}, {Nvz}\n")
		file.write(f"Lx, Ly, Lz: {Lx}, {Ly}, {Lz}\n")
		file.write(f"Nn, Nm, Np, Ns: {Nn}, {Nm}, {Np}, {Ns}\n")
		file.write(f"mi_me: {mi_me}\n")
		file.write(f"Omega_cs: {Omega_cs.tolist()}\n")
		file.write(f"qs: {qs.tolist()}\n")
		file.write(f"alpha_s: {alpha_s.tolist()}\n")
		file.write(f"u_s: {u_s.tolist()}\n")
		file.write(f"nu: {nu}\n")
		file.write(f"t_steps, t_max: {t_steps}, {t_max}\n")

	# Load initial conditions.
	Ck_0, Fk_0 = initialize_system(Omega_cs[0], mi_me, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np)
	

	# Combine initial conditions.
	initial_conditions = jnp.concatenate([Ck_0.flatten(), Fk_0.flatten()])

	# Define the time array.
	t = jnp.linspace(0, t_max, t_steps)
	x = jnp.linspace(0, Lx, Nx)
	T, X = jnp.meshgrid(t, x, indexing='ij')

	dy_dt = partial(ode_system, qs=qs, nu=nu, Omega_cs=Omega_cs, alpha_s=alpha_s, u_s=u_s, Lx=Lx, Ly=Ly, Lz=Lz, Nx=Nx, Ny=Ny, Nz=Nz, Nn=Nn, Nm=Nm, Np=Np, Ns=Ns)

	# Solve the ODE system (I have to rewrite this part of the code).
	result = odeint(dy_dt, initial_conditions, t)
	
	Ck = result[:,:(-6 * Nx * Ny * Nz)].reshape(len(t), Ns * Nn * Nm * Np, Nx, Ny, Nz)
	Fk = result[:,(-6 * Nx * Ny * Nz):].reshape(len(t), 6, Nx, Ny, Nz)
	
	# Define wave vectors.
	kx = (jnp.arange(-Nx//2, Nx//2) + 1) * 2 * jnp.pi
	ky = (jnp.arange(-Ny//2, Ny//2) + 1) * 2 * jnp.pi
	kz = (jnp.arange(-Nz//2, Nz//2) + 1) * 2 * jnp.pi
	
	# Create 3D grids of kx, ky, kz.
	kx_grid, ky_grid, kz_grid = jnp.meshgrid(kx, ky, kz, indexing='ij')
	
	# divBk2_mean = jnp.mean(jnp.array([kx_grid * Fk[i, 3, ...] + ky_grid * Fk[i, 4, ...] + kz_grid * Fk[i, 5, ...] for i in jnp.arange(len(t))]) ** 2, axis=[1, 2, 3])
	
	B, E, Ce, Ci, plasma_energy, EM_energy = anti_transform(Ck, Fk, mi_me, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nvx, Nvy, Nvz, Nn, Nm, Np)
	
	# Cn002 = jnp.mean(Ce[:, :Nn, ...], axis=[2, 3, 4])
	
	# B_exact, E_exact, fe_exact_0, fe_exact_2, fe_exact_5, C0_exact= density_perturbation_solution(Lx, Omega_cs[0], mi_me)    
	# C0_x_t_exact = C0_exact(T, X)
	# C1_x_t_exact = C1_exact(T, X)
	
	# Ce_exact_0 = (jax.vmap(
	#     compute_C_nmp, in_axes=(
	#         None, None, None, None, None, None, None, None, None, None, None, None, 0))
	#     (fe_exact_0, alpha_s[:3], u_s[:3], Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, jnp.arange(Nn * Nm * Np)))
	
	# Ce_exact_2 = (jax.vmap(
	#     compute_C_nmp, in_axes=(
	#         None, None, None, None, None, None, None, None, None, None, None, None, 0))
	#     (fe_exact_2, alpha_s[:3], u_s[:3], Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, jnp.arange(Nn * Nm * Np)))
	
	# Ce_exact_5 = (jax.vmap(
	#     compute_C_nmp, in_axes=(
	#         None, None, None, None, None, None, None, None, None, None, None, None, 0))
	#     (fe_exact_5, alpha_s[:3], u_s[:3], Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, jnp.arange(Nn * Nm * Np)))
	
	# Plot magnetic field.
	plt.plot(t, jnp.sqrt(jnp.mean(B[:, 0, :, 1, 1].real ** 2, axis=1)), label='$B_{x,rms}$', linestyle='-', color='red')
	plt.plot(t, jnp.sqrt(jnp.mean(B[:, 1, :, 1, 1].real ** 2, axis=1)), label='$B_{y,rms}$', linestyle='--', color='blue')
	plt.plot(t, jnp.sqrt(jnp.mean(B[:, 2, :, 1, 1].real ** 2, axis=1)), label='$B_{z,rms}$', linestyle='-.', color='green')

	plt.xlabel('$t\omega_{pe}$')
	plt.ylabel('$B_{rms}$')
	plt.title('Magnetic field vs. Time')

	plt.legend()

	plt.show()
	
	# C0e vs C0i.
	plt.figure(figsize=(8, 6))
	plt.plot(x, Ce[0, :, 1, 1].real, label='$C_{e,0}$, $t\omega_{pe} = 0$', linestyle='-', color='red', linewidth=3.0)
	plt.plot(x, Ci[0, :, 1, 1].real, label='$C_{i,0}$, $t\omega_{pe} = 0$', linestyle='--', color='blue', linewidth=3.0)
	plt.xlabel(r'$x/d_e$', fontsize=16)
	plt.ylabel(r'$C_0$', fontsize=16)
	plt.xlim((0,3))
	# plt.ylim((4,12))
	plt.title(rf'$\nu ={nu}, N_x = {Nx}, N_n = {Nn}$', fontsize=16)
	plt.legend().set_draggable(True)

	plt.show()
	
	# C0 vs t.
	plt.figure(figsize=(8, 6))
	plt.plot(x, Ce[0, 0, :, 1, 1].real, label='Approx. solution, $t\omega_{pe} = 0$', linestyle='-', color='red', linewidth=3.0)
	# plt.plot(x, C0_x_t_exact[0, :].real, label='Exact solution, $t\omega_{pe} = 0$', linestyle='--', color='black', linewidth=3.0)
	plt.plot(x, Ce[20, 0, :, 1, 1].real, label='Approx. solution, $t\omega_{pe} = 2$', linestyle='-', color='blue', linewidth=3.0)
	# plt.plot(x, C0_x_t_exact[20, :].real, label='Exact solution, $t\omega_{pe} = 2$', linestyle=':', color='black', linewidth=3.0)
	plt.plot(x, Ce[50, 0, :, 1, 1].real, label='Approx. solution, $t\omega_{pe} = 5$', linestyle='-', color='green', linewidth=3.0)
	# plt.plot(x, C0_x_t_exact[50, :].real, label='Exact solution, $t\omega_{pe} = 5$', linestyle='-.', color='black', linewidth=3.0)

	plt.xlabel(r'$x/d_e$', fontsize=16)
	plt.ylabel(r'$C_{e, 0}$', fontsize=16)
	plt.xlim((0,3))
	# plt.ylim((4,12))
	plt.title(rf'$\nu ={nu}, N_x = {Nx}, N_n = {Nn}$', fontsize=16)
	plt.legend().set_draggable(True)

	plt.show()
	
	# C0 period.
	plt.figure(figsize=(8, 6))
	plt.plot(x, Ce[0 ,0, :, 1, 1].real, label='Approx. solution, $t\omega_{pe} = 0$', linestyle='-', color='red', linewidth=3.0)
	plt.plot(x, Ce[243 ,0, :, 1, 1].real, label='Approx. solution, $t\omega_{pe} = 24.3$', linestyle='--', color='black', linewidth=3.0)
	plt.plot(x, Ce[245 ,0, :, 1, 1].real, label='Approx. solution, $t\omega_{pe} = 24.5$', linestyle=':', color='blue', linewidth=3.0)
	# plt.plot(x, Ce[113 ,0, :, 1, 1].real, label='Approx. solution, $t\omega_{pe} = 10$', linestyle='--', color='green', linewidth=3.0)
	# plt.plot(x, Ce[64 ,0, :, 1, 1].real, label='Approx. solution, t\omega_{pe} = 6.4', linestyle='-', color='magenta', linewidth=3.0)
	plt.xlabel(r'$x/d_e$', fontsize=16)
	plt.ylabel(r'$C_{e, 0}$', fontsize=16)
	plt.xlim((0,3))
	# plt.ylim((4,12))
	plt.title(rf'$\nu ={nu}, N_x = {Nx}, N_n = {Nn}$' + r'$, T_{theo}\omega_{pe} = 24.5$', fontsize=16)
	plt.legend().set_draggable(True)

	plt.show()
	
	# Hermite coefficients at fixed times.
	plt.figure(figsize=(8, 6))
	plt.plot(jnp.arange(Nn), jnp.mean(jnp.abs(Ce[0 , :, ...]) ** 2, axis=[-3, -2, -1]), label='Approx. solution, $t\omega_{pe} = 0$', linestyle='-', color='red', linewidth=3.0)
	plt.plot(jnp.arange(Nn), jnp.mean(jnp.abs(Ce_exact_0) ** 2, axis=[-3, -2, -1]), label='Exact solution, $t\omega_{pe} = 0$', linestyle='--', color='black', linewidth=3.0)
	plt.plot(jnp.arange(Nn), jnp.mean(jnp.abs(Ce[20 , :, ...]) ** 2, axis=[-3, -2, -1]), label='Approx. solution, $t\omega_{pe} = 2$', linestyle='-', color='blue', linewidth=3.0)
	plt.plot(jnp.arange(Nn), jnp.mean(jnp.abs(Ce_exact_2) ** 2, axis=[-3, -2, -1]), label='Exact solution, $t\omega_{pe} = 2$', linestyle=':', color='black', linewidth=3.0)
	plt.plot(jnp.arange(Nn), jnp.mean(jnp.abs(Ce[50 , :, ...]) ** 2, axis=[-3, -2, -1]), label='Approx. solution, $t\omega_{pe} = 5$', linestyle='-', color='green', linewidth=3.0)
	plt.plot(jnp.arange(Nn), jnp.mean(jnp.abs(Ce_exact_5) ** 2, axis=[-3, -2, -1]), label='Exact solution, $t\omega_{pe} = 5$', linestyle='-.', color='black', linewidth=3.0)
	plt.xlabel(r'$n$', fontsize=16)
	plt.ylabel(r'$\langle|C_{e, n}|^2\rangle$', fontsize=16)
	plt.title(rf'$\nu ={nu}, N_x = {Nx}, N_n = {Nn}$', fontsize=16)
	plt.legend().set_draggable(True)
	plt.yscale('log')
	plt.show()
	
	# Energy vs t.
	plt.figure(figsize=(8, 6))
	# plt.yscale("log")
	plt.plot(t, plasma_energy, label='Plasma energy', linestyle='-', color='red', linewidth=3.0)
	plt.plot(t, EM_energy, label='EM energy', linestyle='-', color='blue', linewidth=3.0)
	plt.plot(t, plasma_energy + EM_energy, label='Total energy', linestyle='-', color='green', linewidth=3.0)

	plt.xlabel(r'$t\omega_{pe}$', fontsize=16)
	plt.ylabel(r'Energy', fontsize=16)
	plt.xlim((0,t_max))
	# plt.ylim((4,12))
	plt.title(rf'$\nu ={nu}, N_x = {Nx}, N_n = {Nn}$', fontsize=16)
	plt.legend()

	plt.show()
	
	# C vs n vs t.

	Ce = Ce.at[:, 0, ...].add(-8)
	C2 = jnp.mean(jnp.abs(Ce) ** 2, axis=[-3, -2, -1])
	
	plt.imshow(jnp.log10(C2), aspect='auto', cmap='viridis', interpolation='none', origin='lower', extent=(0, Nn, 0, t_max))
	plt.colorbar(label=r'$log_{10}(\langle |C_{n}|^2\rangle (t))$').ax.yaxis.label.set_size(16)
	# plt.colorbar(label=r'$\langle |C_{n}|^2\rangle (t)$').ax.yaxis.label.set_size(16)
	plt.xlabel('n', fontsize=16)
	plt.ylabel('t', fontsize=16)
	plt.title(rf'$\nu ={nu}, N_x = {Nx}, N_n = {Nn}$', fontsize=16)
	plt.show()

	# C vs n vs k.
	plt.imshow(jnp.transpose(jnp.log10(Ck[100, :Nn * Nm * Np, :, 1, 1].real)), aspect='auto', cmap='viridis', 
			interpolation='none', origin='lower', extent=(0, Nn, jnp.min(kx), jnp.max(kx)))
	plt.colorbar(label=r'$log_{10}(Real(C_{n, k}))$').ax.yaxis.label.set_size(16)  # Add a color bar to show the mapping of values to colors
	plt.xlabel('n', fontsize=16)
	plt.ylabel('k', fontsize=16)
	plt.title(rf'$\nu ={nu}, N_x = {Nx}, N_n = {Nn}$' + r'$, t\omega_{pe}=10$', fontsize=16)
	plt.show()


if __name__ == "__main__":
	main()
