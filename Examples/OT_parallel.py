"""Example: 2D Orszag–Tang vortex, split across several devices.

Same setup as ``2D_Orszag_Tang.py``, run in parallel. The ``[parallel]`` table in ``input_OT_parallel.toml``
chooses the device type (``device = "cpu"`` or ``"gpu"``), how many devices to use (``num_devices``), and how
the simulation is divided between them: here the 2 species and the 64 x 64 elements are split over a
2 x 2 x 2 mesh of 8 devices, so each device holds one species and a 32 x 32 block of elements.

With ``device = "cpu"`` the CPU is split into virtual devices. That checks the parallel setup, but gives little
speedup, because a single CPU device already uses most of the cores; real speedups need several GPUs.

The returned ``Ck`` and ``Fk`` stay distributed over the devices. The post-processing below works on them
directly, and converting them with ``numpy.asarray`` (as the plotting helpers do) gathers them.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from time import time
from jax import block_until_ready, config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import factorial
from spectrax import simulation, load_parameters, DG_project
import matplotlib.pyplot as plt

from orszag_tang_data_analysis import (
        plot_energy_timeseries, plot_relative_energy_error,
        plot_Jz_slice, animate_Jz
    )

# Read from input_OT_parallel.toml. This has to happen before any other JAX computation, because the devices
# chosen in its [parallel] table can only be set up while JAX has not run anything yet.
toml_file = os.path.join(os.path.dirname(__file__), 'input_OT_parallel.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

mesh = solver_parameters.get("mesh")
if mesh is None:
    print('Running on a single device.')
else:
    print(f'Splitting the simulation over {mesh.devices.size} {mesh.devices.flat[0].platform.upper()} devices: '
          f'{mesh.shape["s"]} along species, {mesh.shape["x"]} along x, {mesh.shape["y"]} along y, {mesh.shape["z"]} along z.')

print('Setting up simulation parameters...')
deltaB = 0.2 # In-plane magnetic field amplitude.
U0 = deltaB * input_parameters["Omega_ce"] / jnp.sqrt(input_parameters["ms"][1] / input_parameters["ms"][0])

kx = 2 * jnp.pi / input_parameters["Lx"]
ky = 2 * jnp.pi / input_parameters["Ly"]

Lx = input_parameters["Lx"]
Ly = input_parameters["Ly"]
Nx = solver_parameters["Nx"]
Ny = solver_parameters["Ny"]
Nz = solver_parameters["Nz"]
Nn = solver_parameters["Nn"]
Nm = solver_parameters["Nm"]
Np = solver_parameters["Np"]
Ns = solver_parameters["Ns"]
N_DG = solver_parameters["N_DG"]
dims = solver_parameters["dims"]
alpha_s = input_parameters["alpha_s"]

alpha = jnp.array(alpha_s).reshape(Ns, 3)
alpha_x = alpha[:, 0, None, None, None, None, None, None, None, None, None]
alpha_y = alpha[:, 1, None, None, None, None, None, None, None, None, None]
alpha_z = alpha[:, 2, None, None, None, None, None, None, None, None, None]

# Electron and ion fluid velocities.
Ue = lambda x, y, z: jnp.array([-U0 * jnp.sin(ky * y), U0 * jnp.sin(kx * x), -deltaB * input_parameters["Omega_ce"] * (2 * kx * jnp.cos(2 * kx * x) + ky * jnp.cos(ky * y))])
Ui = lambda x, y, z: jnp.array([-U0 * jnp.sin(ky * y), U0 * jnp.sin(kx * x), jnp.zeros_like(x)])

# Magnetic and electric fields.
B = lambda x, y, z: jnp.array([-deltaB * jnp.sin(ky * y), deltaB * jnp.sin(2 * kx * x), jnp.ones_like(x)])
E = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)])

p = jnp.arange(Np)[None, :, None, None, None, None, None, None, None, None]
m = jnp.arange(Nm)[None, None, :, None, None, None, None, None, None, None]
n = jnp.arange(Nn)[None, None, None, :, None, None, None, None, None, None]

Us = lambda x, y, z: jnp.stack([Ue(x, y, z), Ui(x, y, z)], axis=0)[:, :, None, None, None]
U_x = lambda x, y, z: Us(x, y, z)[:, 0]
U_y = lambda x, y, z: Us(x, y, z)[:, 1]
U_z = lambda x, y, z: Us(x, y, z)[:, 2]

C0 = lambda x, y, z: (jnp.sqrt(2 ** (n + m + p) / (factorial(n) * factorial(m) * factorial(p)))
     * (1 / (alpha_x ** (n + 1) * alpha_y ** (m + 1) * alpha_z ** (p + 1)))
     * U_x(x, y, z) ** n * U_y(x, y, z) ** m * U_z(x, y, z) ** p)

# The initial conditions are built on one device; `simulation` distributes them over the mesh.
input_parameters["Ck_0"] = DG_project(C0, input_parameters, solver_parameters) # shape (Ns, Np, Nm, Nn, Ny, Nx, Nz, N_DG)

F0 = lambda x, y, z: jnp.concatenate([E(x, y, z), B(x, y, z)])
input_parameters["Fk_0"] = DG_project(F0, input_parameters, solver_parameters) # shape (6, Ny, Nx, Nz, N_DG)

C = input_parameters["Ck_0"][:, :, :, :, :, :, 0, 0] # (Ns, Np, Nm, Nn, Ny, Nx)
F = input_parameters["Fk_0"][:, :, :, 0, 0] # (6, Ny, Nx)

ne = alpha_s[0] * alpha_s[1] * alpha_s[2] * C[0, 0, 0, 0, :, :] # (Ny, Nx)
ni = alpha_s[3] * alpha_s[4] * alpha_s[5] * C[1, 0, 0, 0, :, :]

Jz = (1 / jnp.sqrt(2)) * (alpha_s[3] * alpha_s[4] * (alpha_s[5]**2) * C[1, 1, 0, 0, :, :] -
                          alpha_s[0] * alpha_s[1] * (alpha_s[2]**2) * C[0, 1, 0, 0, :, :]) # (Ny, Nx)

fig, ax = plt.subplots(1, 3, figsize=(16, 6))
im0 = ax[0].imshow(ne, aspect='auto', cmap='jet',
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))
fig.colorbar(im0, label=r'$n_e$').ax.yaxis.label.set_size(16)

im1 = ax[1].imshow(ni, aspect='auto', cmap='jet',
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))
fig.colorbar(im1, label=r'$n_i$').ax.yaxis.label.set_size(16)

im2 = ax[2].imshow(Jz, aspect='auto', cmap='jet',
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))
fig.colorbar(im2, label=r'$J_z$').ax.yaxis.label.set_size(16)

fig, ax = plt.subplots(2, 3, figsize=(16, 6))
im0 = ax[0, 0].imshow(F[0], aspect='auto', cmap='jet',
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))
fig.colorbar(im0, label=r'$Ex$').ax.yaxis.label.set_size(14)

im1 = ax[0, 1].imshow(F[1], aspect='auto', cmap='jet',
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))
fig.colorbar(im1, label=r'$Ey$').ax.yaxis.label.set_size(14)

im2 = ax[0, 2].imshow(F[2], aspect='auto', cmap='jet',
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))
fig.colorbar(im2, label=r'$Ez$').ax.yaxis.label.set_size(14)

im3 = ax[1, 0].imshow(F[3], aspect='auto', cmap='jet',
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))
fig.colorbar(im3, label=r'$Bx$').ax.yaxis.label.set_size(14)

im4 = ax[1, 1].imshow(F[4], aspect='auto', cmap='jet',
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))
fig.colorbar(im4, label=r'$By$').ax.yaxis.label.set_size(14)

im5 = ax[1, 2].imshow(F[5], aspect='auto', cmap='jet',
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))
fig.colorbar(im5, label=r'$Bz$').ax.yaxis.label.set_size(14)

print('Starting simulation...')
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

# Results. output["Ck"] and output["Fk"] are still distributed over the devices; the operations below
# run on them as they are.
alpha_s = input_parameters["alpha_s"]
Lx = input_parameters["Lx"]
Ly = input_parameters["Ly"]
Nx = solver_parameters["Nx"]
Ny = solver_parameters["Ny"]
Nz = solver_parameters["Nz"]
Nn = solver_parameters["Nn"]
Nm = solver_parameters["Nm"]
Np = solver_parameters["Np"]
t = output["time"]
nu = output["nu"]
uz = output["u_s"][2]
Ck = output["Ck"][:, :, :, :, 0, :] # (Nt, Ns*Np*Nm*Nn, Ny, Nx, Nl)
C = output["Ck"][:, :, :, :, 0, 0] # (Nt, Ns*Np*Nm*Nn, Ny, Nx)
F = output["Fk"][:, :, :, :, 0, :] # (Nt, 6, Ny, Nx, Nl)
basis_idx = output["basis_idx"] # Legendre orders of each DG basis function, shape (Nl, 3)

# Densities
ne = alpha_s[0] * alpha_s[1] * alpha_s[2] * C[:, 0, :, :]
ni = alpha_s[3] * alpha_s[4] * alpha_s[5] * C[:, Nn * Nm * Np, :, :]

# Currents and velocities
Jz = (1 / jnp.sqrt(2)) * (alpha_s[3] * alpha_s[4] * (alpha_s[5]**2) * C[:, Nn * Nm * Np + Nn * Nm, :, :] - alpha_s[0] * alpha_s[1] * (alpha_s[2]**2) * C[:, Nn * Nm, :, :])
Uex = (1 / jnp.sqrt(2)) * alpha_s[0]**2 * alpha_s[1] * alpha_s[2] * C[:, 1, :, :] / ne
Uey = (1 / jnp.sqrt(2)) * alpha_s[0] * alpha_s[1]**2 * alpha_s[2] * C[:, Nn, :, :] / ne
Uix = (1 / jnp.sqrt(2)) * alpha_s[3]**2 * alpha_s[4] * alpha_s[5] * C[:, Nn * Nm * Np + 1, :, :] / ni
Uiy = (1 / jnp.sqrt(2)) * alpha_s[3] * alpha_s[4]**2 * alpha_s[5] * C[:, Nn * Nm * Np + Nn, :, :] / ni

Ue_perp = jnp.stack([Uex, Uey], axis=1)
Ui_perp = jnp.stack([Uix, Uiy], axis=1)

# EM Energies. Size (Nt, 6)
U_EM = 0.5 * input_parameters["Omega_ce"]**2 * jnp.sum(jnp.abs(F) ** 2 / ((1 + 2 * basis_idx[:, 0]) * (1 + 2 * basis_idx[:, 1])), axis=(-3, -2, -1)) / (Nx * Ny)
U_E, U_B = jnp.sum(U_EM[:, :3], axis=-1), jnp.sum(U_EM[:, 3:], axis=-1) # Size Nt
U_EM_tot = U_E + U_B

# Single fluid MHD variables and alignment
rho = (ne + ni * input_parameters["ms"][1] / input_parameters["ms"][0])[:, None, :, :]
U_perp = (Ue_perp * ne[:, None, :, :] + Ui_perp * ni[:, None, :, :] * input_parameters["ms"][1] / input_parameters["ms"][0]) / rho
VA_perp = input_parameters["Omega_ce"] * F[:, 3:5, :, :, 0] / jnp.sqrt(rho)

cross_helicity = jnp.sum(rho * VA_perp * U_perp, axis=(-3, -2, -1)) / (Nx * Ny)
total_fluctuation_energy = jnp.sum(rho * (U_perp**2 + VA_perp**2), axis=(-3, -2, -1)) / (Nx * Ny)
alignment_corr = cross_helicity / total_fluctuation_energy

plot_energy_timeseries(output, savepath="OT_parallel_energy.png")
plot_relative_energy_error(output, savepath="OT_parallel_energy_error.png")
plot_Jz_slice(output, input_parameters, solver_parameters, t_query=output["time"][-1], savepath="OT_parallel_Jz_slice.png")
animate_Jz(output, input_parameters, solver_parameters, tmax=None, out_gif="OT_parallel_Jz.gif")
