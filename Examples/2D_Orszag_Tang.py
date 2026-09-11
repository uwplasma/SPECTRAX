"""Example: 2D Orszag–Tang vortex.

Loads the 2D Orszag–Tang setup from ``input_2D_orszag_tang.toml``, constructs
initial distribution-function moments consistent with the prescribed fluid
velocity and electromagnetic fields, runs the simulation, and generates plots
via ``orszag_tang_data_analysis.py``.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from time import time
from jax import block_until_ready, config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from spectrax import simulation, load_parameters, compute_C_nmp
from jax.numpy.fft import fftn, fftshift
from orszag_tang_data_analysis import (
        plot_energy_timeseries, plot_relative_energy_error,
        plot_Jz_slice, animate_Jz
    )

# Read from input.toml
toml_file = os.path.join(os.path.dirname(__file__), 'input_2D_orszag_tang.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

print('Setting up simulation parameters...')
deltaB = 0.2 # In-plane magnetic field amplitude. 
U0 = deltaB * input_parameters["Omega_cs"][0] / jnp.sqrt(input_parameters["mi_me"])

kx = 2 * jnp.pi / input_parameters["Lx"]
ky = 2 * jnp.pi / input_parameters["Ly"]
mi_me = input_parameters["mi_me"]
Nx = solver_parameters["Nx"]
Ny = solver_parameters["Ny"]
Nz = solver_parameters["Nz"]
Nn = solver_parameters["Nn"]
Nm = solver_parameters["Nm"]
Np = solver_parameters["Np"]
Ns = solver_parameters["Ns"]
alpha_s = input_parameters["alpha_s"]
u_s = input_parameters["u_s"]

# Electron and ion fluid velocities.
Ue = lambda x, y, z: jnp.array([- U0 * jnp.sin(ky * y), U0 * jnp.sin(kx * x), -deltaB * input_parameters["Omega_cs"][0] * (2 * kx * jnp.cos(2 * kx * x) + ky * jnp.cos(ky * y))])
Ui = lambda x, y, z: jnp.array([- U0 * jnp.sin(ky * y), U0 * jnp.sin(kx * x), jnp.zeros_like(x)])
# Magnetic and electric fields.
B = lambda x, y, z: jnp.array([-deltaB * jnp.sin(ky * y), deltaB * jnp.sin(2 * kx * x), jnp.ones_like(x)])
E = lambda x, y, z: jnp.array([jnp.zeros_like(x), 
                               jnp.zeros_like(x), 
                               jnp.zeros_like(x)])

x = jnp.linspace(0, input_parameters["Lx"], Nx)
y = jnp.linspace(0, input_parameters["Ly"], Ny)
z = jnp.linspace(0, input_parameters["Lz"], Nz)
X, Y, Z = jnp.meshgrid(x, y, z, indexing='xy')

Us_grid = jnp.stack([Ue(X, Y, Z), Ui(X, Y, Z)], axis=0)  # shape (Ns, 3, Ny, Nx, Nz)


input_parameters["Ck_0"] = compute_C_nmp(Us_grid, alpha_s, u_s, Nn, Nm, Np, Ns)

B_grid = B(X, Y, Z)  # shape (3, Ny, Nx, Nz)
E_grid = E(X, Y, Z)  # shape (3, Ny, Nx, Nz)
F_grid = jnp.concatenate((E_grid, B_grid), axis=0)  # shape (6, Ny, Nx, Nz)
input_parameters["Fk_0"] = fftshift(fftn(F_grid, axes=(-3, -2, -1), norm="forward"), axes=(-3, -2, -1))  # shape (6, Ny, Nx, Nz)


print('Starting simulation...')
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

# print('Saving results...')
# jnp.savez('/Users/csvega/Desktop/Madison/Code/Simulations/Orszag_Tang/S17/output_orszag.npz', **output)

# Results.

plot_energy_timeseries(output, savepath="energy.png")
plot_relative_energy_error(output, savepath="energy_error.png")
plot_Jz_slice(output, input_parameters, solver_parameters, t_query=output["time"][-1], savepath="Jz_slice.png")
animate_Jz(output, input_parameters, solver_parameters, tmax=None, out_gif="Jz.gif")
