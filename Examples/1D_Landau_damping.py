"""Example: 1D Landau damping.

Runs a 1D Landau damping setup defined in ``input_1D_landau_damping.toml`` and
plots standard diagnostics via ``spectrax.plot``.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from time import time
import jax.numpy as jnp
from jax import block_until_ready
from spectrax import simulation, load_parameters, plot

# Read from input.toml
toml_file = os.path.join(os.path.dirname(__file__), 'input_1D_landau_damping.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

alpha_s = input_parameters["alpha_s"]
nx = input_parameters["nx"]
ny = input_parameters["ny"]
nz = input_parameters["nz"]
Lx = input_parameters["Lx"]
Ly = input_parameters["Ly"]
Lz = input_parameters["Lz"]
Omega_cs = input_parameters["Omega_cs"]
dn = input_parameters["dn1"]
Nx = solver_parameters["Nx"]
Ny = solver_parameters["Ny"]
Nz = solver_parameters["Nz"]
Nn = solver_parameters["Nn"]
Nm = solver_parameters["Nm"]
Np = solver_parameters["Np"]
Nt = solver_parameters["timesteps"]

n = nx + ny + nz
L = Lx * jnp.sign(nx) + Ly * jnp.sign(ny) + Lz * jnp.sign(nz)
E_field_component = int(jnp.sign(ny) + 2 * jnp.sign(nz))

# Fourier components of magnetic and electric fields.
Fk_0 = jnp.zeros((6, Ny, Nx//2+1, Nz), dtype=jnp.complex128)
Fk_0 = Fk_0.at[E_field_component, ny, nx, nz].set(dn * L / (4 * jnp.pi * n * Omega_cs[0]))

# Hermite-Fourier components of electron and ion distribution functions.
Ce0_mk, Ce0_0, Ce0_k = 0 + 1j * (1 / (2 * alpha_s[0] ** 3)) * dn, 1 / (alpha_s[0] ** 3) + 0 * 1j, 0 - 1j * (1 / (2 * alpha_s[0] ** 3)) * dn
Ci0_0 = 1 / (alpha_s[3] ** 3) + 0 * 1j
Ck_0 = jnp.zeros((2 * Nn * Nm * Np, Ny, Nx//2+1, Nz), dtype=jnp.complex128)
if nx == 0 and (ny != 0 or nz != 0):
    Ck_0 = Ck_0.at[0, -ny, 0, -nz].set(Ce0_mk) # The negative nx, ny, nz index only exists if nx=0
Ck_0 = Ck_0.at[0, 0, 0, 0].set(Ce0_0)
Ck_0 = Ck_0.at[0, ny, nx, nz].set(Ce0_k)
Ck_0 = Ck_0.at[Nn * Nm * Np, 0, 0, 0].set(Ci0_0)
input_parameters["Ck_0"] = Ck_0

# Simulate
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

# Plot results
plot(output)

# print('Saving results...')
# jnp.savez('output_landau-damping.npz', **output)
