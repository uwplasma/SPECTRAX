import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
number_of_processors_to_use = 5
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
from jax import block_until_ready
from spectrax import simulation, load_parameters, plot

toml_file = os.path.join(os.path.dirname(__file__), 'input_1D_landau_damping.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

alpha_s = input_parameters["alpha_s"]
nx = input_parameters["nx"]
Lx = input_parameters["Lx"]
Omega_cs = input_parameters["Omega_cs"]
dn = input_parameters["dn1"]
Nx = solver_parameters["Nx"]
Nn = solver_parameters["Nn"]
Ns = solver_parameters["Ns"]

# Fourier components of magnetic and electric fields.
Fk_0 = jnp.zeros((6, 1, Nx, 1), dtype=jnp.complex128)
Fk_0 = Fk_0.at[0, 0, int((Nx-1)/2-nx), 0].set(dn * Lx / (4 * jnp.pi * nx * Omega_cs[0]))
Fk_0 = Fk_0.at[0, 0, int((Nx-1)/2+nx), 0].set(dn * Lx / (4 * jnp.pi * nx * Omega_cs[0]))
input_parameters["Fk_0"] = Fk_0

ncps_1d = jnp.array(Nn[:Ns])
total_c = jnp.sum(ncps_1d)
offsets = jnp.cumsum(jnp.concatenate([jnp.array([0]), ncps_1d[:-1]]))

# Hermite-Fourier components of electron and ion distribution functions.
Ce0_mk = 0 + 1j * (1 / (2 * alpha_s[0] ** 3)) * dn
Ce0_0 = 1 / (alpha_s[0] ** 3) + 0 * 1j
Ce0_k = 0 - 1j * (1 / (2 * alpha_s[0] ** 3)) * dn
Ci0_0 = 1 / (alpha_s[3] ** 3) + 0 * 1j
Ck_0 = jnp.zeros((total_c, 1, Nx, 1), dtype=jnp.complex128)
electron_offset = offsets[0]
Ck_0 = Ck_0.at[electron_offset, 0, int((Nx-1)/2-1), 0].set(Ce0_mk)
Ck_0 = Ck_0.at[electron_offset, 0, int((Nx-1)/2), 0].set(Ce0_0)
Ck_0 = Ck_0.at[electron_offset, 0, int((Nx-1)/2+1), 0].set(Ce0_k)
ion_offset = offsets[1]
Ck_0 = Ck_0.at[ion_offset, 0, int((Nx-1)/2), 0].set(Ci0_0)

input_parameters["Ck_0"] = Ck_0

# Simulate
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

# Plot results
plot(output)

# print('Saving results...')
# jnp.savez('output_landau-damping.npz', **output)
