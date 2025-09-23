import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
number_of_processors_to_use = 4 # Parallelization, this should divide total resolution
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
from jax import block_until_ready
from spectrax import simulation, load_parameters, plot

# Read from input.toml
# input_parameters, solver_parameters = load_parameters('input_1D_two_stream.toml')
toml_file = os.path.join(os.path.dirname(__file__), 'input_1D_two_stream.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

alpha_s = input_parameters["alpha_s"]
nx = input_parameters["nx"]
Lx = input_parameters["Lx"]
Omega_cs = input_parameters["Omega_cs"]
dn1 = input_parameters["dn1"]
dn2 = input_parameters["dn2"]
Nx = solver_parameters["Nx"]
Nn = solver_parameters["Nn"]

# Initialize distribution function as a two-stream instability
indices = jnp.array([int((Nx-1)/2-nx), int((Nx-1)/2+nx)])
values  = (dn1 + dn2) * Lx / (4 * jnp.pi * nx * Omega_cs[0])
Fk_0    = jnp.zeros((6, 1, Nx, 1), dtype=jnp.complex128).at[0, 0, indices, 0].set(values)
input_parameters["Fk_0"] = Fk_0

C10     = jnp.array([
        0 + 1j * (1 / (2 * alpha_s[0] ** 3)) * dn1,
        1 / (alpha_s[0] ** 3) + 0 * 1j,
        0 - 1j * (1 / (2 * alpha_s[0] ** 3)) * dn1
])
C20     = jnp.array([
        0 + 1j * (1 / (2 * alpha_s[3] ** 3)) * dn2,
        1 / (alpha_s[3] ** 3) + 0 * 1j,
        0 - 1j * (1 / (2 * alpha_s[3] ** 3)) * dn2
])
indices = jnp.array([int((Nx-1)/2-nx), int((Nx-1)/2), int((Nx-1)/2+nx)])
Ck_0    = jnp.zeros((2 * Nn, 1, Nx, 1), dtype=jnp.complex128)
Ck_0    = Ck_0.at[0,  0, indices, 0].set(C10)
Ck_0    = Ck_0.at[Nn, 0, indices, 0].set(C20)
input_parameters["Ck_0"] = Ck_0

# Simulate
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

# Plot results
plot(output)

# print('Saving results...')
# jnp.savez('output_two-stream.npz', **output)

# print("Loading results...")
# output = jnp.load('output_orszag.npz')