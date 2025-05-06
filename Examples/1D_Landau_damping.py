import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
number_of_processors_to_use = 5 # Parallelization, this should divide total resolution
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
from jax import block_until_ready
from spectrax import simulation, load_parameters, plot

# Read from input.toml
# input_parameters, solver_parameters = load_parameters('input_1D_landau_damping.toml')
toml_file = os.path.join(os.path.dirname(__file__), 'input_1D_landau_damping.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

alpha_s = input_parameters["alpha_s"]
dn = input_parameters["dn1"]
Nx = solver_parameters["Nx"]
Nn = solver_parameters["Nn"]
kx = input_parameters["kx"]
Omega_cs = input_parameters["Omega_cs"]

# Fourier components of magnetic and electric fields.
Fk_0 = jnp.zeros((6, 1, Nx, 1), dtype=jnp.complex128)
Fk_0 = Fk_0.at[0, 0, int((Nx-1)/2-1), 0].set(dn / (2 * kx * Omega_cs[0]))
Fk_0 = Fk_0.at[0, 0, int((Nx-1)/2+1), 0].set(dn / (2 * kx * Omega_cs[0]))
input_parameters["Fk_0"] = Fk_0

# Hermite-Fourier components of electron and ion distribution functions.
Ce0_mk, Ce0_0, Ce0_k = 0 + 1j * (1 / (2 * alpha_s[0] ** 3)) * dn, 1 / (alpha_s[0] ** 3) + 0 * 1j, 0 - 1j * (1 / (2 * alpha_s[0] ** 3)) * dn
Ci0_0 = 1 / (alpha_s[3] ** 3) + 0 * 1j
Ck_0 = jnp.zeros((2 * Nn, 1, Nx, 1), dtype=jnp.complex128)
Ck_0 = Ck_0.at[0, 0, int((Nx-1)/2-1), 0].set(Ce0_mk)
Ck_0 = Ck_0.at[0, 0, int((Nx-1)/2), 0].set(Ce0_0)
Ck_0 = Ck_0.at[0, 0, int((Nx-1)/2+1), 0].set(Ce0_k)
Ck_0 = Ck_0.at[Nn, 0, int((Nx-1)/2), 0].set(Ci0_0)
input_parameters["Ck_0"] = Ck_0

# Simulate
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

# Plot results
plot(output)

# print('Saving results...')
# jnp.savez('output_landau-damping.npz', **output)