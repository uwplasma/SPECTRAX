import os
number_of_processors_to_use = 2 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
from jax import block_until_ready
import jax.numpy as jnp
from _simulation import simulation
from _initialization import load_parameters
from _plot import plot
from _initialize_xv import initialize_xv

# Read from input.toml
input_parameters, solver_parameters = load_parameters('input_orszag.toml')

print('Setting up simulation parameters...')
deltaB = 0.2 # In-plane magnetic field amplitude. 
U0 = deltaB * input_parameters["Omega_cs"][0] / jnp.sqrt(input_parameters["mi_me"])
kx = input_parameters["kx"]
ky = input_parameters["ky"]

# Electron and ion fluid velocities.
Ue = lambda x, y, z: U0 * jnp.array([-jnp.sin(ky * y), jnp.sin(kx * x), -deltaB * input_parameters["Omega_cs"][0] * (2 * kx * jnp.cos(2 * kx * x) + ky * jnp.cos(ky * y))])
Ui = lambda x, y, z: U0 * jnp.array([-jnp.sin(ky * y), jnp.sin(kx * x), jnp.zeros_like(x)])

# Magnetic and electric fields.
B = lambda x, y, z: jnp.array([-deltaB * jnp.sin(ky * y), deltaB * jnp.sin(2 * kx * x), jnp.ones_like(x)])
E = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)])

# Electron and ion distribution functions.
fe = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * input_parameters["vte"] ** 3) * 
                                jnp.exp(-((vx - Ue(x, y, z)[0])**2 + (vy - Ue(x, y, z)[1])**2 + (vz - Ue(x, y, z)[2])**2) / (2 * input_parameters["vte"] ** 2))))
fi = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * input_parameters["vti"] ** 3) * 
                                jnp.exp(-((vx - Ui(x, y, z)[0])**2 + (vy - Ui(x, y, z)[1])**2 + (vz - Ui(x, y, z)[2])**2) / (2 * input_parameters["vti"] ** 2))))

start_time = time()
input_parameters["Ck_0"], input_parameters["Fk_0"] =  block_until_ready(initialize_xv(B, E, fe, fi, input_parameters, **solver_parameters))


print('Simulation parameters set up successfully.')
print('Starting simulation...')
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

print('Simulation completed successfully.')
print('Plotting results...')
plot(output)