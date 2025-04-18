import os
number_of_processors_to_use = 5 # Parallelization, this should divide ntheta*nphi
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
from jax import block_until_ready
from _simulation import simulation
from _initialization import load_parameters
from _plot import plot

# Read from input.toml
input_parameters, solver_parameters = load_parameters('input.toml')

# Simulate
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

# Plot results
plot(output)