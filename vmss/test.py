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
end_time = time()
print(f"Runtime: {end_time - start_time} seconds")

# Plot results
plot(output)