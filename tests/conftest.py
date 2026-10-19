import jax

# Virtual CPU devices for the parallel tests. This has to happen before any JAX computation runs.
jax.config.update("jax_num_cpu_devices", 8)
