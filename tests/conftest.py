"""Expose four logical CPU devices before the JAX backend initializes.

Tests that need another device count, or that exercise bootstrap behaviour, run in fresh subprocesses.
"""
import jax

jax.config.update("jax_num_cpu_devices", 4)
jax.config.update("jax_enable_x64", True)
