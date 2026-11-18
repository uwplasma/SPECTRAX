"""SPECTRAX: a Hermite–Fourier spectral Vlasov–Maxwell solver.

This package re-exports the main user-facing functions from internal modules for
convenient access (e.g. ``from spectrax import simulation, load_parameters``).
"""

import os
from pathlib import Path

import jax

# Persistent compilation cache. Set through jax.config (not only the environment)
# so it also applies when jax was imported before spectrax; user-set
# JAX_COMPILATION_CACHE_DIR / JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS win.
jax.config.update(
    "jax_compilation_cache_dir",
    os.environ.get("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/spectrax/jax")),
)
jax.config.update(
    "jax_persistent_cache_min_compile_time_secs",
    float(os.environ.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")),
)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

from .version import __version__

from ._diagnostics import *
from ._initialization import *
from ._initialize_maxwellian import *
from ._inverse_transform import *
from ._model import *
from ._plot import *
from ._simulation import *
