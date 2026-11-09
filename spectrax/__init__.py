"""SPECTRAX: a Hermite–Fourier spectral Vlasov–Maxwell solver.

This package re-exports the main user-facing functions from internal modules for
convenient access (e.g. ``from spectrax import simulation, load_parameters``).
"""

from pathlib import Path
import jax

# Persistent compilation cache. It must be configured before the first compile,
# which happens during the imports below, so it lives here rather than in main().
# A user-set JAX_COMPILATION_CACHE_DIR (or jax.config setting) is left untouched.
if jax.config.jax_compilation_cache_dir is None:
    jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache/spectrax/jax"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.1)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

from .version import __version__

from ._diagnostics import *
from ._initialization import *
from ._initialize_maxwellian import *
from ._inverse_transform import *
from ._model import *
from ._plot import *
from ._simulation import *
