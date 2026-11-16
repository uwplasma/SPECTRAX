"""SPECTRAX: a Hermite–Fourier spectral Vlasov–Maxwell solver.

This package re-exports the main user-facing functions from internal modules for
convenient access (e.g. ``from spectrax import simulation, load_parameters``).
"""

import os
from pathlib import Path

# Persistent compilation cache. JAX reads these variables when it is first
# imported (during the imports below); values already set by the user win.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/spectrax/jax"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

from .version import __version__

from ._diagnostics import *
from ._initialization import *
from ._initialize_maxwellian import *
from ._inverse_transform import *
from ._model import *
from ._plot import *
from ._simulation import *
