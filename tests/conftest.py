import os
from pathlib import Path

# Reuse compiled XLA programs across test runs (honours a user-set directory).
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/spectrax/jax-tests"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.1")
