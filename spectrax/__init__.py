"""SPECTRAX: a Hermite–Fourier spectral Vlasov–Maxwell solver.

This package re-exports the main user-facing functions from internal modules for
convenient access (e.g. ``from spectrax import simulation, load_parameters``).
"""

from .version import __version__

from ._diagnostics import *
from ._initialization import *
from ._initialize_maxwellian import *
from ._inverse_transform import *
from ._model import *
from ._plot import *
from ._simulation import *
