"""SPECTRAX: a Hermite–Discontinuous_Galerkin spectral Vlasov–Maxwell solver.

This package lazily re-exports the main user-facing names from internal modules (e.g.
``from spectrax import simulation, load_parameters``).  ``import spectrax`` does not import JAX.
"""
from importlib import import_module

from .version import __version__

_EXPORTS = {
    "_initialization": "load_parameters initialize_simulation_parameters construct_idx_array DG_project",
    "_diagnostics": "diagnostics",
    "_initialize_maxwellian": "compute_C_nmp",
    "_inverse_transform": "inverse_HDG_transform",
    "_model": "Hermite_DG_system Hermite_DG_rhs",
    "_plot": "plot",
    "_simulation": "ode_system simulation",
}
_LAZY = {name: module for module, names in _EXPORTS.items() for name in names.split()}
__all__ = ["__version__", *_LAZY]

def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value = getattr(import_module(f"{__name__}.{_LAZY[name]}"), name)
    return value

def __dir__():
    return sorted([*globals(), *_LAZY])
