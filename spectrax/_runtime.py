"""Parallel runtime configuration: validation, CPU/GPU device selection and meshes (JAX-free at import).

One Python process drives ``num_devices`` LOCAL devices arranged as a mesh ``("species","x","y","z")`` of shape
``(ps,px,py,pz)``.  ``spectrax.load_parameters`` calls :func:`read_input`, which bootstraps JAX before any array exists.
"""
import operator
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import cache
from itertools import product
import tomllib

__all__ = ["ParallelConfig", "parallel_config", "bootstrap", "select_devices", "make_mesh", "named_sharding", "read_input"]

@dataclass(frozen=True)
class ParallelConfig:
    """Resolved settings: ``device`` in {"cpu","gpu","auto"}, TOTAL ``num_devices``, ``species_partitions`` (ps) and the
    spatial factors ``mesh_shape=(px,py,pz)``, with ``num_devices == ps*px*py*pz``."""
    device: str = "auto"
    num_devices: int = 1
    species_partitions: int = 1
    mesh_shape: tuple = (1, 1, 1)

    @property
    def shape(self):
        return (self.species_partitions, *self.mesh_shape)

def _count(name, v):
    try:
        if isinstance(v, bool):  # bool is an int subclass but never a valid count
            raise TypeError
        v = operator.index(v)  # Python and NumPy integers
    except TypeError:
        raise TypeError(f"parallel.{name} must be an integer, got {v!r} ({type(v).__name__})") from None
    if v < 1:
        raise ValueError(f"parallel.{name} must be positive, got {v}")
    return v

def parallel_config(parallel=None, Nx=33, Ny=1, Nz=1, Ns=2):
    """Validate ``parallel`` (None, a ``[parallel]`` mapping or a ParallelConfig) against the grid and species count.

    Requires ``num_devices = ps*px*py*pz``, ``ps | Ns`` (ps <= Ns), ``px|Nx``, ``py|Ny``, ``pz|Nz``.  An omitted
    ``mesh_shape`` is chosen among the valid triples with product ``num_devices/ps`` by minimizing
    ``sum(p/N for p>1)`` (boundary traffic); ties go to fewer split directions, then to splitting x before y before z.
    Species sharding is never enabled automatically.  Raises ValueError/TypeError naming the failed rule.
    """
    parallel = asdict(parallel) if isinstance(parallel, ParallelConfig) else {} if parallel is None else parallel
    if not isinstance(parallel, Mapping):
        raise TypeError(f"parallel must be a [parallel] table (mapping) or ParallelConfig, got {parallel!r}")
    parallel = dict(parallel)
    if unknown := set(parallel) - {"device", "num_devices", "species_partitions", "mesh_shape"}:
        raise ValueError(f"Unknown [parallel] keys {sorted(unknown)}; allowed: device, num_devices, species_partitions, mesh_shape")
    device = parallel.get("device", "auto")
    if device not in ("cpu", "gpu", "auto"):
        raise ValueError(f"parallel.device must be 'cpu', 'gpu' or 'auto', got {device!r}")
    n = _count("num_devices", parallel.get("num_devices", 1))
    ps = _count("species_partitions", parallel.get("species_partitions", 1))
    if ps > Ns or Ns % ps:
        raise ValueError(f"species_partitions={ps} must divide Ns={Ns} (and not exceed it)")
    if n % ps:
        raise ValueError(f"num_devices={n} is not divisible by species_partitions={ps}")
    grid, mesh = (Nx, Ny, Nz), parallel.get("mesh_shape")
    if mesh is None:
        cands = [p for p in product(*[[d for d in range(1, n // ps + 1) if N % d == 0] for N in grid])
                 if p[0] * p[1] * p[2] == n // ps]
        if not cands:
            raise ValueError(f"No valid mesh_shape: need px*py*pz=num_devices/species_partitions={n // ps} with px|Nx, py|Ny, "
                             f"pz|Nz for (Nx,Ny,Nz)={grid}")
        mesh = min(cands, key=lambda p: (sum(Fraction(q, N) for q, N in zip(p, grid) if q > 1), sum(q > 1 for q in p), [-q for q in p]))
    else:
        if isinstance(mesh, (str, bytes)) or not hasattr(mesh, "__len__") or len(mesh) != 3:
            raise ValueError(f"parallel.mesh_shape must have 3 entries [px,py,pz], got {mesh!r}")
        mesh = tuple(_count("mesh_shape entries", q) for q in mesh)
        if mesh[0] * mesh[1] * mesh[2] * ps != n:
            raise ValueError(f"num_devices={n} must equal species_partitions*px*py*pz = {ps}*{mesh[0]}*{mesh[1]}*{mesh[2]}")
        for q, N, ax in zip(mesh, grid, "xyz"):
            if N % q:
                raise ValueError(f"mesh_shape factor p{ax}={q} must divide N{ax}={N}" + (" (singleton axes require 1)" if N == 1 else ""))
    return ParallelConfig(device, n, ps, tuple(mesh))

def bootstrap(parallel=None, Nx=33, Ny=1, Nz=1, Ns=2):
    """Validate ``parallel`` and, for "cpu"/"auto", make at least ``num_devices`` logical CPU devices available.

    Must run before the JAX backend initializes (``load_parameters`` does).  Never lowers a larger
    ``jax_num_cpu_devices`` and leaves XLA_FLAGS alone; if the backend already started, :func:`select_devices` reports.
    """
    cfg = parallel_config(parallel, Nx, Ny, Nz, Ns)
    if cfg.device != "gpu" and cfg.num_devices > 1 and "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
        import jax
        if jax.config.jax_num_cpu_devices < cfg.num_devices:
            try:
                jax.config.update("jax_num_cpu_devices", cfg.num_devices)
            except RuntimeError:  # backend already initialized
                pass
    return cfg

def select_devices(cfg):
    """Exactly ``cfg.num_devices`` local devices of the requested backend ("auto": JAX's preferred one); never falls back."""
    import jax
    try:
        devices = jax.local_devices() if cfg.device == "auto" else jax.local_devices(backend=cfg.device)
    except RuntimeError as e:
        raise RuntimeError(f"parallel.device={cfg.device!r} is not available on this host (no fallback is made): {e}") from e
    if len(devices) < cfg.num_devices:
        hint = (" The CPU backend was initialized before the device count could be set: start a fresh Python process and call "
                "spectrax.load_parameters() (or spectrax.bootstrap()) before any JAX computation." if devices[0].platform == "cpu" else "")
        raise RuntimeError(f"parallel.num_devices={cfg.num_devices} requested but only {len(devices)} local "
                           f"{devices[0].platform} device(s) available.{hint}")
    return devices[:cfg.num_devices]

@cache
def make_mesh(cfg):
    """Cached 4-axis ``Mesh`` named ("species","x","y","z") with shape ``cfg.shape`` and AxisType.Auto."""
    import jax
    return jax.make_mesh(cfg.shape, ("species", "x", "y", "z"), axis_types=(jax.sharding.AxisType.Auto,) * 4,
                         devices=select_devices(cfg))

def named_sharding(mesh, ndim, species=False, lead=0, tail=1):
    """Sharding of an ``ndim``-array laid out ``[lead..., species slot, ..., Ny, Nx, Nz, tail...]``: the spatial axes map to
    mesh axes y,x,z and the species slot ("species" if ``species``, else replicated) exists only when
    ``ndim > lead+3+tail``; everything else is replicated."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    if ndim < lead + 3 + tail:
        raise ValueError(f"ndim={ndim} too small for lead={lead}, tail={tail} (need >= {lead + 3 + tail})")
    slot = ndim > lead + 3 + tail
    if species and not slot:
        raise ValueError(f"species=True needs a species axis: ndim={ndim} must exceed lead+3+tail={lead + 3 + tail}")
    return NamedSharding(mesh, P(*[None] * lead, *(["species" if species else None] if slot else []),
                                 *[None] * max(ndim - lead - tail - 4, 0), "y", "x", "z", *[None] * tail))

def read_input(path):
    """Read a run TOML: returns ``(input_parameters, solver_parameters)``; ``solver_parameters["parallel"]`` is the
    bootstrapped ParallelConfig (always present).  The ``solver`` string is left for ``load_parameters`` to resolve."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    input_parameters, solver_parameters = data["input_parameters"], data["solver_parameters"]
    grid = {k: solver_parameters[k] for k in ("Nx", "Ny", "Nz", "Ns") if k in solver_parameters}
    solver_parameters["parallel"] = bootstrap(data.get("parallel"), **grid)
    return input_parameters, solver_parameters
