"""Device selection, device mesh and sharding layout for splitting a SPECTRAX simulation across devices.

A run is set up in two steps: `setup_devices` chooses the device type (``"cpu"`` or ``"gpu"``) and how many
devices to use, and `make_mesh` arranges those devices into a mesh with four named axes: ``'s'`` (species) and
``'z'``, ``'y'``, ``'x'`` (spatial elements). Each axis defaults to size 1.

GPUs are limited to those present in the machine (and need a GPU-enabled JAX, e.g. ``pip install
"spectrax[gpu]"`` on Linux with NVIDIA GPUs). A CPU can be split into any number of virtual devices; these
share the same cores, so they are mainly useful for testing a parallel setup.
"""

import jax
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

__all__ = ["setup_devices", "make_mesh", "check_mesh"]

DEVICE_TYPES = ("cpu", "gpu")

# Devices are numbered fastest along the last axis, so neighbouring x blocks (the most common split) sit on
# consecutive devices.
MESH_AXES = ("s", "z", "y", "x")

# `Hermite_DG_system` arguments with a leading species axis; these are split over 's', all others are replicated.
SPECIES_SHARDED_ARGS = ("alpha_s", "u_s", "ms", "qs", "Ax_p", "Ax_m", "Ay_p", "Ay_m", "Az_p", "Az_m")

def setup_devices(device_type, num_devices):
    """
    Choose the devices a simulation runs on, and make the first of them JAX's default device.

    Call this before any other JAX computation in the process (creating an array counts): splitting a CPU into
    several virtual devices is only possible before JAX has started.

    Parameters
    ----------
    device_type : {"cpu", "gpu"}
        Type of device to run on.
    num_devices : int
        Number of devices to use. On a CPU this many virtual devices are created (or, once JAX has started,
        taken from those already created); for GPUs, at least this many must be present.

    Returns
    -------
    list of jax.Device
        The `num_devices` devices to build a mesh from (see `make_mesh`).
    """
    if device_type not in DEVICE_TYPES:
        raise ValueError(f"device_type must be one of {DEVICE_TYPES}, not {device_type!r}.")
    if device_type == "cpu":
        try:
            jax.config.update("jax_num_cpu_devices", num_devices)
        except RuntimeError as error:
            # JAX has started, so the CPU devices are fixed; using fewer of them than exist is still fine.
            if len(jax.devices("cpu")) < num_devices:
                raise RuntimeError(
                    f"Cannot create {num_devices} CPU devices because JAX has already run a computation in this "
                    "process. Call setup_devices (or load_parameters with a [parallel] table) before any other JAX "
                    "operation."
                ) from error
    try:
        available = jax.devices(device_type)
    except RuntimeError as error:
        raise RuntimeError(f"No {device_type.upper()} devices are available to JAX on this machine.") from error
    if num_devices > len(available):
        raise ValueError(f"{num_devices} {device_type.upper()} devices were requested, but only {len(available)} are available.")
    jax.config.update("jax_default_device", available[0])
    return available[:num_devices]

def make_mesh(devices, species=1, x=1, y=1, z=1):
    """
    Arrange devices into the mesh used to split a simulation.

    Parameters
    ----------
    devices : sequence of jax.Device
        Devices to place on the mesh, as returned by `setup_devices`.
    species, x, y, z : int, optional
        Number of devices along the species axis and along the x, y and z element axes. Their product must
        equal the number of `devices`.

    Returns
    -------
    jax.sharding.Mesh
        Mesh with axes ``('s', 'z', 'y', 'x')``.
    """
    devices = list(devices)
    if species * x * y * z != len(devices):
        raise ValueError(f"The split species={species}, x={x}, y={y}, z={z} uses {species * x * y * z} devices, "
                         f"but {len(devices)} devices were given; the two must match.")
    # Auto axes let XLA propagate shardings through operations that carry no sharding annotation.
    return jax.make_mesh((species, z, y, x), MESH_AXES, devices=devices, axis_types=(AxisType.Auto,) * len(MESH_AXES))

def check_mesh(mesh, Nx, Ny, Nz, Ns):
    """
    Check that a simulation can be split evenly over `mesh`, raising ValueError if not.

    Parameters
    ----------
    mesh : jax.sharding.Mesh
        Mesh built by `make_mesh`.
    Nx, Ny, Nz : int
        Number of elements along x, y and z.
    Ns : int
        Number of species.
    """
    if tuple(mesh.axis_names) != MESH_AXES:
        raise ValueError(f"The mesh must have axes {MESH_AXES} (build it with make_mesh), but has {tuple(mesh.axis_names)}.")
    for axis, label, count in (("s", "Ns (number of species)", Ns), ("x", "Nx (elements along x)", Nx),
                               ("y", "Ny (elements along y)", Ny), ("z", "Nz (elements along z)", Nz)):
        if count % mesh.shape[axis]:
            raise ValueError(f"{label} = {count} cannot be split evenly over the {mesh.shape[axis]} devices of mesh axis '{axis}'.")

def state_specs():
    """Partition specs for the state ``{"Ck": (Ns, Np, Nm, Nn, Ny, Nx, Nz, Nl), "Fk": (6, Ny, Nx, Nz, Nl)}``."""
    return {"Ck": P("s", None, None, None, "y", "x", "z"), "Fk": P(None, "y", "x", "z")}

def args_specs(args):
    """Partition specs matching the `Hermite_DG_system` keyword arguments `args` built in `simulation`."""
    return {name: P("s") if name in SPECIES_SHARDED_ARGS else P() for name in args}

def _constrain(tree, specs, mesh):
    """Apply a sharding constraint to each entry of the flat dict `tree`, using the matching entry of `specs`."""
    return jax.lax.with_sharding_constraint(tree, {name: NamedSharding(mesh, spec) for name, spec in specs.items()})

def shard_state(state, mesh):
    """Inside `jit`, lay out the state over `mesh` according to `state_specs`; unchanged if `mesh` is None."""
    return state if mesh is None else _constrain(state, state_specs(), mesh)

def shard_args(args, mesh):
    """Inside `jit`, lay out the `Hermite_DG_system` arguments over `mesh` according to `args_specs`; unchanged if `mesh` is None."""
    return args if mesh is None else _constrain(args, args_specs(args), mesh)

def shard_solution(solution, mesh):
    """
    Inside `jit`, keep the saved solution distributed over `mesh`, as `simulation` returns it: ``"Ck"`` with shape
    `(Nt, Ns * Np * Nm * Nn, Ny, Nx, Nz, Nl)` and ``"Fk"`` with shape `(Nt, 6, Ny, Nx, Nz, Nl)`. Unchanged if `mesh` is None.
    """
    specs = {"Ck": P(None, "s", "y", "x", "z"), "Fk": P(None, None, "y", "x", "z")}
    return solution if mesh is None else _constrain(solution, specs, mesh)
