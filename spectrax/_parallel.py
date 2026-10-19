"""Device mesh and sharding layout for splitting a SPECTRAX simulation across several devices.

The mesh has four named axes: ``'s'`` (species) and ``'z'``, ``'y'``, ``'x'`` (spatial elements). Each defaults
to size 1, so a mesh of all ones describes a single-device run.
"""

import jax
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

__all__ = ["setup_cpu_devices", "make_mesh", "check_mesh"]

# Devices are numbered fastest along the last axis, so neighbouring x blocks (the most common split) sit on
# consecutive devices.
MESH_AXES = ("s", "z", "y", "x")

# `Hermite_DG_system` arguments with a leading species axis; these are split over 's', all others are replicated.
SPECIES_SHARDED_ARGS = ("alpha_s", "u_s", "ms", "qs", "Ax_p", "Ax_m", "Ay_p", "Ay_m", "Az_p", "Az_m")

def setup_cpu_devices(n):
    """
    Expose `n` virtual devices on the CPU, so that a mesh can be built on a single machine.

    This must be called before JAX runs any computation in the process (creating an array counts).
    Setting the environment variable ``JAX_NUM_CPU_DEVICES=n`` before starting Python is equivalent.

    Parameters
    ----------
    n : int
        Number of CPU devices.
    """
    try:
        jax.config.update("jax_num_cpu_devices", n)
    except RuntimeError as error:
        raise RuntimeError(
            f"Cannot switch to {n} CPU devices because JAX has already run a computation in this process. "
            "Call setup_cpu_devices (or load_parameters with `cpu_devices` in its [parallel] table) before any "
            f"other JAX operation, or set the environment variable JAX_NUM_CPU_DEVICES={n} before starting Python."
        ) from error

def make_mesh(species=1, x=1, y=1, z=1, devices=None):
    """
    Build the device mesh used to split a simulation across devices.

    Parameters
    ----------
    species, x, y, z : int, optional
        Number of devices along the species axis and along the x, y and z element axes. The mesh uses
        `species * x * y * z` devices.
    devices : sequence of jax.Device, optional
        Devices to place on the mesh. Defaults to the devices of JAX's default backend.

    Returns
    -------
    jax.sharding.Mesh
        Mesh with axes ``('s', 'z', 'y', 'x')``.
    """
    available = jax.devices() if devices is None else list(devices)
    n_devices = species * x * y * z
    if n_devices > len(available):
        raise ValueError(
            f"A mesh with species={species}, x={x}, y={y}, z={z} needs {n_devices} devices, but only {len(available)} "
            "are available. On a CPU-only machine, more devices can be exposed with setup_cpu_devices."
        )
    # Auto axes let XLA propagate shardings through operations that carry no sharding annotation.
    return jax.make_mesh((species, z, y, x), MESH_AXES, devices=available[:n_devices],
                         axis_types=(AxisType.Auto,) * len(MESH_AXES))

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
