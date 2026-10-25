"""Time integration driver for the spectral Vlasov–Discontinuous_Galerkin system."""

import inspect
import jax
import jax.numpy as jnp
from jax import jit, config
config.update("jax_enable_x64", True)
from functools import lru_cache, partial
from jax.sharding import NamedSharding, PartitionSpec as P
import equinox as eqx
from diffrax import (diffeqsolve, Dopri8, ODETerm, is_okay,
                     SaveAt, PIDController, TqdmProgressMeter, NoProgressMeter, ConstantStepSize)
from ._runtime import bootstrap, make_mesh, named_sharding
from ._initialization import initialize_simulation_parameters
from ._model import Hermite_DG_system, Hermite_DG_rhs
from ._diagnostics import diagnostics

__all__ = ["ode_system", "simulation", "state_rms_norm", "sharded_rhs"]

def _place(tree, shardings, mesh):
    """device_put concrete leaves on a multi-device mesh; traced leaves (under jax.grad or jit) are resharded by the
    jitted solve. One device needs no placement, and committed arrays break custom JVPs on JAX < 0.7."""
    if mesh.size == 1:
        return tree
    return jax.tree.map(lambda x, s: x if isinstance(x, jax.core.Tracer) else jax.device_put(x, s), tree, shardings)

RHS_ARGS = list(inspect.signature(Hermite_DG_rhs).parameters)[2:-4] # Arrays passed to the RHS by name, between (Ck, Fk) and (Nn, Nm, Np, comm).
SPECIES_ARGS = {"Ax_p", "Ax_m", "Ay_p", "Ay_m", "Az_p", "Az_m", "alpha_s", "u_s", "ms", "qs"} # Species on the first axis.

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'Nl'])
def ode_system(Nx, Ny, Nz, Nn, Nm, Np, Ns, Nl, t, Ck_Fk, args):
    """
    Legacy flat right-hand side for the coupled Vlasov-Maxwell system (not used by `simulation`).

    Parameters
    ----------
    Nx, Ny, Nz : int
        Number of elements per spatial dimension.
    Nn, Nm, Np : int
        Number of Hermite modes per velocity-space dimension.
    Ns : int
        Number of species.
    Nl : int
        Total number of element basis modes.
    t : float
        Integration time (unused but required by Diffrax interface).
    Ck_Fk : jnp.ndarray
        Flattened state vector containing concatenated Hermite coefficients followed
        by electromagnetic field coefficients.
    args : tuple
        Parameter tuple `(Nx, Ny, Nz, Nn, Nm, Np, Ns, Nl, ms, qs, nu, D, Omega_ce, alpha_s, u_s,
        Lx, Ly, Lz, col, sqrt_*, basis_idx, inner_*, di_inner_product, triple_product, A*_p/m, R_p, R_m)`.

    Returns
    -------
    jnp.ndarray
        Flattened derivative vector matching the shape of `Ck_Fk`.
    """

    (ms, qs, nu, D, Omega_ce, alpha_s, u_s, Lx, Ly, Lz, col,
    sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus,
    basis_idx, inner_mm, inner_pm, inner_mp, inner_pp, di_inner_product, triple_product,
    Ax_p, Ax_m, Ay_p, Ay_m, Az_p, Az_m, R_p, R_m
    ) = args[8:]

    total_Ck_size = Nn * Nm * Np * Ns * Nx * Ny * Nz * Nl
    Ck = Ck_Fk[:total_Ck_size].reshape(Nn * Nm * Np * Ns, Ny, Nx, Nz, Nl)
    Fk = Ck_Fk[total_Ck_size:].reshape(6, Ny, Nx, Nz, Nl)

    dy_dt = Hermite_DG_system(Ck, Fk, col, sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus, basis_idx,
                                inner_mm, inner_pm, inner_mp, inner_pp, di_inner_product, triple_product,
                                Ax_p, Ax_m, Ay_p, Ay_m, Az_p, Az_m, R_p, R_m,
                                    Lx, Ly, Lz, nu, D, alpha_s, u_s, ms, qs, Omega_ce, Nn, Nm, Np, Ns)

    return dy_dt

def _sum_dot(x, y):
    """Real part of sum(conj(x) * y) over all leaves, reducing each leaf in its own (possibly sharded) shape."""
    return sum(jnp.sum(jnp.real(jnp.conj(a) * b)) for a, b in zip(jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(y)))

@jax.custom_jvp
def state_rms_norm(error):
    """
    Global RMS norm sqrt(sum_i |e_i|^2 / n) over all leaves of a PyTree, e.g. `(Ck, Fk)`.

    Same value as the default flattened Diffrax/Optimistix RMS norm, but no leaf is raveled
    or concatenated, so sharded leaves need only scalar all-reduces and a field replicated
    across species groups is counted once. The derivative is zero-safe like lineax's `two_norm`.
    """
    count = sum(jnp.size(e) for e in jax.tree_util.tree_leaves(error))
    return jnp.sqrt(_sum_dot(error, error) / count) if count else jnp.array(0.0)

@state_rms_norm.defjvp
def _state_rms_norm_jvp(primals, tangents):
    (error,), (t_error,) = primals, tangents
    out = state_rms_norm(error)
    pred = (out == 0) | jnp.isinf(out) # Zero rather than NaN derivative, as in lineax/optimistix.
    count = max(sum(jnp.size(e) for e in jax.tree_util.tree_leaves(error)), 1)
    return out, jnp.where(pred, 0, _sum_dot(error, t_error) / (count * jnp.where(pred, 1, out)))

def sharded_rhs(mesh, Nn, Nm, Np):
    """`Hermite_DG_rhs` inside `shard_map`: one packed ppermute of the Ck and Fk boundary slabs per sharded axis and
    direction builds the ghost cells, and the species current is a psum. Returns rhs(Ck, Fk, args)."""
    size = dict(zip(mesh.axis_names, mesh.devices.shape))
    spatial = {"y": -4, "x": -3, "z": -2} # Array axis of each spatial mesh axis in Ck and Fk.

    def local(Ck, Fk, args):
        def ghosts(name, ax):
            """Ghost slabs (below, above) of Ck and Fk along `ax`: periodic locally, else from the neighbours."""
            lo = [jax.lax.slice_in_dim(a, a.shape[ax] - 1, a.shape[ax], axis=ax) for a in (Ck, Fk)]
            hi = [jax.lax.slice_in_dim(a, 0, 1, axis=ax) for a in (Ck, Fk)]
            if size[name] == 1:
                return lo, hi
            n = size[name]
            pack = lambda slabs: jnp.concatenate([a.ravel() for a in slabs])
            unpack = lambda buf: [buf[:lo[0].size].reshape(lo[0].shape), buf[lo[0].size:].reshape(lo[1].shape)]
            below = jax.lax.ppermute(pack(lo), name, [(i, (i + 1) % n) for i in range(n)]) # My last slab is my right neighbour's ghost below.
            above = jax.lax.ppermute(pack(hi), name, [(i, (i - 1) % n) for i in range(n)])
            return unpack(below), unpack(above)

        padded = {}
        for name, ax in spatial.items():
            (lc, lf), (hc, hf) = ghosts(name, ax)
            padded[ax] = (jnp.concatenate([lc, Ck, hc], axis=ax), jnp.concatenate([lf, Fk, hf], axis=ax))

        def shift(C, dx, dy, dz):
            """Periodic element shift (`out[i] = C[i + d]`) read from the ghost-padded copy of Ck or Fk."""
            ax, d = next((a, d) for a, d in ((-3, dx), (-4, dy), (-2, dz)) if d)
            Cp = padded[ax][0 if C.ndim == 8 else 1]
            Cp = Cp.reshape(*C.shape[:-4], *Cp.shape[-4:]) # Ck is used with its Hermite axis split as (Np, Nm, Nn).
            return jax.lax.slice_in_dim(Cp, 1 + d, 1 + d + C.shape[ax], axis=ax)

        species_sum = lambda x: jax.lax.psum(jnp.sum(x, axis=0), "species") if size["species"] > 1 else jnp.sum(x, axis=0)
        args = {**args, "Lx": args["Lx"] / size["x"], "Ly": args["Ly"] / size["y"], "Lz": args["Lz"] / size["z"]} # Local box: dx unchanged.
        return Hermite_DG_rhs(Ck, Fk, **args, Nn=Nn, Nm=Nm, Np=Np, comm=(shift, species_sum))

    ck, fk = P("species", None, "y", "x", "z", None), P(None, "y", "x", "z", None)
    return lambda Ck, Fk, args: jax.shard_map(local, mesh=mesh, out_specs=(ck, fk), check_vma=False, # Fk ghosts ride in the Ck buffer, identical per species group.
        in_specs=(ck, fk, {k: P("species") if k in SPECIES_ARGS else P() for k in args}))(Ck, Fk, args)

@lru_cache(maxsize=None)
def _compiled_solve(mesh, solver, adaptive_time_step, timesteps, progress_meter, max_steps, Nn, Nm, Np):
    """Jitted global solve of Y = (Ck, Fk) on `mesh`, cached on static settings so repeated runs reuse it."""
    replicated, by_species = NamedSharding(mesh, P()), NamedSharding(mesh, P("species"))
    in_shardings = ((named_sharding(mesh, 6, species=True), named_sharding(mesh, 5)),
                    {k: by_species if k in SPECIES_ARGS else replicated for k in RHS_ARGS}, replicated, replicated, replicated)
    out_shardings = (named_sharding(mesh, 6, species=True, lead=1), named_sharding(mesh, 6, lead=1), replicated, replicated, replicated)

    rhs = sharded_rhs(mesh, Nn, Nm, Np) if mesh.size > 1 else lambda Ck, Fk, args: Hermite_DG_rhs(Ck, Fk, **args, Nn=Nn, Nm=Nm, Np=Np)

    def solve(y0, args, t_max, dt, tolerance):
        time = jnp.linspace(0, t_max, timesteps)
        stepsize_controller = PIDController(rtol=tolerance, atol=tolerance, norm=state_rms_norm) if adaptive_time_step else ConstantStepSize()
        sol = diffeqsolve(
            ODETerm(lambda t, y, args: rhs(*y, args)), solver=solver,
            stepsize_controller=stepsize_controller, t0=0, t1=t_max, dt0=dt, y0=y0, args=args,
            saveat=SaveAt(ts=time), max_steps=max_steps, progress_meter=progress_meter, throw=False)
        Ck, Fk = sol.ys # Merge only the (sharded) species and (local) Hermite axes for the public history shape.
        return Ck.reshape(timesteps, -1, *Ck.shape[-4:]), Fk, time, sol.result, sol.stats["num_steps"]

    return jit(solve, **(dict(in_shardings=in_shardings, out_shardings=out_shardings) if mesh.size > 1 else {})), in_shardings

def simulation(input_parameters={}, Nx=33, Ny=1, Nz=1, Nn=20, Nm=1, Np=1, Ns=2, N_DG=2, dims=1, timesteps=200, dt = 0.01, solver=Dopri8(), adaptive_time_step=True,
               parallel=None, progress_meter=TqdmProgressMeter(), max_steps=1000000):
    """
    Run a spectral Vlasov-Maxwell simulation and return the solution together with
    the parameter dictionary used to produce it.

    Parameters
    ----------
    input_parameters : dict, optional
        User-specified overrides passed to `initialize_simulation_parameters`.
    Nx, Ny, Nz : int, optional
        Number of elements per spatial dimension.
    Nn, Nm, Np : int, optional
        Number of Hermite modes per velocity-space axis.
    Ns : int, optional
        Number of species.
    N_DG : int, optional
        Number of DG element basis function modes in each direction.
    dims : int, optional
        Spatial dimensionality of the simulation. Number of dimensions in which more than one basis modes are to be used.
    timesteps : int, optional
        Number of solution snapshots to save between `t=0` and `t_max`.
    dt : float, optional
        Initial integration step size.
    solver : diffrax.AbstractSolver, optional
        Diffrax solver instance controlling the time integration.
    adaptive_time_step : bool, optional
        Toggle for using an adaptive time step versus a fixed one.
    parallel : dict or spectrax.ParallelConfig, optional
        Device layout (`device`, `num_devices`, `species_partitions`, `mesh_shape`), as in a
        TOML `[parallel]` section. Default: one device of JAX's preferred backend.
    progress_meter : diffrax.AbstractProgressMeter, optional
        Use `diffrax.NoProgressMeter()` for headless runs and benchmarks.
    max_steps : int, optional
        Maximum number of integration steps.

    Returns
    -------
    dict
        Dictionary containing the evolved coefficients (`Ck`, `Fk`), time samples,
        number of steps, energy diagnostics, and all simulation parameters. `Ck` and `Fk`
        stay distributed over the selected devices.

    Notes
    -----
    This is a host-side wrapper (setup, placement, status check). Called eagerly it synchronizes
    once, on the scalar solver status; under `jax.jit` or `jax.grad` a failed solve raises at run
    time through `eqx.error_if` instead. The state
    Y = (Ck, Fk) is integrated by one global Diffrax solve with Ck of shape
    (Ns, Np*Nm*Nn, Ny, Nx, Nz, Nl) sharded over species and space, and Fk of shape
    (6, Ny, Nx, Nz, Nl) sharded over space and replicated across species groups.
    """
    parallel = bootstrap(parallel, Nx, Ny, Nz, Ns) # Validated device layout (configures CPU devices if JAX is not yet initialized).
    mesh = make_mesh(parallel)
    # Supplied initial states keep their public shapes and go straight to their shards (species prefix, spatial suffix);
    # default ones are projected sharded.
    input_parameters = {**input_parameters, **{k: _place(v, named_sharding(mesh, jnp.ndim(v), species=k == "Ck_0"), mesh)
                                               for k, v in input_parameters.items() if k in ("Ck_0", "Fk_0")}}
    with jax.default_device(mesh.devices.flat[0]): # Small setup tables live on the selected backend.
        parameters = initialize_simulation_parameters(input_parameters, Nx, Ny, Nz, Nn, Nm, Np, Ns, N_DG, dims, timesteps, dt, parallel=parallel)
    y0 =(parameters["Ck_0"].reshape(Ns, -1, *parameters["Fk_0"].shape[-4:]), parameters["Fk_0"])
    args = {k: parameters["collision_matrix" if k == "col" else k] for k in RHS_ARGS}
    args["alpha_s"], args["u_s"] = args["alpha_s"].reshape(Ns, 3), args["u_s"].reshape(Ns, 3)

    # Solve the ODE system
    solve, in_shardings = _compiled_solve(mesh, solver, adaptive_time_step, timesteps, progress_meter, max_steps, Nn, Nm, Np)
    Ck, Fk, time, result, num_steps = solve(*_place((y0, args, parameters["t_max"], dt, parameters["ode_tolerance"]), in_shardings, mesh))
    if isinstance(result._value, jax.core.Tracer): # Under a user jit: fail at run time instead of on the host.
        Ck = eqx.error_if(Ck, ~is_okay(result), "Diffrax solve failed")
    elif not is_okay(result): # One scalar synchronization; the solution itself is not transferred.
        raise RuntimeError(f"Diffrax solve failed after {int(num_steps)} steps: {result}")

    # Output results
    output = {"Ck": Ck, "Fk": Fk, "time": time, "num_steps": num_steps, **parameters, "Nn": Nn, "Nm": Nm, "Np": Np, "Ns": Ns}
    diagnostics(output)

    return output
