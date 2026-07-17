import jax
import jax.numpy as jnp
from jax import config, devices, jit, lax, shard_map
config.update("jax_enable_x64", True)
from functools import partial
from diffrax import (diffeqsolve, Tsit5, Dopri5, ODETerm,
                     SaveAt, PIDController, TqdmProgressMeter, NoProgressMeter, ConstantStepSize)
from ._initialization import initialize_simulation_parameters
from ._model import Hermite_DG_system
from ._diagnostics import diagnostics

# Parallelize the simulation using JAX
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P
mesh = jax.make_mesh(
    (len(devices()),), ("cell",), axis_types=(AxisType.Explicit,), devices=devices()
)

def _place(x, shardings):
    return jax.device_put(jax.device_get(x), shardings)

__all__ = ["ode_system", "simulation"]

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'Nl', 'shard_axis', 'shards'])
def ode_system(Nx, Ny, Nz, Nn, Nm, Np, Ns, Nl, shard_axis, shards, t, Ck_Fk, args):

    (ms, qs, nu, D, Omega_ce, alpha_s, u_s, Lx, Ly, Lz, col, 
    sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus, 
    basis_idx, inner_mm, inner_pm, inner_mp, inner_pp, di_inner_product, tripple_product, 
    Ax_p, Ax_m, Ay_p, Ay_m, Az_p, Az_m, R_p, R_m
    ) = args[8:]

    Ck, Fk = Ck_Fk

    return Hermite_DG_system(Ck, Fk, col, sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus, basis_idx,
                                inner_mm, inner_pm, inner_mp, inner_pp, di_inner_product, tripple_product, 
                                Ax_p, Ax_m, Ay_p, Ay_m, Az_p, Az_m, R_p, R_m,
                                    Lx, Ly, Lz, nu, D, alpha_s, u_s, ms, qs, Omega_ce, Nn, Nm, Np, Ns,
                                    shard_axis, shards)

def _sharded_rms_norm(error):
    squared = sum(jnp.vdot(x, x) for x in error)
    size = sum(x.size for x in error) * lax.axis_size("cell")
    return jnp.sqrt(jnp.maximum(jnp.real(lax.psum(squared, "cell")), 0) / size)

@partial(jit, static_argnames=['shape', 'solver', 'shard_axis', 'adaptive'])
def _solve_sharded(initial_conditions, args, solver_args, shape, solver,
                    shard_axis, adaptive):
    Nx, Ny, Nz, Nn, Nm, Np, Ns, Nl = shape
    shards = len(devices())
    local_sizes = {"x": (Nx // shards, Ny, Nz),
                   "y": (Nx, Ny // shards, Nz),
                   "z": (Nx, Ny, Nz // shards)}[shard_axis]
    spatial_axis = {"y": 1, "x": 2, "z": 3}[shard_axis]
    state_spec = [None] * 5
    state_spec[spatial_axis] = "cell"

    def local_solve(y0, local_args, local_solver_args):
        local_time, t0, local_dt, t1, tolerance = local_solver_args
        lengths = list(local_args[7:10])
        lengths[{"x": 0, "y": 1, "z": 2}[shard_axis]] /= shards
        local_args = local_args[:7] + tuple(lengths) + local_args[10:]
        vector_field = partial(
            ode_system, *local_sizes, Nn, Nm, Np, Ns, Nl, shard_axis, shards
        )
        controller = (PIDController(rtol=tolerance, atol=tolerance,
                                    norm=_sharded_rms_norm)
                      if adaptive else ConstantStepSize())
        sol = diffeqsolve(
            ODETerm(vector_field), solver=solver, stepsize_controller=controller,
            t0=t0, t1=t1, dt0=local_dt, y0=y0,
            args=(*local_sizes, Nn, Nm, Np, Ns, Nl) + local_args,
            saveat=SaveAt(ts=local_time), max_steps=1000000, throw=False,
            progress_meter=NoProgressMeter(),
        )
        return sol.ys, sol.result

    saved_spec = P(None, *state_spec)
    return shard_map(
        local_solve, mesh=mesh,
        in_specs=((P(*state_spec), P(*state_spec)),
                  (P(),) * len(args), (P(),) * 5),
        out_specs=((saved_spec, saved_spec), P()), check_vma=False,
    )(initial_conditions, args, solver_args)

def simulation(input_parameters={}, Nx=33, Ny=1, Nz=1, Nn=20, Nm=1, Np=1, Ns=2, N_DG=2, dims=1, timesteps=200, dt = 0.01, solver=Dopri5(), shard_axis=None, adaptive=True):
    """
    Simulates the Vlasov-Maxwell system using a mixed spectral-Galerkin method.
    This function initializes simulation parameters, sets up initial conditions,
    and solves the system of ordinary differential equations (ODEs) representing
    the Vlasov-Maxwell equations. The solution is returned as time-evolving
    coefficients for the distribution function (Ck) and electromagnetic fields (Fk).
    Args:
        input_parameters (dict, optional): Dictionary of user-defined simulation parameters.
        Nx (int, optional): Number of grid points in the x-direction. Default is 33.
        Ny (int, optional): Number of grid points in the y-direction. Default is 1.
        Nz (int, optional): Number of grid points in the z-direction. Default is 1.
        Nn (int, optional): Number of Hermite modes in the x direction in velocity space. Default is 20.
        Nm (int, optional): Number of Hermite modes in the y direction in velocity space.
        Np (int, optional): Number of Hermite modes in the z direction in velocity space.
        Ns (int, optional): Number of particle species. Default is 2.
        timesteps (int, optional): Number of time steps for the simulation. Default is 200.
        shard_axis (str, optional): Spatial cell axis to shard over all JAX devices.
        adaptive (bool, optional): Use adaptive time steps. Default is True.
    Returns:
        tuple: A tuple containing:
            - Ck (jnp.ndarray): Time-evolving coefficients for the distribution function.
            - Fk (jnp.ndarray): Time-evolving coefficients for the electromagnetic fields.
            - sol.ts (jnp.ndarray): Array of time points corresponding to the solution.
    Notes:
        - The simulation uses the `diffeqsolve` function to integrate the ODEs.
        - The solution is reshaped to separate the coefficients for the distribution
          function (Ck) and the electromagnetic fields (Fk).
        - The function relies on JAX for numerical computations and efficient array operations.
    """
    
    # **Initialize simulation parameters**
    parameters = initialize_simulation_parameters(input_parameters, Nx, Ny, Nz, Nn, Nm, Np, Ns, N_DG, dims, timesteps, dt)

    # Keep distributions and fields separate so their layouts can be controlled independently.
    initial_conditions = (
        parameters["Ck_0"].reshape(Ns * Nn * Nm * Np, Ny, Nx, Nz, -1),
        parameters["Fk_0"].reshape(6, Ny, Nx, Nz, -1),
    )
    if shard_axis is not None:
        spatial_axis = {"y": 1, "x": 2, "z": 3}[shard_axis]
        if initial_conditions[0].shape[spatial_axis] % len(devices()):
            raise ValueError(f"{shard_axis} cells must be divisible by the device count")
        state_spec = [None] * 5
        state_spec[spatial_axis] = "cell"
        state_sharding = NamedSharding(mesh, P(*state_spec))
        initial_conditions = _place(
            initial_conditions, (state_sharding,) * 2
        )

    # Define the time array for data output.
    time = jnp.linspace(0, parameters["t_max"], timesteps)
    
    # Define number of basis modes
    Nl = parameters["basis_idx"].shape[0]

    # Arguments for the ODE system.
    args = (Nx, Ny, Nz, Nn, Nm, Np, Ns, Nl, parameters["ms"], parameters["qs"], parameters["nu"], parameters["D"], parameters["Omega_ce"], parameters["alpha_s"],
            parameters["u_s"], parameters["Lx"], parameters["Ly"], parameters["Lz"], parameters["collision_matrix"], 
            parameters["sqrt_n_plus"], parameters["sqrt_n_minus"], parameters["sqrt_m_plus"], parameters["sqrt_m_minus"], 
            parameters["sqrt_p_plus"], parameters["sqrt_p_minus"], parameters["basis_idx"], 
            parameters["inner_mm"], parameters["inner_pm"], parameters["inner_mp"], parameters["inner_pp"], parameters["di_inner_product"],
            parameters["tripple_product"], parameters["Ax_p"], parameters["Ax_m"], 
            parameters["Ay_p"], parameters["Ay_m"], parameters["Az_p"], parameters["Az_m"], parameters["R_p"], parameters["R_m"]
            )
    if shard_axis is not None:
        replicated = NamedSharding(mesh, P())
        args = args[:8] + _place(args[8:], (replicated,) * len(args[8:]))

    if shard_axis is not None:
        solver_args = (time, 0.0, dt, parameters["t_max"], parameters["ode_tolerance"])
        solver_args = _place(solver_args, (replicated,) * 5)
        solution, solver_result = _solve_sharded(
            initial_conditions, args[8:], solver_args,
            (Nx, Ny, Nz, Nn, Nm, Np, Ns, Nl), solver, shard_axis, adaptive,
        )
    else:
        vector_field = partial(
            ode_system, Nx, Ny, Nz, Nn, Nm, Np, Ns, Nl, None, 1
        )
        sol = diffeqsolve(
            ODETerm(vector_field), solver=solver,
            stepsize_controller=(PIDController(rtol=parameters["ode_tolerance"], atol=parameters["ode_tolerance"])
                                 if adaptive else ConstantStepSize()),
            t0=0, t1=parameters["t_max"], dt0=dt,
            y0=initial_conditions, args=args, saveat=SaveAt(ts=time),
            max_steps=1000000, progress_meter=TqdmProgressMeter(1),
        )
        solution, solver_result = sol.ys, sol.result

    Ck, Fk = solution
    
    # Set n = 0, k = 0 mode to zero to get array with time evolution of perturbation.
    mode, y, x, z, basis = jnp.indices(Ck.shape[1:], sparse=True)
    zero_mode = (((mode == 0) | (mode == Nn * Nm * Np)) & (y == 0)
                 & (x == 1) & (z == 0) & (basis == 0))
    dCk = jnp.where(zero_mode[None], 0, Ck)
    
    # Output results
    temporary_output = {"Ck": Ck, "Fk": Fk, "time": time, "dCk": dCk,
                        "solver_result": solver_result}
    output = {**temporary_output, **parameters}
    # diagnostics(output)
    return output
