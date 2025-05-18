import jax.numpy as jnp
from jax import jit, config, vmap
import inspect
config.update("jax_enable_x64", True)
from jax.debug import print as jprint
from functools import partial
from diffrax import (diffeqsolve, Tsit5, Dopri5, ODETerm,
                     SaveAt, PIDController, TqdmProgressMeter)
from ._initialization import initialize_simulation_parameters
from ._model import plasma_current, Hermite_Fourier_system
from ._diagnostics import diagnostics

# Parallelize the simulation using JAX
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax import devices, make_mesh, NamedSharding, device_put
mesh = make_mesh((len(devices()),), ("batch"))
spec = P("batch")
sharding = NamedSharding(mesh, spec)

__all__ = ["cross_product", "ode_system", "simulation"]

@jit
def cross_product(k_vec, F_vec):
    """
    Computes the cross product of two 3D vectors.
    Args:
        k_vec (array-like): A 3-element array representing the first vector (k).
        F_vec (array-like): A 3-element array representing the second vector (F).
    Returns:
        jnp.ndarray: A 3-element array representing the cross product k x F.
    """
    kx, ky, kz = k_vec
    Fx, Fy, Fz = F_vec
    return jnp.array([ky * Fz - kz * Fy, kz * Fx - kx * Fz, kx * Fy - ky * Fx])

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns'])
def ode_system(Nx, Ny, Nz, Nn, Nm, Np, Ns, t, Ck_Fk, args):
    (qs, nu, D, Omega_cs, alpha_s, u_s,
     Lx, Ly, Lz, kx_grid, ky_grid, kz_grid
    ) = args

    sum_of_per_species_modes = sum(
        Nn[s] * Nm[s] * Np[s] for s in range(Ns)
    )
    total_Ck_size = sum_of_per_species_modes * Nx * Ny * Nz
    Ck = Ck_Fk[:total_Ck_size].reshape(sum_of_per_species_modes, Ny, Nx, Nz)
    Fk = Ck_Fk[total_Ck_size:].reshape(6, Ny, Nx, Nz)

    # dCk_s_dt = vmap(
    #     Hermite_Fourier_system,
    #     in_axes=(None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0)
    # )(Ck, Fk, kx_grid, ky_grid, kz_grid, Lx, Ly, Lz, nu, D, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, jnp.arange(Nn * Nm * Np * Ns))

    partial_Hermite_Fourier_system = partial(
        Hermite_Fourier_system,
        Ck, Fk, kx_grid, ky_grid, kz_grid, Lx, Ly, Lz, nu, D, alpha_s, u_s, qs, Omega_cs,
        Nn, Nm, Np)
    sharded_fun = jit(shard_map(vmap(partial_Hermite_Fourier_system), mesh, in_specs=spec, out_specs=spec, check_rep=False))
    indices_sharded = device_put(jnp.arange(sum_of_per_species_modes), sharding)
    dCk_s_dt = sharded_fun(indices_sharded)
    nabla = jnp.array([kx_grid / Lx, ky_grid / Ly, kz_grid / Lz])

    dBk_dt = -cross_product(nabla, Fk[:3])
    current = plasma_current(qs, alpha_s, u_s, Ck, Nn, Nm, Np, Ns)
    dEk_dt = cross_product(nabla, Fk[3:]) - current
    dFk_dt = jnp.concatenate([dEk_dt, dBk_dt], axis=0)  # Shape: (6, Ny, Nx, Nz)
    dy_dt = jnp.concatenate([dCk_s_dt.reshape(-1), dFk_dt.reshape(-1)])
    return dy_dt

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'timesteps', 'solver'])
def simulation(input_parameters={}, Nx=33, Ny=1, Nz=1,
               Nn=(20, 20), Nm=(1, 1), Np=(1, 1), # Default to tuples for Ns=2
               Ns=2, timesteps=200, solver=Tsit5): # Using Tsit5 as a common Diffrax solver
    """
    Simulates the Vlasov-Maxwell system using spectral methods.
    Nn, Nm, Np are expected to be tuples of integers, one per species.

    Args:
        input_parameters (dict, optional): Dictionary of user-defined simulation parameters.
        Nx (int, optional): Number of grid points in the x-direction.
        Ny (int, optional): Number of grid points in the y-direction.
        Nz (int, optional): Number of grid points in the z-direction.
        Nn (tuple, optional): Tuple of Hermite modes in v_parallel for each species.
        Nm (tuple, optional): Tuple of Hermite modes in v_perp_1 for each species.
        Np (tuple, optional): Tuple of Hermite modes in v_perp_2 for each species.
        Ns (int, optional): Number of particle species. Must match len(Nn), len(Nm), len(Np).
        timesteps (int, optional): Number of time steps for the simulation output.
        solver (diffrax.AbstractSolver, optional): The ODE solver from Diffrax.
    Returns:
        dict: A dictionary containing Ck, Fk, time, dCk, and other parameters.
    """
    parameters = initialize_simulation_parameters(input_parameters, Nx, Ny, Nz, Nn, Nm, Np, Ns, timesteps)
    initial_conditions = jnp.concatenate([parameters["Ck_0"].flatten(), parameters["Fk_0"].flatten()])

    # Define the time array for data output.
    time_array = jnp.linspace(0, parameters["t_max"], timesteps)


    ode_args = (parameters["qs"], parameters["nu"], parameters["D"], parameters["Omega_cs"],
                parameters["alpha_s"], parameters["u_s"],
                parameters["Lx"], parameters["Ly"], parameters["Lz"],
                parameters["kx_grid"], parameters["ky_grid"], parameters["kz_grid"])


    ode_fixed = partial(ode_system, Nx, Ny, Nz, Nn, Nm, Np, Ns)

    # Ensure solver is an instance if a class was passed (e.g. Tsit5 vs Tsit5())
    current_solver = solver() if inspect.isclass(solver) else solver


    sol = diffeqsolve(
        ODETerm(ode_fixed),
        solver=current_solver,
        stepsize_controller=PIDController(rtol=parameters["ode_tolerance"], atol=parameters["ode_tolerance"]),
        t0=0.0, t1=parameters["t_max"], dt0=None, # dt0=None for adaptive stepping
        y0=initial_conditions, args=ode_args, saveat=SaveAt(ts=time_array),
        max_steps=100000, # Increased max_steps
        progress_meter=TqdmProgressMeter()
    )

    # Calculate the total number of Hermite coefficients across all species
    sum_of_per_species_modes = sum(Nn[s] * Nm[s] * Np[s] for s in range(Ns))
    total_Ck_elements = sum_of_per_species_modes * Nx * Ny * Nz

    # Reshape the solution to extract Ck and Fk
    # Ck part of the solution
    Ck_flat = sol.ys[:, :total_Ck_elements]
    Ck = Ck_flat.reshape(len(sol.ts), sum_of_per_species_modes, Nx, Ny, Nz)
    # Fk part of the solution
    Fk_flat = sol.ys[:, total_Ck_elements:]
    Fk = Fk_flat.reshape(len(sol.ts), 6, Nx, Ny, Nz)
    # Set n=0, k=0 mode to zero to get array with time evolution of perturbation.
    kx0_idx = (Nx - 1) // 2 if Nx > 0 else 0
    ky0_idx = (Ny - 1) // 2 if Ny > 0 else 0
    kz0_idx = (Nz - 1) // 2 if Nz > 0 else 0

    if sum_of_per_species_modes > 0:
         dCk = Ck.at[:, 0, ky0_idx, kx0_idx, kz0_idx].set(0)

    if Ns >= 2:
        offset_species_1 = Nn[0] * Nm[0] * Np[0]
        if offset_species_1 < sum_of_per_species_modes: # Ensure index is valid
            dCk = Ck.at[:, offset_species_1, ky0_idx, kx0_idx, kz0_idx].set(0)



    temporary_output = {"Ck": Ck, "Fk": Fk, "time": sol.ts, "dCk": dCk}
    output = {**temporary_output, **parameters}

    diagnostics(output)
    return output
