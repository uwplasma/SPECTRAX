import jax.numpy as jnp
from jax import jit, config, vmap
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

    total_count = 0
    for s_idx in range(Ns):
        total_count += Nn[s_idx] * Nm[s_idx] * Np[s_idx]
    total_Ck_size = total_count * Nx * Ny * Nz

    Ck = Ck_Fk[:total_Ck_size].reshape(total_count, Ny, Nx, Nz)
    Fk = Ck_Fk[total_Ck_size:].reshape(6, Ny, Nx, Nz)

    partial_Hermite_Fourier_system = partial(
        Hermite_Fourier_system,
        Ck, Fk, kx_grid, ky_grid, kz_grid, Lx, Ly, Lz, nu, D, alpha_s, u_s, qs, Omega_cs,
        Nn, Nm, Np)
    sharded_fun = jit(shard_map(vmap(partial_Hermite_Fourier_system), mesh, in_specs=spec, out_specs=spec, check_rep=False))
    indices_sharded = device_put(jnp.arange(total_count), sharding)
    dCk_dt = sharded_fun(indices_sharded)
    nabla = jnp.array([kx_grid / Lx, ky_grid / Ly, kz_grid / Lz])

    dBk_dt = -cross_product(nabla, Fk[:3])
    current = plasma_current(qs, alpha_s, u_s, Ck, Nn, Nm, Np, Ns)
    dEk_dt = cross_product(nabla, Fk[3:]) - current
    dFk_dt = jnp.concatenate([dEk_dt, dBk_dt], axis=0)
    dy_dt = jnp.concatenate([dCk_dt.reshape(-1), dFk_dt.reshape(-1)])


    return dy_dt

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'timesteps', 'solver'])
def simulation(input_parameters={}, Nx=33, Ny=1, Nz=1, Nn=20, Nm=1, Np=1, Ns=2, timesteps=200, solver=Dopri5):
    """
    Simulates the Vlasov-Maxwell system using spectral methods.
    This function initializes simulation parameters, sets up initial conditions,
    and solves the system of ordinary differential equations (ODEs) representing
    the Vlasov-Maxwell equations. The solution is returned as time-evolving
    coefficients for the distribution function (Ck) and electromagnetic fields (Fk).
    Args:
        input_parameters (dict, optional): Dictionary of user-defined simulation parameters.
        Nx (int, optional): Number of grid points in the x-direction. Default is 33.
        Ny (int, optional): Number of grid points in the y-direction. Default is 1.
        Nz (int, optional): Number of grid points in the z-direction. Default is 1.
        Nn (int, optional): Number of velocity space harmonics. Default is 20.
        Nn (tuple, optional): Tuple of Hermite modes in v_x for each species.
        Nm (tuple, optional): Tuple of Hermite modes in v_y for each species.
        Np (tuple, optional): Tuple of Hermite modes in v_z for each species.
        Ns (int, optional): Number of particle species. Must match len(Nn), len(Nm), len(Np).
        timesteps (int, optional): Number of time steps for the simulation. Default is 200.
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
    
    parameters = initialize_simulation_parameters(input_parameters, Nx, Ny, Nz, Nn, Nm, Np, Ns, timesteps)

    initial_conditions = jnp.concatenate([parameters["Ck_0"].flatten(), parameters["Fk_0"].flatten()])

    time = jnp.linspace(0, parameters["t_max"], timesteps)
    

    args = (parameters["qs"], parameters["nu"], parameters["D"], parameters["Omega_cs"], parameters["alpha_s"],
            parameters["u_s"], parameters["Lx"], parameters["Ly"], parameters["Lz"],
            parameters["kx_grid"], parameters["ky_grid"], parameters["kz_grid"])

    ode_system_partial = partial(ode_system, Nx, Ny, Nz, Nn, Nm, Np, Ns)
    sol = diffeqsolve(
        ODETerm(ode_system_partial), solver=solver(),
        stepsize_controller=PIDController(rtol=parameters["ode_tolerance"], atol=parameters["ode_tolerance"]),
        t0=0, t1=parameters["t_max"], dt0=parameters["t_max"]/timesteps,
        y0=initial_conditions, args=args, saveat=SaveAt(ts=time),
        max_steps=1000000, progress_meter=TqdmProgressMeter())
    
    ## Idea: take the eigenvalues of ODE_system to determine the stability of the system.

    total_count = 0
    for s in range(Ns):
        total_count += Nn[s] * Nm[s] * Np[s]
    total_Ck_size = total_count * Nx * Ny * Nz
    Ck = sol.ys[:, :total_Ck_size].reshape(len(sol.ts), total_count, Ny, Nx, Nz)
    Fk = sol.ys[:, total_Ck_size:].reshape(len(sol.ts), 6, Ny, Nx, Nz)

    dCk = Ck

    ncps_list = [Nn[s_idx] * Nm[s_idx] * Np[s_idx] for s_idx in range(Ns)]
    ncps = jnp.asarray(ncps_list)
    offsets = jnp.cumsum(jnp.concatenate([jnp.array([0]), ncps[:-1]]))

    if Ns > 0 and ncps_list[0] > 0:
        dCk = dCk.at[:, offsets[0], (Ny-1)//2, (Nx-1)//2, (Nz-1)//2].set(0)

    if Ns > 1 and ncps_list[1] > 0:
        dCk = dCk.at[:, offsets[1], (Ny-1)//2, (Nx-1)//2, (Nz-1)//2].set(0)


    # Output results
    temporary_output = {"Ck": Ck, "Fk": Fk, "time": time, "dCk": dCk}
    output = {**temporary_output, **parameters}
    diagnostics(output)
    return output
