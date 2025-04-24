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

    total_Ck_size = Nn * Nm * Np * Ns * Nx * Ny * Nz
    Ck = Ck_Fk[:total_Ck_size].reshape(Nn * Nm * Np * Ns, Ny, Nx, Nz)
    Fk = Ck_Fk[total_Ck_size:].reshape(6, Ny, Nx, Nz)

    # dCk_s_dt = vmap(
    #     Hermite_Fourier_system,
    #     in_axes=(None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0)
    # )(Ck, Fk, kx_grid, ky_grid, kz_grid, Lx, Ly, Lz, nu, D, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, jnp.arange(Nn * Nm * Np * Ns))
    
    partial_Hermite_Fourier_system = partial(
        Hermite_Fourier_system,
        Ck, Fk, kx_grid, ky_grid, kz_grid, Lx, Ly, Lz, nu, D, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np)
    sharded_fun = jit(shard_map(vmap(partial_Hermite_Fourier_system), mesh, in_specs=spec, out_specs=spec, check_rep=False))
    indices_sharded = device_put(jnp.arange(Nn * Nm * Np * Ns), sharding)
    dCk_s_dt = sharded_fun(indices_sharded)

    nabla = jnp.array([kx_grid / Lx, ky_grid / Ly, kz_grid / Lz])
    dBk_dt = -1j * cross_product(nabla, Fk[:3])
    
    current = plasma_current(qs, alpha_s, u_s, Ck, Nn, Nm, Np, Ns)
    dEk_dt = 1j * cross_product(nabla, Fk[3:]) - current / Omega_cs[0]

    dFk_dt = jnp.concatenate([dEk_dt, dBk_dt], axis=0)
    dy_dt  = jnp.concatenate([dCk_s_dt.reshape(-1), dFk_dt.reshape(-1)])
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
        Nm (int, optional): Number of azimuthal harmonics. Default is 1.
        Np (int, optional): Number of poloidal harmonics. Default is 1.
        Ns (int, optional): Number of particle species. Default is 2.
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
    
    # **Initialize simulation parameters**
    parameters = initialize_simulation_parameters(input_parameters, Nx, Ny, Nz, Nn, Nm, Np, Ns, timesteps)

    # Combine initial conditions.
    initial_conditions = jnp.concatenate([parameters["Ck_0"].flatten(), parameters["Fk_0"].flatten()])

    # Define the time array for data output.
    time = jnp.linspace(0, parameters["t_max"], timesteps)
    
    # Arguments for the ODE system.
    args = (parameters["qs"], parameters["nu"], parameters["D"], parameters["Omega_cs"], parameters["alpha_s"],
            parameters["u_s"], parameters["Lx"], parameters["Ly"], parameters["Lz"],
            parameters["kx_grid"], parameters["ky_grid"], parameters["kz_grid"])

    # Solve the ODE system
    ode_system_partial = partial(ode_system, Nx, Ny, Nz, Nn, Nm, Np, Ns)
    sol = diffeqsolve(
        ODETerm(ode_system_partial), solver=solver(),
        stepsize_controller=PIDController(rtol=parameters["ode_tolerance"], atol=parameters["ode_tolerance"]),
        t0=0, t1=parameters["t_max"], dt0=parameters["t_max"]/timesteps,
        y0=initial_conditions, args=args, saveat=SaveAt(ts=time),
        max_steps=100000, progress_meter=TqdmProgressMeter())
    
    ## Idea: take the eigenvalues of ODE_system to determine the stability of the system.
    
    # Reshape the solution to extract Ck and Fk
    Ck = sol.ys[:,:(-6 * Nx * Ny * Nz)].reshape(len(sol.ts), Ns * Nn * Nm * Np, Ny, Nx, Nz)
    Fk = sol.ys[:,(-6 * Nx * Ny * Nz):].reshape(len(sol.ts), 6, Ny, Nx, Nz)
    
    # Set n = 0, k = 0 mode to zero to get array with time evolution of perturbation.
    dCk = Ck.at[:, 0, 0, 1, 0].set(0)
    dCk = dCk.at[:, Nn * Nm * Np, 0, 1, 0].set(0)
    
    # Output results
    temporary_output = {"Ck": Ck, "Fk": Fk, "time": time, "dCk": dCk}
    output = {**temporary_output, **parameters}
    diagnostics(output)
    return output