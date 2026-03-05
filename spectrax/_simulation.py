import jax.numpy as jnp
from jax import jit, config
config.update("jax_enable_x64", True)
from functools import partial
from diffrax import (diffeqsolve, Tsit5, Dopri5, ODETerm,
                     SaveAt, PIDController, TqdmProgressMeter, NoProgressMeter, ConstantStepSize)
from ._initialization import initialize_simulation_parameters
from ._model import Hermite_DG_system
from ._diagnostics import diagnostics

# Parallelize the simulation using JAX
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax import devices, make_mesh, NamedSharding, device_put
mesh = make_mesh((len(devices()),), ("batch"))
spec = P("batch")
sharding = NamedSharding(mesh, spec)

__all__ = ["ode_system", "simulation"]

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'Nl'])
def ode_system(Nx, Ny, Nz, Nn, Nm, Np, Ns, Nl, t, Ck_Fk, args):

    (ms, qs, nu, D, Omega_ce, alpha_s, u_s, Lx, Ly, Lz, col, 
    sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus, 
    basis_idx, inner_mm, inner_pm, inner_mp, inner_pp, di_inner_product, tripple_product, 
    Ax_p, Ax_m, Ay_p, Ay_m, Az_p, Az_m, R_p, R_m
    ) = args[8:]

    total_Ck_size = Nn * Nm * Np * Ns * Nx * Ny * Nz * Nl
    Ck = Ck_Fk[:total_Ck_size].reshape(Nn * Nm * Np * Ns, Ny, Nx, Nz, Nl)
    Fk = Ck_Fk[total_Ck_size:].reshape(6, Ny, Nx, Nz, Nl)

    dy_dt = Hermite_DG_system(Ck, Fk, col, sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus, basis_idx, 
                                inner_mm, inner_pm, inner_mp, inner_pp, di_inner_product, tripple_product, 
                                Ax_p, Ax_m, Ay_p, Ay_m, Az_p, Az_m, R_p, R_m,
                                    Lx, Ly, Lz, nu, D, alpha_s, u_s, ms, qs, Omega_ce, Nn, Nm, Np, Ns)

    return dy_dt

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'N_DG', 'dims', 'timesteps', 'solver'])
def simulation(input_parameters={}, Nx=33, Ny=1, Nz=1, Nn=20, Nm=1, Np=1, Ns=2, N_DG=2, dims=1, timesteps=200, dt = 0.01, solver=Dopri5()):
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

    # Combine initial conditions.
    initial_conditions = jnp.concatenate([parameters["Ck_0"].flatten(), parameters["Fk_0"].flatten()])

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
    
    # Solve the ODE system
    ode_system_partial = partial(ode_system, Nx, Ny, Nz, Nn, Nm, Np, Ns, Nl)
    sol = diffeqsolve(
        ODETerm(ode_system_partial), solver=solver,
        stepsize_controller=PIDController(rtol=parameters["ode_tolerance"], atol=parameters["ode_tolerance"]),
        # stepsize_controller=ConstantStepSize(),
        t0=0, t1=parameters["t_max"], dt0=dt,
        y0=initial_conditions, args=args, saveat=SaveAt(ts=time),
        max_steps=1000000, progress_meter=TqdmProgressMeter())
    
    ## Idea: take the eigenvalues of ODE_system to determine the stability of the system.
    
    # Reshape the solution to extract Ck and Fk
    Ck = sol.ys[:,:(-6 * Nx * Ny * Nz * Nl)].reshape(len(sol.ts), Ns * Nn * Nm * Np, Ny, Nx, Nz, Nl)
    Fk = sol.ys[:,(-6 * Nx * Ny * Nz * Nl):].reshape(len(sol.ts), 6, Ny, Nx, Nz, Nl) # 
    
    # Set n = 0, k = 0 mode to zero to get array with time evolution of perturbation.
    dCk = Ck.at[:, 0, 0, 1, 0, 0].set(0)
    dCk = dCk.at[:, Nn * Nm * Np, 0, 1, 0, 0].set(0)
    
    # Output results
    temporary_output = {"Ck": Ck, "Fk": Fk, "time": time, "dCk": dCk}
    output = {**temporary_output, **parameters}
    # diagnostics(output)
    return output