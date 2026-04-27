"""Time integration driver for the spectral Vlasov–Discontinuous_Galerkin system."""

import jax.numpy as jnp
from jax import jit, config
config.update("jax_enable_x64", True)
from functools import partial
from diffrax import (diffeqsolve, Dopri8, ODETerm,
                     SaveAt, PIDController, TqdmProgressMeter, NoProgressMeter, ConstantStepSize)
from ._initialization import initialize_simulation_parameters
from ._model import Hermite_DG_system
from ._diagnostics import diagnostics

__all__ = ["ode_system", "simulation"]

@jit
def cross_product(k_vec, F_vec):
    """
    Compute the cross product `k × F` for length-3 vectors or broadcastable arrays.

    Parameters
    ----------
    k_vec : array-like
        First vector with leading dimension 3.
    F_vec : array-like
        Second vector with leading dimension 3.

    Returns
    -------
    jnp.ndarray
        Array representing the cross product with the same trailing shape as the inputs.
    """
    kx, ky, kz = k_vec
    Fx, Fy, Fz = F_vec
    return jnp.array([ky * Fz - kz * Fy, kz * Fx - kx * Fz, kx * Fy - ky * Fx])

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'Nl'])
def ode_system(Nx, Ny, Nz, Nn, Nm, Np, Ns, Nl, t, Ck_Fk, args):
    """
    Right-hand side for the coupled Vlasov-Maxwell system expressed in spectral form.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Number of Fourier modes per spatial dimension.
    Nn, Nm, Np : int
        Number of Hermite modes per velocity-space dimension.
    Ns : int
        Number of species.
    t : float
        Integration time (unused but required by Diffrax interface).
    Ck_Fk : jnp.ndarray
        Flattened state vector containing concatenated Hermite coefficients followed
        by electromagnetic field coefficients.
    args : tuple
        Cached parameter tuple produced in `simulation` providing physical constants,
        grids, and helper arrays.

    Returns
    -------
    jnp.ndarray
        Flattened derivative vector matching the shape of `Ck_Fk`.
    """

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

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'N_DG', 'dims', 'timesteps', 'solver', 'adaptive_time_step'])
def simulation(input_parameters={}, Nx=33, Ny=1, Nz=1, Nn=20, Nm=1, Np=1, Ns=2, N_DG=2, dims=1, timesteps=200, dt = 0.01, solver=Dopri8(), adaptive_time_step=True):
    """
    Run a spectral Vlasov-Maxwell simulation and return the solution together with
    the parameter dictionary used to produce it.

    Parameters
    ----------
    input_parameters : dict, optional
        User-specified overrides passed to `initialize_simulation_parameters`.
    Nx, Ny, Nz : int, optional
        Number of retained Fourier modes per spatial direction.
    Nn, Nm, Np : int, optional
        Number of Hermite modes per velocity-space axis.
    Ns : int, optional
        Number of species.
    timesteps : int, optional
        Number of solution snapshots to save between `t=0` and `t_max`.
    dt : float, optional
        Initial integration step size.
    solver : diffrax.AbstractSolver, optional
        Diffrax solver instance controlling the time integration.

    Returns
    -------
    dict
        Dictionary containing the evolved coefficients (`Ck`, `Fk`), time samples,
        perturbation diagnostics, and all simulation parameters.
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
    
    controllers = {
    True: PIDController(
        rtol=parameters["ode_tolerance"],
        atol=parameters["ode_tolerance"],
    ),
    False: ConstantStepSize(),
    }
    stepsize_controller = controllers[adaptive_time_step]
    
    # Solve the ODE system
    ode_system_partial = partial(ode_system, Nx, Ny, Nz, Nn, Nm, Np, Ns, Nl)
    sol = diffeqsolve(
        ODETerm(ode_system_partial), solver=solver,
        stepsize_controller=stepsize_controller,
        t0=0, t1=parameters["t_max"], dt0=dt,
        y0=initial_conditions, args=args, saveat=SaveAt(ts=time),
        max_steps=1000000, progress_meter=TqdmProgressMeter(1))
    
    # Reshape the solution to extract Ck and Fk
    Ck = sol.ys[:,:(-6 * Nx * Ny * Nz * Nl)].reshape(len(sol.ts), Ns * Nn * Nm * Np, Ny, Nx, Nz, Nl)
    Fk = sol.ys[:,(-6 * Nx * Ny * Nz * Nl):].reshape(len(sol.ts), 6, Ny, Nx, Nz, Nl) # 
    
    # Output results
    temporary_output = {"Ck": Ck, "Fk": Fk, "time": time}
    output = {**temporary_output, **parameters}
    diagnostics(output)
    
    return output
