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

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'Nl'])
def ode_system(Nx, Ny, Nz, Nn, Nm, Np, Ns, Nl, t, Ck_Fk, args):
    """
    Right-hand side for the coupled Vlasov-Maxwell system expressed in spectral-FE form.

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
    Ck_Fk : tuple[jnp.ndarray, jnp.ndarray]
        Hermite coefficients with shape `(Ns, Np, Nm, Nn, Ny, Nx, Nz, Nl)` and
        electromagnetic field coefficients with shape `(6, Ny, Nx, Nz, Nl)`.
    args : tuple
        Cached parameter tuple produced in `simulation` providing physical constants,
        grids, and helper arrays.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Derivatives matching the Hermite and field coefficient arrays.
    """

    (ms, qs, nu, D, Omega_ce, alpha_s, u_s, Lx, Ly, Lz, col, 
    sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus, 
    basis_idx, inner_mm, inner_pm, inner_mp, inner_pp, di_inner_product, triple_product, 
    Ax_p, Ax_m, Ay_p, Ay_m, Az_p, Az_m, R_p, R_m
    ) = args[8:]

    Ck, Fk = Ck_Fk
    Ck_shape = Ck.shape
    Ck = Ck.reshape(Nn * Nm * Np * Ns, Ny, Nx, Nz, Nl)

    dy_dt = Hermite_DG_system(Ck, Fk, col, sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus, basis_idx, 
                                inner_mm, inner_pm, inner_mp, inner_pp, di_inner_product, triple_product, 
                                Ax_p, Ax_m, Ay_p, Ay_m, Az_p, Az_m, R_p, R_m,
                                    Lx, Ly, Lz, nu, D, alpha_s, u_s, ms, qs, Omega_ce, Nn, Nm, Np, Ns)

    # The model returns the flattened (dCk, dFk); split it back into the state's axes.
    dCk, dFk = jnp.split(dy_dt, [Ck.size])
    return dCk.reshape(Ck_shape), dFk.reshape(Fk.shape)

def _stepsize_controller(adaptive_time_step, tolerance, dtmin=None):
    """PID control at `tolerance` with an optional step floor `dtmin` (None: no floor), or constant steps."""
    if not adaptive_time_step:
        return ConstantStepSize()
    return PIDController(rtol=tolerance, atol=tolerance, dtmin=dtmin, force_dtmin=False)

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'N_DG', 'dims', 'timesteps', 'solver', 'adaptive_time_step',
                                'dtmin', 'max_steps', 'throw'])
def simulation(input_parameters={}, Nx=33, Ny=1, Nz=1, Nn=20, Nm=1, Np=1, Ns=2, N_DG=2, dims=1, timesteps=200, dt = 0.01, solver=Dopri8(), adaptive_time_step=True,
               dtmin=None, max_steps=1000000, throw=True):
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
    dtmin : float, optional
        Smallest adaptive step. If the step would fall below it the solve stops with an
        error, instead of crawling towards `max_steps`. A collapsing step usually signals an
        under-resolved run (filamentation reaching the Hermite cutoff, or a species whose
        Hermite width is narrower than it becomes). Default `None`:
        no floor. A short physical transient may need steps far below `t_max / max_steps` and
        still finish within the budget, so no floor is inferred from them.
    max_steps : int, optional
        Step budget; exhausting it stops the solve (`RESULTS.max_steps_reached`), reported
        separately from a step below `dtmin` (`RESULTS.dt_min_reached`).
    throw : bool, optional
        Raise on integration failure; with False the failure is reported in `solver_result`,
        and only the first `num_valid_times` saved times (`valid_times`) hold a solution.

    Returns
    -------
    dict
        Dictionary containing the evolved coefficients (`Ck`, `Fk`), time samples,
        perturbation diagnostics, and all simulation parameters.
    """
    
    # **Initialize simulation parameters**
    parameters = initialize_simulation_parameters(input_parameters, Nx, Ny, Nz, Nn, Nm, Np, Ns, N_DG, dims, timesteps, dt)

    # Preserve the species, Hermite and element axes for the integrator.
    initial_conditions = (
        parameters["Ck_0"].reshape(Ns, Np, Nm, Nn, Ny, Nx, Nz, -1),
        parameters["Fk_0"].reshape(6, Ny, Nx, Nz, -1),
    )

    # Define the time array for data output.
    time = jnp.linspace(0, parameters["t_max"], timesteps)
    
    # Define total number of basis modes
    Nl = parameters["basis_idx"].shape[0]

    # Arguments for the ODE system.
    args = (Nx, Ny, Nz, Nn, Nm, Np, Ns, Nl, parameters["ms"], parameters["qs"], parameters["nu"], parameters["D"], parameters["Omega_ce"], parameters["alpha_s"],
            parameters["u_s"], parameters["Lx"], parameters["Ly"], parameters["Lz"], parameters["collision_matrix"], 
            parameters["sqrt_n_plus"], parameters["sqrt_n_minus"], parameters["sqrt_m_plus"], parameters["sqrt_m_minus"], 
            parameters["sqrt_p_plus"], parameters["sqrt_p_minus"], parameters["basis_idx"], 
            parameters["inner_mm"], parameters["inner_pm"], parameters["inner_mp"], parameters["inner_pp"], parameters["di_inner_product"],
            parameters["triple_product"], parameters["Ax_p"], parameters["Ax_m"], 
            parameters["Ay_p"], parameters["Ay_m"], parameters["Az_p"], parameters["Az_m"], parameters["R_p"], parameters["R_m"]
            )
    
    stepsize_controller = _stepsize_controller(adaptive_time_step, parameters["ode_tolerance"], dtmin)
    
    # Solve the ODE system
    ode_system_partial = partial(ode_system, Nx, Ny, Nz, Nn, Nm, Np, Ns, Nl)
    sol = diffeqsolve(
        ODETerm(ode_system_partial), solver=solver,
        stepsize_controller=stepsize_controller,
        t0=0, t1=parameters["t_max"], dt0=dt,
        y0=initial_conditions, args=args, saveat=SaveAt(ts=time),
        max_steps=max_steps, progress_meter=TqdmProgressMeter(), throw=throw)
    
    # Reshape the solution to extract Ck and Fk
    Ck = sol.ys[0].reshape(len(sol.ts), Ns * Nn * Nm * Np, Ny, Nx, Nz, Nl)
    Fk = sol.ys[1]
    
    # Output results
    # Diffrax fills saves after a failed solve with inf; flag them so they are never read as data.
    valid_times = jnp.isfinite(sol.ts)
    temporary_output = {"Ck": Ck, "Fk": Fk, "time": time, "solver_stats": sol.stats,
                        "solver_result": sol.result, "valid_times": valid_times,
                        "num_valid_times": jnp.sum(valid_times)}
    output = {**temporary_output, **parameters}
    diagnostics(output)
    
    return output
