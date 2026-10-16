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

# Parameters passed to `Hermite_DG_system` under the same name as in the parameter dictionary.
_RHS_PARAMETER_NAMES = (
    "sqrt_n_plus", "sqrt_n_minus", "sqrt_m_plus", "sqrt_m_minus", "sqrt_p_plus", "sqrt_p_minus", "basis_idx",
    "inner_mm", "inner_pm", "inner_mp", "inner_pp", "di_inner_product", "triple_product",
    "Ax_p", "Ax_m", "Ay_p", "Ay_m", "Az_p", "Az_m", "R_p", "R_m",
    "Lx", "Ly", "Lz", "nu", "D", "alpha_s", "u_s", "ms", "qs", "Omega_ce",
)

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'dims'])
def ode_system(t, y, args, Nx, Ny, Nz, Nn, Nm, Np, Ns, dims):
    """
    Right-hand side for the coupled Vlasov-Maxwell system expressed in spectral-FE form.

    Parameters
    ----------
    t : float
        Integration time (unused but required by Diffrax interface).
    y : dict
        State with entries `"Ck"` (Hermite-DG coefficients, shape `(Ns, Np, Nm, Nn, Ny, Nx, Nz, Nl)`) and
        `"Fk"` (electromagnetic field coefficients, shape `(6, Ny, Nx, Nz, Nl)`).
    args : dict
        Keyword arguments of `Hermite_DG_system` other than the state and the grid sizes, built in `simulation`.
    Nx, Ny, Nz : int
        Number of elements per spatial dimension.
    Nn, Nm, Np : int
        Number of Hermite modes per velocity-space dimension.
    Ns : int
        Number of species.
    dims : int
        Number of spatial dimensions in which the DG basis varies.

    Returns
    -------
    dict
        Time derivative with the same structure and shapes as `y`.
    """
    dCk, dFk = Hermite_DG_system(y["Ck"], y["Fk"], **args, Nx=Nx, Ny=Ny, Nz=Nz, Nn=Nn, Nm=Nm, Np=Np, Ns=Ns, dims=dims)
    return {"Ck": dCk, "Fk": dFk}

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

    Returns
    -------
    dict
        Dictionary containing the evolved coefficients (`Ck`, `Fk`), time samples,
        perturbation diagnostics, and all simulation parameters.
    """
    
    # **Initialize simulation parameters**
    parameters = initialize_simulation_parameters(input_parameters, Nx, Ny, Nz, Nn, Nm, Np, Ns, N_DG, dims, timesteps, dt)

    # Define total number of basis modes
    Nl = parameters["basis_idx"].shape[0]

    # Initial state, with species and Hermite modes on separate axes.
    initial_conditions = {"Ck": parameters["Ck_0"].reshape(Ns, Np, Nm, Nn, Ny, Nx, Nz, Nl),
                          "Fk": parameters["Fk_0"].reshape(6, Ny, Nx, Nz, Nl)}

    # Define the time array for data output.
    time = jnp.linspace(0, parameters["t_max"], timesteps)

    # Arguments for the ODE system.
    args = {"col": parameters["collision_matrix"], **{name: parameters[name] for name in _RHS_PARAMETER_NAMES}}
    
    controllers = {
    True: PIDController(
        rtol=parameters["ode_tolerance"],
        atol=parameters["ode_tolerance"],
    ),
    False: ConstantStepSize(),
    }
    stepsize_controller = controllers[adaptive_time_step]
    
    # Solve the ODE system
    ode_system_partial = partial(ode_system, Nx=Nx, Ny=Ny, Nz=Nz, Nn=Nn, Nm=Nm, Np=Np, Ns=Ns, dims=dims)
    sol = diffeqsolve(
        ODETerm(ode_system_partial), solver=solver,
        stepsize_controller=stepsize_controller,
        t0=0, t1=parameters["t_max"], dt0=dt,
        y0=initial_conditions, args=args, saveat=SaveAt(ts=time),
        max_steps=1000000, progress_meter=TqdmProgressMeter())

    # Output Ck with species and Hermite modes flattened into one axis.
    Ck = sol.ys["Ck"].reshape(len(sol.ts), Ns * Nn * Nm * Np, Ny, Nx, Nz, Nl)
    Fk = sol.ys["Fk"]
    
    # Output results
    temporary_output = {"Ck": Ck, "Fk": Fk, "time": time}
    output = {**temporary_output, **parameters}
    diagnostics(output)
    
    return output
