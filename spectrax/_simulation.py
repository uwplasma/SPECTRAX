"""Time integration driver for the spectral Vlasov–Discontinuous_Galerkin system."""

import jax.numpy as jnp
from jax import jit, config
config.update("jax_enable_x64", True)
from functools import partial
from diffrax import (diffeqsolve, Dopri8, ODETerm,
                     SaveAt, PIDController, TqdmProgressMeter, NoProgressMeter, ConstantStepSize, is_okay)
from ._initialization import initialize_simulation_parameters
from ._model import Hermite_DG_system
from ._diagnostics import diagnostics
from ._parallel import check_mesh, shard_state, shard_args, shard_solution

__all__ = ["ode_system", "simulation"]

# Parameters passed to `Hermite_DG_system` under the same name as in the parameter dictionary.
_RHS_PARAMETER_NAMES = (
    "sqrt_n_plus", "sqrt_n_minus", "sqrt_m_plus", "sqrt_m_minus", "sqrt_p_plus", "sqrt_p_minus", "basis_idx",
    "inner_mm", "inner_pm", "inner_mp", "inner_pp", "di_inner_product", "triple_product",
    "Ax_p", "Ax_m", "Ay_p", "Ay_m", "Az_p", "Az_m", "R_p", "R_m",
    "Lx", "Ly", "Lz", "nu", "D", "alpha_s", "u_s", "ms", "qs", "Omega_ce",
)

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'dims', 'mesh'])
def ode_system(t, y, args, Nx, Ny, Nz, Nn, Nm, Np, Ns, dims, mesh=None):
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
    mesh : jax.sharding.Mesh, optional
        Device mesh the state is split over (see `simulation`).

    Returns
    -------
    dict
        Time derivative with the same structure and shapes as `y`.
    """
    y = shard_state(y, mesh)
    dCk, dFk = Hermite_DG_system(y["Ck"], y["Fk"], **args, Nx=Nx, Ny=Ny, Nz=Nz, Nn=Nn, Nm=Nm, Np=Np, Ns=Ns, dims=dims)
    return shard_state({"Ck": dCk, "Fk": dFk}, mesh)

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'N_DG', 'dims', 'timesteps', 'solver', 'adaptive_time_step', 'mesh'])
def simulation(input_parameters={}, Nx=33, Ny=1, Nz=1, Nn=20, Nm=1, Np=1, Ns=2, N_DG=2, dims=1, timesteps=200, dt = 0.01, solver=Dopri8(), adaptive_time_step=True, mesh=None):
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
    mesh : jax.sharding.Mesh, optional
        Device mesh from `make_mesh` to split the simulation over: the species and the elements along
        x, y and z are divided between the devices of the corresponding mesh axes, and XLA partitions the
        computation accordingly. The grid must split evenly over the mesh. The returned `Ck` and `Fk` stay
        distributed in the same way (converting them with `numpy.asarray` gathers them). Default None runs
        on one device.

    Returns
    -------
    dict
        Dictionary containing the evolved coefficients (`Ck`, `Fk`), time samples,
        perturbation diagnostics, and all simulation parameters.
    """
    if mesh is not None:
        check_mesh(mesh, Nx, Ny, Nz, Ns)

    # **Initialize simulation parameters**
    parameters = initialize_simulation_parameters(input_parameters, Nx, Ny, Nz, Nn, Nm, Np, Ns, N_DG, dims, timesteps, dt)

    # Define total number of basis modes
    Nl = parameters["basis_idx"].shape[0]

    # Initial state, with species and Hermite modes on separate axes, laid out over the mesh if there is one.
    initial_conditions = shard_state({"Ck": parameters["Ck_0"].reshape(Ns, Np, Nm, Nn, Ny, Nx, Nz, Nl),
                                      "Fk": parameters["Fk_0"].reshape(6, Ny, Nx, Nz, Nl)}, mesh)

    # Define the time array for data output.
    time = jnp.linspace(0, parameters["t_max"], timesteps)

    # Arguments for the ODE system.
    args = shard_args({"col": parameters["collision_matrix"], **{name: parameters[name] for name in _RHS_PARAMETER_NAMES}}, mesh)

    controllers = {
    True: PIDController(
        rtol=parameters["ode_tolerance"],
        atol=parameters["ode_tolerance"],
    ),
    False: ConstantStepSize(),
    }
    stepsize_controller = controllers[adaptive_time_step]
    
    # Solve the ODE system
    ode_system_partial = partial(ode_system, Nx=Nx, Ny=Ny, Nz=Nz, Nn=Nn, Nm=Nm, Np=Np, Ns=Ns, dims=dims, mesh=mesh)
    sol = diffeqsolve(
        ODETerm(ode_system_partial), solver=solver,
        stepsize_controller=stepsize_controller,
        t0=0, t1=parameters["t_max"], dt0=dt,
        y0=initial_conditions, args=args, saveat=SaveAt(ts=time),
        max_steps=1000000, progress_meter=TqdmProgressMeter(), throw=False)

    # Raise if the integration failed (e.g. too many steps). The check is attached to `time` rather than to the
    # solution, as `throw=True` would do, because checking the solution gathers all of it onto one device.
    time = sol.result.error_if(time, jnp.invert(is_okay(sol.result)))

    # Output Ck with species and Hermite modes flattened into one axis, still distributed over the mesh if there is one.
    solution = shard_solution({"Ck": sol.ys["Ck"].reshape(len(sol.ts), Ns * Nn * Nm * Np, Ny, Nx, Nz, Nl),
                               "Fk": sol.ys["Fk"]}, mesh)

    # Output results
    temporary_output = {**solution, "time": time}
    output = {**temporary_output, **parameters}
    diagnostics(output)
    
    return output
