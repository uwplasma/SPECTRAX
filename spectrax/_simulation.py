"""Time integration driver for the spectral Vlasov–Maxwell system."""

import numpy as np
import jax.numpy as jnp
from jax import jit, config
config.update("jax_enable_x64", True)
from functools import partial
from diffrax import (diffeqsolve, Dopri8, ODETerm, SaveAt, PIDController, TqdmProgressMeter,
                     NoProgressMeter, ConstantStepSize, RecursiveCheckpointAdjoint)
from ._initialization import initialize_simulation_parameters
from ._model import plasma_current, Hermite_Fourier_system, uniform_acceleration
from ._diagnostics import diagnostics

__all__ = ["cross_product", "ode_system", "simulation"]

def _twothirds_mask(Ny: int, Nx: int, Nz: int):
    """Boolean 2/3 de-aliasing mask on the solver's (Ny, Nx//2+1, Nz) Fourier layout.

    Keeps integer mode indices with ``3*|k| < N`` in each direction, i.e. ``|k| <= (N-1)//3``.
    With this strict bound a product of two retained modes never aliases back onto a retained
    mode (``2*K < N - K``). The inclusive bound ``|k| <= N//3`` fails when ``N`` is divisible by
    three: ``K + K = 2K`` aliases onto ``-K``. The real FFT runs along x, so only ``k_x >= 0`` is stored.
    """
    ky = np.fft.fftfreq(Ny, d=1.0 / Ny).round().astype(int)[:, None, None]
    kx = np.arange(Nx // 2 + 1)[None, :, None]
    kz = np.fft.fftfreq(Nz, d=1.0 / Nz).round().astype(int)[None, None, :]
    return (3 * np.abs(ky) < Ny) & (3 * kx < Nx) & (3 * np.abs(kz) < Nz)

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

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'frame'])
def ode_system(Nx, Ny, Nz, Nn, Nm, Np, Ns, t, Ck_Fk, args, frame="fixed"):
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
    Ck_Fk : tuple[jnp.ndarray, jnp.ndarray] or tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Hermite and electromagnetic field coefficients, optionally followed by the basis state
        ``stack([u_s, alpha_s])`` of shape ``(2, 3*Ns)``. When present it replaces the
        ``alpha_s, u_s`` in ``args`` for the kinetic operator and the current.
    frame : {"fixed", "pump"}
        With a basis state: ``"fixed"`` keeps it constant; ``"pump"`` moves each centre with the
        uniform acceleration ``(q/m)(E0 + u x B0)`` and removes that force from the kinetic
        operator (exact change of variables, see :func:`spectrax._model.uniform_acceleration`).
    args : tuple
        Cached parameter tuple produced in `simulation` providing physical constants,
        grids, and helper arrays.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Derivatives matching the Hermite and field coefficient arrays.
    """

    (qs, nu, D, Omega_cs, alpha_s, u_s,
     Lx, Ly, Lz, kx_grid, ky_grid, kz_grid, k2_grid, nabla, col,
     sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus
    ) = args[7:]

    Ck, Fk = Ck_Fk[:2]
    moving = len(Ck_Fk) == 3
    if moving:  # basis state (u_s, alpha_s), shape (2, 3*Ns); complex like (Ck, Fk), imaginary part zero
        u_s, alpha_s = jnp.real(Ck_Fk[2][0]), jnp.real(Ck_Fk[2][1])

    # Build the 2/3 mask once per call (JIT will constant-fold it since Nx/Ny/Nz are static)
    mask23 = _twothirds_mask(Ny, Nx, Nz)

    F = jnp.fft.irfftn(Fk * mask23, s=(Nz, Ny, Nx), axes=(-1, -3, -2), norm="forward")
    C = jnp.fft.irfftn(Ck * mask23, s=(Nz, Ny, Nx), axes=(-1, -3, -2), norm="forward")

    pump = moving and frame == "pump"
    u_dot, F0 = uniform_acceleration(F, u_s, qs, Omega_cs, Ns) if pump else (None, None)
    dCk_s_dt = Hermite_Fourier_system(Ck, C, F, kx_grid, ky_grid, kz_grid, k2_grid, col, 
                                      sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus, 
                                      Lx, Ly, Lz, nu, D, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, Ns, mask23=mask23, F0=F0)

    dBk_dt = -1j * cross_product(nabla, Fk[:3])
    
    current = plasma_current(qs, alpha_s, u_s, Ck, Nn, Nm, Np, Ns)
    dEk_dt = 1j * cross_product(nabla, Fk[3:]) - current / Omega_cs[0]

    dFk_dt = jnp.concatenate([dEk_dt, dBk_dt], axis=0)
    if pump:
        return dCk_s_dt, dFk_dt, jnp.stack([u_dot, jnp.zeros_like(u_dot)]).astype(Ck_Fk[2].dtype)
    if moving:
        return dCk_s_dt, dFk_dt, jnp.zeros_like(Ck_Fk[2])
    return dCk_s_dt, dFk_dt

def _stepsize_controller(adaptive_time_step, tolerance, dtmin=None):
    """PID control at `tolerance` with an optional step floor `dtmin` (None: no floor), or constant steps."""
    if not adaptive_time_step:
        return ConstantStepSize()
    return PIDController(rtol=tolerance, atol=tolerance, dtmin=dtmin, force_dtmin=False)

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'timesteps', 'solver', 'adaptive_time_step',
                                'dtmin', 'max_steps', 'throw', 'adjoint', 'progress_meter', 'frame'])
def simulation(input_parameters={}, Nx=33, Ny=1, Nz=1, Nn=20, Nm=1, Np=1, Ns=2, 
               timesteps=200, dt = 0.01, solver=Dopri8(), adaptive_time_step=True, dtmin=None, max_steps=1000000,
               throw=True, adjoint=RecursiveCheckpointAdjoint(), progress_meter=TqdmProgressMeter(), frame=None):
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
    adjoint : diffrax.AbstractAdjoint, optional
        How gradients are taken through the solve. Default `RecursiveCheckpointAdjoint()`;
        pass `RecursiveCheckpointAdjoint(checkpoints=k)` to bound reverse-mode memory, or
        `ForwardMode()` for `jax.jacfwd`. The objective must be a real scalar.
    progress_meter : diffrax.AbstractProgressMeter, optional
        Default `TqdmProgressMeter()`; `NoProgressMeter()` inside optimisation loops.
    frame : {None, "fixed", "pump"}, optional
        None (default): the Hermite basis centres ``u_s`` and widths ``alpha_s`` are constants.
        ``"fixed"``: they are carried in the solver state as ``stack([u_s, alpha_s])`` and
        returned per saved time as ``basis_u`` and ``basis_alpha`` (shape ``(Nt, 3*Ns)``);
        with zero rates the kinetic right-hand side is bitwise that of ``frame=None``.
        ``"pump"``: each centre follows the uniform acceleration ``(q/m)(E0 + u x B0)`` (oscillating-
        centre frame), and that uniform force is removed from the kinetic operator.

    Returns
    -------
    dict
        Dictionary containing the evolved coefficients (`Ck`, `Fk`), time samples,
        solver statistics, perturbation diagnostics, and all simulation parameters.
    """
    
    # **Initialize simulation parameters**
    parameters = initialize_simulation_parameters(input_parameters, Nx, Ny, Nz, Nn, Nm, Np, Ns, timesteps, dt)

    # Project the initial state onto the de-aliased active subspace so distribution, current and
    # fields share one set of Fourier modes; report the removed content instead of hiding it.
    mask23 = _twothirds_mask(Ny, Nx, Nz)
    Ck_0, Fk_0 = jnp.asarray(parameters["Ck_0"]), jnp.asarray(parameters["Fk_0"])
    initial_projection_residual = jnp.sqrt(jnp.sum(jnp.abs(Ck_0 * ~mask23) ** 2)
                                           + jnp.sum(jnp.abs(Fk_0 * ~mask23) ** 2))
    # Preserve the species and Hermite axes for the integrator.
    initial_conditions = (
        (Ck_0 * mask23).reshape(Ns, Np, Nm, Nn, Ny, Nx//2+1, Nz),
        (Fk_0 * mask23).reshape(6, Ny, Nx//2+1, Nz),
    )
    if frame not in (None, "fixed", "pump"):
        raise ValueError(f"frame must be None, 'fixed' or 'pump', got {frame!r}")
    if frame is not None:
        basis_0 = jnp.stack([parameters["u_s"], parameters["alpha_s"]]).astype(jnp.complex128)
        initial_conditions = initial_conditions + (basis_0,)

    # Define the time array for data output.
    time = jnp.linspace(0, parameters["t_max"], timesteps)
    
    # Arguments for the ODE system.
    args = (Nx, Ny, Nz, Nn, Nm, Np, Ns, parameters["qs"], parameters["nu"], parameters["D"],
            parameters["Omega_cs"], parameters["alpha_s"], parameters["u_s"], 
            parameters["Lx"], parameters["Ly"], parameters["Lz"],
            parameters["kx_grid"], parameters["ky_grid"], parameters["kz_grid"], 
            parameters["k2_grid"], parameters["nabla"], parameters["collision_matrix"], 
            parameters["sqrt_n_plus"], parameters["sqrt_n_minus"],
            parameters["sqrt_m_plus"], parameters["sqrt_m_minus"],
            parameters["sqrt_p_plus"], parameters["sqrt_p_minus"])
    

    stepsize_controller = _stepsize_controller(adaptive_time_step, parameters["ode_tolerance"], dtmin)

    # Solve the ODE system
    ode_system_partial = partial(ode_system, Nx, Ny, Nz, Nn, Nm, Np, Ns, frame=frame or "fixed")
    sol = diffeqsolve(
        ODETerm(ode_system_partial), solver=solver,
        stepsize_controller=stepsize_controller,
        t0=0, t1=parameters["t_max"], dt0=dt,
        y0=initial_conditions, args=args, saveat=SaveAt(ts=time),
        max_steps=max_steps, adjoint=adjoint, progress_meter=progress_meter, throw=throw)
        
    # Reshape the solution to extract Ck and Fk
    Ck = sol.ys[0].reshape(len(sol.ts), Ns * Nn * Nm * Np, Ny, Nx//2+1, Nz)
    Fk = sol.ys[1]
    
    # Set n = 0, k = 0 mode to zero to get array with time evolution of perturbation.
    dCk = Ck.at[:, jnp.arange(Ns) * Nn * Nm * Np, 0, 0, 0].set(0)
    
    # Output results
    # Diffrax fills saves after a failed solve with inf; flag them so they are never read as data.
    valid_times = jnp.isfinite(sol.ts)
    temporary_output = {"Ck": Ck, "Fk": Fk, "time": time, "dCk": dCk, "solver_stats": sol.stats,
                        "initial_projection_residual": initial_projection_residual,
                        "solver_result": sol.result, "valid_times": valid_times,
                        "num_valid_times": jnp.sum(valid_times)}
    if frame is not None:
        temporary_output.update(basis_u=jnp.real(sol.ys[2][:, 0]), basis_alpha=jnp.real(sol.ys[2][:, 1]))
    output = {**temporary_output, **parameters, "Nx": Nx}  # static grid length for the rFFT weights
    diagnostics(output)
    return output
