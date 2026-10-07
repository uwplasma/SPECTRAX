"""Time integration driver for the spectral Vlasov–Maxwell system."""

import numpy as np
import jax.numpy as jnp
from jax import jit, config
config.update("jax_enable_x64", True)
from functools import partial
from diffrax import (diffeqsolve, Dopri8, ODETerm,
                     SaveAt, PIDController, TqdmProgressMeter, NoProgressMeter, ConstantStepSize)
from ._initialization import initialize_simulation_parameters
from ._model import plasma_current, Hermite_Fourier_system
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

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns'])
def ode_system(Nx, Ny, Nz, Nn, Nm, Np, Ns, t, Ck_Fk, args):
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

    (qs, nu, D, Omega_cs, alpha_s, u_s,
     Lx, Ly, Lz, kx_grid, ky_grid, kz_grid, k2_grid, nabla, col,
     sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus
    ) = args[7:]

    total_Ck_size = Nn * Nm * Np * Ns * (Nx//2+1) * Ny * Nz
    Ck = Ck_Fk[:total_Ck_size].reshape(Nn * Nm * Np * Ns, Ny, Nx//2+1, Nz)
    Fk = Ck_Fk[total_Ck_size:].reshape(6, Ny, Nx//2+1, Nz)


    # Build the 2/3 mask once per call (JIT will constant-fold it since Nx/Ny/Nz are static)
    mask23 = _twothirds_mask(Ny, Nx, Nz)

    F = jnp.fft.irfftn(Fk * mask23, s=(Nz, Ny, Nx), axes=(-1, -3, -2), norm="forward")
    C = jnp.fft.irfftn(Ck * mask23, s=(Nz, Ny, Nx), axes=(-1, -3, -2), norm="forward")

    dCk_s_dt = Hermite_Fourier_system(Ck, C, F, kx_grid, ky_grid, kz_grid, k2_grid, col, 
                                      sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus, 
                                      Lx, Ly, Lz, nu, D, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, Ns, mask23=mask23)

    dBk_dt = -1j * cross_product(nabla, Fk[:3])
    
    current = plasma_current(qs, alpha_s, u_s, Ck, Nn, Nm, Np, Ns)
    dEk_dt = 1j * cross_product(nabla, Fk[3:]) - current / Omega_cs[0]

    dFk_dt = jnp.concatenate([dEk_dt, dBk_dt], axis=0)
    dy_dt  = jnp.concatenate([dCk_s_dt.reshape(-1), dFk_dt.reshape(-1)])

    return dy_dt

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'timesteps', 'solver', 'adaptive_time_step'])
def simulation(input_parameters={}, Nx=33, Ny=1, Nz=1, Nn=20, Nm=1, Np=1, Ns=2, 
               timesteps=200, dt = 0.01, solver=Dopri8(), adaptive_time_step=True):
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
        solver statistics, perturbation diagnostics, and all simulation parameters.
    """
    
    # **Initialize simulation parameters**
    parameters = initialize_simulation_parameters(input_parameters, Nx, Ny, Nz, Nn, Nm, Np, Ns, timesteps, dt)

    # Combine initial conditions.
    # Project the initial state onto the de-aliased active subspace so distribution, current and
    # fields share one set of Fourier modes; report the removed content instead of hiding it.
    mask23 = _twothirds_mask(Ny, Nx, Nz)
    Ck_0, Fk_0 = jnp.asarray(parameters["Ck_0"]), jnp.asarray(parameters["Fk_0"])
    initial_projection_residual = jnp.sqrt(jnp.sum(jnp.abs(Ck_0 * ~mask23) ** 2)
                                           + jnp.sum(jnp.abs(Fk_0 * ~mask23) ** 2))
    initial_conditions = jnp.concatenate([(Ck_0 * mask23).flatten(), (Fk_0 * mask23).flatten()])

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
    

    controllers = {
    True: PIDController(
        rtol=parameters["ode_tolerance"],
        atol=parameters["ode_tolerance"],
    ),
    False: ConstantStepSize(),
    }
    stepsize_controller = controllers[adaptive_time_step]

    # Solve the ODE system
    ode_system_partial = partial(ode_system, Nx, Ny, Nz, Nn, Nm, Np, Ns)
    sol = diffeqsolve(
        ODETerm(ode_system_partial), solver=solver,
        stepsize_controller=stepsize_controller,
        t0=0, t1=parameters["t_max"], dt0=dt,
        y0=initial_conditions, args=args, saveat=SaveAt(ts=time),
        max_steps=1000000, progress_meter=TqdmProgressMeter())
        
    # Reshape the solution to extract Ck and Fk
    Ck = sol.ys[:,:(-6 * (Nx//2+1) * Ny * Nz)].reshape(len(sol.ts), Ns * Nn * Nm * Np, Ny, Nx//2+1, Nz)
    Fk = sol.ys[:,(-6 * (Nx//2+1) * Ny * Nz):].reshape(len(sol.ts), 6, Ny, Nx//2+1, Nz)
    
    # Set n = 0, k = 0 mode to zero to get array with time evolution of perturbation.
    dCk = Ck.at[:, 0, 0, 0, 0].set(0)
    dCk = dCk.at[:, Nn * Nm * Np, 0, 0, 0].set(0)
    
    # Output results
    temporary_output = {"Ck": Ck, "Fk": Fk, "time": time, "dCk": dCk, "solver_stats": sol.stats,
                         "initial_projection_residual": initial_projection_residual}
    output = {**temporary_output, **parameters}
    diagnostics(output)
    return output
