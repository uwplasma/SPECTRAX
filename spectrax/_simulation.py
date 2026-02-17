import jax.numpy as jnp
from jax import jit, config
config.update("jax_enable_x64", True)
from functools import partial
from diffrax import (diffeqsolve, Tsit5, Dopri8, ODETerm,
                     SaveAt, PIDController, TqdmProgressMeter, NoProgressMeter, ConstantStepSize)
from ._initialization import initialize_simulation_parameters
from ._model import plasma_current, Hermite_Fourier_system
from ._diagnostics import diagnostics

__all__ = ["cross_product", "ode_system", "simulation"]

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz'])
def _twothirds_mask(Ny: int, Nx: int, Nz: int):
    """Return a boolean mask in fftshifted (zero-centered) k-ordering that keeps |k|<=N//3 in each dim."""
    def centered_modes(N):
        # integer mode numbers in fftshift ordering: [-N//2, ..., -1, 0, 1, ..., N//2-1]
        k = jnp.fft.fftfreq(N) * N
        return jnp.fft.fftshift(k)

    ky = centered_modes(Ny)[:, None, None]
    kx = centered_modes(Nx)[None, :, None]
    kz = centered_modes(Nz)[None, None, :]

    # cutoffs (keep indices with |k| <= floor(N/3)); if N<3 this naturally keeps only k=0
    cy = Ny // 3
    cx = Nx // 3
    cz = Nz // 3

    return (jnp.abs(ky) <= cy) & (jnp.abs(kx) <= cx) & (jnp.abs(kz) <= cz)

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
        Number of retained Fourier modes per spatial dimension.
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

    total_Ck_size = Nn * Nm * Np * Ns * Nx * Ny * Nz
    Ck = Ck_Fk[:total_Ck_size].reshape(Nn * Nm * Np * Ns, Ny, Nx, Nz)
    Fk = Ck_Fk[total_Ck_size:].reshape(6, Ny, Nx, Nz)


    # Build the 2/3 mask once per call (JIT will constant-fold it since Nx/Ny/Nz are static)
    mask23 = _twothirds_mask(Ny, Nx, Nz)

    F = jnp.fft.ifftn(jnp.fft.ifftshift(Fk * mask23, axes=(-3, -2, -1)), axes=(-3, -2, -1), norm="forward")
    C = jnp.fft.ifftn(jnp.fft.ifftshift(Ck * mask23, axes=(-3, -2, -1)), axes=(-3, -2, -1), norm="forward")

    dCk_s_dt = Hermite_Fourier_system(Ck, C, F, kx_grid, ky_grid, kz_grid, k2_grid, col, 
                                      sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus, 
                                      Lx, Ly, Lz, nu, D, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, Ns, mask23=mask23)

    # nabla = jnp.array([kx_grid / Lx, ky_grid / Ly, kz_grid / Lz])
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
        perturbation diagnostics, and all simulation parameters.
    """
    
    # **Initialize simulation parameters**
    parameters = initialize_simulation_parameters(input_parameters, Nx, Ny, Nz, Nn, Nm, Np, Ns, timesteps, dt)

    # Combine initial conditions.
    initial_conditions = jnp.concatenate([parameters["Ck_0"].flatten(), parameters["Fk_0"].flatten()])

    # Define the time array for data output.
    time = jnp.linspace(0, parameters["t_max"], timesteps)
    
    # Arguments for the ODE system.
    args = (Nx, Ny, Nz, Nn, Nm, Np, Ns, parameters["qs"], parameters["nu"], parameters["D"], parameters["Omega_cs"], parameters["alpha_s"],
            parameters["u_s"], parameters["Lx"], parameters["Ly"], parameters["Lz"],
            parameters["kx_grid"], parameters["ky_grid"], parameters["kz_grid"], parameters["k2_grid"], parameters["nabla"], parameters["collision_matrix"], 
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
