import jax.numpy as jnp
from jax import jit
from functools import partial
try: import tomllib
except ModuleNotFoundError: import pip._vendor.tomli as tomllib
import diffrax
import inspect
from .midpoint_solver import ImplicitMidpoint

__all__ = ["load_parameters", "initialize_simulation_parameters"]

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz','Nn', 'Nm', 'Np', 'Ns', 'timesteps'])
def initialize_simulation_parameters(user_parameters={}, Nx=33, Ny=1, Nz=1, Nn=50, Nm=1, Np=1, Ns=2, timesteps=500, dt=0.01):
    """
    Assemble the parameter dictionary used to run a Hermite-Fourier Vlasov-Maxwell
    simulation, starting from library defaults and overriding them with user input.
    The defaults include a two-stream perturbation, precomputed spectral grids, and
    helper tables required by the RHS evaluation. Derived quantities are evaluated
    after merging with any user-provided overrides so that dependent fields remain
    consistent.

    Parameters
    ----------
    user_parameters : Mapping, optional
        Optional dictionary of parameter overrides. Any key present here replaces
        the corresponding default before derived quantities are computed.
    Nx, Ny, Nz : int, optional
        Number of Fourier modes retained along each spatial direction.
    Nn, Nm, Np : int, optional
        Number of Hermite modes along each velocity-space axis.
    Ns : int, optional
        Number of particle species represented in the simulation.
    timesteps : int, optional
        Number of time samples to store in the solution.
    dt : float, optional
        Initial guess for the integrator time step.

    Returns
    -------
    dict
        Dictionary containing the merged parameters, derived helper arrays, and
        initial spectral coefficients such as `Ck_0` and `Fk_0`.
    """
    # Define all default parameters in a single dictionary
    default_parameters = {
        "Lx": 4 * jnp.pi,
        "Ly": 1.0,
        "Lz": 1.0,
        "mi_me": 1.0,
        "Ti_Te": 1.0,
        "qs": jnp.array([-1, -1]),
        "alpha_e": jnp.array([0.707107, 0.707107, 0.707107]),
        "alpha_s": lambda p: jnp.concatenate([
            p["alpha_e"],
            p["alpha_e"] * jnp.sqrt(p["Ti_Te"] / p["mi_me"])
        ]),
        "u_s": jnp.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
        "Omega_cs": lambda p: jnp.array([1.0, 1.0 / p["mi_me"]]),
        "nu": 3.0,
        "D": 0.0,
        "t_max": 0.3,
        "nx": 1,
        "ny": 0,
        "nz": 0,
        "dn1": 0.001,
        "dn2": 0.001,
        "ode_tolerance": 1e-12,
        "vte": lambda p: p["alpha_s"][0] / jnp.sqrt(2),
        "vti": lambda p: p["vte"] * jnp.sqrt(1 / p["mi_me"]),
    }
    
    # Initialize distribution function as a two-stream instability
    indices = jnp.array([int((Nx-1)/2-default_parameters["nx"]), int((Nx-1)/2+default_parameters["nx"])])
    dn1     = default_parameters["dn1"]
    dn2     = default_parameters["dn2"]
    alpha_e = default_parameters["alpha_e"]
    values  = (dn1 + dn2) * default_parameters["Lx"] / (4 * jnp.pi * default_parameters["nx"] * default_parameters["Omega_cs"](default_parameters)[0])
    Fk_0    = jnp.zeros((6, 1, Nx, 1), dtype=jnp.complex128).at[0, 0, indices, 0].set(values)
    C10     = jnp.array([
            0 + 1j * (1 / (2 * alpha_e[0] ** 3)) * dn1,
            1 / (alpha_e[0] ** 3) + 0 * 1j,
            0 - 1j * (1 / (2 * alpha_e[0] ** 3)) * dn1
    ])
    C20     = jnp.array([
            0 + 1j * (1 / (2 * alpha_e[0] ** 3)) * dn2,
            1 / (alpha_e[0] ** 3) + 0 * 1j,
            0 - 1j * (1 / (2 * alpha_e[0] ** 3)) * dn2
    ])
    indices = jnp.array([int((Nx-1)/2-default_parameters["nx"]), int((Nx-1)/2), int((Nx-1)/2+default_parameters["nx"])])
    Ck_0    = jnp.zeros((2 * Nn, 1, Nx, 1), dtype=jnp.complex128)
    Ck_0    = Ck_0.at[0,  0, indices, 0].set(C10)
    Ck_0    = Ck_0.at[Nn, 0, indices, 0].set(C20)
    
    default_parameters.update({
        "Ck_0": Ck_0, "Fk_0": Fk_0, "Ns": Ns,
        "timesteps": timesteps, "dt": dt, 
        "Nx": Nx, "Ny": Ny, "Nz": Nz,
        "Nn": Nn, "Nm": Nm, "Np": Np,
    })

    # Merge user-provided parameters into the default dictionary
    parameters = {**default_parameters, **user_parameters}

    Lx, Ly, Lz = parameters["Lx"], parameters["Ly"], parameters["Lz"]
    
    # Compute derived parameters based on user-provided or default values
    for key, value in parameters.items():
        if callable(value):  # If the value is a lambda function, compute it
            parameters[key] = value(parameters)
        if isinstance(value, list):
            parameters[key] = jnp.array(value)
    kx_simulation = (jnp.arange(-Nx//2, Nx//2) + 1) * 2 * jnp.pi
    ky_simulation = (jnp.arange(-Ny//2, Ny//2) + 1) * 2 * jnp.pi
    kz_simulation = (jnp.arange(-Nz//2, Nz//2) + 1) * 2 * jnp.pi  
    kx_grid, ky_grid, kz_grid = jnp.meshgrid(kx_simulation, ky_simulation, kz_simulation, indexing='xy')
    k2_grid = kx_grid**2 + ky_grid**2 + kz_grid**2
    nabla = jnp.array([kx_grid / Lx, ky_grid / Ly, kz_grid / Lz])

    def precompute_collisions(Nn, Nm, Np):
        p = jnp.arange(Np)[:, None, None]
        m = jnp.arange(Nm)[None, :, None]
        n = jnp.arange(Nn)[None, None, :]
        def safe(N, i):
            term = i * (i - 1) * (i - 2)
            denom = (N - 1) * (N - 2) * (N - 3)
            return jnp.where(N > 3, term / denom, 0.0)
        col = safe(Nn, n) + safe(Nm, m) + safe(Np, p)
        return col
    
    def build_coeff_tables(Nn, Nm, Np):
        p = jnp.arange(Np)[None, :, None, None, None, None, None]
        m = jnp.arange(Nm)[None, None, :, None, None, None, None]
        n = jnp.arange(Nn)[None, None, None, :, None, None, None]

        sqrt_n_plus  = jnp.sqrt(n+1)
        sqrt_n_minus = jnp.sqrt(n) 
        sqrt_m_plus  = jnp.sqrt(m+1)
        sqrt_m_minus = jnp.sqrt(m)
        sqrt_p_plus  = jnp.sqrt(p+1)
        sqrt_p_minus = jnp.sqrt(p)

        return sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus
    
    sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus = build_coeff_tables(Nn, Nm, Np)

    parameters.update({
        "kx_grid": kx_grid, "ky_grid": ky_grid, "kz_grid": kz_grid, "k2_grid": k2_grid, 
        "nabla": nabla, "collision_matrix": precompute_collisions(Nn, Nm, Np),
        "sqrt_n_plus": sqrt_n_plus, "sqrt_n_minus": sqrt_n_minus,
        "sqrt_m_plus": sqrt_m_plus, "sqrt_m_minus": sqrt_m_minus,
        "sqrt_p_plus": sqrt_p_plus, "sqrt_p_minus": sqrt_p_minus,
    })

    return parameters

def load_parameters(input_file):
    """
    Load simulation input parameters and solver configuration from a TOML file.

    Parameters
    ----------
    input_file : str or pathlib.Path
        Path to the TOML file containing simulation parameters.

    Returns
    -------
    tuple[dict, dict]
        A pair `(input_parameters, solver_parameters)` where `solver_parameters`
        includes an instantiated Diffrax solver ready for `diffeqsolve`.
    """
    parameters = tomllib.load(open(input_file, "rb"))
    input_parameters = parameters['input_parameters']
    solver_parameters = parameters['solver_parameters']

    # NEW: whether to use adaptive time-stepping or constant dt
    # Default is True to preserve the current behavior.
    adaptive_time_step = solver_parameters.get("adaptive_time_step", True)
    solver_parameters["adaptive_time_step"] = adaptive_time_step


    def get_solver_class(name: str):
        for cls_name, cls in inspect.getmembers(diffrax, inspect.isclass):
            if issubclass(cls, diffrax.AbstractSolver) and cls is not diffrax.AbstractSolver and cls_name == name: return cls()
            elif name == "ImplicitMidpoint": return ImplicitMidpoint(rtol=input_parameters["ode_tolerance"], atol=input_parameters["ode_tolerance"])
        raise ValueError(f"Solver '{name}' is not supported. Choose from Diffrax solvers.")
    solver_parameters["solver"] = get_solver_class(solver_parameters.get("solver", "Tsit5"))
    
    return input_parameters, solver_parameters
