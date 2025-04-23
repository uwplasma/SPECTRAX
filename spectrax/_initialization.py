import jax.numpy as jnp
try: import tomllib
except ModuleNotFoundError: import pip._vendor.tomli as tomllib
import diffrax
import inspect

__all__ = ["load_parameters", "initialize_simulation_parameters"]

def initialize_simulation_parameters(user_parameters={}, Nx=33, Ny=1, Nz=1, Nn=50, Nm=1, Np=1, Ns=2, timesteps=500):
    """
    Initialize the simulation parameters for a Vlasov solver with Hermite polynomials, 
    combining user-provided values with predefined defaults. This function 
    ensures all required parameters are set and automatically calculates 
    derived parameters based on the inputs.

    The function uses lambda functions to define derived parameters that 
    depend on other parameters. These lambda functions are evaluated after 
    merging user-provided parameters with the defaults, ensuring derived 
    parameters are consistent with any overrides.

    Parameters:
    ----------
    user_parameters : dict
        Dictionary containing user-specified parameters. Any parameter not provided
        will default to predefined values.

    Returns:
    -------
    parameters : dict
        Dictionary containing all simulation parameters, with user-provided values
        overriding defaults.
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
        "nu": 2.0,
        "D": 0.0,
        "t_max": 0.3,
        "kx": lambda p: 2 * jnp.pi / p["Lx"],
        "dn1": 0.001,
        "dn2": 0.001,
        "ode_tolerance": 1e-6,
        "vte": lambda p: p["alpha_s"][0] / jnp.sqrt(2),
        "vti": lambda p: p["vte"] * jnp.sqrt(1 / p["mi_me"]),
    }
    
    # Initialize distribution function as a two-stream instability
    indices = jnp.array([int((Nx-1)/2-1), int((Nx-1)/2+1)])
    dn1     = default_parameters["dn1"]
    dn2     = default_parameters["dn2"]
    alpha_e = default_parameters["alpha_e"]
    values  = (dn1 + dn2) / (2 * default_parameters["kx"](default_parameters) * default_parameters["Omega_cs"](default_parameters)[0])
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
    indices = jnp.array([int((Nx-1)/2-1), int((Nx-1)/2), int((Nx-1)/2+1)])
    Ck_0    = jnp.zeros((2 * Nn, 1, Nx, 1), dtype=jnp.complex128)
    Ck_0    = Ck_0.at[0,  0, indices, 0].set(C10)
    Ck_0    = Ck_0.at[Nn, 0, indices, 0].set(C20)
    
    default_parameters.update({
        "Ck_0": Ck_0, "Fk_0": Fk_0,
        "timesteps": timesteps, "Ns": Ns,
        "Nx": Nx, "Ny": Ny, "Nz": Nz,
        "Nn": Nn, "Nm": Nm, "Np": Np,
    })

    # Merge user-provided parameters into the default dictionary
    parameters = {**default_parameters, **user_parameters}
    
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
    parameters.update({
        "kx_grid": kx_grid, "ky_grid": ky_grid, "kz_grid": kz_grid
    })

    return parameters

def load_parameters(input_file):
    """
    Load parameters from a TOML file.

    Parameters:
    ----------
    input_file : str
        Path to the TOML file containing simulation parameters.

    Returns:
    -------
    parameters : dict
        Dictionary containing simulation parameters.
    """
    parameters = tomllib.load(open(input_file, "rb"))
    input_parameters = parameters['input_parameters']
    solver_parameters = parameters['solver_parameters']

    def get_solver_class(name: str):
        for cls_name, cls in inspect.getmembers(diffrax, inspect.isclass):
            if issubclass(cls, diffrax.AbstractSolver) and cls is not diffrax.AbstractSolver and cls_name == name: return cls
        raise ValueError(f"Solver '{name}' is not supported. Choose from Diffrax solvers.")
    solver_parameters["solver"] = get_solver_class(solver_parameters.get("solver", "Tsit5"))
    
    return input_parameters, solver_parameters
