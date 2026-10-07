"""Simulation parameter initialization and TOML configuration loading."""

import jax.numpy as jnp
from jax import jit
from functools import partial
try: import tomllib
except ModuleNotFoundError: import pip._vendor.tomli as tomllib
import diffrax
import inspect
from .midpoint_solver import ImplicitMidpoint

__all__ = ["load_parameters", "initialize_simulation_parameters", "hypercollision_spectrum",
           "filter_spectrum", "exponential_filter", "species_initial_state"]


def hypercollision_spectrum(Nn, Nm, Np, order=2):
    """Damping spectrum of the numerical hypercollision ``-nu * col * C``, shape ``(Np, Nm, Nn)``.

    Per velocity axis with N modes, ``col_i(n) = n!/(n-2*order+1)! / [(N-1)!/(N-2*order)!]``: zero for
    ``n <= 2*order - 2`` and 1 for the top mode ``n = N - 1`` (an axis with ``N <= 2*order - 1`` gets
    none). The axes are summed. ``order = 2`` is ``n(n-1)(n-2)/[(N-1)(N-2)(N-3)]``, the default; it
    leaves every moment of total degree <= 2 (density, momentum, full pressure tensor) unchanged for
    any basis centre and width. ``order = 1`` (``n``, Lenard-Bernstein-like) would damp the momentum
    and energy rows and is rejected. Pass the result as ``"collision_matrix"`` to override the default.
    """
    if int(order) != order or order < 2:
        raise ValueError(f"hypercollision order must be an integer >= 2, got {order}")
    r = 2 * order - 1  # number of factors

    def axis(N, i):
        term, denom = i, N - 1
        for j in range(1, r):
            term, denom = term * (i - j), denom * (N - 1 - j)
        return term / denom if N > r else jnp.zeros(i.shape, float)

    p = jnp.arange(Np)[:, None, None]
    m = jnp.arange(Nm)[None, :, None]
    n = jnp.arange(Nn)[None, None, :]
    return axis(Nn, n) + axis(Nm, m) + axis(Np, p)

def filter_spectrum(Nn, Nm, Np, order=36, keep=2):
    """Shape ``s(n)`` of a Hou-Li-type exponential filter, shape ``(Np, Nm, Nn)``.

    Per velocity axis with N modes, ``s_i(n) = (n/(N-1))**order`` for ``n > keep`` and exactly 0 for
    ``n <= keep``; the axes are summed. One filter application is ``C <- exp(-strength * s) C``
    (:func:`exponential_filter`; Hou & Li use ``strength = order = 36``). With ``keep = 2`` every moment
    of total degree <= 2 (density, momentum, full pressure tensor) is untouched for any basis centre and
    width. Applying it as a right-hand-side damping, i.e. passing it as ``"collision_matrix"`` with
    ``nu = rate``, is the continuous form: over a time ``dt`` it equals the filter with
    ``strength = rate * dt``.
    """
    if int(order) != order or order < 1:
        raise ValueError(f"filter order must be a positive integer, got {order}")
    if int(keep) != keep or keep < 2:
        raise ValueError(f"keep must be an integer >= 2 (density, momentum, energy), got {keep}")

    def axis(N, i):
        return jnp.where(i > keep, (i / max(N - 1, 1)) ** order, 0.0)

    p = jnp.arange(Np)[:, None, None]
    m = jnp.arange(Nm)[None, :, None]
    n = jnp.arange(Nn)[None, None, :]
    return axis(Nn, n) + axis(Nm, m) + axis(Np, p)


def exponential_filter(Ck, strength, spectrum):
    """Apply ``C <- exp(-strength * spectrum) C`` to every species and Fourier mode, between steps.

    ``Ck`` has shape ``(Ns * Np * Nm * Nn, ...)`` or ``(Ns, Np, Nm, Nn, ...)``; ``spectrum`` is
    :func:`filter_spectrum`. ``strength = 0`` returns ``Ck`` unchanged (bitwise).
    """
    Np, Nm, Nn = spectrum.shape
    shape = Ck.shape
    C = Ck.reshape(-1, Np, Nm, Nn, *shape[-3:])
    sigma = jnp.exp(-strength * spectrum)[None, :, :, :, None, None, None]
    return (C * sigma).astype(Ck.dtype).reshape(shape)


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
        Number of Fourier modes along each spatial direction.
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
        "t_max": 50.0,
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
    dn1     = default_parameters["dn1"]
    dn2     = default_parameters["dn2"]
    alpha_e = default_parameters["alpha_e"]
    values  = (dn1 + dn2) * default_parameters["Lx"] / (4 * jnp.pi * default_parameters["nx"] * default_parameters["Omega_cs"](default_parameters)[0])
    Fk_0    = jnp.zeros((6, Ny, Nx//2+1, Nz), dtype=jnp.complex128).at[0, 0, default_parameters["nx"], 0].set(values)
    C10     = jnp.array([1 / (alpha_e[0] ** 3) + 0 * 1j,
            0 - 1j * (1 / (2 * alpha_e[0] ** 3)) * dn1
    ])
    C20     = jnp.array([1 / (alpha_e[0] ** 3) + 0 * 1j,
            0 - 1j * (1 / (2 * alpha_e[0] ** 3)) * dn2
    ])
    indices = jnp.array([0, default_parameters["nx"]])
    Ck_0    = jnp.zeros((Ns * Nn * Nm * Np, Ny, Nx//2+1, Nz), dtype=jnp.complex128)
    Ck_0    = Ck_0.at[0,  0, indices, 0].set(C10)
    Ck_0    = Ck_0.at[Nn * Nm * Np, 0, indices, 0].set(C20)
    
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

    # Real-valued fft leaves only the positive-k elements on the last axis given.
    # We choose the x-axis since this gets us the savings in 1D as well as 2D/3D.
    kx_simulation = jnp.fft.rfftfreq(Nx) * Nx * 2 * jnp.pi
    ky_simulation = jnp.fft.fftfreq(Ny) * Ny * 2 * jnp.pi
    kz_simulation = jnp.fft.fftfreq(Nz) * Nz * 2 * jnp.pi

    ky_grid, kx_grid, kz_grid = jnp.meshgrid(ky_simulation, kx_simulation, kz_simulation, indexing='ij')
    k2_grid = kx_grid**2 + ky_grid**2 + kz_grid**2
    nabla = jnp.array([kx_grid / Lx, ky_grid / Ly, kz_grid / Lz])

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
        "nabla": nabla,
        "collision_matrix": (jnp.asarray(user_parameters["collision_matrix"]) if "collision_matrix" in user_parameters
                             else hypercollision_spectrum(Nn, Nm, Np)),
        "sqrt_n_plus": sqrt_n_plus, "sqrt_n_minus": sqrt_n_minus,
        "sqrt_m_plus": sqrt_m_plus, "sqrt_m_minus": sqrt_m_minus,
        "sqrt_p_plus": sqrt_p_plus, "sqrt_p_minus": sqrt_p_minus,
    })

    return parameters

def species_initial_state(species, Lx=4 * jnp.pi, Ly=1.0, Lz=1.0, Nx=33, Ny=1, Nz=1, Nn=20, Nm=1, Np=1,
                          Omega_ce=1.0, B0=(0.0, 0.0, 0.0)):
    """Build the SPECTRAX initial state from a physical description of each species.

    Each species is a mapping with ``charge`` and ``mass`` (in units of the electron charge and mass),
    ``density`` (in units of the reference density that defines the electron plasma frequency),
    ``vth`` (thermal speed sqrt(2 T / m) over c, a number or one value per axis, which is also the Hermite
    width), ``drift`` (mean velocity over c, a number for x or one value per axis) and optionally a density
    perturbation ``density * (1 + perturbation_amplitude * cos(2 pi perturbation_mode x / L))`` along
    ``perturbation_axis`` ("x", "y" or "z"). The first species must have unit mass: it sets the field units,
    in which the electric field is c B and ``Omega_ce`` is the gyrofrequency of the reference field over the
    plasma frequency. ``B0`` is a uniform background magnetic field in those units.

    Each species is expanded in its own Maxwellian, so only its zeroth Hermite coefficient is nonzero. The
    electric field solves Gauss's law, i k . E_k = rho_k / Omega_ce, so the state is consistent from t = 0.
    Returns the ``input_parameters`` entries ``qs``, ``ms``, ``Omega_cs``, ``alpha_s``, ``u_s``, ``Ck_0`` and
    ``Fk_0``.
    """
    species = list(species.values()) if isinstance(species, dict) else list(species)
    if float(species[0].get("mass", 1.0)) != 1.0:
        raise ValueError("The first species sets the field units and must have unit mass (electrons).")
    per_axis = lambda value: [float(value), 0.0, 0.0] if jnp.ndim(jnp.asarray(value)) == 0 else [float(v) for v in value]
    qs = jnp.array([float(sp.get("charge", -1.0)) for sp in species])
    ms = jnp.array([float(sp.get("mass", 1.0)) for sp in species])
    vth = [jnp.asarray(sp.get("vth", 0.707107)) for sp in species]
    alpha_s = jnp.concatenate([jnp.broadcast_to(v, (3,)) for v in vth]).astype(float)
    u_s = jnp.array(sum((per_axis(sp.get("drift", 0.0)) for sp in species), []))
    lengths, sizes = (Lx, Ly, Lz), (Nx, Ny, Nz)
    y, x, z = jnp.meshgrid(*(jnp.arange(N) * L / N for L, N in ((Ly, Ny), (Lx, Nx), (Lz, Nz))), indexing="ij")
    position = dict(x=(x, Lx), y=(y, Ly), z=(z, Lz))
    H = Nn * Nm * Np
    Ck_0 = jnp.zeros((len(species) * H, Ny, Nx // 2 + 1, Nz), dtype=jnp.complex128)
    rho_k = 0.0
    for s, sp in enumerate(species):
        coordinate, L = position[sp.get("perturbation_axis", "x")]
        n = float(sp.get("density", 1.0)) * (1 + float(sp.get("perturbation_amplitude", 0.0))
                                                * jnp.cos(2 * jnp.pi * float(sp.get("perturbation_mode", 1)) * coordinate / L))
        n_k = jnp.fft.rfftn(n, axes=(-1, -3, -2), norm="forward")
        Ck_0 = Ck_0.at[s * H].set(n_k / jnp.prod(alpha_s[3 * s:3 * s + 3]))
        rho_k = rho_k + qs[s] * n_k
    kx, ky, kz = (2 * jnp.pi * f(N) * N / L for f, N, L in ((jnp.fft.rfftfreq, Nx, Lx), (jnp.fft.fftfreq, Ny, Ly), (jnp.fft.fftfreq, Nz, Lz)))
    ky, kx, kz = jnp.meshgrid(ky, kx, kz, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2
    E_k = -1j * jnp.stack([kx, ky, kz]) * rho_k / (Omega_ce * jnp.where(k2 > 0, k2, 1.0))
    Fk_0 = jnp.zeros((6, Ny, Nx // 2 + 1, Nz), dtype=jnp.complex128).at[:3].set(E_k)
    Fk_0 = Fk_0.at[3:, 0, 0, 0].set(jnp.asarray(B0, dtype=float))
    return dict(qs=qs, ms=ms, Omega_cs=Omega_ce / ms, alpha_s=alpha_s, u_s=u_s, Ck_0=Ck_0, Fk_0=Fk_0)


def load_parameters(input_file):
    """
    Load simulation input parameters and solver configuration from a TOML file.

    The file has an ``[input_parameters]`` table (box lengths ``Lx``, ``Ly``, ``Lz``, ``t_max``, ``nu``,
    ``D``, ``ode_tolerance``, and with species also ``Omega_ce`` and ``B0``) and a ``[solver_parameters]``
    table (``Nx``, ``Ny``, ``Nz``, ``Nn``, ``Nm``, ``Np``, ``timesteps``, ``dt``, ``solver``). Species may be
    given physically, one ``[species.<name>]`` table each, and the initial state is then built by
    :func:`species_initial_state`; otherwise ``qs``, ``alpha_s``, ``u_s``, ``Omega_cs`` and the initial
    coefficients are taken as given.

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
    input_parameters = parameters.get('input_parameters', {})
    solver_parameters = parameters.get('solver_parameters', {})
    if "species" in parameters:              # physical species description: build the initial state here
        grid = {key: solver_parameters.get(key, default) for key, default in
                dict(Nx=33, Ny=1, Nz=1, Nn=20, Nm=1, Np=1).items()}
        geometry = {key: input_parameters[key] for key in ("Lx", "Ly", "Lz", "Omega_ce", "B0") if key in input_parameters}
        input_parameters.update(species_initial_state(parameters["species"], **geometry, **grid))
        input_parameters.pop("Omega_ce", None), input_parameters.pop("B0", None)
        solver_parameters["Ns"] = len(parameters["species"])

    # Whether to use adaptive time-stepping or constant dt.
    # Default is True.
    adaptive_time_step = solver_parameters.get("adaptive_time_step", True)
    solver_parameters["adaptive_time_step"] = adaptive_time_step


    def get_solver_class(name: str):
        for cls_name, cls in inspect.getmembers(diffrax, inspect.isclass):
            if issubclass(cls, diffrax.AbstractSolver) and cls is not diffrax.AbstractSolver and cls_name == name: return cls()
            elif name == "ImplicitMidpoint": return ImplicitMidpoint(rtol=input_parameters["ode_tolerance"], atol=input_parameters["ode_tolerance"])
        raise ValueError(f"Solver '{name}' is not supported. Choose from Diffrax solvers.")
    solver_parameters["solver"] = get_solver_class(solver_parameters.get("solver", "Dopri8"))
    
    return input_parameters, solver_parameters
