import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
try: import tomllib
except ModuleNotFoundError: import pip._vendor.tomli as tomllib
import diffrax
import inspect
from .midpoint_solver import ImplicitMidpoint
from math import comb
from jax.scipy.special import factorial
from scipy.special import roots_legendre

__all__ = ["load_parameters", "initialize_simulation_parameters", "construct_idx_array", "legT"]

@partial(jit, static_argnames=['Nx', 'Ny', 'Nz','Nn', 'Nm', 'Np', 'Ns', 'N_DG', 'dims', 'timesteps'])
def initialize_simulation_parameters(user_parameters={}, Nx=33, Ny=1, Nz=1, Nn=50, Nm=1, Np=1, Ns=2, N_DG=2, dims=1, timesteps=500, dt=0.01):
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
        "ms": jnp.array([1.0, 1.0]),
        "qs": jnp.array([-1, -1]),
        "alpha_e": jnp.array([0.707107, 0.707107, 0.707107]),
        "alpha_s": lambda p: jnp.concatenate([
            p["alpha_e"],
            p["alpha_e"] * jnp.sqrt(1.0 / p["ms"][1])
        ]),
        "u_s": jnp.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
        "Omega_ce": 1.0,
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
        "vti": lambda p: p["vte"] * jnp.sqrt(1 / p["ms"][1]),
    }
    
    # Initialize distribution function as a two-stream instability
    basis_idx = construct_idx_array(dims, N_DG) # Construct array of combinations of Legendre coefficients
    dn1     = default_parameters["dn1"]
    dn2     = default_parameters["dn2"]
    alpha_e = default_parameters["alpha_e"]
    Lx, Ly, Lz = default_parameters["Lx"], default_parameters["Ly"], default_parameters["Lz"]
    values  = (dn1 + dn2) * Lx / (4 * jnp.pi * default_parameters["nx"] * default_parameters["Omega_ce"])
    F0 = lambda x, y, z: jnp.concatenate([jnp.array([2 * values * jnp.cos(2*jnp.pi*x/Lx)]), jnp.broadcast_to(jnp.zeros_like(x), (5,) + jnp.zeros_like(x).shape)])
    Fk_0 = legT(F0, basis_idx, N_DG, Lx, Nx)

    C10 = lambda x: 1 / (alpha_e[0] ** 3) - dn1 * (1 / (alpha_e[0] ** 3)) * jnp.sin(2*jnp.pi*x/Lx)
    C20 = lambda x: 1 / (alpha_e[0] ** 3) - dn2 * (1 / (alpha_e[0] ** 3)) * jnp.sin(2*jnp.pi*x/Lx)
    C0 = lambda x, y, z: jnp.concatenate([jnp.array([C10(x)]), jnp.broadcast_to(jnp.zeros_like(x), (Nn-1,) + jnp.zeros_like(x).shape), jnp.array([C20(x)]), jnp.broadcast_to(jnp.zeros_like(x), (Nn-1,) + jnp.zeros_like(x).shape)])
    Ck_0 = legT(C0, basis_idx, N_DG, Lx, Nx)
    
    default_parameters.update({
        "Ck_0": Ck_0, "Fk_0": Fk_0, "Ns": Ns,
        "timesteps": timesteps, "dt": dt, 
        "Nx": Nx, "Ny": Ny, "Nz": Nz,
        "Nn": Nn, "Nm": Nm, "Np": Np,
        "N_DG": N_DG, "dims": dims
    })

    # Merge user-provided parameters into the default dictionary
    parameters = {**default_parameters, **user_parameters}

    
    # Compute derived parameters based on user-provided or default values
    for key, value in parameters.items():
        if callable(value):  # If the value is a lambda function, compute it
            parameters[key] = value(parameters)
        if isinstance(value, list):
            parameters[key] = jnp.array(value)

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
        p = jnp.arange(Np)[None, :, None, None, None, None, None, None] # (s, p, m, n, y, x, z, l)
        m = jnp.arange(Nm)[None, None, :, None, None, None, None, None]
        n = jnp.arange(Nn)[None, None, None, :, None, None, None, None]

        sqrt_n_plus  = jnp.sqrt(n+1)
        sqrt_n_minus = jnp.sqrt(n) 
        sqrt_m_plus  = jnp.sqrt(m+1)
        sqrt_m_minus = jnp.sqrt(m)
        sqrt_p_plus  = jnp.sqrt(p+1)
        sqrt_p_minus = jnp.sqrt(p)

        return sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus
    
    sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus = build_coeff_tables(Nn, Nm, Np)
    Lx, Ly, Lz = parameters["Lx"], parameters["Ly"], parameters["Lz"]
    # Precompute matrices for product coefficients between basis functions
    inner_mm, inner_pm, inner_mp, inner_pp, di_inner_product, tripple_product = compute_basis_products(Lx/Nx, Ly/Ny, Lz/Nz, basis_idx)
    
    alpha = parameters["alpha_s"].reshape(Ns, 3)
    u = parameters["u_s"].reshape(Ns, 3)
    Ax_p, Ax_m, Ay_p, Ay_m, Az_p, Az_m = compute_A_pm_matrices(Nn, Nm, Np, alpha, u)
    R_p, R_m = compute_R_pm_matrices(False)

    parameters.update({
        "collision_matrix": precompute_collisions(Nn, Nm, Np),
        "sqrt_n_plus": sqrt_n_plus, "sqrt_n_minus": sqrt_n_minus,
        "sqrt_m_plus": sqrt_m_plus, "sqrt_m_minus": sqrt_m_minus,
        "sqrt_p_plus": sqrt_p_plus, "sqrt_p_minus": sqrt_p_minus,
        "basis_idx": basis_idx, "inner_mm": inner_mm,
        "inner_pm": inner_pm, "inner_mp": inner_mp,
        "inner_pp": inner_pp, "di_inner_product": di_inner_product,
        "tripple_product": tripple_product, "Ax_p": Ax_p,
        "Ax_m": Ax_m, "Ay_p": Ay_p, "Ay_m": Ay_m, "Az_p": Az_p, "Az_m": Az_m,
        "R_p": R_p, "R_m": R_m
    })

    return parameters

def construct_idx_array(dims=1, N_DG=2):
        Nl = comb(N_DG + dims - 1, dims)
        l = 0
        indices = jnp.zeros((Nl, 3))

        if dims == 1:
            for i in range(N_DG):
                indices = indices.at[l, :].set([i, 0, 0])
                l += 1
        else:
            for i in range(N_DG):
                for j in range(N_DG - i):
                    for k in range(N_DG - i - j):
                        indices = indices.at[l, :].set([i, j, k])
                        l += 1
                        if dims < 3:
                            break
        
        return indices.astype(jnp.int32)

def compute_basis_products(dx, dy, dz, basis_idx):
    def wigner(j1, j2, j3): # Special case of wigner 3j symbol where bottom row is zero
        l = (j1 + j2 + j3) // 2
        return jnp.where(((j1 + j2 + j3) % 2 == 0) & (j3 <= j1 + j2) & (jnp.abs(j1 - j2) <= j3), (-1)**l * factorial(l) / (factorial(l-j1) * factorial(l-j2) * factorial(l-j3)) * jnp.sqrt(factorial(2*l-2*j1) * factorial(2*l-2*j2) * factorial(2*l-2*j3) / factorial(2*l+1)), 0)
        
    def construct_di_inner(idx): # Function for computing inner product between a basis function and the derivative of another basis function
        b, pl, ql = idx
        di = jnp.array([dx, dy, dz])
        p = basis_idx[pl]
        q = basis_idx[ql]

        return di[b-1] * di[b-2] / ((2 * p[b-1] + 1) * (2 * p[b-2] + 1)) * jnp.where((p[b-1] == q[b-1]) & (p[b-2] == q[b-2]) & (p[b] > q[b]) & ((p[b] - q[b]) % 2 == 1), 2, 0)
    
    def compute_tp(idx): # Compute integrals of a triple product of basis functions based on Wigner 3j identities
        i, j, k = idx
        ix, iy, iz = basis_idx[i]
        jx, jy, jz = basis_idx[j]
        kx, ky, kz = basis_idx[k]
        return dx * dy * dz * (wigner(ix, jx, kx) * wigner(jy, jy, ky) * wigner(iz, jz, kz)) ** 2

    def inner_compute(idx): # Function for computing the surface integrals on each face
        b, pl, ql = idx
        di = jnp.array([dx, dy, dz])
        p = basis_idx[pl]
        q = basis_idx[ql]

        return di[b-1] * di[b-2] / ((2 * p[b-1] + 1) * (2 * p[b-2] + 1)) * jnp.where((p[b-1] == q[b-1]) & (p[b-2] == q[b-2]), 1, 0)

    i = jnp.transpose(basis_idx[:, None, :], (2, 0, 1))
    j = jnp.transpose(basis_idx[None, :, :], (2, 0, 1))
    Nl = basis_idx.shape[0]

    inner_mm = vmap(vmap(vmap(inner_compute)))(jnp.stack(jnp.indices((3, Nl, Nl)), axis=-1)) # Construct matrices of surface integral coefficients
    inner_pm = (inner_mm * ((-1) ** j)) # "p" and "m" represent basis functions evaluated above (p) and below (m) face
    inner_mp = jnp.transpose(inner_pm, (0, 2, 1))
    inner_pp = inner_mm * ((-1) ** (i + j))

    di_inner_product = vmap(vmap(vmap(construct_di_inner)))(jnp.stack(jnp.indices((3, Nl, Nl)), axis=-1)) # Construct matrix of basis-derivative products
    tripple_product = vmap(vmap(vmap(compute_tp)))(jnp.stack(jnp.indices((Nl, Nl, Nl)), axis=-1)) # Construct matrix of triple products

    return inner_mm, inner_pm, inner_mp, inner_pp, di_inner_product, tripple_product

def compute_A_pm_matrices(Nn, Nm, Np, alpha, u):
    def compute_A_off(idx):
        i, j = idx
        return jnp.where((j == i + 1) | (j == i - 1), jnp.sqrt(jnp.maximum(i, j) / 2), 0)
    
    Ax_mat = alpha[:, 0, None, None] * vmap(vmap(compute_A_off))(jnp.stack(jnp.indices((Nn, Nn)), axis=-1)) + u[:, 0, None, None] * jnp.eye(Nn)
    Ay_mat = alpha[:, 1, None, None] * vmap(vmap(compute_A_off))(jnp.stack(jnp.indices((Nm, Nm)), axis=-1)) + u[:, 1, None, None] * jnp.eye(Nm)
    Az_mat = alpha[:, 2, None, None] * vmap(vmap(compute_A_off))(jnp.stack(jnp.indices((Np, Np)), axis=-1)) + u[:, 2, None, None] * jnp.eye(Np)

    Lam, U = jnp.linalg.eig(Ax_mat)
    D_mat = Lam[:, :, None] * jnp.eye(Nn)
    Ax_p = jnp.real(U @ (D_mat + jnp.abs(D_mat)) @ jnp.transpose(U, (0, 2, 1)) / 2)[:, None, None, :, :, None, None, None, None, None]
    Ax_m = jnp.real(U @ (D_mat - jnp.abs(D_mat)) @ jnp.transpose(U, (0, 2, 1)) / 2)[:, None, None, :, :, None, None, None, None, None]

    Lam, U = jnp.linalg.eig(Ay_mat)
    D_mat = Lam[:, :, None] * jnp.eye(Nm)
    Ay_p = jnp.real(U @ (D_mat + jnp.abs(D_mat)) @ jnp.transpose(U, (0, 2, 1)) / 2)[:, None, :, :, None, None, None, None, None, None]
    Ay_m = jnp.real(U @ (D_mat - jnp.abs(D_mat)) @ jnp.transpose(U, (0, 2, 1)) / 2)[:, None, :, :, None, None, None, None, None, None]

    Lam, U = jnp.linalg.eig(Az_mat)
    D_mat = Lam[:, :, None] * jnp.eye(Np)
    Az_p = jnp.real(U @ (D_mat + jnp.abs(D_mat)) @ jnp.transpose(U, (0, 2, 1)) / 2)[:, :, :, None, None, None, None, None, None, None]
    Az_m = jnp.real(U @ (D_mat - jnp.abs(D_mat)) @ jnp.transpose(U, (0, 2, 1)) / 2)[:, :, :, None, None, None, None, None, None, None]

    return Ax_p, Ax_m, Ay_p, Ay_m, Az_p, Az_m

def compute_R_pm_matrices(upwind=False):
    Rx_mat = jnp.block([[jnp.zeros((3, 3)), jnp.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])], [jnp.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]]), jnp.zeros((3, 3))]])
    Ry_mat = jnp.block([[jnp.zeros((3, 3)), jnp.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]])], [jnp.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]), jnp.zeros((3, 3))]])
    Rz_mat = jnp.block([[jnp.zeros((3, 3)), jnp.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])], [jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]), jnp.zeros((3, 3))]])
    R_mat = jnp.stack([Rx_mat, Ry_mat, Rz_mat])

    if upwind: # upwind flux
        Lam, U = jnp.linalg.eig(R_mat)
        D = Lam[:, :, None] * jnp.eye(6)
        R_p = jnp.real(U @ (D + jnp.abs(D)) @ jnp.transpose(U, (0, 2, 1)) / 2)[:, :, :, None, None, None, None, None]
        R_m = jnp.real(U @ (D - jnp.abs(D)) @ jnp.transpose(U, (0, 2, 1)) / 2)[:, :, :, None, None, None, None, None]
    else: # central flux
        R_p = (R_mat / 2)[:, :, :, None, None, None, None, None]
        R_m = (R_mat / 2)[:, :, :, None, None, None, None, None]

    return R_p, R_m

def legT(f, basis_idx, N_DG, Lx, Nx, Ly=1, Ny=1, Lz=1, Nz=1): # Transform initial values into finite element basis
        Nq = int(N_DG // 2 + 1)
        dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz
        x_elements = dx / 2 + jnp.linspace(0, Lx, Nx, endpoint=False)
        y_elements = dy / 2 + jnp.linspace(0, Ly, Ny, endpoint=False)
        z_elements = dz / 2 + jnp.linspace(0, Lz, Nz, endpoint=False)
        quad_points, w_quad = jnp.array(roots_legendre(Nq))
        def legendre(n, x):
            def term(k, acc):
                return acc + (-1)**k * jnp.round(factorial(2*n-2*k) / (factorial(k) * factorial(n-k) * factorial(n-2*k))) * x**(n-2*k)
            return lax.fori_loop(0, (n // 2) + 1, term, jnp.zeros_like(x)) / 2**n
        legendre_array = vmap(vmap(legendre, in_axes=(None, 0)), in_axes=(0, None))
        leg_vals = jnp.array(legendre_array(jnp.arange(N_DG), quad_points)) # Create array of shape (N_DG, Nq) evaluating the n'th Legendre polynomial at each quadrature point
        leg_vals3d = leg_vals[basis_idx[:, 1], :][:, :, None, None] * leg_vals[basis_idx[:, 0], :][:, None, :, None] * leg_vals[basis_idx[:, 2], :][:, None, None, :] # create values of evaluated Legendre polynomials on 3d local grid. Shape (Nl, Nq, Nq, Nq) where Nq is the number of quadrature points.
        leg_vals3d = jnp.transpose(leg_vals3d, (1, 2, 3, 0))
        w_array = w_quad[:, None, None] * w_quad[None, :, None] * w_quad[None, None, :]

        X = x_elements[:, None, None, None, None, None] + quad_points[None, None, None, :, None, None] * dx / 2
        Y = y_elements[None, :, None, None, None, None] + quad_points[None, None, None, None, :, None] * dy / 2
        Z = z_elements[None, None, :, None, None, None] + quad_points[None, None, None, None, None, :] * dz / 2
        X, Y, Z = jnp.broadcast_arrays(X, Y, Z)
        Uk_0 = (basis_idx[:, 0] + 0.5) * (basis_idx[:, 1] + 0.5) * (basis_idx[:, 2] + 0.5) * jnp.sum(f(X, Y, Z)[..., None] * leg_vals3d * w_array[:, :, :, None], axis=(-4, -3, -2))

        return Uk_0

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
            if issubclass(cls, diffrax.AbstractSolver) and cls is not diffrax.AbstractSolver and cls_name == name: return cls()
            elif name == "ImplicitMidpoint": return ImplicitMidpoint(rtol=input_parameters["ode_tolerance"], atol=input_parameters["ode_tolerance"])
        raise ValueError(f"Solver '{name}' is not supported. Choose from Diffrax solvers.")
    solver_parameters["solver"] = get_solver_class(solver_parameters.get("solver", "Tsit5"))
    
    return input_parameters, solver_parameters
