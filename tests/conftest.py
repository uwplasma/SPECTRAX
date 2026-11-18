"""Shared test helpers: Hermite tables, random states and the raw ODE right-hand side."""
import numpy as np
import jax
import jax.numpy as jnp
from spectrax._initialization import initialize_simulation_parameters
from spectrax._simulation import ode_system

jax.config.update("jax_enable_x64", True)


def h_table(N, x):
    """Orthonormal Hermite functions h_n(x), n < N, by the three-term recurrence."""
    h = np.zeros((N, x.size))
    h[0] = 1.0
    if N > 1:
        h[1] = np.sqrt(2) * x
    for n in range(1, N - 1):
        h[n + 1] = np.sqrt(2 / (n + 1)) * x * h[n] - np.sqrt(n / (n + 1)) * h[n - 1]
    return h


ORDERS = (4, 3, 2)  # (Nn, Nm, Np): unequal, so an axis permutation cannot pass
GRID = (5, 4, 3)    # (Nx, Ny, Nz)


def axis_moments(N, u, a, kmax=2):
    """M[k, n] = integral of v^k psi_n((v - u)/a) dv by Gauss-Hermite quadrature."""
    x, w = np.polynomial.hermite.hermgauss(N + kmax + 2)
    h = h_table(N, x)
    return np.array([(h * w * (u + a * x) ** k).sum(axis=1) * a / np.sqrt(np.pi) for k in range(kmax + 1)])


def random_state(Ns, seed=0):
    """Coefficients, fields and an anisotropic shifted basis for Ns species on the unequal layout."""
    Nn, Nm, Np = ORDERS
    Nx, Ny, Nz = GRID
    rng = np.random.default_rng(seed)
    rfft = lambda x: jnp.fft.rfftn(jnp.asarray(x), axes=(-1, -3, -2), norm="forward")  # real fields
    Ck = rfft(0.1 * rng.normal(size=(Ns, Np, Nm, Nn, Ny, Nx, Nz)))
    Ck = Ck.at[:, 0, 0, 0, 0, 0, 0].set(1.0)
    Fk = rfft(0.1 * rng.normal(size=(6, Ny, Nx, Nz)))
    u = jnp.asarray(rng.normal(size=3 * Ns)) * 0.3
    a = jnp.asarray(rng.uniform(0.5, 1.5, size=3 * Ns))
    return Ck, Fk, u, a


def ode_args(Ns, u, a, nu=0.5):
    Nn, Nm, Np = ORDERS
    Nx, Ny, Nz = GRID
    p = initialize_simulation_parameters({"alpha_s": a, "u_s": u, "qs": jnp.array([-1.0, 1.0, 2.0][:Ns]),
                                          "Omega_cs": jnp.array([1.0, 0.1, 0.05][:Ns]), "nu": nu},
                                         Nx, Ny, Nz, Nn, Nm, Np, Ns)
    keys = ("qs", "nu", "D", "Omega_cs", "alpha_s", "u_s", "Lx", "Ly", "Lz", "kx_grid", "ky_grid", "kz_grid",
            "k2_grid", "nabla", "collision_matrix", "sqrt_n_plus", "sqrt_n_minus", "sqrt_m_plus",
            "sqrt_m_minus", "sqrt_p_plus", "sqrt_p_minus")
    return (Nx, Ny, Nz, Nn, Nm, Np, Ns) + tuple(p[k] for k in keys)


def rhs(Ns, state, args):
    return ode_system(*GRID, *ORDERS, Ns, 0.0, state, args)


B0 = jnp.array([[0.0, 0, 0], [1.0, 1, 1]])  # unit basis (u, alpha) for one species


def maxwellian_state(N, d, r):
    """Single-species spatially uniform state: the Maxwellian centred at d of width r projected by quadrature
    onto the unit basis B0, Ck of shape (1, 1, 1, N, 1, 1, 1)."""
    x, w = np.polynomial.hermite.hermgauss(N + 40)
    g = (h_table(N, x * r + d) * w).sum(axis=1) / np.sqrt(np.pi)
    return jnp.zeros((1, 1, 1, N, 1, 1, 1)).at[0, 0, 0, :, 0, 0, 0].set(g)
