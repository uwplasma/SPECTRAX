"""Hermite basis with time-dependent centres u_s and widths alpha_s carried in the solver state."""
import numpy as np
import jax
import jax.numpy as jnp
from diffrax import NoProgressMeter
from spectrax import simulation
from spectrax._initialization import initialize_simulation_parameters
from spectrax._model import plasma_current
from spectrax._simulation import ode_system
from spectrax._diagnostics import diagnostics

jax.config.update("jax_enable_x64", True)

ORDERS = (4, 3, 2)  # (Nn, Nm, Np): unequal, so an axis permutation cannot pass
GRID = (5, 4, 3)    # (Nx, Ny, Nz)


def axis_moments(N, u, a, kmax=2):
    """M[k, n] = integral of v^k psi_n((v - u)/a) dv by Gauss-Hermite quadrature."""
    x, w = np.polynomial.hermite.hermgauss(N + kmax + 2)
    h = np.zeros((N, x.size))
    h[0] = 1.0
    if N > 1:
        h[1] = np.sqrt(2) * x
    for n in range(1, N - 1):
        h[n + 1] = np.sqrt(2 / (n + 1)) * x * h[n] - np.sqrt(n / (n + 1)) * h[n - 1]
    return np.array([(h * w * (u + a * x) ** k).sum(axis=1) * a / np.sqrt(np.pi) for k in range(kmax + 1)])


def random_state(Ns, seed=0):
    """Coefficients, fields and an anisotropic shifted basis for Ns species on the unequal layout."""
    Nn, Nm, Np = ORDERS
    Nx, Ny, Nz = GRID
    rng = np.random.default_rng(seed)
    shape = (Ns, Np, Nm, Nn, Ny, Nx // 2 + 1, Nz)
    Ck = jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape)) * 0.1
    Ck = Ck.at[:, 0, 0, 0, 0, 0, 0].set(1.0)
    Fk = jnp.asarray(rng.normal(size=(6, Ny, Nx // 2 + 1, Nz)) + 1j * rng.normal(size=(6, Ny, Nx // 2 + 1, Nz))) * 0.1
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


def test_basis_state_rhs_is_bitwise_the_fixed_basis():
    """With the basis in the state and zero rates, dCk and dFk are bitwise those of the fixed basis."""
    Ns = 3
    Ck, Fk, u, a = random_state(Ns)
    args = ode_args(Ns, u, a)
    dC0, dF0 = rhs(Ns, (Ck, Fk), args)
    dC1, dF1, dB = rhs(Ns, (Ck, Fk, jnp.stack([u, a])), args)
    assert jnp.array_equal(dC0, dC1) and jnp.array_equal(dF0, dF1)
    assert jnp.array_equal(dB, jnp.zeros((2, 3 * Ns)))


def test_basis_state_overrides_the_configured_basis():
    """The state's (u, alpha), not the configured ones, enter the kinetic operator and the current."""
    Ns = 2
    Ck, Fk, u, a = random_state(Ns, seed=1)
    u2, a2 = u + 0.2, a * 1.3
    dC_state, dF_state, _ = rhs(Ns, (Ck, Fk, jnp.stack([u2, a2])), ode_args(Ns, u, a))
    dC_ref, dF_ref = rhs(Ns, (Ck, Fk), ode_args(Ns, u2, a2))
    assert jnp.array_equal(dC_state, dC_ref) and jnp.array_equal(dF_state, dF_ref)


def test_current_and_energy_with_time_dependent_basis_match_quadrature():
    """J and the kinetic energy computed with a per-time (u, alpha) agree with velocity quadrature."""
    Ns = 2
    Nn, Nm, Np = ORDERS
    Nx, Ny, Nz = GRID
    qs, masses = np.array([-1.0, 1.0]), np.array([1.0, 7.0])
    Cks, us, As, J_quad, K_quad = [], [], [], [], []
    for seed in range(3):  # three "saved times", each with its own basis
        Ck, _, u, a = random_state(Ns, seed=seed)
        C = np.asarray(Ck[:, :, :, :, 0, 0, 0]).real  # k = 0 moments, (Ns, Np, Nm, Nn)
        J, K = np.zeros(3), np.zeros(Ns)
        for s in range(Ns):
            M = [axis_moments(N, float(u[3 * s + i]), float(a[3 * s + i])) for i, N in enumerate(ORDERS)]
            full = lambda kx, ky, kz: np.einsum("pmn,n,m,p->", C[s], M[0][kx], M[1][ky], M[2][kz])
            J += qs[s] * np.array([full(1, 0, 0), full(0, 1, 0), full(0, 0, 1)])
            K[s] = 0.5 * masses[s] * (full(2, 0, 0) + full(0, 2, 0) + full(0, 0, 2))
        J_mean = plasma_current(jnp.asarray(qs), a, u, Ck.reshape(Ns * Nn * Nm * Np, Ny, Nx // 2 + 1, Nz),
                                Nn, Nm, Np, Ns)[:, 0, 0, 0]
        np.testing.assert_allclose(np.real(J_mean), J, rtol=1e-13, atol=1e-14)
        Cks.append(Ck.reshape(Ns * Nn * Nm * Np, Ny, Nx // 2 + 1, Nz))
        us.append(u), As.append(a), K_quad.append(K)
    out = {"Ck": jnp.stack(Cks), "Fk": jnp.zeros((3, 6, Ny, Nx // 2 + 1, Nz), complex), "alpha_s": As[0],
           "u_s": us[0], "basis_alpha": jnp.stack(As), "basis_u": jnp.stack(us), "Omega_cs": jnp.ones(Ns),
           "Lx": 1.0, "Nx": Nx, "Nn": Nn, "Nm": Nm, "Np": Np, "masses": masses}
    diagnostics(out)
    np.testing.assert_allclose(out["kinetic_energy_species"], np.array(K_quad), rtol=1e-13)


def test_fixed_frame_simulation_is_bitwise_the_constant_basis_run():
    """With constant steps, carrying the basis in the state reproduces the constant-basis run bitwise.

    (An adaptive controller's RMS error norm also counts the basis components, so adaptive step
    sequences differ; those agree to the solver tolerance instead, checked below.)"""
    kw = dict(Nx=7, Nn=6, Nm=2, Np=1, Ns=2, timesteps=3, progress_meter=NoProgressMeter(),
              adaptive_time_step=False, dt=0.01)
    params = {"t_max": 0.5, "nu": 0.3}
    ref = simulation(params, **kw)
    mov = simulation(params, frame="fixed", **kw)
    for key in ("Ck", "Fk", "kinetic_energy", "EM_energy"):
        assert jnp.array_equal(ref[key], mov[key]), key
    assert mov["basis_u"].shape == (3, 6)
    assert jnp.array_equal(mov["basis_alpha"], jnp.broadcast_to(ref["alpha_s"], (3, 6)))


def test_fixed_frame_adaptive_run_agrees_to_tolerance():
    kw = dict(Nx=7, Nn=6, Ns=2, timesteps=3, progress_meter=NoProgressMeter())
    params = {"t_max": 0.5, "nu": 0.3, "ode_tolerance": 1e-10}
    ref, mov = simulation(params, **kw), simulation(params, frame="fixed", **kw)
    np.testing.assert_allclose(mov["Ck"], ref["Ck"], rtol=0, atol=1e-9 * float(jnp.abs(ref["Ck"]).max()))
