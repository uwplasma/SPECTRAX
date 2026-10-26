"""Hermite basis with time-dependent centres u_s and widths alpha_s carried in the solver state."""
import numpy as np
import jax
import jax.numpy as jnp
from diffrax import NoProgressMeter
from spectrax import simulation
from spectrax._initialization import initialize_simulation_parameters
from spectrax._model import plasma_current
from spectrax._simulation import ode_system, _twothirds_mask
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


# ---------------------------------------------------------------- moving-basis terms and pump frame
from spectrax._model import Hermite_Fourier_system, basis_rate_terms, uniform_acceleration  # noqa: E402


def _axis_coefficients(N, u, a, fv):
    """c_n = (1/a) integral f(v) h_n((v - u)/a) dv for a 1D f, so f = sum_n c_n psi_n((v - u)/a)."""
    x, w = np.polynomial.hermite.hermgauss(200)
    v = 2.0 * x  # integrate f exactly enough: f has Gaussian factors narrower than e^{-x^2/4}
    h = np.zeros((N, x.size))
    h[0] = 1.0
    h[1] = np.sqrt(2) * (v - u) / a
    for n in range(1, N - 1):
        h[n + 1] = np.sqrt(2 / (n + 1)) * (v - u) / a * h[n] - np.sqrt(n / (n + 1)) * h[n - 1]
    return (h * fv(v) * np.exp(x ** 2) * w * 2.0).sum(axis=1) / a


def test_basis_rate_terms_match_finite_differences_of_quadrature_coefficients():
    """d/dt of the exact projection of a fixed non-Gaussian f onto a moving, rescaled basis."""
    Nn, Nm, Np = ORDERS
    fs = [lambda v: np.exp(-(v - 0.3) ** 2) + 0.5 * np.exp(-(v + 0.4) ** 2 / 0.5),
          lambda v: np.exp(-(v + 0.1) ** 2 / 1.5) * (1 + 0.3 * v),
          lambda v: np.exp(-v ** 2 / 0.8) * (1 + 0.2 * v ** 2)]
    u0, a0 = np.array([0.2, -0.1, 0.05]), np.array([1.1, 0.9, 1.3])
    ud, ad = np.array([0.7, -0.4, 0.3]), np.array([0.2, -0.15, 0.1])

    def C(t):
        c = [_axis_coefficients(N, u0[i] + ud[i] * t, a0[i] + ad[i] * t, fs[i]) for i, N in enumerate(ORDERS)]
        return np.einsum("n,m,p->pmn", *c)[None, :, :, :, None, None, None]

    h = 1e-4
    fd = (-C(2 * h) + 8 * C(h) - 8 * C(-h) + C(-2 * h)) / (12 * h)
    rate = basis_rate_terms(jnp.asarray(C(0.0)), jnp.asarray(ud), jnp.asarray(ad), jnp.asarray(a0), Nn, Nm, Np, 1)
    np.testing.assert_allclose(rate, fd, rtol=0, atol=1e-10 * np.abs(fd).max())


def _kinetic(Ns, Ck, F, args, F0=None):
    Nn, Nm, Np = ORDERS
    Nx, Ny, Nz = GRID
    (qs, nu, D, Om, a, u, Lx, Ly, Lz, kx, ky, kz, k2, _, col, *sq) = args[7:]
    mask = _twothirds_mask(Ny, Nx, Nz)
    C = jnp.fft.irfftn(Ck * mask, s=(Nz, Ny, Nx), axes=(-1, -3, -2), norm="forward")
    return Hermite_Fourier_system(Ck, C, F, kx, ky, kz, k2, col, *sq, Lx, Ly, Lz, nu, D, a, u, qs, Om,
                                  Nn, Nm, Np, Ns, mask23=mask, F0=F0)


def test_fused_pump_force_equals_full_force_plus_shift_term():
    """Removing (q/m)(E0 + u x B0) from the force equals adding -(u_dot/a) sqrt(2n) C_{n-1}."""
    Ns = 3
    Nn, Nm, Np = ORDERS
    Nx, Ny, Nz = GRID
    Ck, Fk, u, a = random_state(Ns, seed=4)
    args = ode_args(Ns, u, a)
    F = jnp.fft.irfftn(Fk, s=(Nz, Ny, Nx), axes=(-1, -3, -2), norm="forward") + jnp.arange(1.0, 7.0)[:, None, None, None]
    u_dot, F0 = uniform_acceleration(F, u, args[7], args[10], Ns)
    fused = _kinetic(Ns, Ck, F, args, F0=F0)
    full = _kinetic(Ns, Ck, F, args)
    mask = _twothirds_mask(Ny, Nx, Nz)  # the force acts on, and lands in, the de-aliased band
    shift = basis_rate_terms(Ck * mask, u_dot, jnp.zeros(3 * Ns), a, Nn, Nm, Np, Ns) * mask
    np.testing.assert_allclose(fused, full + shift, rtol=0, atol=1e-12 * float(jnp.abs(full).max()))


def test_drifting_maxwellian_stays_a_single_mode_in_the_pump_frame():
    """A uniform electron-ion plasma oscillation: in the pump frame each species stays g = delta_n0 while
    its centre follows the analytic two-fluid oscillation; total energy is conserved."""
    Nn, Nx, U, Om_i, a_e, a_i = 8, 4, 0.5, 0.1, 0.2, 0.05
    Ck_0 = jnp.zeros((2 * Nn, 1, Nx // 2 + 1, 1), complex).at[0, 0, 0, 0].set(a_e ** -3).at[Nn, 0, 0, 0].set(a_i ** -3)
    params = {"Ck_0": Ck_0, "Fk_0": jnp.zeros((6, 1, Nx // 2 + 1, 1), complex), "qs": jnp.array([-1.0, 1.0]),
              "alpha_s": jnp.array([a_e] * 3 + [a_i] * 3), "u_s": jnp.array([U, 0, 0, 0, 0, 0.0]),
              "Omega_cs": jnp.array([1.0, Om_i]), "mi_me": 1 / Om_i, "nu": 0.0, "t_max": 1000.0, "ode_tolerance": 1e-12}
    out = simulation(params, Nx=Nx, Nn=Nn, Ns=2, timesteps=101, frame="pump", progress_meter=NoProgressMeter())
    C = out["Ck"][:, :, 0, 0, 0]
    assert float(jnp.abs(C[:, 1:Nn]).max()) * a_e ** 3 < 1e-13
    assert float(jnp.abs(C[:, Nn + 1:]).max()) * a_i ** 3 < 1e-13
    t, M = out["time"], 1 / Om_i
    w = U * jnp.cos(jnp.sqrt(1 + Om_i) * t)  # u_e - u_i
    u_i = (U - w) / (1 + M)
    np.testing.assert_allclose(out["basis_u"][:, 3], u_i, atol=1e-8)
    np.testing.assert_allclose(out["basis_u"][:, 0], w + u_i, atol=1e-8)
    E = out["total_energy"]
    assert float(jnp.abs(E - E[0]).max() / E[0]) < 1e-8  # measured 3e-9 at tolerance 1e-12, 3.6k steps


def test_unknown_frame_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="frame"):
        simulation(frame="lab", progress_meter=NoProgressMeter())
