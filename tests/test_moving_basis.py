"""Hermite basis with time-dependent centres u_s and widths alpha_s carried in the solver state."""
import numpy as np
import jax
import jax.numpy as jnp
from diffrax import NoProgressMeter
from spectrax import simulation
from spectrax._model import plasma_current
from spectrax._diagnostics import diagnostics
from tests.conftest import ORDERS, GRID, axis_moments, random_state, ode_args, rhs

jax.config.update("jax_enable_x64", True)


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
