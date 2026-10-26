"""Conservative Dougherty collision operator in the asymmetric, shifted and scaled Hermite basis."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from diffrax import NoProgressMeter
from spectrax import simulation
from spectrax._model import (dougherty_moments, dougherty_rate_real, dougherty_collisions,
                             dougherty_cross_moments)
from spectrax._simulation import _twothirds_mask
from spectrax._initialize_maxwellian import compute_C_nmp

jax.config.update("jax_enable_x64", True)
r2 = np.sqrt(2.0)


def hermite_projection(xi, N):
    """C_n = int f H_n(xi) / sqrt(2^n n!) dxi on a uniform grid (spectrally accurate for decaying f)."""
    h = np.zeros((N, xi.size))
    h[0] = 1.0
    h[1] = r2 * xi
    for n in range(1, N - 1):
        h[n + 1] = np.sqrt(2 / (n + 1)) * xi * h[n] - np.sqrt(n / (n + 1)) * h[n - 1]
    return h


def project_2v(F, ux, ax, uy, ay, Nn, Nm, L=9.0, M=301):
    """Hermite coefficients (Nm, Nn) of a function F(vx, vy), basis (ux, ax), (uy, ay)."""
    xi = np.linspace(-L, L, M)
    dxi = xi[1] - xi[0]
    VX, VY = np.meshgrid(ux + ax * xi, uy + ay * xi, indexing="xy")  # rows: vy, cols: vx
    vals = np.asarray(F(jnp.asarray(VX), jnp.asarray(VY)))
    hx, hy = hermite_projection(xi, Nn), hermite_projection(xi, Nm)
    return np.einsum("mj,ji,ni->mn", hy, vals, hx) * dxi ** 2


def f_test(vx, vy):
    """Drifting anisotropic Gaussian with a polynomial skew: not a Maxwellian."""
    g = jnp.exp(-(vx - 0.4) ** 2 / (2 * 0.55) - (vy + 0.2) ** 2 / (2 * 0.3))
    return 1.3 * g * (1 + 0.3 * (vx - 0.4) ** 3 / 0.55 ** 1.5 - 0.2 * (vy + 0.2) * (vx - 0.4))


def test_projection_matches_direct_evaluation_on_an_anisotropic_drifting_distribution():
    Nn, Nm = 9, 7
    ux, ax, uy, ay = 0.1, 0.9, -0.3, 0.7
    C = project_2v(f_test, ux, ax, uy, ay, Nn, Nm)
    alpha, u = jnp.array([ax, ay, 1.0]), jnp.array([ux, uy, 0.0])
    Cs = jnp.asarray(C)[None, None, :, :, None, None, None]
    dens, U, theta = dougherty_moments(Cs, alpha, u, Nn, Nm, 1, 1)
    # Direct moments of f by quadrature
    vx = jnp.linspace(-8, 8, 801); dv = vx[1] - vx[0]
    VX, VY = jnp.meshgrid(vx, vx, indexing="xy")
    fv = f_test(VX, VY)
    n0 = fv.sum() * dv ** 2
    Ux, Uy = (VX * fv).sum() * dv ** 2 / n0, (VY * fv).sum() * dv ** 2 / n0
    th = (((VX - Ux) ** 2 + (VY - Uy) ** 2) * fv).sum() * dv ** 2 / (2 * n0)
    np.testing.assert_allclose([dens.ravel()[0], U[0, 0].ravel()[0], U[0, 1].ravel()[0], theta.ravel()[0]],
                               [n0, Ux, Uy, th], rtol=1e-9, atol=1e-11)
    nu = 0.7

    def collide(vx, vy):
        def flux_x(vx, vy):
            return (vx - Ux) * f_test(vx, vy) + th * jax.grad(f_test, 0)(vx, vy)

        def flux_y(vx, vy):
            return (vy - Uy) * f_test(vx, vy) + th * jax.grad(f_test, 1)(vx, vy)
        return nu * (jax.grad(flux_x, 0)(vx, vy) + jax.grad(flux_y, 1)(vx, vy))

    Cf = jax.vmap(jax.vmap(collide))
    direct = project_2v(Cf, ux, ax, uy, ay, Nn, Nm)
    stencil = np.asarray(dougherty_rate_real(Cs, U, theta, nu, alpha, u, Nn, Nm, 1, 1))[0, 0, :, :, 0, 0, 0]
    np.testing.assert_allclose(stencil, direct, atol=1e-9 * np.abs(direct).max())


def random_masked_state(Ns, N=(5, 4, 3), grid=(6, 4, 3), seed=1):
    Nn, Nm, Np = N
    Nx, Ny, Nz = grid
    rng = np.random.default_rng(seed)
    mask = _twothirds_mask(Ny, Nx, Nz)
    Cr = 0.05 * rng.normal(size=(Ns, Np, Nm, Nn, Ny, Nx, Nz))
    Cr[:, 0, 0, 0] += 1.0
    Ck = jnp.fft.rfftn(jnp.asarray(Cr), axes=(-1, -3, -2), norm="forward") * mask
    C = jnp.fft.irfftn(Ck, s=(Nz, Ny, Nx), axes=(-1, -3, -2), norm="forward")
    return Ck, C, mask


def physical_rates(dC, alpha, u, Ns, ms):
    """Per-x rates of density, momentum (3) and energy summed over species, from real-space coefficient rates."""
    a, uu = alpha.reshape(Ns, 3), u.reshape(Ns, 3)
    out = []
    for s in range(Ns):
        J = np.prod(a[s])
        d0 = dC[s, 0, 0, 0]
        d1 = [dC[s, 0, 0, 1], dC[s, 0, 1, 0], dC[s, 1, 0, 0]]
        d2 = [dC[s, 0, 0, 2], dC[s, 0, 2, 0], dC[s, 2, 0, 0]]
        mom = [ms[s] * J * (uu[s, i] * d0 + a[s, i] * d1[i] / r2) for i in range(3)]
        en = 0.5 * ms[s] * J * sum(uu[s, i] ** 2 * d0 + r2 * uu[s, i] * a[s, i] * d1[i]
                                     + a[s, i] ** 2 * (d0 / 2 + d2[i] / r2) for i in range(3))
        out.append(np.stack([J * d0] + mom + [en]))
    return np.array(out)


@pytest.mark.parametrize("Ns", [1, 3])
def test_conservation_to_round_off_at_finite_N(Ns):
    N, grid = (5, 4, 3), (6, 4, 3)
    Ck, C, mask = random_masked_state(Ns, N, grid)
    rng = np.random.default_rng(2)
    alpha = jnp.asarray(rng.uniform(0.6, 1.4, 3 * Ns)); u = jnp.asarray(rng.normal(size=3 * Ns) * 0.3)
    Omega = jnp.array([1.0, 0.25, 0.04][:Ns])
    nu = jnp.asarray(rng.uniform(0.2, 1.0, (Ns, Ns)))
    dCk = dougherty_collisions(C, nu, alpha, u, Omega, *N, Ns, mask)
    dC = np.asarray(jnp.fft.irfftn(dCk, s=(grid[2], grid[1], grid[0]), axes=(-1, -3, -2), norm="forward"))
    rates = physical_rates(dC, np.asarray(alpha), np.asarray(u), Ns, 1 / np.asarray(Omega))
    scale = np.abs(dC).max()
    np.testing.assert_allclose(rates[:, 0], 0, atol=1e-13 * scale)       # density, per species
    np.testing.assert_allclose(rates[:, 1:].sum(axis=0), 0, atol=1e-12 * scale * Ns)  # total momentum, energy
    if Ns == 1:  # self collisions conserve each species' momentum and energy
        np.testing.assert_allclose(rates[0], 0, atol=1e-12 * scale)
    else:  # but pair collisions do exchange them
        assert np.abs(rates[:, 1:]).max() > 1e-4 * scale


def test_cross_moments_make_exchange_antisymmetric():
    dens = jnp.array([1.0, 0.7]); U = jnp.array([[0.3, -0.1, 0.2], [-0.5, 0.4, 0.0]])
    theta = jnp.array([0.5, 0.02]); ms = jnp.array([1.0, 25.0]); nu_sr = 0.8
    active = (0, 1, 2)
    usr, th_sr, th_rs = dougherty_cross_moments(dens, U, theta, ms, 0, 1, active)
    nu_rs = nu_sr * ms[0] * dens[0] / (ms[1] * dens[1])
    R = [ms[0] * dens[0] * nu_sr * (usr - U[0]), ms[1] * dens[1] * nu_rs * (usr - U[1])]
    np.testing.assert_allclose(R[0] + R[1], 0, atol=1e-15)
    Q = [-ms[k] * dens[k] * nu * (3 * theta[k] + U[k] @ U[k] - usr @ U[k] - 3 * th)
         for k, nu, th in ((0, nu_sr, th_sr), (1, nu_rs, th_rs))]
    np.testing.assert_allclose(Q[0] + Q[1], 0, atol=1e-15)
    assert ms[0] * th_sr == pytest.approx(ms[1] * th_rs)


def maxwellian_state(N, u_basis, a_basis, drift, vth, grid=(1, 1, 1)):
    Nn, Nm, Np = N
    Us = jnp.broadcast_to(jnp.asarray(drift).reshape(1, 3, 1, 1, 1), (1, 3, grid[1], grid[0], grid[2]))
    Ck = compute_C_nmp(Us, a_basis, u_basis, Nn, Nm, Np, 1, vth_s=vth)
    return jnp.fft.irfftn(Ck, s=(grid[2], grid[1], grid[0]), axes=(-1, -3, -2), norm="forward")


def test_maxwellian_with_matching_moments_is_an_exact_equilibrium():
    N = (12, 10, 8)
    u_b, a_b = jnp.array([0.2, -0.1, 0.0]), jnp.array([1.0, 0.9, 1.1])
    vth = jnp.full(3, 0.6)  # isotropic, narrower than every basis width, drifting off-centre
    C = maxwellian_state(N, u_b, a_b, drift=[0.5, 0.1, -0.2], vth=vth)
    dens, U, theta = dougherty_moments(C, a_b, u_b, *N, 1)
    np.testing.assert_allclose(theta.ravel(), 0.36, rtol=1e-12)
    np.testing.assert_allclose(U[0, :, 0, 0, 0], [0.5, 0.1, -0.2], rtol=1e-12)
    rate = dougherty_rate_real(C, U, theta, 1.0, a_b, u_b, *N, 1)
    assert np.abs(rate).max() < 1e-13 * np.abs(C).max()
    # an anisotropic Maxwellian is not in the kernel
    C2 = maxwellian_state(N, u_b, a_b, drift=[0.5, 0.1, -0.2], vth=jnp.array([0.5, 0.6, 0.7]))
    _, U2, th2 = dougherty_moments(C2, a_b, u_b, *N, 1)
    assert np.abs(dougherty_rate_real(C2, U2, th2, 1.0, a_b, u_b, *N, 1)).max() > 1e-3 * np.abs(C2).max()


def test_shifted_centred_basis_gives_mode_decay_rates_n_nu():
    """With u = U and a^2 = 2 theta, every retained mode decays at exactly n nu (sum over axes)."""
    N = (7, 6, 1)
    rng = np.random.default_rng(3)
    C = jnp.asarray(rng.normal(size=(1, 1, 6, 7, 1, 1, 1))) * 0.1
    C = C.at[:, 0, 0, 0].set(1.0)
    C = C.at[:, 0, 0, 1].set(0.0).at[:, 0, 1, 0].set(0.0).at[:, 0, 0, 2].set(0.3).at[:, 0, 2, 0].set(-0.3)
    a, u = jnp.array([1.0, 1.0, 1.0]), jnp.zeros(3)
    _, U, theta = dougherty_moments(C, a, u, *N, 1)
    np.testing.assert_allclose(theta.ravel(), 0.5, rtol=1e-14)
    nu = 0.4
    rate = dougherty_rate_real(C, U, theta, nu, a, u, *N, 1)
    n = np.arange(7)[None, :] + np.arange(6)[:, None]
    np.testing.assert_allclose(rate[0, 0, :, :, 0, 0, 0], -nu * n * C[0, 0, :, :, 0, 0, 0], atol=1e-15)


def run(params, **kw):
    base = dict(Nx=3, Ny=1, Nz=1, Ns=1, timesteps=6, progress_meter=NoProgressMeter(), dt=1e-2)
    base.update(kw)
    return simulation(params, **base)


def homogeneous_params(Ns, N, drifts, vths, a, u, mi=1.0):
    Ck = []
    for s in range(Ns):
        Us = jnp.broadcast_to(jnp.asarray(drifts[s]).reshape(1, 3, 1, 1, 1), (1, 3, 1, 3, 1))
        Ck.append(compute_C_nmp(Us, a[3 * s:3 * s + 3], u[3 * s:3 * s + 3], *N, 1, vth_s=vths[s]))
    Ck = jnp.concatenate(Ck).reshape(Ns * N[0] * N[1] * N[2], 1, 2, 1)
    return {"Ck_0": Ck, "Fk_0": jnp.zeros((6, 1, 2, 1), dtype=complex), "alpha_s": a, "u_s": u,
            "qs": jnp.zeros(Ns), "Omega_cs": jnp.array([1.0, 1.0 / mi][:Ns]), "nu": 0.0, "ode_tolerance": 1e-10}


def test_temperature_anisotropy_relaxes_at_the_dougherty_rate():
    """d = 2 active axes: T_x - T_y decays like exp(-2 nu t)."""
    N = (12, 12, 1)
    nu, t_max = 0.5, 1.0
    p = homogeneous_params(1, N, [[0, 0, 0]], [jnp.array([0.75, 0.55, 0.707107])], jnp.ones(3), jnp.zeros(3))
    p.update(t_max=t_max, nu_dougherty=nu)
    out = run(p, Nn=12, Nm=12, Np=1, collision_model="dougherty")
    C = jnp.fft.irfftn(out["Ck"].reshape(-1, 1, 12, 12, 1, 2, 1), s=(1, 1, 3), axes=(-1, -3, -2), norm="forward")
    Tx = 0.5 + C[:, 0, 0, 2, 0, 0, 0] / (r2 * C[:, 0, 0, 0, 0, 0, 0])
    Ty = 0.5 + C[:, 0, 2, 0, 0, 0, 0] / (r2 * C[:, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(Tx - Ty, (0.75 ** 2 - 0.55 ** 2) * np.exp(-2 * nu * out["time"]), rtol=1e-7)
    np.testing.assert_allclose(Tx + Ty, Tx[0] + Ty[0], rtol=1e-10)


def test_two_species_drift_and_temperature_equilibrate_conservatively():
    N = (8, 1, 1)
    a = jnp.array([1.0, 1, 1, 0.3, 1, 1]); u = jnp.array([0.2, 0, 0, -0.1, 0, 0])
    p = homogeneous_params(2, N, [[0.3, 0, 0], [-0.2, 0, 0]], [jnp.full(3, 0.6), jnp.full(3, 0.15)], a, u, mi=4.0)
    nu_sr = 0.6
    p.update(t_max=0.6, nu_dougherty=jnp.array([[0.3, nu_sr], [0.0, 0.1]]))
    out = run(p, Nn=8, Nm=1, Np=1, Ns=2, collision_model="dougherty")
    C = jnp.fft.irfftn(out["Ck"].reshape(-1, 2, 1, 1, 8, 1, 2, 1), s=(1, 1, 3), axes=(-1, -3, -2), norm="forward")
    C = C[:, :, 0, 0, :, 0, 0, 0]  # (t, s, n)
    aa = np.array([1.0, 0.3]); uu = np.array([0.2, -0.1]); ms = np.array([1.0, 4.0])
    dens = aa * C[:, :, 0]
    U = uu + aa * C[:, :, 1] / (r2 * C[:, :, 0])
    W = ms * aa * (uu ** 2 * C[..., 0] / 2 + uu * aa * C[..., 1] / r2 + aa ** 2 * (C[..., 0] / 2 + C[..., 2] / r2) / 2)
    P = ms * dens * U
    np.testing.assert_allclose(P.sum(1), P[0].sum(), atol=1e-11)
    np.testing.assert_allclose(W.sum(1), W[0].sum(), rtol=1e-11)
    dU = U[:, 0] - U[:, 1]
    nu_rs = nu_sr * ms[0] * dens[0, 0] / (ms[1] * dens[0, 1])
    np.testing.assert_allclose(dU, dU[0] * np.exp(-(nu_sr + nu_rs) / 2 * out["time"]), rtol=1e-7)
    assert abs(dU[-1]) < abs(dU[0])


def test_off_is_bitwise_unchanged_and_zero_nu_matches():
    p = {"t_max": 0.3, "nu": 0.2}
    kw = dict(Nx=7, Nn=6, Nm=1, Np=1, Ns=2, adaptive_time_step=False, timesteps=3)
    ref = run(p, **kw)
    zero = run({**p, "nu_dougherty": 0.0}, collision_model="dougherty", **kw)
    np.testing.assert_allclose(zero["Ck"], ref["Ck"], atol=1e-14)
    again = run(p, collision_model=None, **kw)
    assert jnp.array_equal(again["Ck"], ref["Ck"])


def test_jit_and_grad_through_a_collisional_run():
    N = (6, 1, 1)
    base = homogeneous_params(1, N, [[0.0, 0, 0]], [jnp.full(3, 0.6)], jnp.ones(3), jnp.zeros(3))
    base["Ck_0"] = base["Ck_0"].at[3, 0, 0, 0].add(0.05)  # non-Maxwellian: relaxes

    def loss(nu):
        out = run({**base, "t_max": 0.2, "nu_dougherty": nu}, Nn=6, Nm=1, Np=1, collision_model="dougherty",
                  timesteps=2)
        return jnp.real(out["Ck"][-1, 3, 0, 0, 0])

    g = jax.grad(loss)(0.5)
    eps = 1e-5
    fd = (loss(0.5 + eps) - loss(0.5 - eps)) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-5)
    assert np.isfinite(g) and g != 0


def test_toml_collisions_table(tmp_path):
    from spectrax import load_parameters
    f = tmp_path / "in.toml"
    f.write_text('[input_parameters]\nnu = 1.0\node_tolerance = 1e-8\n[solver_parameters]\nNx = 3\n'
                 '[collisions]\nmodel = "dougherty"\nnu = [[0.5, 0.2], [0.0, 0.1]]\n')
    inp, sol = load_parameters(f)
    assert sol["collision_model"] == "dougherty" and inp["nu_dougherty"] == [[0.5, 0.2], [0.0, 0.1]]
    f.write_text('[input_parameters]\nnu = 1.0\n[solver_parameters]\nNx = 3\n')
    inp, sol = load_parameters(f)
    assert "collision_model" not in sol and "nu_dougherty" not in inp


# ---- Francisquez et al. (JPP 88, 905880303, 2022) LBO-G cross-species moments ----

def _pair(beta, drift=True):
    dens = jnp.array([1.0, 0.7]); U = jnp.array([[0.3, -0.1, 0.2], [-0.5, 0.4, 0.0]]) * (1.0 if drift else 0.0)
    theta = jnp.array([0.5, 0.02]); ms = jnp.array([1.0, 25.0])
    return dens, U, theta, ms, dougherty_cross_moments(dens, U, theta, ms, 0, 1, (0, 1, 2), beta)


@pytest.mark.parametrize("beta", [-0.5, 0.0, 0.7])
def test_lbo_g_exchange_conserves_momentum_and_energy(beta):
    dens, U, theta, ms, (usr, th_sr, th_rs, urs) = _pair(beta)
    nu_sr = 0.8; nu_rs = nu_sr * ms[0] * dens[0] / (ms[1] * dens[1])
    R = [ms[0] * dens[0] * nu_sr * (usr - U[0]), ms[1] * dens[1] * nu_rs * (urs - U[1])]
    np.testing.assert_allclose(R[0] + R[1], 0, atol=1e-15)
    Q = [-ms[k] * dens[k] * nu * (3 * theta[k] + U[k] @ U[k] - u @ U[k] - 3 * th)
         for k, nu, th, u in ((0, nu_sr, th_sr, usr), (1, nu_rs, th_rs, urs))]
    np.testing.assert_allclose(Q[0] + Q[1], 0, atol=1e-14)


def test_lbo_g_matches_paper_formulas_eq_2_8_2_9():
    dens, U, theta, ms, (usr, th_sr, th_rs, urs) = _pair(0.3)
    d = 3; ms_, mr_ = 1.0, 25.0; delta = 1.0; beta = 0.3
    du2 = float(jnp.sum((U[0] - U[1]) ** 2))
    np.testing.assert_allclose(usr, U[0] + delta * (1 + beta) / 2 * (U[1] - U[0]), rtol=1e-14)
    vts2, vtr2 = float(theta[0]), float(theta[1])
    ref = vts2 + delta * (1 + beta) / 2 / (1 + ms_ / mr_) * (vtr2 - ms_ / mr_ * vts2 + du2 / d)
    np.testing.assert_allclose(th_sr, ref, rtol=1e-14)


def test_lbo_g_beta0_matches_morse_rates_and_em_doubles_the_temperature_rate():
    # Morse (1963) / Francisquez eq. 2.4 with alpha_E = 1, no drift: dT_s/dt = (T_r - T_s)/n_s;
    # LBO rate dT_s/dt = 2 nu_sr (T_sr - T_s), nu_sr = (m_s + m_r)/(m_s n_s)
    dens, U, theta, ms, g = _pair(0.0, drift=False)
    ns = float(dens[0]); mr = float(ms[1]); m_s = float(ms[0])
    nu_sr = (m_s + mr) / (m_s * ns)
    Ts, Tr = m_s * float(theta[0]), mr * float(theta[1])
    morse = (Tr - Ts) / ns
    np.testing.assert_allclose(2 * nu_sr * (m_s * g[1] - Ts), morse, rtol=1e-13)
    em = dougherty_cross_moments(dens, U, theta, ms, 0, 1, (0, 1, 2))
    np.testing.assert_allclose(2 * nu_sr * (m_s * em[1] - Ts), 2 * morse, rtol=1e-13)
    # momentum rate (no thermal part): nu_sr (u_sr - U_s) = (m_s + m_r)(U_r - U_s)/(2 m_s n_s) for both
    dens, U, theta, ms, g = _pair(0.0)
    np.testing.assert_allclose(ms[0] * dens[0] * nu_sr * (g[0] - U[0]), (ms[0] + ms[1]) * (U[1] - U[0]) / 2, rtol=1e-13)
    np.testing.assert_allclose(g[0], 0.5 * (U[0] + U[1]), rtol=1e-14)


def test_default_cross_moments_are_unchanged_by_the_beta_option():
    dens, U, theta, ms, _ = _pair(0.0)
    a = dougherty_cross_moments(dens, U, theta, ms, 0, 1, (0, 1, 2))
    b = dougherty_cross_moments(dens, U, theta, ms, 0, 1, (0, 1, 2), None)
    assert all(jnp.array_equal(x, y) for x, y in zip(a, b)) and len(a) == 3


@pytest.mark.parametrize("beta", [0.0, 0.6])
def test_lbo_g_spectral_conservation_to_round_off(beta):
    N, grid, Ns = (5, 4, 3), (6, 4, 3), 3
    Ck, C, mask = random_masked_state(Ns, N, grid)
    rng = np.random.default_rng(2)
    alpha = jnp.asarray(rng.uniform(0.6, 1.4, 3 * Ns)); u = jnp.asarray(rng.normal(size=3 * Ns) * 0.3)
    Omega = jnp.array([1.0, 0.25, 0.04]); nu = jnp.asarray(rng.uniform(0.2, 1.0, (Ns, Ns)))
    dCk = dougherty_collisions(C, nu, alpha, u, Omega, *N, Ns, mask, beta=beta)
    dC = np.asarray(jnp.fft.irfftn(dCk, s=(grid[2], grid[1], grid[0]), axes=(-1, -3, -2), norm="forward"))
    rates = physical_rates(dC, np.asarray(alpha), np.asarray(u), Ns, 1 / np.asarray(Omega))
    scale = np.abs(dC).max()
    np.testing.assert_allclose(rates[:, 0], 0, atol=1e-13 * scale)
    np.testing.assert_allclose(rates[:, 1:].sum(axis=0), 0, atol=1e-12 * scale * Ns)


def test_lbo_g_run_relaxes_temperature_at_the_morse_rate():
    # two Maxwellians, no drift, mass ratio 4: d(T_s - T_r)/dt = -alpha_E (1/n_s + 1/n_r)(T_s - T_r), alpha_E = n_s m_s nu_sr/(m_s+m_r)
    N = (8, 1, 1)
    a = jnp.array([0.6, 1, 1, 0.3, 1, 1]); u = jnp.zeros(6)
    p = homogeneous_params(2, N, [[0, 0, 0], [0, 0, 0]], [jnp.full(3, 0.6), jnp.full(3, 0.3)], a, u, mi=4.0)
    nu_sr = 0.5
    p.update(t_max=0.4, nu_dougherty=jnp.array([[0.0, nu_sr], [0.0, 0.0]]), beta_dougherty=0.0)
    out = run(p, Nn=8, Nm=1, Np=1, Ns=2, collision_model="dougherty_g", timesteps=40)
    C = jnp.fft.irfftn(out["Ck"].reshape(-1, 2, 1, 1, 8, 1, 2, 1), s=(1, 1, 3), axes=(-1, -3, -2), norm="forward")
    C = np.asarray(C[:, :, 0, 0, :, 0, 0, 0]); aa = np.array([0.6, 0.3]); ms = np.array([1.0, 4.0])
    T = ms * aa ** 2 * (C[..., 0] / 2 + C[..., 2] / r2) / C[..., 0]  # T = m alpha^2 (1/2 + C2/(sqrt2 C0))
    n = aa * C[:, :, 0]
    alphaE = n[0, 0] * ms[0] * nu_sr / (ms[0] + ms[1])
    rate = alphaE * (1 / n[0, 0] + 1 / n[0, 1])
    t = np.asarray(out["time"])
    np.testing.assert_allclose(T[:, 0] - T[:, 1], (T[0, 0] - T[0, 1]) * np.exp(-rate * t), rtol=2e-3)


def test_toml_francisquez_cross_species(tmp_path):
    from spectrax import load_parameters
    f = tmp_path / "in.toml"
    f.write_text('[input_parameters]\nnu = 1.0\n[solver_parameters]\nNx = 3\n'
                 '[collisions]\nmodel = "dougherty"\ncross_species = "francisquez_g"\nbeta = 0.2\nnu = 0.5\n')
    inp, sol = load_parameters(f)
    assert sol["collision_model"] == "dougherty_g" and inp["beta_dougherty"] == 0.2


def test_lbo_g_rejects_beta_at_or_below_minus_one(tmp_path):
    from spectrax import load_parameters
    toml = tmp_path / "in.toml"
    toml.write_text('[input_parameters]\n[solver_parameters]\n[collisions]\nmodel = "dougherty"\n'
                    'cross_species = "francisquez_g"\nbeta = -1.0\n')
    with pytest.raises(ValueError, match="beta > -1"):
        load_parameters(toml)
