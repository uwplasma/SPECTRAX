"""Exact triangular remap of the Hermite basis (u, a) -> (u', a')."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from diffrax import NoProgressMeter
from spectrax import simulation
from spectrax._model import basis_rate_terms
from spectrax._remap import (remap_matrix, remap, moment_target, cap_target, remap_event, low_moments,
                             tail_fraction)
from tests.test_moving_basis import ORDERS, random_state, axis_moments

jax.config.update("jax_enable_x64", True)
Nn, Nm, Np = ORDERS


def h_table(N, x):
    h = np.zeros((N, x.size))
    h[0] = 1.0
    if N > 1:
        h[1] = np.sqrt(2) * x
    for n in range(1, N - 1):
        h[n + 1] = np.sqrt(2 / (n + 1)) * x * h[n] - np.sqrt(n / (n + 1)) * h[n - 1]
    return h


@pytest.mark.parametrize("N,A,B", [(1, 1.3, 0.2), (2, 0.8, -0.4), (12, 1.1, 0.5), (40, 0.95, -0.3)])
def test_remap_matrix_is_the_gauss_hermite_projection(N, A, B):
    """T[m, n] = pi^-1/2 integral e^{-xi^2} h_n(xi) h_m(A xi + B) dxi, exactly lower-triangular, T_nn = A^n."""
    x, w = np.polynomial.hermite.hermgauss(N + 2)
    Q = (h_table(N, A * x + B) * w) @ h_table(N, x).T / np.sqrt(np.pi)
    T = np.asarray(remap_matrix(N, A, B))
    assert np.abs(T - Q).max() < 1e-13 * max(1.0, np.abs(Q).max())
    assert np.all(np.triu(T, 1) == 0)
    np.testing.assert_allclose(np.diag(T), A ** np.arange(N), rtol=1e-14)


def _moves(Ns, seed):
    Ck, _, u, a = random_state(Ns, seed=seed)
    rng = np.random.default_rng(seed + 10)
    a2 = a * jnp.asarray(rng.uniform(0.95, 1.4, 3 * Ns))
    u2 = u + a2 * jnp.asarray(rng.uniform(-0.5, 0.5, 3 * Ns))
    return Ck, jnp.stack([u, a]), jnp.stack([u2, a2])


def test_remap_preserves_every_resolved_moment():
    """All mixed moments v_x^i v_y^j v_z^k with i < Nn, j < Nm, k < Np agree with quadrature to 1e-12."""
    Ns = 2
    Ck, B, B2 = _moves(Ns, seed=3)
    C2 = remap(Ck, B, B2, Nn, Nm, Np, Ns)

    def moments(C, basis):
        C = np.asarray(C)[..., 0, 0, 0].real  # k = 0, (Ns, Np, Nm, Nn); remap acts pointwise in x
        out = []
        for s in range(Ns):
            M = [axis_moments(N, float(basis[0, 3 * s + i]), float(basis[1, 3 * s + i]), kmax=N - 1)
                 for i, N in enumerate(ORDERS)]
            out.append(np.einsum("pmn,in,jm,kp->ijk", C[s], *M))
        return np.array(out)

    m0, m1 = moments(Ck, B), moments(C2, B2)
    assert np.abs(m1 - m0).max() < 1e-12 * np.abs(m0).max()
    lm0, lm1 = low_moments(Ck, B, Nn, Nm, Np, Ns), low_moments(C2, B2, Nn, Nm, Np, Ns)
    assert float(jnp.abs(lm1 - lm0).max()) < 1e-13 * float(jnp.abs(lm0).max())  # every Fourier mode


def test_round_trip_is_the_identity():
    Ns = 3
    Ck, B, B2 = _moves(Ns, seed=5)
    back = remap(remap(Ck, B, B2, Nn, Nm, Np, Ns), B2, B, Nn, Nm, Np, Ns)
    assert float(jnp.abs(back - Ck).max()) < 1e-13


def test_generator_of_the_remap_is_the_moving_basis_rate():
    """d/ds remap(C; (u, a) -> (u + u_dot s, a + a_dot s)) at s = 0 equals basis_rate_terms."""
    Ns = 2
    Ck, B, _ = _moves(Ns, seed=7)
    rates = jnp.asarray(np.random.default_rng(1).normal(size=(2, 3 * Ns)))
    f = lambda s: remap(Ck, B, B + s * rates, Nn, Nm, Np, Ns)
    deriv = jax.jacfwd(f)(0.0)
    expect = basis_rate_terms(Ck, rates[0], rates[1], B[1], Nn, Nm, Np, Ns)
    assert float(jnp.abs(deriv - expect).max()) < 1e-12 * float(jnp.abs(expect).max())


def _maxwellian(N, d, r):
    """Exact coefficients (in a basis of centre 0, width 1) of a Maxwellian centred at d with basis-matched
    width r: g_n = projection computed by quadrature."""
    x, w = np.polynomial.hermite.hermgauss(N + 40)
    v = x * r + d  # integrate f(v) h_n(v) dv with f = e^{-(v-d)^2/r^2}/(r sqrt(pi))
    return (h_table(N, v) * w).sum(axis=1) / np.sqrt(np.pi)


@pytest.mark.parametrize("d,r", [(0.5, 1.0), (0.9, 1.2), (-0.7, 0.9)])
def test_target_recentres_a_shifted_heated_maxwellian_to_one_mode(d, r):
    """moment_target recovers (U, sqrt(2) sigma) exactly and the remap returns g = delta_n0."""
    N = 24
    g = _maxwellian(N, d, r)
    Ck = jnp.zeros((1, 1, 1, N, 1, 1, 1)).at[0, 0, 0, :, 0, 0, 0].set(g)
    B = jnp.array([[0.0, 0, 0], [1.0, 1, 1]])
    target, sigma = moment_target(Ck, B, N, 1, 1, 1)
    np.testing.assert_allclose(target[:, 0], [d, r], rtol=1e-13)
    np.testing.assert_allclose(sigma[0], r / np.sqrt(2), rtol=1e-13)
    out = np.asarray(remap(Ck, B, target, N, 1, 1, 1))[0, 0, 0, :, 0, 0, 0] * r  # C' = (a/a') T C
    np.testing.assert_allclose(out, np.eye(N)[0], atol=1e-13)


def test_caps_are_enforced_and_targets_are_clipped():
    B = jnp.array([[0.0, 0, 0], [1.0, 1, 1]])
    Ck = jnp.zeros((1, 1, 1, 6, 1, 1, 1)).at[0, 0, 0, 0, 0, 0, 0].set(1.0)
    with pytest.raises(ValueError, match="caps"):
        remap_event(Ck, B, jnp.array([[1.5, 0, 0], [1.0, 1, 1]]), 6, 1, 1, 1)  # |du|/a' = 1.5
    with pytest.raises(ValueError, match="caps"):
        remap_event(Ck, B, jnp.array([[0.0, 0, 0], [0.8, 1, 1]]), 6, 1, 1, 1)  # narrows by 1.25
    capped = cap_target(B, jnp.array([[3.0, 0, 0], [0.5, 4.0, 1.0]]), sigma=jnp.array([0.2, 1.0, 1.0]))
    np.testing.assert_allclose(capped[1], [1 / 1.1, 4.0, 1.1])  # narrowing cap, free widening, sigma floor
    np.testing.assert_allclose(capped[0], [1 / 1.1, 0, 0])      # |du| <= a'
    _, rec = remap_event(Ck, B, capped, 6, 1, 1, 1, t=2.0)       # inside the caps: accepted and recorded
    assert rec["t"] == 2.0 and rec["moment_defect"] < 1e-14 and rec["new_basis"] == np.asarray(capped).tolist()
    assert float(tail_fraction(Ck, 6, 1, 1, 1)[0]) == 0.0


def test_landau_damping_is_unchanged_by_a_mid_run_remap():
    """Landau damping k lambda_D = 0.5: a run remapped at t = 5 to a shifted (0.3 a) and widened (1.1 a)
    electron basis tracks the fixed-basis run, and both damp at gamma = -0.1534."""
    N, Nx, k = 128, 4, 0.5
    a_e, a_i = np.sqrt(2.0), np.sqrt(2.0 / 1836)
    Ck0 = (jnp.zeros((2 * N, 1, Nx // 2 + 1, 1), complex).at[0, 0, 0, 0].set(a_e ** -3)
           .at[N, 0, 0, 0].set(a_i ** -3).at[0, 0, 1, 0].set(0.5e-3 * a_e ** -3))
    Fk0 = jnp.zeros((6, 1, Nx // 2 + 1, 1), complex).at[0, 0, 1, 0].set(0.5e-3j / k)
    B = jnp.array([[0.0] * 6, [a_e] * 3 + [a_i] * 3])

    def run(C, F, basis, T, n):
        p = {"Ck_0": C, "Fk_0": F, "qs": jnp.array([-1.0, 1.0]), "alpha_s": basis[1], "u_s": basis[0],
             "Omega_cs": jnp.array([1.0, 1 / 1836]), "mi_me": 1836.0, "nu": 0.0, "t_max": T, "Lx": 2 * np.pi / k,
             "ode_tolerance": 1e-10}
        return simulation(p, Nx=Nx, Nn=N, Ns=2, timesteps=n, frame="fixed", progress_meter=NoProgressMeter())

    ref = run(Ck0, Fk0, B, 15.0, 301)
    half = run(Ck0, Fk0, B, 5.0, 2)
    B2 = B.at[0, 0].set(0.3 * a_e).at[1, 0].set(1.1 * a_e)
    C1, rec = remap_event(half["Ck"][-1], B, B2, N, 1, 1, 2, t=5.0)
    rest = run(C1, half["Fk"][-1], B2, 10.0, 201)
    E_ref, E_new = np.abs(ref["Fk"][100:, 0, 0, 1, 0]), np.abs(rest["Fk"][:, 0, 0, 1, 0])
    assert rec["moment_defect"] < 1e-13
    assert np.abs(E_new - E_ref).max() < 1e-5 * E_ref.max()  # measured 5e-7
    t = np.asarray(ref["time"])
    E = np.abs(np.asarray(ref["Fk"][:, 0, 0, 1, 0]))
    peaks = [i for i in range(1, t.size - 1) if E[i] > E[i - 1] and E[i] > E[i + 1] and t[i] > 3]
    gamma = np.polyfit(t[peaks], np.log(E[peaks]), 1)[0]
    assert abs(gamma + 0.1534) < 2e-3
