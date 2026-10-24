import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import factorial

from spectrax import compute_C_nmp

jax.config.update("jax_enable_x64", True)


def test_basis_width_maxwellian_keeps_the_closed_form():
    """Default vth: C_nmp = sqrt(2^(n+m+p) / (n! m! p!)) (U - u)^n ... / alpha^(n+1) ... (the previous formula)."""
    Nn, Nm, Np, Ns, alpha, u = 6, 3, 2, 2, np.array([1.0, 0.7, 1.3, 0.5, 0.9, 1.1]), np.array([0.1, -0.2, 0.3, 1.0, 0.0, -0.5])
    U = np.random.default_rng(1).normal(size=(Ns, 3, 2, 5, 1))
    a, w = alpha.reshape(Ns, 3), u.reshape(Ns, 3)
    n, m, p = np.meshgrid(np.arange(Nn), np.arange(Nm), np.arange(Np), indexing="ij")
    C = np.empty((Ns, Np, Nm, Nn, 2, 5, 1))
    for s in range(Ns):
        for idx in np.ndindex(Nn, Nm, Np):
            i, j, k = idx
            C[s, k, j, i] = (np.sqrt(2.0 ** (i + j + k) / (factorial(i) * factorial(j) * factorial(k)))
                             * (U[s, 0] - w[s, 0]) ** i * (U[s, 1] - w[s, 1]) ** j * (U[s, 2] - w[s, 2]) ** k
                             / (a[s, 0] ** (i + 1) * a[s, 1] ** (j + 1) * a[s, 2] ** (k + 1)))
    expected = np.fft.rfftn(C, axes=(-1, -3, -2), norm="forward")
    np.testing.assert_allclose(compute_C_nmp(U, alpha, u, Nn, Nm, Np, Ns), expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(compute_C_nmp(U, alpha, u, Nn, Nm, Np, Ns, vth_s=alpha / np.sqrt(2)), expected,
                               rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("vth, drift", [(0.5, 0.0), (0.5, 0.8), (0.7, -0.4), (1.0, 0.3)])
def test_narrow_maxwellian_in_a_wide_basis_is_reconstructed(vth, drift):
    """A Maxwellian of width vth drifting by `drift` in a basis of width alpha = sqrt(2) (vth <= 1): the Hermite
    series sum_n C_n H_n(xi) e^{-xi^2} / sqrt(2^n n! pi) must return the Gaussian (1D, x only). The series
    converges like (1 - vth^2)^(n/2), so 200 modes reach round-off for vth >= 0.5."""
    Nn, alpha = 200, np.sqrt(2.0)
    U = np.full((1, 3, 1, 1, 1), 0.0)
    U[0, 0] = drift
    C = np.asarray(compute_C_nmp(U, [alpha, alpha, alpha], [0.0] * 3, Nn, 1, 1, 1,
                                 vth_s=[vth, alpha / np.sqrt(2), alpha / np.sqrt(2)]))[0, 0, 0, :, 0, 0, 0].real
    v = np.linspace(-4, 4, 81)
    xi = v / alpha
    h = np.zeros((Nn, v.size))  # orthonormal Hermite functions
    h[0], h[1] = np.pi**-0.25 * np.exp(-xi**2 / 2), np.sqrt(2) * xi * np.pi**-0.25 * np.exp(-xi**2 / 2)
    for n in range(2, Nn):
        h[n] = np.sqrt(2 / n) * xi * h[n - 1] - np.sqrt((n - 1) / n) * h[n - 2]
    f = alpha**2 * np.pi**-0.25 * np.exp(-xi**2 / 2) * (C @ h)  # 1D marginal, as in inverse_HF_transform
    exact = np.exp(-(v - drift) ** 2 / (2 * vth**2)) / (np.sqrt(2 * np.pi) * vth)  # unit density
    np.testing.assert_allclose(f, exact, atol=1e-12)


def test_high_order_maxwellian_is_finite():
    """Nn = 256 overflowed the factorial form (inf / inf = NaN) before the recurrence."""
    U = jnp.linspace(-0.1, 0.1, 6).reshape(2, 3, 1, 1, 1)
    alpha = jnp.array([0.25, 0.3, 0.35, 0.05, 0.06, 0.07])
    assert jnp.all(jnp.isfinite(compute_C_nmp(U, alpha, jnp.zeros(6), 256, 4, 4, 2)))


def _projected_coefficients(sigma, shift, alpha, N):
    """Independent quadrature: g_n = int f(v) H_n(v/alpha) / sqrt(2^n n!) dv / alpha for a unit-density
    1D Maxwellian of standard deviation sigma centred at `shift` (basis centred at 0)."""
    v = np.linspace(shift - 60 * alpha, shift + 60 * alpha, 24001)
    xi = v / alpha
    # H_n(xi) / sqrt(2^n n!) = pi^(1/4) e^(xi^2/2) h_n(xi); w = f e^(xi^2/2), exponents combined (sigma < alpha).
    w = np.exp(-(v - shift) ** 2 / (2 * sigma**2) + xi**2 / 2) / (np.sqrt(2 * np.pi) * sigma)
    g, h_prev, h = np.empty(N), np.zeros_like(xi), np.pi**-0.25 * np.exp(-xi**2 / 2)
    for n in range(N):
        g[n] = np.trapezoid(w * np.pi**0.25 * h, v) / alpha
        h_prev, h = h, np.sqrt(2 / (n + 1)) * xi * h - np.sqrt(n / (n + 1)) * h_prev
    return g


@pytest.mark.parametrize("ratio", [0.5, 1 / np.sqrt(2), 0.85, 0.95])
def test_shifted_anisotropic_maxwellian_matches_quadrature_up_to_the_weighted_norm_bound(ratio):
    """sigma / alpha in {0.5, 1/sqrt(2) (matched), 0.85, 0.95}: inside the weighted-norm bound sigma < alpha,
    including broader-than-matched Maxwellians. Two species, three anisotropic axes, shifted centres,
    high order on x; every C_nmp equals the product of 1D coefficients projected by independent quadrature."""
    Ns, Nn, Nm, Np = 2, 96, 6, 4
    alpha = np.array([[1.0, 0.6, 1.7], [0.3, 0.45, 0.2]])
    sigma = ratio * alpha * np.array([1.0, 0.9, 0.8])[None, :] ** (ratio > 0.9)  # keep sigma < alpha
    shift = np.array([[0.4, -0.3, 0.9], [-0.1, 0.05, 0.12]])  # (U - u), in velocity units
    U = np.broadcast_to(shift[:, :, None, None, None], (Ns, 3, 1, 1, 1)).copy()
    C = np.asarray(compute_C_nmp(U, alpha.ravel(), np.zeros(6), Nn, Nm, Np, Ns, vth_s=sigma.ravel()))
    for s in range(Ns):
        gx, gy, gz = (_projected_coefficients(sigma[s, i], shift[s, i], alpha[s, i], N)
                      for i, N in enumerate((Nn, Nm, Np)))
        expected = np.einsum('p,m,n->pmn', gz, gy, gx)
        np.testing.assert_allclose(C[s, :, :, :, 0, 0, 0].real, expected, rtol=0, atol=1e-11 * np.abs(expected).max())
        # Geometric decay at the rate |1 - 2 sigma^2 / alpha^2|^(n/2) predicted for the weighted norm.
        c2 = abs(1 - 2 * (sigma[s, 0] / alpha[s, 0]) ** 2)
        if c2 > 0.05:
            g = np.abs(C[s, 0, 0, :, 0, 0, 0])
            rate = (g[-1] / g[-21]) ** (1 / 20)  # the drift adds a slowly varying prefactor
            assert abs(rate / np.sqrt(c2) - 1) < 0.1


def test_coefficients_stop_decaying_at_and_beyond_sigma_equal_alpha():
    """At sigma = alpha the even coefficients only decay like n^(-1/4) (weighted norm diverges);
    beyond it they grow. The recurrence stays finite; the expansion is not convergent."""
    Nn, alpha = 400, 1.0
    U = np.zeros((1, 3, 1, 1, 1))
    for ratio, grows in [(1.0, False), (1.1, True)]:
        g = np.asarray(compute_C_nmp(U, [alpha] * 3, [0.0] * 3, Nn, 1, 1, 1,
                                     vth_s=[ratio * alpha, alpha / np.sqrt(2), alpha / np.sqrt(2)]))[0, 0, 0, ::2, 0, 0, 0]
        assert np.all(np.isfinite(g))
        weighted_norm = np.cumsum(np.abs(g) ** 2)
        assert weighted_norm[-1] > 2 * weighted_norm[len(g) // 8]  # no convergence of sum |g_n|^2
        assert (abs(g[-1]) > abs(g[len(g) // 2])) == grows
