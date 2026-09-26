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
