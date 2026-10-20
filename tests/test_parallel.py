"""Tests of the Hermite streaming and Maxwell flux splittings used by the DG fluxes."""
import jax.numpy as jnp
import numpy as np
import pytest

from spectrax._initialization import compute_A_pm_matrices, compute_R_pm_matrices


# ------------------------------------------------------------------ A_pm

def eig_split(M):
    """Base-commit construction: eigendecomposition with a general `eig`."""
    Lam, U = jnp.linalg.eig(M)
    D = Lam[..., :, None] * jnp.eye(M.shape[-1])
    part = lambda s: np.real(np.asarray(U @ (D + s * jnp.abs(D)) @ jnp.swapaxes(U, -1, -2) / 2))
    return part(1), part(-1)


@pytest.mark.parametrize("N", [(3, 2, 1), (5, 4, 3)])
def test_A_pm_symmetry_sum_and_eig_reference(N):
    Ns = 3
    rng = np.random.default_rng(1)
    alpha, u = jnp.array(rng.uniform(.3, 1.2, (Ns, 3))), jnp.array(rng.normal(0, .5, (Ns, 3)))
    Ap = compute_A_pm_matrices(*N, alpha, u)
    for k, n in enumerate(N):
        A_p, A_m = (np.asarray(Ap[2 * k + j]).reshape(Ns, n, n) for j in range(2))  # all other axes are size one
        off = np.diag(np.sqrt(np.arange(1, n) / 2), 1) + np.diag(np.sqrt(np.arange(1, n) / 2), -1)
        A = np.asarray(alpha)[:, k, None, None] * off + np.asarray(u)[:, k, None, None] * np.eye(n)
        for M in (A_p, A_m):
            np.testing.assert_allclose(M, np.swapaxes(M, -1, -2), atol=1e-14)
        np.testing.assert_allclose(A_p + A_m, A, atol=1e-13)
        np.testing.assert_allclose(A_p @ A_m, 0, atol=1e-12)  # orthogonal positive/negative parts
        assert np.linalg.eigvalsh(A_p).min() > -1e-12 and np.linalg.eigvalsh(A_m).max() < 1e-12
        ref_p, ref_m = eig_split(jnp.array(A))
        np.testing.assert_allclose(A_p, ref_p, atol=1e-12), np.testing.assert_allclose(A_m, ref_m, atol=1e-12)


def test_R_pm_sum_and_central_flux():
    for upwind in (False, True):
        R_p, R_m = (np.asarray(R).reshape(3, 6, 6) for R in compute_R_pm_matrices(upwind))
        np.testing.assert_allclose(R_p + R_m, np.asarray(compute_R_pm_matrices(False)[0]).reshape(3, 6, 6) * 2, atol=1e-13)
        np.testing.assert_allclose(R_p, np.swapaxes(R_p, -1, -2), atol=1e-14)
