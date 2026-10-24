import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from spectrax import inverse_HDG_transform

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("Nx, Ny, Nz", [(8, 1, 1), (7, 1, 1), (6, 5, 1), (5, 4, 3)])
def test_inverse_transform_moments_match_the_solver_convention(Nx, Ny, Nz):
    """Random real C_nmp per element -> Ck with the element mean in the first Legendre slot, then back
    through inverse_HDG_transform: the density and first velocity moments of f must return C_000 and
    C_100/sqrt(2) (C_010, C_001) on every element."""
    Nn, Nm, Np = 3, 2 if Ny > 1 else 1, 2 if Nz > 1 else 1
    C = np.random.default_rng(0).standard_normal((1, Np, Nm, Nn, Ny, Nx, Nz))
    Ck = jnp.asarray(C.reshape(1, Np * Nm * Nn, Ny, Nx, Nz))[..., None]  # (1, H, Ny, Nx, Nz, Nl=1)
    node, weight = hermgauss(4)  # exact for the polynomial degrees involved
    xi_y, xi_x, xi_z = np.meshgrid(node, node, node, indexing="ij")
    f = inverse_HDG_transform(Ck, Nn, Nm, Np, xi_x, xi_y, xi_z)[0]  # (Ny, Nx, Nz, 4, 4, 4)
    quadrature = np.einsum("i,j,k->ijk", weight, weight, weight) * np.exp(xi_x**2 + xi_y**2 + xi_z**2)
    moment = lambda g: np.einsum("yxzijk,ijk->yxz", f * g, quadrature)
    grid = lambda n, m, p: C[0, p, m, n]  # (Ny, Nx, Nz)
    np.testing.assert_allclose(moment(1.0), grid(0, 0, 0), atol=1e-12)
    np.testing.assert_allclose(moment(xi_x), grid(1, 0, 0) / np.sqrt(2), atol=1e-12)
    if Ny > 1:
        np.testing.assert_allclose(moment(xi_y), grid(0, 1, 0) / np.sqrt(2), atol=1e-12)
    if Nz > 1:
        np.testing.assert_allclose(moment(xi_z), grid(0, 0, 1) / np.sqrt(2), atol=1e-12)
