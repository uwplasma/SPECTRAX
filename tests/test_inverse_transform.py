import jax
import jax.numpy as jnp
import numpy as np

from spectrax import inverse_HF_transform

jax.config.update("jax_enable_x64", True)


def test_inverse_transform_recovers_perturbed_maxwellian():
    """C_000 = n / alpha^3 with a cos(kx) perturbation reconstructs n (1 + eps cos kx) exp(-xi^2) / (pi^1.5 alpha^3)."""
    Nn, Nx, alpha, eps = 3, 8, np.sqrt(2.0), 0.1
    Ck = jnp.zeros((1, Nn, 1, Nx // 2 + 1, 1), dtype=jnp.complex128)
    Ck = Ck.at[0, 0, 0, 0, 0].set(1 / alpha**3).at[0, 0, 0, 1, 0].set(eps / 2 / alpha**3)
    xi = jnp.linspace(-3, 3, 7)
    xi_y, xi_x, xi_z = jnp.meshgrid(jnp.zeros(1), xi, jnp.zeros(1), indexing="ij")
    f = inverse_HF_transform(Ck, Nn, 1, 1, Nx, 1, 1, xi_x, xi_y, xi_z)[0, 0, :, 0, 0, :, 0]
    x = np.arange(Nx) / Nx * 2 * np.pi
    expected = (1 + eps * np.cos(x))[:, None] * np.exp(-xi**2)[None, :] / (np.pi**1.5 * alpha**3)
    np.testing.assert_allclose(f, expected, rtol=1e-12)
