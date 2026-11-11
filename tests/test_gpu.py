"""Small smoke tests for the self-hosted GPU CI job (run with `pytest -m gpu`)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from spectrax import inverse_HF_transform, simulation

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.gpu

def _rhs(device):
    """Vector field of a small problem at t=0, evaluated on `device` (no time loop: cheap to compile)."""
    import spectrax._simulation as sim
    captured = {}

    def stop(term, **kwargs):
        captured["rhs"] = jax.jit(term.vector_field)(0.0, kwargs["y0"], kwargs["args"])
        raise RuntimeError("captured")

    mp = pytest.MonkeyPatch()
    mp.setattr(sim, "diffeqsolve", stop)
    try:
        with jax.disable_jit(), jax.default_device(device), pytest.raises(RuntimeError, match="captured"):
            simulation(Nx=7, Nn=4, timesteps=3)
    finally:
        mp.undo()
    return [np.asarray(y) for y in captured["rhs"]]


def test_vector_field_is_finite_on_default_device():
    assert all(np.all(np.isfinite(y)) for y in _rhs(jax.devices()[0]))


def test_vector_field_matches_cpu():
    gpu, cpu = _rhs(jax.devices()[0]), _rhs(jax.devices("cpu")[0])
    for g, c in zip(gpu, cpu):
        np.testing.assert_allclose(g, c, rtol=1e-8, atol=1e-12)


def test_inverse_transform_density_moment():
    Nx, Nn = 8, 3
    C = np.random.default_rng(0).standard_normal((1, 1, 1, Nn, 1, Nx, 1))
    Ck = jnp.fft.rfftn(C, axes=(-1, -3, -2), norm="forward").reshape(1, Nn, 1, Nx // 2 + 1, 1)
    node, weight = hermgauss(4)
    xi_y, xi_x, xi_z = np.meshgrid(node, node, node, indexing="ij")
    f = inverse_HF_transform(Ck, Nn, 1, 1, Nx, 1, 1, xi_x, xi_y, xi_z)[0]
    quadrature = np.einsum("i,j,k->ijk", weight, weight, weight) * np.exp(xi_x**2 + xi_y**2 + xi_z**2)
    density = np.einsum("yxzijk,ijk->yxz", np.asarray(f), quadrature)
    np.testing.assert_allclose(density, C[0, 0, 0, 0], atol=1e-12)
