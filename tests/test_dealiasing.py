"""Strict 2/3 de-aliasing: a product of retained modes must never alias back onto a retained mode."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectrax import simulation
from spectrax._simulation import _twothirds_mask

jax.config.update("jax_enable_x64", True)

SIZES = [3, 4, 5, 6, 7, 8, 9, 12, 33, 34, 129]


def _active_1d(N, axis):
    """Retained integer mode indices along one axis of the solver layout."""
    shape = {0: (N, 1, 1), 1: (1, N, 1), 2: (1, 1, N)}[axis]
    mask = np.asarray(_twothirds_mask(*shape)).reshape(-1)
    k = np.arange(N // 2 + 1) if axis == 1 else np.fft.fftfreq(N, d=1.0 / N).round().astype(int)
    return k[mask]


@pytest.mark.parametrize("N", SIZES)
@pytest.mark.parametrize("axis", [0, 1, 2])  # y (full FFT), x (rFFT), z (full FFT)
def test_mask_keeps_exactly_three_k_less_than_N(N, axis):
    kept = set(np.abs(_active_1d(N, axis)).tolist())
    assert kept == set(range((N - 1) // 3 + 1))
    assert all(3 * k < N for k in kept)


@pytest.mark.parametrize("N", SIZES)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_no_retained_sum_aliases_onto_a_retained_mode(N, axis):
    """Exhaustive triad check: for retained k1, k2 with k1+k2 not retained, its alias is not retained."""
    full = set(_active_1d(N, axis).tolist()) | set((-_active_1d(N, axis)).tolist())
    for k1, k2 in itertools.product(full, repeat=2):
        s = k1 + k2
        if s in full:
            continue
        alias = (s + N // 2) % N - N // 2  # representative of s mod N in [-N//2, N//2)
        assert alias not in full


@pytest.mark.parametrize("N", [6, 7, 8, 12, 33, 34, 129])
def test_square_of_boundary_cosine_has_only_its_mean_on_both_transforms(N):
    """cos(K x) at the largest retained K: the projected square is exactly 1/2 (the old mask left 1/4 at K)."""
    K = (N - 1) // 3
    x = np.arange(N) / N
    u = np.cos(2 * np.pi * K * x)
    # x axis: real FFT layout used by the solver
    mx = np.asarray(_twothirds_mask(1, N, 1))[0, :, 0]
    assert np.allclose(np.fft.irfft(np.fft.rfft(u, norm="forward") * mx, n=N, norm="forward"), u)
    pk = np.fft.rfft(u * u, norm="forward") * mx
    np.testing.assert_allclose(pk[0], 0.5, atol=1e-14)
    np.testing.assert_allclose(np.abs(pk[1:]).max(), 0.0, atol=1e-14)
    # y axis: full complex FFT
    my = np.asarray(_twothirds_mask(N, 1, 1))[:, 0, 0]
    qk = np.fft.fft(u * u, norm="forward") * my
    np.testing.assert_allclose(np.abs(qk[1:]).max(), 0.0, atol=1e-14)


@pytest.mark.parametrize("Ny, Nx", [(6, 6), (9, 7), (8, 12), (7, 9)])
def test_2d_mixed_triads_do_not_alias(Ny, Nx):
    """Product of two retained 2D modes (one at the x/y boundary) stays out of the active set unless exact."""
    mask = np.asarray(_twothirds_mask(Ny, Nx, 1))[:, :, 0]
    Kx, Ky = (Nx - 1) // 3, (Ny - 1) // 3
    y, x = np.meshgrid(np.arange(Ny) / Ny, np.arange(Nx) / Nx, indexing="ij")
    a = np.cos(2 * np.pi * (Kx * x + Ky * y))
    b = np.cos(2 * np.pi * (Kx * x - Ky * y))
    for f in (a, b):
        assert np.allclose(np.fft.irfft2(np.fft.rfft2(f) * mask, s=(Ny, Nx)), f)
    prod = np.fft.rfft2(a * b, norm="forward") * mask
    # exact product: 1/2 cos(2Kx x) + 1/2 cos(2Ky y); both are outside the active set for Kx, Ky > 0
    expected = np.zeros_like(prod)
    if Kx == 0:
        expected[0, 0] += 0.5
    if Ky == 0:
        expected[0, 0] += 0.5
    np.testing.assert_allclose(prod, expected, atol=1e-14)


def test_simulation_projects_initial_state_and_reports_it():
    """Excluded initial content is removed from the evolved state and its norm is reported."""
    Nx = 9  # active |kx| <= 2; kx = 3 is excluded (the old inclusive mask kept it)
    params = {"t_max": 0.1}
    from spectrax import initialize_simulation_parameters
    p = initialize_simulation_parameters(params, Nx=Nx, Nn=4, timesteps=2)
    Ck_0 = p["Ck_0"].at[0, 0, 3, 0].set(1e-3)
    out = simulation({"t_max": 0.1, "Ck_0": Ck_0}, Nx=Nx, Nn=4, timesteps=2)
    np.testing.assert_allclose(out["initial_projection_residual"], 1e-3, rtol=1e-12)
    assert jnp.all(out["Ck"][:, :, :, 3:, :] == 0)
    assert jnp.all(out["Fk"][:, :, :, 3:, :] == 0)
