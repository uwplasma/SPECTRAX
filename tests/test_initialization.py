import jax.numpy as jnp
import pytest

from spectrax import periodic_grid


def test_periodic_grid_rejects_empty_grid():
    with pytest.raises(ValueError, match="size must be positive"):
        periodic_grid(1.0, 0)


@pytest.mark.parametrize("size", [32, 33])
def test_periodic_grid_preserves_fourier_mode(size):
    length = 5.0
    mode = 3
    grid = periodic_grid(length, size)
    signal = jnp.sin(2 * jnp.pi * mode * grid / length)
    spectrum = jnp.fft.rfft(signal, norm="forward")

    assert grid.shape == (size,)
    assert grid[-1] < length
    assert jnp.max(jnp.abs(spectrum.at[mode].set(0))) < 1e-12
