import jax.numpy as jnp
import pytest

from spectrax import diagnostics


@pytest.mark.parametrize(("Nx", "expected_energy"), [(1, 0.5), (4, 0.5), (5, 1.0)])
def test_rfft_field_energy_weights_nyquist_only_for_even_grid(Nx, expected_energy):
    Nx_kept = Nx // 2 + 1
    output = {
        "Nx": Nx,
        "Nn": 3,
        "Nm": 1,
        "Np": 1,
        "alpha_s": jnp.ones(6),
        "u_s": jnp.zeros(6),
        "Omega_cs": jnp.ones(2),
        "mi_me": 1.0,
        "Lx": 1.0,
        "Ck": jnp.zeros((1, 6, 1, Nx_kept, 1), dtype=jnp.complex128),
        "Fk": jnp.zeros((1, 6, 1, Nx_kept, 1), dtype=jnp.complex128)
        .at[0, 0, 0, -1, 0]
        .set(1.0),
    }

    diagnostics(output)

    assert output["EM_energy"][0] == pytest.approx(expected_energy)
