import jax.numpy as jnp
import pytest

from spectrax import diagnostics, hermite_tail_fraction


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


def test_hermite_tail_fraction_uses_odd_rfft_weights():
    dCk = jnp.zeros((1, 4, 1, 3, 1), dtype=jnp.complex128)
    dCk = dCk.at[0, 0, 0, -1, 0].set(1.0)
    dCk = dCk.at[0, 3, 0, 0, 0].set(1.0)
    output = {"dCk": dCk, "Nx": 5, "Nn": 4, "Nm": 1, "Np": 1,
              "Ns": 1, "u_s": jnp.zeros(3)}

    assert hermite_tail_fraction(output)[0, 0] == pytest.approx(1 / 3)


def test_hermite_tail_fraction_resolves_species_and_active_axes():
    Nn, Nm, Np, Ns = 3, 2, 2, 2
    modes = Nn * Nm * Np
    dCk = jnp.zeros((1, Ns * modes, 1, 1, 1), dtype=jnp.complex128)
    dCk = dCk.at[0, 0, 0, 0, 0].set(2.0)
    dCk = dCk.at[0, Nn, 0, 0, 0].set(1.0)
    dCk = dCk.at[0, modes, 0, 0, 0].set(1.0)
    output = {"dCk": dCk, "Nx": 1, "Nn": Nn, "Nm": Nm, "Np": Np,
              "Ns": Ns, "u_s": jnp.zeros(3 * Ns)}

    assert hermite_tail_fraction(output)[0] == pytest.approx(jnp.array([0.2, 0.0]))


def test_hermite_tail_fraction_ignores_inactive_axes_and_zero_norm():
    output = {"dCk": jnp.zeros((1, 1, 1, 1, 1)), "Nx": 1,
              "Nn": 1, "Nm": 1, "Np": 1, "Ns": 1}

    assert hermite_tail_fraction(output)[0, 0] == 0.0


def test_hermite_tail_fraction_rejects_invalid_width():
    with pytest.raises(ValueError, match="positive integer"):
        hermite_tail_fraction({}, width=0)
