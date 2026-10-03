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


def _field_output(Fk, Nx, Omega=1.3):
    Nt, _, Ny, Nx_kept, Nz = Fk.shape
    return {"Nx": Nx, "Nn": 1, "Nm": 1, "Np": 1, "alpha_s": jnp.ones(3), "u_s": jnp.zeros(3),
            "Omega_cs": jnp.array([Omega]), "Lx": 1.0,
            "Ck": jnp.zeros((Nt, 1, Ny, Nx_kept, Nz), dtype=jnp.complex128), "Fk": Fk}


@pytest.mark.parametrize("Ny, Nx, Nz", [(1, 1, 1), (1, 2, 1), (1, 7, 1), (1, 8, 1), (5, 6, 1),
                                        (4, 9, 1), (1, 6, 5), (3, 5, 4), (4, 4, 4), (2, 3, 7)])
def test_field_energy_matches_independent_full_fft_parseval(Ny, Nx, Nz):
    """EM_energy from the stored half spectrum equals numpy's full-FFT sum and the real-space mean,
    for odd and even Nx, full-FFT y/z axes, and singleton axes."""
    import numpy as np
    F = np.random.default_rng(Nx + 10 * Ny + 100 * Nz).standard_normal((6, Ny, Nx, Nz))
    Fk = jnp.fft.rfftn(F, axes=(-1, -3, -2), norm="forward")[None]
    output = _field_output(Fk, Nx)
    diagnostics(output)
    full = np.sum(np.abs(np.fft.fftn(F, axes=(1, 2, 3), norm="forward")) ** 2)
    np.testing.assert_allclose(output["EM_energy"][0], 0.5 * 1.3**2 * full, rtol=1e-12)
    np.testing.assert_allclose(output["EM_energy"][0], 0.5 * 1.3**2 * np.sum(np.mean(F**2, axis=(1, 2, 3))),
                               rtol=1e-12)


@pytest.mark.parametrize("Nx, Nx_kept", [(9, 3), (8, 4), (9, 6)])
def test_field_energy_rejects_cropped_or_mismatched_rfft_axis(Nx, Nx_kept):
    """A cropped/dealiased array is not a full rfft: the weights cannot be inferred and must be refused."""
    output = _field_output(jnp.zeros((1, 6, 1, Nx_kept, 1), dtype=jnp.complex128), Nx)
    with pytest.raises(ValueError, match="full rfft"):
        diagnostics(output)


def test_field_energy_requires_original_Nx():
    output = _field_output(jnp.zeros((1, 6, 1, 5, 1), dtype=jnp.complex128), 9)
    del output["Nx"]
    with pytest.raises(KeyError):
        diagnostics(output)


def test_hermite_tail_fraction_is_per_species_for_three_species():
    """Each species' tail fraction uses only its own coefficients (3 species, unequal tails)."""
    Nn, Ns = 4, 3
    dCk = jnp.zeros((1, Ns * Nn, 1, 2, 1), dtype=jnp.complex128)
    for s, tail in enumerate([0.0, 1.0, 3.0]):
        dCk = dCk.at[0, s * Nn + 1, 0, 1, 0].set(1.0)       # bulk, kx=1 (weight 2 at Nx=3)
        dCk = dCk.at[0, s * Nn + Nn - 1, 0, 1, 0].set(tail)  # outermost Hermite mode
    out = hermite_tail_fraction({"dCk": dCk, "Nx": 3, "Nn": Nn, "Nm": 1, "Np": 1, "Ns": Ns})
    assert out[0] == pytest.approx(jnp.array([0.0, 0.5, 0.9]))
