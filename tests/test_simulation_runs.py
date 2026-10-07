import pytest
import jax.numpy as jnp
from spectrax import simulation

def test_simulation_runs():
    """Test if the simulation runs without errors with default parameters."""
    result = simulation()
    assert isinstance(result, dict), "Simulation did not return a dictionary."
    assert "Ck" in result, "Missing Ck in output."
    assert "Fk" in result, "Missing Fk in output."
    assert result["solver_stats"]["num_steps"] == (
        result["solver_stats"]["num_accepted_steps"]
        + result["solver_stats"]["num_rejected_steps"]
    )

def test_electric_field_update():
    """Test that the electric field updates and does not remain zero."""
    result = simulation()
    assert not jnp.all(result["Fk"] == 0), "Electric field did not update."

if __name__ == "__main__":
    pytest.main()


def test_species_toml_builds_a_gauss_consistent_state_and_damps_at_the_landau_rate(tmp_path):
    """A physical species table gives the right densities, satisfies Gauss's law, and Landau damps at k lambda_D = 0.5."""
    import numpy as np
    from spectrax import load_parameters, simulation
    toml = tmp_path / "landau.toml"
    toml.write_text("[input_parameters]\nLx = 1.2566370614359172\nt_max = 20.0\nnu = 2.0\n"
                    "[solver_parameters]\nNx = 9\nNn = 64\ntimesteps = 201\n"
                    "[species.electrons]\ncharge = -1\nvth = 0.1414213562\nperturbation_amplitude = 0.01\n"
                    "[species.protons]\ncharge = 1\nmass = 1836\nvth = 0.0033002\n")
    p, s = load_parameters(toml)
    assert s["Ns"] == 2
    density = [np.asarray(p["Ck_0"][i * 64, 0, :, 0]) * np.prod(np.asarray(p["alpha_s"][3 * i:3 * i + 3])) for i in range(2)]
    np.testing.assert_allclose([density[0][0], density[0][1], density[1][1]], [1.0, 0.005, 0.0], atol=1e-12)
    k = 2 * np.pi / p["Lx"]
    np.testing.assert_allclose(1j * k * p["Fk_0"][0, 0, 1, 0], -density[0][1] + density[1][1], atol=1e-14)
    out = simulation(p, **s)
    t, E = np.asarray(out["time"]), np.abs(np.asarray(out["Fk"][:, 0, 0, 1, 0]))
    peaks = [i for i in range(1, len(E) - 1) if E[i] > E[i - 1] and E[i] > E[i + 1]]
    np.testing.assert_allclose(np.polyfit(t[peaks], np.log(E[peaks]), 1)[0], -0.1533, rtol=0.02)
    np.testing.assert_allclose(np.pi / np.mean(np.diff(t[peaks])), 1.4156, rtol=0.01)


def test_the_first_species_must_set_the_field_units():
    import pytest
    from spectrax import species_initial_state
    with pytest.raises(ValueError):
        species_initial_state({"ions": dict(charge=1, mass=1836)})
