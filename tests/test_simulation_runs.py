import pytest
import jax.numpy as jnp
from spectrax import DG_project, simulation


def test_DG_project_retains_highest_mode():
    coefficients = DG_project(lambda x, y, z: 0.5 * (3 * (2 * x - 1) ** 2 - 1),
                              {"Lx": 1.0, "Ly": 1.0, "Lz": 1.0},
                              {"Nx": 1, "Ny": 1, "Nz": 1, "N_DG": 3, "dims": 1})
    assert jnp.allclose(coefficients, jnp.array([0.0, 0.0, 1.0]), atol=1e-14)

def test_simulation_runs():
    """Test if the simulation runs without errors with default parameters."""
    result = simulation()
    assert isinstance(result, dict), "Simulation did not return a dictionary."
    assert "Ck" in result, "Missing Ck in output."
    assert "Fk" in result, "Missing Fk in output."

def test_electric_field_update():
    """Test that the electric field updates and does not remain zero."""
    result = simulation()
    assert not jnp.all(result["Fk"] == 0), "Electric field did not update."

if __name__ == "__main__":
    pytest.main()
