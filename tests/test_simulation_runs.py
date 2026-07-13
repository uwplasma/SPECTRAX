import pytest
import jax.numpy as jnp
from diffrax import RESULTS
from spectrax import simulation

def test_simulation_runs():
    """Test if the simulation runs without errors with default parameters."""
    result = simulation(throw=False)
    assert isinstance(result, dict), "Simulation did not return a dictionary."
    assert "Ck" in result, "Missing Ck in output."
    assert "Fk" in result, "Missing Fk in output."
    assert result["Ck"].shape[1:] == (result["Ns"] * result["Nn"] * result["Nm"] * result["Np"],
                                      result["Ny"], result["Nx"] // 2 + 1, result["Nz"])
    assert result["Fk"].shape[1:] == (6, result["Ny"], result["Nx"] // 2 + 1, result["Nz"])
    assert result["solver_result"] == RESULTS.successful

def test_electric_field_update():
    """Test that the electric field updates and does not remain zero."""
    result = simulation()
    assert not jnp.all(result["Fk"] == 0), "Electric field did not update."

def test_max_steps_failure_reported():
    """Report a solver step limit when exceptions are disabled."""
    result = simulation({"t_max": 0.02}, dt=0.01, adaptive_time_step=False,
                        max_steps=1, throw=False)
    assert result["solver_result"] == RESULTS.max_steps_reached

if __name__ == "__main__":
    pytest.main()
