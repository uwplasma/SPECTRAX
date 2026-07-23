import pytest
import jax.numpy as jnp
from diffrax import ConstantStepSize, ODETerm, diffeqsolve
from spectrax import SSPRK3, simulation

def test_ssprk3_is_third_order():
    term = ODETerm(lambda t, y, args: -y)

    def error(dt):
        solution = diffeqsolve(
            term, SSPRK3(), t0=0, t1=1, dt0=dt, y0=jnp.array(1.0),
            stepsize_controller=ConstantStepSize(),
        )
        return abs(solution.ys[0] - jnp.exp(-1.0))

    assert error(0.05) < error(0.1) / 7.5

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
