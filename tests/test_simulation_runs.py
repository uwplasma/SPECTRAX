import pytest
import diffrax
import jax.numpy as jnp
import optimistix as optx
from spectrax import simulation
from spectrax.midpoint_solver import ImplicitMidpoint

def test_simulation_runs():
    """Test if the simulation runs without errors with default parameters."""
    result = simulation()
    assert isinstance(result, dict), "Simulation did not return a dictionary."
    assert "Ck" in result, "Missing Ck in output."
    assert "Fk" in result, "Missing Fk in output."
    assert result["Ck"].shape[1:] == (result["Ns"] * result["Nn"] * result["Nm"] * result["Np"],
                                      result["Ny"], result["Nx"] // 2 + 1, result["Nz"])
    assert result["Fk"].shape[1:] == (6, result["Ny"], result["Nx"] // 2 + 1, result["Nz"])

def test_electric_field_update():
    """Test that the electric field updates and does not remain zero."""
    result = simulation()
    assert not jnp.all(result["Fk"] == 0), "Electric field did not update."


def test_implicit_midpoint_runs_with_structured_state():
    result = simulation(
        {"t_max": 0.01, "ode_tolerance": 1e-8}, Nx=5, Nn=3,
        timesteps=2, dt=0.01,
        solver=ImplicitMidpoint(max_iters=4, linear_restart=2, linear_max_restarts=2),
        adaptive_time_step=False,
    )

    assert jnp.all(jnp.isfinite(result["Ck"]))
    assert jnp.all(jnp.isfinite(result["Fk"]))


def test_implicit_midpoint_reports_newton_exhaustion():
    term = diffrax.ODETerm(lambda t, y, args: y + 1)
    result = ImplicitMidpoint(max_iters=0).step(
        term, 0.0, 0.1, jnp.array(0.0), None, None, False
    )[-1]

    assert result == diffrax.RESULTS.promote(optx.RESULTS.nonlinear_max_steps_reached)


def test_implicit_midpoint_evaluates_rhs_at_midpoint_time():
    term = diffrax.ODETerm(lambda t, y, args: t)
    y1 = ImplicitMidpoint(max_iters=2, linear_restart=1, linear_max_restarts=1).step(
        term, 0.0, 0.2, jnp.array(0.0), None, None, False
    )[0]

    assert jnp.allclose(y1, 0.02, rtol=0, atol=1e-12)

if __name__ == "__main__":
    pytest.main()
