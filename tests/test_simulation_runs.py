import pytest
import diffrax
import jax.numpy as jnp
import optimistix as optx
from spectrax import simulation
from spectrax.midpoint_solver import (
    ImplicitMidpoint,
    collision_diffusion_x_streaming_preconditioner,
)

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
    assert result["Ck"].shape[1:] == (result["Ns"] * result["Nn"] * result["Nm"] * result["Np"],
                                      result["Ny"], result["Nx"] // 2 + 1, result["Nz"])
    assert result["Fk"].shape[1:] == (6, result["Ny"], result["Nx"] // 2 + 1, result["Nz"])

def test_electric_field_update():
    """Test that the electric field updates and does not remain zero."""
    result = simulation()
    assert not jnp.all(result["Fk"] == 0), "Electric field did not update."

def test_implicit_midpoint_step_structured_state():
    term = diffrax.ODETerm(lambda t, y, args: (-y[0], 2 * y[1]))
    y1, _, _, _, result = ImplicitMidpoint().step(
        term, 0.0, 0.1, (jnp.ones(2), jnp.ones((2, 2))), None, None, False
    )
    assert result == diffrax.RESULTS.successful
    assert jnp.allclose(y1[0], 0.95 / 1.05, rtol=1e-5)
    assert jnp.allclose(y1[1], 1.1 / 0.9, rtol=1e-5)

def test_implicit_midpoint_reports_newton_exhaustion():
    term = diffrax.ODETerm(lambda t, y, args: y + 1)
    result = ImplicitMidpoint(max_iters=0).step(
        term, 0.0, 0.1, jnp.array(0.0), None, None, False
    )[-1]
    assert result == diffrax.RESULTS.promote(optx.RESULTS.nonlinear_max_steps_reached)

def test_implicit_midpoint_evaluates_rhs_at_midpoint_time():
    term = diffrax.ODETerm(lambda t, y, args: t)
    y1 = ImplicitMidpoint(max_iters=2).step(
        term, 0.0, 0.2, jnp.array(0.0), None, None, False
    )[0]
    assert jnp.allclose(y1, 0.02, rtol=0, atol=1e-12)

def test_implicit_midpoint_simulation_reports_iterations():
    result = simulation(
        {"t_max": 0.01, "ode_tolerance": 1e-8}, Nx=5, Nn=3,
        timesteps=2, dt=0.01, adaptive_time_step=False,
        solver=ImplicitMidpoint(
            max_iters=4, linear_restart=2, linear_max_restarts=2,
            preconditioner=collision_diffusion_x_streaming_preconditioner,
        ),
    )
    assert jnp.all(jnp.isfinite(result["Ck"]))
    stats = result["midpoint_stats"]
    assert stats.newton_iterations >= stats.max_newton_iterations > 0
    assert stats.linear_iterations >= stats.max_linear_iterations > 0

def test_collision_diffusion_x_streaming_line():
    args = [None] * 21
    args[-20], args[-19] = 2.0, 3.0
    args[-17], args[-16], args[-15] = jnp.arange(4.0, 7.0), jnp.arange(7.0, 10.0), 10.0
    args[-12], args[-9] = jnp.full((1, 1, 1), 2.0), jnp.full((1, 1, 1), 5.0)
    args[-7] = jnp.arange(3.0).reshape(1, 1, 3)
    args[-6] = jnp.sqrt(jnp.arange(1.0, 4.0)).reshape(1, 1, 1, 3, 1, 1, 1)
    args[-5] = jnp.sqrt(jnp.arange(3.0)).reshape(1, 1, 1, 3, 1, 1, 1)
    coefficients = jnp.arange(1.0, 4.0).astype(complex).reshape(1, 1, 1, 3, 1, 1, 1)
    fields = jnp.ones((6, 1, 1, 1))

    actual = collision_diffusion_x_streaming_preconditioner(tuple(args), 0.2)(
        (coefficients, fields)
    )
    phase = 0.1j * 2 / 10
    diagonal = 1 + 0.1 * (2 * jnp.arange(3.0) + 15) + phase * 7
    coupling = phase * 4 / jnp.sqrt(2)
    matrix = (jnp.diag(diagonal)
              + jnp.diag(coupling * jnp.sqrt(jnp.arange(1.0, 3.0)), 1)
              + jnp.diag(coupling * jnp.sqrt(jnp.arange(1.0, 3.0)), -1))
    expected = jnp.linalg.solve(matrix, coefficients.reshape(3))
    assert jnp.allclose(actual[0].reshape(3), expected, rtol=1e-6, atol=1e-6)
    assert jnp.array_equal(actual[1], fields)

if __name__ == "__main__":
    pytest.main()
