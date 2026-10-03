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

@pytest.mark.parametrize("limits", [{"dtmin": 10.0}, {"max_steps": 2}])
def test_step_limits_stop_the_solve_with_an_error(limits):
    """A step below dtmin (here: any step) or too many steps stops the solve instead of crawling on."""
    with pytest.raises(Exception, match="minimum step size|maximum number of solver steps"):
        simulation(**limits)

def test_throw_false_reports_the_failure_instead_of_raising():
    """With throw=False a stopped solve returns, says why, and marks which saves hold a solution."""
    from diffrax import RESULTS
    ok = simulation(throw=False)
    assert ok["solver_result"] == RESULTS.successful
    assert int(ok["num_valid_times"]) == ok["time"].size and bool(jnp.all(ok["valid_times"]))
    for limits, result, n_min in [({"max_steps": 20}, RESULTS.max_steps_reached, 1),
                                  ({"dtmin": 10.0}, RESULTS.dt_min_reached, 0)]:
        failed = simulation(throw=False, **limits)
        assert failed["solver_result"] == result
        n = int(failed["num_valid_times"])
        assert n_min <= n < failed["time"].size
        assert bool(jnp.all(failed["valid_times"][:n])) and not bool(jnp.any(failed["valid_times"][n:]))
        assert bool(jnp.all(jnp.isfinite(failed["Fk"][:n]))) and not bool(jnp.any(jnp.isfinite(failed["Fk"][n:])))

def test_short_transient_finishes_within_the_step_budget_without_a_default_floor():
    """A transient needing steps ~1e-6 t_max finishes in far fewer than t_max/dt_transient steps.

    y' = -exp(-t/tau)/tau + cos t, y(0) = 1, has the solution exp(-t/tau) + sin t. With tau = 1e-5 and
    t_max = 10 the first steps are ~tau, far below t_max / max_steps; a floor inferred from the budget
    would stop this run, which in fact finishes in well under max_steps steps.
    """
    from diffrax import ODETerm, Dopri8, diffeqsolve, RESULTS
    from spectrax._simulation import _stepsize_controller
    tau, t_max, max_steps = 1e-5, 10.0, 400
    term = ODETerm(lambda t, y, args: -jnp.exp(-t / tau) / tau + jnp.cos(t))
    solve = lambda controller: diffeqsolve(term, Dopri8(), 0.0, t_max, 1e-3, jnp.array(1.0),
                                           stepsize_controller=controller, max_steps=max_steps, throw=False)
    sol = solve(_stepsize_controller(True, 1e-10))
    assert sol.result == RESULTS.successful and int(sol.stats["num_steps"]) < max_steps
    assert abs(float(sol.ys[-1]) - (jnp.exp(-t_max / tau) + jnp.sin(t_max))) < 1e-8
    assert solve(_stepsize_controller(True, 1e-10, dtmin=t_max / max_steps)).result == RESULTS.dt_min_reached

if __name__ == "__main__":
    pytest.main()
