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

def test_separate_rtol_atol_default_to_ode_tolerance():
    """ode_rtol/ode_atol default to ode_tolerance (bitwise the old run); a looser atol alone takes fewer steps."""
    kw = dict(Nx=9, Nn=8, timesteps=3)
    base = {"t_max": 2.0, "ode_tolerance": 1e-9}
    ref = simulation(base, **kw)
    same = simulation({**base, "ode_rtol": 1e-9, "ode_atol": 1e-9}, **kw)
    assert bool(jnp.all(ref["Ck"] == same["Ck"])) and bool(jnp.all(ref["Fk"] == same["Fk"]))
    assert float(ref["ode_rtol"]) == float(ref["ode_atol"]) == 1e-9
    loose = simulation({**base, "ode_atol": 1e-3}, **kw)
    assert int(loose["solver_stats"]["num_steps"]) < int(ref["solver_stats"]["num_steps"])

def test_floored_pid_controller():
    """floor=None (or all-None leaves) is bitwise PIDController; a floor on a tiny component stops it driving dt."""
    from diffrax import ODETerm, Dopri8, diffeqsolve, PIDController, SaveAt
    from spectrax import FlooredPIDController
    term = ODETerm(lambda t, y, args: {"big": jnp.cos(t) * y["big"], "tiny": -30.0 * jnp.sin(30.0 * t) * y["tiny"]})
    y0 = {"big": jnp.array([1.0, 2.0]), "tiny": jnp.array([1e-12])}
    solve = lambda c: diffeqsolve(term, Dopri8(), 0.0, 5.0, 1e-3, y0, stepsize_controller=c, saveat=SaveAt(t1=True))
    tol = dict(rtol=1e-10, atol=1e-16)
    ref = solve(PIDController(**tol))
    for floor in (None, {"big": None, "tiny": None}):
        sol = solve(FlooredPIDController(**tol, floor=floor))
        assert int(sol.stats["num_steps"]) == int(ref.stats["num_steps"])
        assert all(bool(jnp.all(a == b)) for a, b in zip(sol.ys.values(), ref.ys.values()))
    floored = solve(FlooredPIDController(**tol, floor={"big": None, "tiny": jnp.array(1e-10 * 1.0)}))
    assert int(floored.stats["num_steps"]) < int(ref.stats["num_steps"]) // 2
    assert bool(jnp.allclose(floored.ys["big"], ref.ys["big"], rtol=1e-8, atol=0))

if __name__ == "__main__":
    pytest.main()
