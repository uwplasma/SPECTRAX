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
    assert result["solver_stats"]["num_steps"] == (
        result["solver_stats"]["num_accepted_steps"]
        + result["solver_stats"]["num_rejected_steps"]
    )

def test_solver_state_keeps_species_and_hermite_axes(monkeypatch):
    """The integrator sees (Ck, Fk) with their species, Hermite, grid and element axes, not one flat vector."""
    import spectrax._simulation as sim
    captured = {}
    def stop(term, **kwargs):
        captured.update(kwargs, rhs=term.vector_field(0.0, kwargs["y0"], kwargs["args"]))
        raise RuntimeError("captured")
    monkeypatch.setattr(sim, "diffeqsolve", stop)
    with pytest.raises(RuntimeError, match="captured"):
        simulation(Nx=7, Nn=4, timesteps=3)
    shapes = [(2, 1, 1, 4, 1, 7, 1, 2), (6, 1, 7, 1, 2)]
    assert [y.shape for y in captured["y0"]] == shapes
    assert [y.shape for y in captured["rhs"]] == shapes

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



def _three_species_unequal_orders():
    import numpy as np
    Ns, Nn, Nm, Np, Ny, Nx, Nz, Nl = 3, 3, 2, 4, 2, 5, 3, 2
    H = Nn * Nm * Np
    rng = np.random.default_rng(3)
    shape = (Ns * H, Ny, Nx, Nz, Nl)
    Ck_0 = jnp.asarray(1e-2 * rng.standard_normal(shape))
    Ck_0 = Ck_0.at[jnp.arange(Ns) * H, :, :, :, 0].set(1.0)
    params = {"Ck_0": Ck_0, "Fk_0": jnp.zeros((6,) + shape[1:]),
              "ms": jnp.array([1.0, 2.0, 4.0]), "qs": jnp.array([-1.0, 1.0, 2.0]),
              "alpha_s": jnp.linspace(0.4, 0.9, 3 * Ns), "u_s": jnp.linspace(-0.2, 0.2, 3 * Ns), "t_max": 0.2}
    grid = dict(Nx=Nx, Ny=Ny, Nz=Nz, Nn=Nn, Nm=Nm, Np=Np, Ns=Ns, N_DG=Nl)
    return params, grid


def test_structured_state_maps_flat_hermite_index_to_species_p_m_n(monkeypatch):
    """With unequal (Nn, Nm, Np) and 3 species, the solver state axis order is (s, p, m, n): flat
    output index s*Nn*Nm*Np + p*Nn*Nm + m*Nn + n lands at state[s, p, m, n]."""
    import spectrax._simulation as sim
    params, g = _three_species_unequal_orders()
    captured = {}
    def stop(term, **kwargs):
        captured["y0"] = kwargs["y0"]
        raise RuntimeError("captured")
    monkeypatch.setattr(sim, "diffeqsolve", stop)
    import jax
    with jax.disable_jit(), pytest.raises(RuntimeError, match="captured"):
        simulation(params, **g, timesteps=2)
    Ck = captured["y0"][0]
    assert Ck.shape == (g["Ns"], g["Np"], g["Nm"], g["Nn"], g["Ny"], g["Nx"], g["Nz"], g["N_DG"])
    Nn, Nm, Np = g["Nn"], g["Nm"], g["Np"]
    for s, p, m, n in [(0, 0, 0, 1), (1, 3, 1, 2), (2, 2, 0, 0)]:
        flat = s * Nn * Nm * Np + p * Nn * Nm + m * Nn + n
        assert jnp.array_equal(Ck[s, p, m, n], params["Ck_0"][flat])


def test_continuation_from_output_matches_single_run():
    """Restarting from a saved (Ck, Fk) snapshot reproduces the uninterrupted fixed-step trajectory."""
    from diffrax import Tsit5
    params, g = _three_species_unequal_orders()
    kw = dict(dt=0.01, solver=Tsit5(), adaptive_time_step=False)
    full = simulation({**params, "t_max": 0.2}, **g, timesteps=3, **kw)        # t = 0, 0.1, 0.2
    first = simulation({**params, "t_max": 0.1}, **g, timesteps=2, **kw)
    second = simulation({**params, "t_max": 0.1, "Ck_0": first["Ck"][-1], "Fk_0": first["Fk"][-1]},
                        **g, timesteps=2, **kw)
    assert jnp.allclose(second["Ck"][-1], full["Ck"][-1], rtol=0, atol=1e-13)
    assert jnp.allclose(second["Fk"][-1], full["Fk"][-1], rtol=0, atol=1e-13)


def test_implicit_midpoint_accepts_the_structured_state():
    """ImplicitMidpoint works on the (Ck, Fk) pytree and agrees with Tsit5 on a short run."""
    from diffrax import Tsit5
    from spectrax.midpoint_solver import ImplicitMidpoint
    params = {"t_max": 0.05, "ode_tolerance": 1e-8}
    g = dict(Nx=5, Nn=4, timesteps=2, dt=0.005, adaptive_time_step=False)
    explicit = simulation(params, solver=Tsit5(), **g)
    implicit = simulation(params, solver=ImplicitMidpoint(rtol=1e-10, atol=1e-10), **g)
    assert bool(jnp.all(jnp.isfinite(implicit["Ck"])))
    assert jnp.allclose(implicit["Ck"][-1], explicit["Ck"][-1], rtol=0, atol=1e-5)

if __name__ == "__main__":
    pytest.main()
