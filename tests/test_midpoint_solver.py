"""Failure handling of the implicit midpoint Newton-GMRES step: deterministic edge cases."""
import diffrax
import jax
import jax.numpy as jnp
import optimistix as optx
import pytest
from spectrax.midpoint_solver import ImplicitMidpoint

SUCCESS = diffrax.RESULTS.successful
MAX_ITERS = diffrax.RESULTS.promote(optx.RESULTS.nonlinear_max_steps_reached)
DIVERGED = diffrax.RESULTS.promote(optx.RESULTS.nonlinear_divergence)


def step(vf, y0, dt=0.1, **kwargs):
    y1, y_error, _, _, result = ImplicitMidpoint(**kwargs).step(
        diffrax.ODETerm(vf), 0.0, dt, y0, None, None, False)
    return y1, y_error, result


@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf, -jnp.inf])
def test_nonfinite_rhs_is_a_failure_not_success(bad):
    """A NaN residual compares False with both < 1 and >= 1; it must never read as converged."""
    _, y_error, result = step(lambda t, y, args: y * bad, jnp.ones(3))
    assert result == DIVERGED
    assert bool(jnp.all(jnp.isinf(y_error)))


def test_nonfinite_only_after_an_update_is_a_failure():
    """The first residual is finite, the Newton update lands where the RHS is NaN."""
    vf = lambda t, y, args: jnp.where(jnp.abs(y) > 0.5, jnp.nan, -40.0 * y)
    _, _, result = step(vf, jnp.array([0.4]), dt=1.0)
    assert result == DIVERGED


def test_failed_inner_solve_is_a_failure():
    """A NaN Jacobian breaks GMRES; whatever update it returns, the outer residual check refuses it."""
    @jax.custom_jvp
    def rhs(y):
        return -y ** 3

    @rhs.defjvp
    def _(primals, tangents):
        return rhs(primals[0]), jnp.nan * tangents[0]

    _, y_error, result = step(lambda t, y, args: rhs(y), jnp.ones(2), dt=0.5, max_iters=5)
    assert result in (MAX_ITERS, DIVERGED) and bool(jnp.all(jnp.isinf(y_error)))


def test_max_iterations_reached_is_reported():
    """y' = -y^3 with a large step needs several Newton updates; one is not enough."""
    vf = lambda t, y, args: -(y ** 3)
    _, y_error, result = step(vf, jnp.array([2.0]), dt=1.0, max_iters=1, rtol=1e-10, atol=1e-12)
    assert result == MAX_ITERS and bool(jnp.all(jnp.isinf(y_error)))
    y1, _, result = step(vf, jnp.array([2.0]), dt=1.0, max_iters=50, rtol=1e-10, atol=1e-12)
    assert result == SUCCESS
    y_mid = 0.5 * (2.0 + y1[0])
    assert abs(float(y1[0] - 2.0 + y_mid ** 3)) < 1e-9


def test_last_allowed_update_is_accepted_when_it_converges():
    """For linear y' = -y one Newton update is exact: max_iters=1 must report success."""
    y1, _, result = step(lambda t, y, args: -y, jnp.ones(4), dt=0.1, max_iters=1, rtol=1e-12, atol=1e-14)
    assert result == SUCCESS
    assert jnp.allclose(y1, 0.95 / 1.05, rtol=1e-12, atol=0)


def test_zero_iterations_accepts_only_an_already_converged_guess():
    """max_iters=0 checks the explicit predictor: exact for y' = 1, not for y' = -y."""
    y1, _, result = step(lambda t, y, args: jnp.ones_like(y), jnp.zeros(2), dt=0.1, max_iters=0)
    assert result == SUCCESS and jnp.allclose(y1, 0.1)
    _, _, result = step(lambda t, y, args: -y, jnp.ones(2), dt=0.1, max_iters=0)
    assert result == MAX_ITERS


def test_failed_step_ends_a_constant_step_solve_and_is_rejected_by_an_adaptive_one():
    """Constant steps: the failure is the solve's result. Adaptive: the step is retried smaller."""
    vf = lambda t, y, args: jnp.where(t > 0.3, jnp.nan, -y)
    sol = diffrax.diffeqsolve(diffrax.ODETerm(vf), ImplicitMidpoint(), 0.0, 1.0, 0.1, jnp.ones(2),
                              stepsize_controller=diffrax.ConstantStepSize(), throw=False)
    assert sol.result == DIVERGED
    # Stiff-ish y' = -y^3 from y=4: large first steps fail Newton and are rejected, not returned as data.
    sol = diffrax.diffeqsolve(diffrax.ODETerm(lambda t, y, args: -(y ** 3)),
                              ImplicitMidpoint(max_iters=3), 0.0, 1.0, 1.0, jnp.array([4.0]),
                              stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-3),
                              throw=False)
    assert sol.result == SUCCESS and int(sol.stats["num_rejected_steps"]) > 0
    assert abs(float(sol.ys[-1, 0]) - 4.0 / (1 + 2 * 16.0) ** 0.5) < 1e-3


def test_oscillator_energy_to_solve_tolerance_and_midpoint_phase():
    """Converged midpoint conserves the quadratic energy to solve tolerance and rotates at
    (2/dt) arctan(w dt/2), not w: the phase lag accumulates as ~ w^3 dt^2 t / 12."""
    w, dt, n = 2.0, 0.2, 250
    vf = lambda t, y, args: (y[1], -w ** 2 * y[0])
    sol = diffrax.diffeqsolve(diffrax.ODETerm(vf), ImplicitMidpoint(rtol=1e-12, atol=1e-14),
                              0.0, n * dt, dt, (jnp.array(1.0), jnp.array(0.0)),
                              stepsize_controller=diffrax.ConstantStepSize(),
                              saveat=diffrax.SaveAt(ts=dt * jnp.arange(n + 1)), throw=False)
    assert sol.result == SUCCESS
    x, v = sol.ys
    energy = x ** 2 + (v / w) ** 2
    assert float(jnp.max(jnp.abs(energy - 1.0))) < 1e-11
    w_mid = 2.0 / dt * jnp.arctan(w * dt / 2)
    phase = jnp.unwrap(jnp.arctan2(-v / w, x))
    t = sol.ts
    assert float(jnp.max(jnp.abs(phase - w_mid * t))) < 1e-9
    lag = float(w * t[-1] - phase[-1])
    assert abs(lag - float((w - w_mid) * t[-1])) < 1e-9
    # Leading-order estimate; its relative correction is 3 (w dt)^2 / 20 = 0.024 here.
    assert abs(lag / (w ** 3 * dt ** 2 * t[-1] / 12) - 1) < 0.03
