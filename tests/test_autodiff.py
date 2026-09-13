"""Independent derivative checks on a real-linear, complex plasma trajectory."""


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectrax import initialize_simulation_parameters, simulation_final
from spectrax._simulation import _solver_args, ode_system



def test_checkpointed_complex_gradients():
    # Random Fourier coefficients exercise both real and imaginary dependencies.
    rng = np.random.default_rng(4)
    C = rng.normal(size=(8, 1, 4, 1)) + 1j * rng.normal(size=(8, 1, 4, 1))
    F = rng.normal(size=(6, 1, 4, 1)) + 1j * rng.normal(size=(6, 1, 4, 1))

    def objective(theta, checkpointing=True, checkpoint_size=None, checkpoints=None):
        parameters = dict(t_max=theta[0], nu=theta[1],
                          Ck_0=jnp.asarray(C) * theta[2], Fk_0=jnp.asarray(F) * 0.01,
                          alpha_s=jnp.array([0.7] * 6), u_s=jnp.zeros(6))
        C1, F1 = simulation_final(parameters, Nx=6, Nn=4, steps=7,
                                  checkpointing=checkpointing, checkpoint_size=checkpoint_size, checkpoints=checkpoints)
        return jnp.sum(jnp.abs(F1) ** 2) + 0.01 * jnp.sum(jnp.abs(C1) ** 2)

    theta = jnp.array([0.2, 0.3, 0.02])
    value, reference = jax.jit(jax.value_and_grad(lambda p: objective(p, False)))(theta)
    for segment in (None, 1, 3, 7, 20):
        got_value, gradient = jax.jit(jax.value_and_grad(
            lambda p: objective(p, checkpoint_size=segment)))(theta)
        np.testing.assert_allclose(got_value, value, rtol=1e-13)
        np.testing.assert_allclose(gradient, reference, rtol=1e-11, atol=1e-13)
    for budget in (1, 4):
        got_value, gradient = jax.jit(jax.value_and_grad(
            lambda p: objective(p, checkpoints=budget)))(theta)
        np.testing.assert_allclose(got_value, value, rtol=1e-13)
        np.testing.assert_allclose(gradient, reference, rtol=1e-10, atol=1e-13)
    direction = jnp.array([0.3, -0.2, 0.7])
    _, tangent = jax.jvp(objective, (theta,), (direction,))
    np.testing.assert_allclose(tangent, reference @ direction, rtol=1e-11)
    epsilon = 1e-5
    finite_difference = (objective(theta + epsilon * direction)
                         - objective(theta - epsilon * direction)) / (2 * epsilon)
    np.testing.assert_allclose(tangent, finite_difference, rtol=1e-7, atol=1e-11)


def test_rk4_against_independent_dopri8():
    import diffrax
    p = initialize_simulation_parameters(dict(t_max=0.15), Nx=6, Nn=4)
    y0 = jnp.concatenate((p["Ck_0"].ravel(), p["Fk_0"].ravel()))
    args = _solver_args(p, 6, 1, 1, 4, 1, 1, 2)
    rhs = lambda t, y, a: ode_system(6, 1, 1, 4, 1, 1, 2, t, y, a)
    reference = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs), diffrax.Dopri8(), t0=0.0, t1=0.15, dt0=0.01,
        y0=y0, args=args, stepsize_controller=diffrax.PIDController(rtol=1e-12, atol=1e-12),
        saveat=diffrax.SaveAt(t1=True),
    ).ys[0]
    errors = []
    for steps in (2, 4, 8):
        C, F = simulation_final(p, Nx=6, Nn=4, steps=steps)
        errors.append(float(jnp.linalg.norm(jnp.concatenate((C.ravel(), F.ravel())) - reference)))
    assert errors[0] / errors[1] > 12
    assert errors[1] / errors[2] > 12




@pytest.mark.parametrize("steps", [0, -1])
def test_invalid_steps(steps):
    with pytest.raises(ValueError, match="positive integer"):
        simulation_final(steps=steps)


def test_streaming_integral_and_time_derivative():
    # Polynomial integrands have analytic integrals, including endpoint derivatives.
    def solve(time, checkpoints=None):
        return simulation_final(dict(t_max=time), Nx=6, Nn=4, steps=7,
                                 checkpoints=checkpoints,
                                 integrand=lambda t, C, F: jnp.array([t, t ** 2]))[2]
    time = jnp.array(0.2)
    expected = jnp.array([time ** 2 / 2, time ** 3 / 3])
    for checkpoints in (None, 3):
        np.testing.assert_allclose(jax.jit(lambda t: solve(t, checkpoints))(time), expected, atol=1e-14)
        np.testing.assert_allclose(jax.jacrev(lambda t: solve(t, checkpoints))(time),
                                   jnp.array([time, time ** 2]), atol=1e-13)
    np.testing.assert_allclose(jax.jacfwd(solve)(time), jnp.array([time, time ** 2]), atol=1e-13)
