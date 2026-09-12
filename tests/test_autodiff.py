"""Independent derivative checks on a real-linear, complex plasma trajectory."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectrax import initialize_simulation_parameters, simulation_final
from spectrax._simulation import _solver_args, ode_system

spec = importlib.util.spec_from_file_location(
    "phase_control", Path(__file__).parents[1] / "Examples" / "2D_phase_control.py")
phase_control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase_control)


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


def test_phase_constraints_and_observables():
    grid, hermite = 12, 3
    phases = jnp.array([0.2, 0.7, -0.3, 0.6])
    p = phase_control.setup(phases, grid, hermite, 0.2)
    q = phase_control.setup(phases + jnp.array([0.8, -0.4, 0.1, 0.6]), grid, hermite, 0.2)
    initial = lambda params: (params["Ck_0"], params["Fk_0"])
    np.testing.assert_allclose(jnp.abs(p["Fk_0"]) ** 2, jnp.abs(q["Fk_0"]) ** 2, atol=1e-16)
    np.testing.assert_allclose(phase_control.quantities(initial(p), p, grid, hermite)[:4],
                               phase_control.quantities(initial(q), q, grid, hermite)[:4], atol=1e-14)
    # Ampere-consistent initial current, div B = 0, and charge neutrality.
    kx = jnp.fft.rfftfreq(grid) * grid * 2 * jnp.pi / p["Lx"]
    ky = jnp.fft.fftfreq(grid) * grid * 2 * jnp.pi / p["Ly"]
    F = p["Fk_0"][..., 0]
    np.testing.assert_allclose(kx[None, :] * F[3] + ky[:, None] * F[4], 0, atol=1e-16)
    current = phase_control.plasma_current(p["qs"], p["alpha_s"], p["u_s"], p["Ck_0"],
                                           hermite, hermite, hermite, 2)
    np.testing.assert_allclose(current[2, ..., 0],
                               0.5j * (kx[None, :] * F[4] - ky[:, None] * F[3]), atol=1e-16)

    def observable(phases):
        params = phase_control.setup(phases, grid, hermite, 0.2)
        final = simulation_final(params, Nx=grid, Ny=grid, Nn=hermite, Nm=hermite,
                                  Np=hermite, steps=4)
        return phase_control.quantities(final, params, grid, hermite)

    # All requested classes of outputs: electric/magnetic/particle energies and current shape.
    reverse = jax.jit(jax.jacrev(observable))(phases)
    forward = jax.jit(jax.jacfwd(observable))(phases)
    assert np.isfinite(reverse).all()
    np.testing.assert_allclose(reverse, forward, atol=1e-12, rtol=1e-8)


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


def test_window_objective_gradient():
    objective, _ = phase_control.problem(grid=12, hermite=3, final_time=0.4,
                                         steps=8, time_window=(0.1, 0.4))
    phases = jnp.array([0.2, 0.7, -0.3, 0.6])
    direction = jnp.array([0.1, -0.3, 0.7, 0.2])
    gradient = jax.jit(jax.grad(objective))(phases)
    epsilon = 1e-4
    scalar = jax.jit(objective)
    finite_difference = (scalar(phases + epsilon * direction)
                         - scalar(phases - epsilon * direction)) / (2 * epsilon)
    np.testing.assert_allclose(gradient @ direction, finite_difference, rtol=3e-5, atol=1e-10)
    for mode in (dict(checkpointing=False), dict(checkpoints=3)):
        reference, _ = phase_control.problem(grid=12, hermite=3, final_time=0.4,
                                             steps=8, time_window=(0.1, 0.4), **mode)
        np.testing.assert_allclose(jax.jit(jax.grad(reference))(phases), gradient,
                                   rtol=1e-9, atol=1e-12)
