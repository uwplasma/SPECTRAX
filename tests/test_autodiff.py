"""Gradients through ``simulation``: reverse mode, forward mode, finite differences, and memory bounds."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffrax import Dopri8, Tsit5, ForwardMode, NoProgressMeter, RecursiveCheckpointAdjoint

from spectrax import simulation, plasma_current, compute_C_nmp

Nx = Ny = 8
Nn = Nm = Np = 4  # four Hermite modes so the collision operator is nonzero
Ns = 2


def orszag_tang(theta, t_max):
    """Tiny two-species Orszag–Tang setup; ``theta = (deltaB, nu)`` are the differentiated parameters."""
    deltaB, nu = theta
    Lx = Ly = 50.0
    x = jnp.arange(Nx) * Lx / Nx
    X, Y = jnp.meshgrid(x, x, indexing="xy")
    k = 2 * jnp.pi / Lx
    B = jnp.stack([-deltaB * jnp.sin(k * Y), deltaB * jnp.sin(2 * k * X), jnp.ones_like(X)])
    Fk_0 = jnp.fft.rfftn(jnp.concatenate([jnp.zeros_like(B), B])[..., None], axes=(-1, -3, -2), norm="forward")
    U0 = 0.02 * deltaB / 0.2
    flow = jnp.stack([-U0 * jnp.sin(k * Y), U0 * jnp.sin(k * X), jnp.zeros_like(X)])
    Jz = deltaB * k * (2 * jnp.cos(2 * k * X) + jnp.cos(k * Y))          # electrons carry the current
    alpha_s = jnp.array([0.25] * 3 + [0.05] * 3)
    Us = jnp.stack([flow.at[2].set(-0.5 * Jz), flow])[..., None]
    Ck_0 = compute_C_nmp(Us, alpha_s, jnp.zeros(6), Nn, Nm, Np, Ns).reshape(Ns * Np * Nm * Nn, Ny, Nx // 2 + 1, 1)
    return dict(Lx=Lx, Ly=Ly, Lz=1.0, mi_me=25.0, qs=jnp.array([-1.0, 1.0]), Omega_cs=jnp.array([0.5, 0.02]),
                alpha_s=alpha_s, u_s=jnp.zeros(6), nu=nu, D=0.0, t_max=t_max, ode_tolerance=1e-8,
                Ck_0=Ck_0, Fk_0=Fk_0)


def run(theta, t_max=2.0, **kwargs):
    kwargs.setdefault("progress_meter", NoProgressMeter())
    return simulation(orszag_tang(theta, t_max), Nx=Nx, Ny=Ny, Nz=1, Nn=Nn, Nm=Nm, Np=Np, Ns=Ns,
                      timesteps=3, **kwargs)


def energy(theta, **kwargs):
    """Field + kinetic energy plus the Hermite free energy, which the collision operator damps directly."""
    output = run(theta, **kwargs)
    return output["EM_energy"][-1] + output["kinetic_energy"][-1] + jnp.sum(jnp.abs(output["Ck"][-1]) ** 2)


def centered_difference(f, theta, eps=1e-5):
    return np.array([float(f(theta + eps * e) - f(theta - eps * e)) / (2 * eps) for e in jnp.eye(theta.size)])


def assert_matches_finite_differences(gradient, f, theta):
    """Finite differences carry cancellation error, so compare with a floor relative to the gradient norm."""
    np.testing.assert_allclose(gradient, centered_difference(f, theta), rtol=1e-5, atol=1e-6 * float(jnp.linalg.norm(gradient)))


theta0 = jnp.array([0.2, 1.0])


def test_reverse_forward_and_finite_differences_agree_adaptive():
    """Adaptive Dopri8 (the default integrator): reverse mode, forward mode and finite differences agree."""
    adaptive = dict(dt=0.01, solver=Dopri8())
    value, reverse = jax.jit(jax.value_and_grad(lambda th: energy(th, **adaptive)))(theta0)
    assert jnp.isrealobj(value) and jnp.isfinite(value)
    forward = jax.jacfwd(lambda th: energy(th, adjoint=ForwardMode(), **adaptive))(theta0)
    assert jnp.all(jnp.abs(reverse) > 0)
    np.testing.assert_allclose(reverse, forward, rtol=1e-9)
    assert_matches_finite_differences(reverse, lambda th: energy(th, **adaptive), theta0)


def test_fixed_step_gradient_is_independent_of_checkpoint_budget():
    """The checkpoint budget changes memory and recomputation only, never the gradient."""
    steps = 20
    fixed = dict(dt=2.0 / steps, solver=Tsit5(), adaptive_time_step=False, max_steps=steps + 1)
    gradients = [jax.grad(lambda th: energy(th, adjoint=RecursiveCheckpointAdjoint(checkpoints=k), **fixed))(theta0)
                 for k in (1, 4, steps)]
    for gradient in gradients[1:]:
        np.testing.assert_allclose(gradient, gradients[0], rtol=1e-10, atol=1e-14)
    assert_matches_finite_differences(gradients[0], lambda th: energy(th, **fixed), theta0)


def test_reverse_pass_workspace_is_bounded_in_the_number_of_steps():
    """With a fixed checkpoint budget the compiled reverse-pass workspace does not grow with the step count."""
    def workspace(steps):
        fixed = dict(dt=2.0 / steps, solver=Tsit5(), adaptive_time_step=False, max_steps=steps + 1,
                     adjoint=RecursiveCheckpointAdjoint(checkpoints=4))
        compiled = jax.jit(jax.value_and_grad(lambda th: energy(th, **fixed))).lower(theta0).compile()
        analysis = compiled.memory_analysis()
        if analysis is None:
            pytest.skip("memory analysis unavailable on this backend")
        return analysis.temp_size_in_bytes
    small, large = workspace(10), workspace(80)
    assert large <= 1.05 * small, (small, large)


def test_current_density_functional_gradient():
    """A smooth functional of the out-of-plane current density is differentiable through ``plasma_current``."""
    fixed = dict(dt=0.1, solver=Tsit5(), adaptive_time_step=False, max_steps=21)

    def peak_current(theta):
        output = run(theta, **fixed)
        Jk = plasma_current(output["qs"], output["alpha_s"], output["u_s"], output["Ck"][-1], Nn, Nm, Np, Ns)
        Jz = jnp.fft.irfftn(Jk[2], s=(1, Ny, Nx), axes=(-1, -3, -2), norm="forward")
        return jnp.mean(Jz**8) ** (1 / 8)

    gradient = jax.jit(jax.grad(peak_current))(theta0)
    assert jnp.all(jnp.isfinite(gradient)) and abs(gradient[0]) > 0
    assert_matches_finite_differences(gradient, peak_current, theta0)
