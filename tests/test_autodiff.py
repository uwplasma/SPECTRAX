"""Gradients through the Hermite-DG ``simulation``: reverse mode, forward mode, finite differences, and memory bounds."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffrax import ForwardMode, NoProgressMeter, RecursiveCheckpointAdjoint
from jax.scipy.special import factorial

from spectrax import construct_idx_array, legT, simulation

Nx = Ny = 4
Nh = 4  # Nn = Nm = Np; the collision matrix vanishes identically for fewer than four Hermite modes
Ns, N_DG, dims = 2, 2, 2
basis_idx = construct_idx_array(dims, N_DG)
mass = (1 + 2 * basis_idx[:, 0]) * (1 + 2 * basis_idx[:, 1])  # inverse Legendre mass of each basis function
theta0 = jnp.array([0.2, 0.05])  # (deltaB, nu): an initial-condition amplitude and a physical parameter


def orszag_tang(theta):
    """Tiny two-species Orszag-Tang setup projected with ``legT``, as in Examples/2D_Orszag_Tang.py."""
    deltaB, nu = theta
    L, Omega_ce, k = 50.0, 0.5, 2 * jnp.pi / 50.0
    alpha_s = jnp.array([0.25] * 3 + [0.05] * 3)
    U0 = deltaB * Omega_ce / 5.0
    a = alpha_s.reshape(Ns, 3, *[1] * 9)
    n, m, p = (jnp.arange(Nh).reshape([1] * i + [Nh] + [1] * (9 - i)) for i in (3, 2, 1))

    def C0(x, y, z):
        flow = [-U0 * jnp.sin(k * y), U0 * jnp.sin(k * x)]
        Jz = -deltaB * Omega_ce * (2 * k * jnp.cos(2 * k * x) + k * jnp.cos(k * y))  # electrons carry the current
        U = jnp.stack([jnp.stack(flow + [Jz]), jnp.stack(flow + [jnp.zeros_like(x)])])[:, :, None, None, None]
        return (jnp.sqrt(2.0 ** (n + m + p) / (factorial(n) * factorial(m) * factorial(p)))
                / (a[:, 0] ** (n + 1) * a[:, 1] ** (m + 1) * a[:, 2] ** (p + 1)) * U[:, 0] ** n * U[:, 1] ** m * U[:, 2] ** p)

    def F0(x, y, z):
        return jnp.stack([jnp.zeros_like(x)] * 3 + [-deltaB * jnp.sin(k * y), deltaB * jnp.sin(2 * k * x), jnp.ones_like(x)])

    return dict(Lx=L, Ly=L, Lz=1.0, ms=jnp.array([1.0, 25.0]), qs=jnp.array([-1.0, 1.0]), alpha_s=alpha_s, u_s=jnp.zeros(6),
                Omega_ce=Omega_ce, nu=nu, D=0.0, t_max=2.0, ode_tolerance=1e-8,
                Ck_0=legT(C0, basis_idx, N_DG, L, Nx, L, Ny), Fk_0=legT(F0, basis_idx, N_DG, L, Nx, L, Ny))


def energy(theta, **kwargs):
    """Final field energy plus the density-normalised Hermite free energy, which the collision operator damps."""
    kwargs.setdefault("progress_meter", NoProgressMeter())
    out = simulation(orszag_tang(theta), Nx=Nx, Ny=Ny, Nz=1, Nn=Nh, Nm=Nh, Np=Nh, Ns=Ns, N_DG=N_DG, dims=dims,
                     timesteps=3, dt=0.01, **kwargs)
    density = jnp.repeat(jnp.prod(out["alpha_s"].reshape(Ns, 3), axis=1), Nh**3)[:, None, None, None, None]
    return 0.5 * jnp.sum(out["Fk"][-1] ** 2 / mass) + jnp.sum((density * out["Ck"][-1]) ** 2 / mass)


def centered_difference(f, theta, eps=1e-5):
    return np.array([float(f(theta + eps * e) - f(theta - eps * e)) / (2 * eps) for e in jnp.eye(theta.size)])


def assert_matches_finite_differences(gradient, f, theta):
    """Finite differences carry cancellation error, so compare with a floor relative to the gradient norm."""
    np.testing.assert_allclose(gradient, centered_difference(f, theta), rtol=1e-5, atol=1e-6 * float(jnp.linalg.norm(gradient)))


def test_reverse_forward_and_finite_differences_agree():
    """Adaptive Dopri5 with the PID controller: reverse mode, forward mode and finite differences agree."""
    value, reverse = jax.jit(jax.value_and_grad(energy))(theta0)
    assert jnp.isrealobj(value) and jnp.isfinite(value) and jnp.all(jnp.abs(reverse) > 0)
    forward = jax.jacfwd(lambda th: energy(th, adjoint=ForwardMode()))(theta0)
    np.testing.assert_allclose(reverse, forward, rtol=1e-9)
    assert_matches_finite_differences(reverse, energy, theta0)


def test_gradient_is_independent_of_checkpoint_budget():
    """The checkpoint budget changes memory and recomputation only; the adaptive step sequence is replayed exactly."""
    gradients = [jax.grad(lambda th: energy(th, adjoint=RecursiveCheckpointAdjoint(checkpoints=k)))(theta0) for k in (2, 8, None)]
    for gradient in gradients[1:]:
        np.testing.assert_allclose(gradient, gradients[0], rtol=1e-10, atol=1e-14)


def test_reverse_pass_workspace_does_not_grow_with_the_step_budget():
    """At fixed checkpoints the compiled reverse pass does not grow with max_steps (the default grows as log2(max_steps))."""
    def workspace(max_steps):
        f = lambda th: energy(th, adjoint=RecursiveCheckpointAdjoint(checkpoints=4), max_steps=max_steps)
        analysis = jax.jit(jax.value_and_grad(f)).lower(theta0).compile().memory_analysis()
        if analysis is None:
            pytest.skip("memory analysis unavailable on this backend")
        return analysis.temp_size_in_bytes
    small, large = workspace(256), workspace(65536)
    assert large <= 1.05 * small, (small, large)


def test_upwind_hermite_matrices_match_host_eigendecomposition():
    """The upwind split A = A+ + A- uses a symmetric eigendecomposition; compare with an independent host LAPACK eig."""
    from spectrax._initialization import compute_A_pm_matrices
    alpha, u = np.array([[0.25, 0.3, 0.35], [0.05, 0.06, 0.07]]), np.array([[0.1, -0.2, 0.0], [0.0, 0.03, -0.01]])
    for N in (1, 2, 3, 5, 8):
        Ax_p, Ax_m, Ay_p, Ay_m, Az_p, Az_m = (np.asarray(M) for M in compute_A_pm_matrices(N, N, N, jnp.asarray(alpha), jnp.asarray(u)))
        blocks = [(Ax_p[:, 0, 0, :, :, 0, 0, 0, 0, 0], Ax_m[:, 0, 0, :, :, 0, 0, 0, 0, 0]),
                  (Ay_p[:, 0, :, :, 0, 0, 0, 0, 0, 0], Ay_m[:, 0, :, :, 0, 0, 0, 0, 0, 0]),
                  (Az_p[:, :, :, 0, 0, 0, 0, 0, 0, 0], Az_m[:, :, :, 0, 0, 0, 0, 0, 0, 0])]
        i, j = np.indices((N, N))
        off_diagonal = np.where(np.abs(i - j) == 1, np.sqrt(np.maximum(i, j) / 2), 0.0)
        for s in range(Ns):
            for d, (plus, minus) in enumerate(blocks):
                lam, vec = np.linalg.eig(alpha[s, d] * off_diagonal + u[s, d] * np.eye(N))
                lam, vec = lam.real, vec.real
                np.testing.assert_allclose(plus[s], vec @ np.diag(np.maximum(lam, 0)) @ vec.T, atol=1e-13)
                np.testing.assert_allclose(minus[s], vec @ np.diag(np.minimum(lam, 0)) @ vec.T, atol=1e-13)


def test_gradients_with_respect_to_thermal_speeds_and_drifts():
    """alpha_s and u_s reach the upwind Hermite matrices through a symmetric eigendecomposition, which is differentiable."""
    base = orszag_tang(theta0)

    def field_energy(x, adjoint=RecursiveCheckpointAdjoint()):   # x = (scale of every thermal speed, common x-drift)
        p = dict(base, alpha_s=base["alpha_s"] * x[0], u_s=base["u_s"] + x[1] * jnp.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
        out = simulation(p, Nx=Nx, Ny=Ny, Nz=1, Nn=Nh, Nm=Nh, Np=Nh, Ns=Ns, N_DG=N_DG, dims=dims, timesteps=3, dt=0.01,
                         progress_meter=NoProgressMeter(), adjoint=adjoint)
        return 0.5 * jnp.sum(out["Fk"][-1] ** 2 / mass)

    x0 = jnp.array([1.0, 0.02])   # nonzero drift: at zero drift the field energy is stationary in it by symmetry
    reverse = jax.jit(jax.grad(field_energy))(x0)
    assert jnp.all(jnp.isfinite(reverse)) and jnp.all(jnp.abs(reverse) > 1e-8)
    forward = jax.jacfwd(lambda x: field_energy(x, ForwardMode()))(x0)
    np.testing.assert_allclose(reverse, forward, rtol=1e-9, atol=1e-12 * float(jnp.linalg.norm(reverse)))
    assert_matches_finite_differences(reverse, field_energy, x0)

