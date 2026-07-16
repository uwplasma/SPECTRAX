"""Custom Diffrax solver: implicit midpoint with a Newton–GMRES nonlinear solve."""

from collections.abc import Callable
from typing import NamedTuple

import diffrax
import jax
import jax.numpy as jnp
import optimistix as optx
from solvax import lu_factor_banded, lu_solve_banded, newton_krylov


def collision_diffusion_preconditioner(args, dt):
    """Build ``(I + dt / 2 * (nu C + D k²))⁻¹`` from simulation arguments."""
    nu = args[-20]
    diffusion = args[-19]
    k2 = args[-9]
    collision = args[-7]
    rate = nu * collision[None, :, :, :, None, None, None] + diffusion * k2
    diagonal = 1 + 0.5 * dt * rate

    def apply(residual):
        coefficients, fields = residual
        return coefficients / diagonal, fields

    return apply


def collision_diffusion_x_streaming_preconditioner(args, dt):
    """Invert collision, diffusion, and x-streaming Hermite line blocks."""
    nu, diffusion = args[-20:-18]
    alpha = args[-17].reshape(-1, 3)[:, 0]
    drift = args[-16].reshape(-1, 3)[:, 0]
    length, kx, k2 = args[-15], args[-12], args[-9]
    collision, sqrt_plus, sqrt_minus = args[-7:-4]
    phase = (0.5j * dt / length * kx)[None, None, None]
    species = (slice(None),) + (None,) * 6
    diagonal = (
        1
        + 0.5 * dt * (
            nu * collision[None, :, :, :, None, None, None]
            + diffusion * k2[None, None, None, None]
        )
        + drift[species] * phase
    )
    coupling = alpha[species] * phase / jnp.sqrt(2)

    def lines(value):
        return jnp.moveaxis(jnp.broadcast_to(value, diagonal.shape), 3, -1)

    bands = jnp.stack(
        (lines(coupling * sqrt_minus), lines(diagonal),
         lines(coupling * sqrt_plus)), axis=-2,
    )
    flat_bands = bands.reshape(-1, 3, bands.shape[-1])
    factors = jax.vmap(lambda band: lu_factor_banded(band, 1, 1))(flat_bands)

    def apply(residual):
        coefficients, fields = residual
        rhs = jnp.moveaxis(coefficients, 3, -1).reshape(-1, coefficients.shape[3])
        solution = jax.vmap(lu_solve_banded)(factors, rhs)
        solution = solution.reshape(bands.shape[:-2] + (-1,))
        return jnp.moveaxis(solution, -1, 3), fields

    return apply


class MidpointSolverState(NamedTuple):
    """Newton--GMRES iteration totals and per-step maxima."""

    newton_iterations: jax.Array
    linear_iterations: jax.Array
    max_newton_iterations: jax.Array
    max_linear_iterations: jax.Array


class ImplicitMidpoint(diffrax.AbstractSolver):
    """Implicit midpoint ODE solver using a JAX-compiled Newton–GMRES iteration.

    This solver implements the implicit midpoint rule:

        y_{n+1} = y_n + Δt * f(t_n + Δt / 2, (y_n + y_{n+1}) / 2)

    The nonlinear equation for ``y_{n+1}`` is solved with Newton iterations,
    where each linearized step is solved via GMRES using JAX's linearization.
    ``rtol`` and ``atol`` control the nonlinear residual, while
    ``linear_rtol`` and ``linear_atol`` control each GMRES solve.
    Newton iteration stops when ``||F|| <= max(atol, rtol * ||F_initial||)``.
    Accepted-step diagnostics are accumulated in :class:`MidpointSolverState`.
    ``inner_product`` controls the reductions used by Newton and GMRES.
    An optional ``preconditioner(args, dt)`` factory supplies the inverse action
    used by GMRES.
    """

    rtol: float = 1e-6
    atol: float = 1e-8
    max_iters: int = 300
    linear_restart: int = 20
    linear_rtol: float = 1e-4
    linear_atol: float = 0.0
    linear_max_restarts: int = 20
    inner_product: Callable | None = None
    preconditioner: Callable | None = None

    term_structure = diffrax.ODETerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def order(self, terms):
        return 2

    def init(self, terms, t0, t1, y0, args):
        del terms, t0, t1, y0, args
        zero = jnp.int32(0)
        return MidpointSolverState(zero, zero, zero, zero)

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del made_jump
        if solver_state is None:
            solver_state = self.init(terms, t0, t1, y0, args)

        δt = t1 - t0
        t_mid = t0 + 0.5 * δt
        f0 = terms.vf(t0, y0, args)
        y1_init = jax.tree.map(lambda y, f: y + δt * f, y0, f0)

        # Define F(y1) = y1 - y0 - δt * f(t_mid, (y0 + y1)/2)
        def F_fn(y1):
            y_mid = jax.tree.map(lambda a, b: 0.5 * (a + b), y0, y1)
            f_mid = terms.vf(t_mid, y_mid, args)
            return jax.tree.map(lambda a, b, f: a - b - δt * f, y1, y0, f_mid)

        precond = None if self.preconditioner is None else self.preconditioner(args, δt)
        solution = newton_krylov(
            F_fn,
            y1_init,
            precond=precond,
            inner_product=self.inner_product,
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_iters,
            linear_restart=self.linear_restart,
            linear_rtol=self.linear_rtol,
            linear_atol=self.linear_atol,
            linear_max_restarts=self.linear_max_restarts,
        )
        y1 = solution.x

        y_error = jax.tree.map(lambda a, b: a - b, y1, y1_init)
        dense_info = dict(y0=y0, y1=y1)
        result = diffrax.RESULTS.where(
            solution.converged,
            diffrax.RESULTS.successful,
            diffrax.RESULTS.promote(optx.RESULTS.nonlinear_max_steps_reached),
        )
        next_solver_state = MidpointSolverState(
            solver_state.newton_iterations + solution.newton_iterations,
            solver_state.linear_iterations + solution.linear_iterations,
            jnp.maximum(solver_state.max_newton_iterations, solution.newton_iterations),
            jnp.maximum(solver_state.max_linear_iterations, solution.linear_iterations),
        )
        return y1, y_error, dense_info, next_solver_state, result
