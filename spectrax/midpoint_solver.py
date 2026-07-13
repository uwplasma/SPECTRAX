"""Custom Diffrax solver: implicit midpoint with a Newton–GMRES nonlinear solve."""

import diffrax
import jax
import optimistix as optx
from solvax import newton_krylov


class ImplicitMidpoint(diffrax.AbstractSolver):
    """Implicit midpoint ODE solver using a JAX-compiled Newton–GMRES iteration.

    This solver implements the implicit midpoint rule:

        y_{n+1} = y_n + Δt * f(t_n + Δt / 2, (y_n + y_{n+1}) / 2)

    The nonlinear equation for ``y_{n+1}`` is solved with Newton iterations,
    where each linearized step is solved via GMRES using JAX's linearization.
    ``rtol`` and ``atol`` control the nonlinear residual, while
    ``linear_rtol`` and ``linear_atol`` control each GMRES solve.
    Newton iteration stops when ``||F|| <= max(atol, rtol * ||F_initial||)``.
    """

    rtol: float = 1e-6
    atol: float = 1e-8
    max_iters: int = 300
    linear_restart: int = 20
    linear_rtol: float = 1e-4
    linear_atol: float = 0.0
    linear_max_restarts: int = 20

    term_structure = diffrax.ODETerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def order(self, terms):
        return 2

    def init(self, terms, t0, t1, y0, args):
        return None

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump

        δt = t1 - t0
        t_mid = t0 + 0.5 * δt
        f0 = terms.vf(t0, y0, args)
        y1_init = jax.tree.map(lambda y, f: y + δt * f, y0, f0)

        # Define F(y1) = y1 - y0 - δt * f(t_mid, (y0 + y1)/2)
        def F_fn(y1):
            y_mid = jax.tree.map(lambda a, b: 0.5 * (a + b), y0, y1)
            f_mid = terms.vf(t_mid, y_mid, args)
            return jax.tree.map(lambda a, b, f: a - b - δt * f, y1, y0, f_mid)

        solution = newton_krylov(
            F_fn,
            y1_init,
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
        return y1, y_error, dense_info, None, result
