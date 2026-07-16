"""Custom Diffrax solver: implicit midpoint with a Newton–GMRES nonlinear solve."""

import diffrax
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import lax
from solvax import gmres


class ImplicitMidpoint(diffrax.AbstractSolver):
    """Implicit midpoint ODE solver using a JAX-compiled Newton–GMRES iteration.

    This solver implements the implicit midpoint rule:

        y_{n+1} = y_n + Δt * f(t_{n+1}, (y_n + y_{n+1}) / 2)

    The nonlinear equation for ``y_{n+1}`` is solved with Newton iterations,
    where each linearized step is solved via GMRES using JAX's linearization.
    """

    rtol: float = 1e-6
    atol: float = 1e-8
    max_iters: int = 300
    linear_restart: int = 20
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
        f0 = terms.vf(t0, y0, args)
        y1_init = jax.tree.map(lambda y, f: y + δt * f, y0, f0)

        # Define F(y1) = y1 - y0 - δt * f(t1, (y0 + y1)/2)
        def F_fn(y1):
            y_mid = jax.tree.map(lambda a, b: 0.5 * (a + b), y0, y1)
            f_mid = terms.vf(t1, y_mid, args)
            return jax.tree.map(lambda a, b, f: a - b - δt * f, y1, y0, f_mid)

        y1, converged = _newton_gmres(
            F_fn, y0, y1_init, self.rtol, self.atol, self.max_iters,
            self.linear_restart, self.linear_max_restarts,
        )

        y_error = jax.tree.map(lambda a, b: a - b, y1, y1_init)
        dense_info = dict(y0=y0, y1=y1)
        result = diffrax.RESULTS.where(
            converged,
            diffrax.RESULTS.successful,
            diffrax.RESULTS.promote(optx.RESULTS.nonlinear_max_steps_reached),
        )
        return y1, y_error, dense_info, None, result


def _newton_gmres(F_fn, y0, y_init, rtol, atol, max_iters,
                  linear_restart, linear_max_restarts):
    """Solve ``F(y)=0`` using Newton iterations with GMRES linear solves.

    Notes
    -----
    - The Jacobian-vector product is obtained via ``jax.linearize``.
    - The GMRES tolerance is chosen adaptively (Eisenstat–Walker style) based on
      the current scaled residual norm.
    """

    @jax.jit
    def loop_fn(y_init):
        def cond_fn(state):
            _, not_converged, i = state
            return (i < max_iters) & not_converged

        def body_fn(state):
            y1, _, i = state

            res, jvp = jax.linearize(F_fn, y1)

            scaled_res = jax.tree.map(
                lambda r, a, b: r / (atol + rtol * jnp.maximum(jnp.abs(a), jnp.abs(b))),
                res, y0, y1,
            )
            norm = jnp.sqrt(sum(jnp.vdot(x, x).real for x in jax.tree.leaves(scaled_res)))

            # Adaptive inner tolerance (Eisenstat–Walker).
            inner_tol = jnp.minimum(0.1, norm * 0.5)

            def update(_):
                linear_solution = gmres(
                    jvp,
                    jax.tree.map(jnp.negative, res),
                    rtol=inner_tol,
                    atol=atol,
                    restart=linear_restart,
                    max_restarts=linear_max_restarts,
                )
                return jax.tree.map(lambda y, d: y + d, y1, linear_solution.x)

            y1_next = lax.cond(
                norm < 1.0, lambda _: y1, update, operand=None
            )
            return y1_next, norm >= 1.0, i + 1

        init_state = (y_init, True, 0)
        y1_final, not_converged, _ = lax.while_loop(cond_fn, body_fn, init_state)
        return y1_final, ~not_converged

    return loop_fn(y_init)
