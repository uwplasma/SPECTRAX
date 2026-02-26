"""Custom Diffrax solver: implicit midpoint with a Newton–GMRES nonlinear solve."""

import diffrax
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.sparse.linalg import gmres


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
        y1_init = y0 + δt * f0

        # Define F(y1) = y1 - y0 - δt * f(t1, (y0 + y1)/2)
        def F_fn(y1):
            y_mid = 0.5 * (y0 + y1)
            return y1 - y0 - δt * terms.vf(t1, y_mid, args)

        y1 = _newton_gmres(F_fn, y0, y1_init, self.rtol, self.atol, self.max_iters)

        y_error = y1 - y1_init
        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, None, diffrax.RESULTS.successful


def _newton_gmres(F_fn, y0, y_init, rtol, atol, max_iters):
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

            scale = atol + rtol * jnp.maximum(jnp.abs(y0), jnp.abs(y1))
            norm = jnp.linalg.norm(res / scale)

            # Adaptive inner tolerance (Eisenstat–Walker).
            inner_tol = jnp.minimum(0.1, norm * 0.5)

            delta, _ = gmres(
                jvp,
                -res,
                tol=inner_tol,
                atol=atol,
                maxiter=min(20, max_iters // 2),
            )

            y1_next = jnp.where(norm < 1.0, y1, y1 + delta)
            return y1_next, norm >= 1.0, i + 1

        init_state = (y_init, True, 0)
        y1_final, _, _ = lax.while_loop(cond_fn, body_fn, init_state)
        return y1_final

    return loop_fn(y_init)
