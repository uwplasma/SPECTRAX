"""Custom Diffrax solver: implicit midpoint with a Newton–GMRES nonlinear solve."""

import diffrax
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import lax
from jax.scipy.sparse.linalg import gmres


class ImplicitMidpoint(diffrax.AbstractSolver):
    """Implicit midpoint ODE solver using a JAX-compiled Newton–GMRES iteration.

    This solver implements the implicit midpoint rule:

        y_{n+1} = y_n + Δt * f(t_n + Δt / 2, (y_n + y_{n+1}) / 2)

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
        t_mid = t0 + 0.5 * δt
        f0 = terms.vf(t0, y0, args)
        y1_init = jax.tree.map(lambda y, f: y + δt * f, y0, f0)

        # Define F(y1) = y1 - y0 - δt * f(t_mid, (y0 + y1)/2)
        def F_fn(y1):
            y_mid = jax.tree.map(lambda a, b: 0.5 * (a + b), y0, y1)
            f_mid = terms.vf(t_mid, y_mid, args)
            return jax.tree.map(lambda a, b, f: a - b - δt * f, y1, y0, f_mid)

        y1, status = _newton_gmres(
            F_fn, y0, y1_init, self.rtol, self.atol, self.max_iters
        )

        result = diffrax.RESULTS.where(
            status == _CONVERGED,
            diffrax.RESULTS.successful,
            diffrax.RESULTS.where(
                status == _MAX_ITERS,
                diffrax.RESULTS.promote(optx.RESULTS.nonlinear_max_steps_reached),
                diffrax.RESULTS.promote(optx.RESULTS.nonlinear_divergence),
            ),
        )
        # As Diffrax's own implicit solvers do: an infinite error estimate makes an adaptive
        # controller reject the failed step and retry with a smaller one; with constant steps
        # the failure result ends the solve.
        y_error = jax.tree.map(
            lambda a, b: jnp.where(status == _CONVERGED, a - b, jnp.inf), y1, y1_init
        )
        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, None, result


def _newton_gmres(F_fn, y0, y_init, rtol, atol, max_iters):
    """Solve ``F(y)=0`` using at most ``max_iters`` Newton updates with GMRES linear solves.

    Returns ``(y, status)`` with status ``_CONVERGED``, ``_MAX_ITERS`` or ``_NONFINITE``.
    Convergence is decided on the residual of the iterate that is returned: the scaled
    residual must be finite and below one. A nonfinite residual or Newton update (e.g. a
    failed inner solve) stops the iteration with ``_NONFINITE``; a NaN never counts as
    converged.

    Notes
    -----
    - The Jacobian-vector product is obtained via ``jax.linearize``.
    - The GMRES tolerance is chosen adaptively (Eisenstat–Walker style) based on
      the current scaled residual norm.
    """

    def scaled_norm(res, y1):
        scaled_res = jax.tree.map(
            lambda r, a, b: r / (atol + rtol * jnp.maximum(jnp.abs(a), jnp.abs(b))),
            res, y0, y1,
        )
        return jnp.sqrt(sum(jnp.vdot(x, x).real for x in jax.tree.leaves(scaled_res)))

    def classify(y1, norm, i):
        finite = jnp.isfinite(norm) & _all_finite(y1)
        return jnp.where(
            ~finite, _NONFINITE,
            jnp.where(norm < 1.0, _CONVERGED, jnp.where(i >= max_iters, _MAX_ITERS, _ITERATING)),
        )

    @jax.jit
    def loop_fn(y_init):
        def cond_fn(state):
            return state[1] == _ITERATING

        def body_fn(state):
            y1, _, i = state
            res, jvp = jax.linearize(F_fn, y1)
            norm = scaled_norm(res, y1)
            # Adaptive inner tolerance (Eisenstat–Walker).
            delta, _ = gmres(
                jvp,
                jax.tree.map(jnp.negative, res),
                tol=jnp.minimum(0.1, norm * 0.5),
                atol=atol,
                maxiter=max(1, min(20, max_iters // 2)),
            )
            y1_next = jax.tree.map(lambda y, d: y + d, y1, delta)
            # Decide on the residual of the iterate that will be returned.
            norm_next = scaled_norm(F_fn(y1_next), y1_next)
            return y1_next, classify(y1_next, norm_next, i + 1), i + 1

        state = (y_init, classify(y_init, scaled_norm(F_fn(y_init), y_init), 0), 0)
        y1_final, status, _ = lax.while_loop(cond_fn, body_fn, state)
        return y1_final, status

    return loop_fn(y_init)


_ITERATING, _CONVERGED, _MAX_ITERS, _NONFINITE = 0, 1, 2, 3


def _all_finite(tree):
    return jnp.all(jnp.array([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)]))
