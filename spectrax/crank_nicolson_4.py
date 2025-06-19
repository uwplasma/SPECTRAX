import diffrax
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.sparse.linalg import gmres  # Jacobian-free Krylov solver

class CrankNicolson(diffrax.AbstractSolver):
    """
    Crank–Nicolson solver for ODEs in JAX using a Jacobian-free Newton–Krylov (JFNK) nonlinear solve.

    This is an implicit second-order method well-suited for stiff problems. We replace
    fixed-point iteration with Newton’s method, using GMRES for the linear solves
    and JAX JVP for Jacobian–vector products.

    Parameters
    ----------
    rtol : float
        Relative tolerance for the Newton–Krylov convergence.
    atol : float
        Absolute tolerance for the Newton–Krylov convergence.
    max_iters : int
        Maximum number of Newton iterations.
    """
    rtol: float
    atol: float
    max_iters: int = 10
    term_structure = diffrax.ODETerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def order(self, terms):
        return 2

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        δt = t1 - t0
        # Euler predictor for initial guess
        f0 = terms.vf(t0, y0, args)
        euler_y1 = y0 + δt * f0

        # Define the nonlinear residual: F(y) = y - y0 - δt * f(t1, (y0 + y)/2)
        def F_fn(y):
            return y - (y0 + δt * terms.vf(t1, 0.5 * (y0 + y), args))

        # Build Jv action via JAX JVP
        def make_Jv(y):
            return lambda v: jax.jvp(F_fn, (y,), (v,))[1]

        # Newton–Krylov loop with fixed iterations, masking updates after convergence
        def newton_body(i, y1):
            residual = F_fn(y1)
            scale = self.atol + self.rtol * jnp.maximum(jnp.abs(y1), jnp.abs(y0))
            norm = jnp.linalg.norm(residual / scale)
            Jv = make_Jv(y1)
            delta, info = gmres(Jv, -residual, tol=self.rtol)
            # Only update where not yet converged
            return jnp.where(norm < 1.0, y1, y1 + delta)

        y1 = lax.fori_loop(0, self.max_iters, newton_body, euler_y1)

        # Local error estimate
        y_error = y1 - euler_y1

        dense_info = dict(y0=y0, y1=y1)
        solver_state = None
        result = diffrax.RESULTS.successful
        return y1, y_error, dense_info, solver_state, result

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)