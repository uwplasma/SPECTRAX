import diffrax
import lineax as lx
import jax
import jax.numpy as jnp
from jax import lax, random
from jax.scipy.sparse.linalg import gmres
from time import time
from functools import partial
import lineax as lx
from jax.flatten_util import ravel_pytree
from ._preconditioning import inverse_preconditioner
import jaxopt
from collections import deque


class ImplicitMidpoint(diffrax.AbstractSolver):
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
        
        # inv_precond = partial(
        # inverse_preconditioner,
        # *args,
        # y0,
        # δt,
        # self.rtol,     
        # self.max_iters  
        # )



        # M_right = lx.FunctionLinearOperator(inv_precond, input_structure=y0)


        y1 = _newton_gmres(F_fn, y0, y1_init, self.rtol, self.atol, self.max_iters, precond_fn=None)
        

        y_error = y1 - y1_init
        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, None, diffrax.RESULTS.successful


def _newton_gmres(F_fn, y0, y_init, rtol, atol, max_iters, precond_fn=None):
    """Optimized Newton-GMRES with adaptive tolerances and single F evaluation."""

    # static_struct = jax.tree_util.tree_map(
    # lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), y_init
# )
    
    @jax.jit
    def loop_fn(y_init):
        def cond_fn(state):
            _, not_converged, i = state
            return (i < max_iters) & not_converged

        def body_fn(state):
            y1, _, i = state
            
            res, jvp = jax.linearize(F_fn, y1)

            # res_tilde = precond_fn(res)

            # def jvp_tilde(v):
            #     return precond_fn(jvp(v))

            # mv_right = lambda z: jvp(precond_fn(z))

            # def mv(v):
#             #     return jax.jvp(F_fn, (y1,), (v,))[1]
            
            scale = atol + rtol * jnp.maximum(jnp.abs(y0), jnp.abs(y1))
            norm = jnp.linalg.norm(res / scale)
            
            # Adaptive inner tolerance (Eisenstat-Walker)
            inner_tol = jnp.minimum(0.1, norm * 0.5)
            
            delta, _ = gmres(jvp, -res, 
                           tol=inner_tol, 
                           atol=atol, 
                           maxiter=min(20, max_iters//2), 
                           M=precond_fn)

            # delta = precond_fn(delta_tilde)  # Apply preconditioner to the solution

#             # _, jvp = jax.linearize(lambda x: F_fn(x), y1)
#             A = lx.FunctionLinearOperator(jvp, input_structure=static_struct)
#             # A = lx.JacobianLinearOperator(lambda x, _: F_fn(x), y1)
# # #           # b = -res
#             solver = lx.GMRES(rtol=inner_tol, atol=atol, max_steps=20 * min(20, max_iters//2), norm=lambda r: jnp.linalg.norm(r))

            # M = lx.FunctionLinearOperator(precond_fn, input_structure=res)

# #             # A_tilde = A @ precond_fn                # (P^{-1} A)δ
#             sol = lx.linear_solve(A, -res, solver, throw=False)
#             delta = sol.value
#             # delta = precond_fn.mv(delta_tilde)  # Apply preconditioner to the solution
#             steps = sol.stats["num_steps"]          # JAX scalar; no Python boolean ops
#             jax.debug.print("Newton {}: GMRES steps = {}", i, steps)

            y1_next = jnp.where(norm < 1.0, y1, y1 + delta)
            
            return y1_next, norm >= 1.0, i + 1

        init_state = (y_init, True, 0)
        y1_final, _, _ = lax.while_loop(cond_fn, body_fn, init_state)
        return y1_final

    return loop_fn(y_init)


# def _newton_gmres(F_fn, y0, y_init, rtol, atol, max_iters, precond_fn=None):
#     """Efficient JAX-compatible Newton–GMRES loop with early convergence."""
    
#     @jax.jit
#     def loop_fn(y_init):
#         def cond_fn(state):
#             _, not_converged, i = state
#             return (i < max_iters) & not_converged

#         def body_fn(state):
#             y1, _, i = state
#             # res = F_fn(y1)
#             res, jvp = jax.linearize(F_fn, y1)
#             scale = atol + rtol * jnp.maximum(jnp.abs(y0), jnp.abs(y1))
#             norm = jnp.linalg.norm(res / scale)

#             # def mv(v):
#             #     return jax.jvp(F_fn, (y1,), (v,))[1]

                        
#             delta, _ = gmres(jvp, -res, tol=rtol, atol=atol, maxiter=max_iters, M=precond_fn)

#             # _, jvp = jax.linearize(lambda x: F_fn(x), y1)
#             # A = lx.FunctionLinearOperator(jvp, input_structure=jax.eval_shape(lambda: y1))
#             # A = lx.JacobianLinearOperator(lambda x, _: F_fn(x), y1)
#             # b = -res
#             # solver = lx.GMRES(rtol=rtol, atol=atol, max_steps=max_iters,
#             #       norm=lambda r: jnp.linalg.norm(r))

#             # # M = lx.FunctionLinearOperator(lambda x, _: precond_fn(x), jnp.zeros_like(b))
#             # delta = lx.linear_solve(A, b, solver).value

#             y1_next = jnp.where(norm < 1.0, y1, y1 + delta)

#             return y1_next, norm >= 1.0, i + 1

#         init_state = (y_init, True, 0)
#         y1_final, _, _ = lax.while_loop(cond_fn, body_fn, init_state)
#         return y1_final

#     return loop_fn(y_init)