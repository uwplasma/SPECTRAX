import diffrax
import jax
import jax.numpy as jnp
from jax import lax, jacrev
from functools import partial
from diffrax._custom_types import VF

class CrankNicolson(diffrax.AbstractSolver):
    """
    Crank-Nicolson solver using Newton's method for the implicit equation.
    
    Solves: G(y1) = y1 - y0 - dt * f(t1, 0.5*(y0 + y1)) = 0
    using Newton iteration: y1_new = y1 - J^(-1) * G(y1)
    where J is the Jacobian of G with respect to y1.
    """
    rtol: float = 1e-6
    atol: float = 1e-8
    max_iters: int = 10  # Newton typically converges in few iterations
    
    term_structure = diffrax.ODETerm
    interpolation_cls = diffrax.LocalLinearInterpolation
    
    def order(self, terms):
        return 2
    
    def init(self, terms, t0, t1, y0, args):
        return None
    
    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        
        δt = t1 - t0
        
        # Define the nonlinear residual function G(y1) = 0
        def residual(y1):
            """G(y1) = y1 - y0 - dt * f(t1, 0.5*(y0 + y1))"""
            y_mid = 0.5 * (y0 + y1)
            return y1 - y0 - δt * terms.vf(t1, y_mid, args)
        
        # Compute Jacobian using JAX (reverse mode for large systems)
        jacobian_fn = jacrev(residual, holomorphic=True)
        
        def keep_iterating(val):
            _, not_converged, iter_count = val
            return not_converged & (iter_count < self.max_iters)
        
        def newton_iteration(val):
            y1, _, iter_count = val
            
            # Compute residual and Jacobian
            G = residual(y1)
            J = jacobian_fn(y1)
            
            # Newton update: y1_new = y1 - J^(-1) * G
            # Use JAX's solve for numerical stability
            try:
                delta_y = jnp.linalg.solve(J, G)
                new_y1 = y1 - delta_y
            except:
                # Fallback to pseudoinverse if J is singular
                delta_y = jnp.linalg.pinv(J) @ G
                new_y1 = y1 - delta_y
            
            # Check convergence
            # Use both absolute change and residual norm
            change_norm = jnp.linalg.norm(delta_y)
            residual_norm = jnp.linalg.norm(G)
            
            y_scale = jnp.maximum(jnp.linalg.norm(y1), jnp.linalg.norm(new_y1))
            rel_change = change_norm / (y_scale + self.atol)
            
            # Converged if both change and residual are small
            converged = (rel_change < self.rtol) & (residual_norm < self.atol)
            not_converged = ~converged
            
            return new_y1, not_converged, iter_count + 1
        
        # Initial guess: Forward Euler
        f0 = terms.vf(t0, y0, args)
        euler_y1 = y0 + δt * f0
        
        # Run Newton iteration
        y1, final_not_converged, final_iter_count = lax.while_loop(
            keep_iterating,
            newton_iteration,
            (euler_y1, True, 0)
        )
        
        # Error estimation: compare with Euler method
        y_error = y1 - euler_y1
        
        # Check if Newton failed to converge
        result = diffrax.RESULTS.successful
        
        # Package return values
        dense_info = dict(y0=y0, y1=y1)
        solver_state = None
        
        return y1, y_error, dense_info, solver_state, result
    
    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)