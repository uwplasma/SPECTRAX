import diffrax
import jax
import jax.numpy as jnp
from jax import lax

from diffrax._custom_types import VF

class CrankNicolson(diffrax.AbstractSolver):
    """
    Implementation of the Crank-Nicolson method for solving ODEs in JAX.
    
    This is an implicit second-order method that provides good stability
    for stiff equations. This implementation uses fixed-point iteration
    to solve the implicit equations.
    
    Parameters:
    -----------
    rtol : float
        Relative tolerance for convergence of the fixed-point iteration.
    atol : float
        Absolute tolerance for convergence of the fixed-point iteration.
    max_iters : int
        Maximum number of iterations for the fixed-point method.
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
        f0 = terms.vf(t0, y0, args)
        
        def keep_iterating(val):
            _, not_converged = val
            return not_converged
        
        def fixed_point_iteration(val):
            y1, _ = val
            
            # Crank-Nicolson update
            new_y1 = y0 + δt * terms.vf(t1, 0.5 * (y0 + y1), args)
            
            # Calculate scaled difference for convergence check
            diff = jnp.abs(new_y1 - y1)
            scale = self.atol + self.rtol * jnp.maximum(jnp.abs(y1), jnp.abs(new_y1))
            
            # Check if all elements have converged
            not_converged = jnp.any(diff > scale)
            
            return new_y1, not_converged
        
        # Initialize counter for iterations
        euler_y1 = y0 + δt * f0  # Euler step as initial guess
        
        # Run fixed-point iteration
        y1, _ = lax.while_loop(keep_iterating, fixed_point_iteration, (euler_y1, True))
        
        # Estimate local error using difference from Euler step
        y_error = y1 - euler_y1
        
        # Package return values expected by diffrax
        dense_info = dict(y0=y0, y1=y1)
        solver_state = None
        result = diffrax.RESULTS.successful
        
        return y1, y_error, dense_info, solver_state, result
    
    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
