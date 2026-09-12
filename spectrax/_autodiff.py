"""Terminal Vlasov–Maxwell solves with exact checkpointed discrete derivatives."""

import operator

import diffrax
import jax
import jax.numpy as jnp
from solvax import checkpointed_fori_loop

from ._initialization import initialize_simulation_parameters
from ._simulation import _solver_args, ode_system

__all__ = ["simulation_final"]


def simulation_final(input_parameters=None, *, steps, Nx=33, Ny=1, Nz=1,
                     Nn=20, Nm=1, Np=1, Ns=2, checkpoint_size=None,
                     checkpointing=True, checkpoints=None, integrand=None):
    """Return final ``(Ck, Fk)`` using fixed-step classical RK4.

    Compose with any real scalar JAX objective and ``jax.value_and_grad``;
    ``jax.jvp``, ``jax.vjp`` and ``jax.jit`` also work. Physical parameters,
    initial coefficients and ``t_max`` may be differentiated. Resolutions,
    ``steps``, ``integrand`` and checkpoint options must be static under JIT.

    Coefficients have the same layout as one snapshot from ``simulation``:
    ``Ck: (Ns*Np*Nm*Nn, Ny, Nx//2+1, Nz)`` and
    ``Fk: (6, Ny, Nx//2+1, Nz)``. No trajectory or diagnostics are retained.
    The step size is ``t_max / steps``; check stability and time convergence.
    There is no adaptive error estimate or implicit nonlinear solve.

    Optionally pass ``integrand(t, Ck, Fk)`` returning a real scalar or array.
    It is integrated with the same RK4 stages, returning ``(Ck, Fk, integral)``
    instead. Only the small accumulator is added to the checkpointed state;
    its shape must remain fixed. This supports time-window objectives without
    storing a trajectory. Smooth time weights permit accurate RK quadrature.

    SOLVAX replays segments during reverse mode. For N steps, state size S,
    and segment width C, retained state is O(S*(ceil(N/C)+C)), plus RHS
    workspace. The default C=ceil(sqrt(N)) balances these terms. This is
    sublinear in time steps, NOT independent of spatial/velocity resolution.
    Set ``checkpointing=False`` only for a small taped reference or timing.
    Alternatively, ``checkpoints=K`` uses Diffrax binomial checkpointing with
    a fixed retained-state budget O(K*S), independent of the step count (plus
    one-step workspace). This path supports reverse mode only; use the
    default SOLVAX path for JVPs. More checkpoints reduce replay work.
    All paths differentiate the identical finite RK4 algorithm, including
    the real-linear Fourier transforms; no continuous adjoint is substituted.
    """
    steps = operator.index(steps)
    if steps < 1:
        raise ValueError("steps must be a positive integer")
    parameters = initialize_simulation_parameters(
        {} if input_parameters is None else input_parameters,
        Nx, Ny, Nz, Nn, Nm, Np, Ns,
    )
    dt = parameters["t_max"] / steps
    if checkpoints is not None:
        checkpoints = operator.index(checkpoints)
        if checkpoints < 1 or not checkpointing or checkpoint_size is not None:
            raise ValueError("checkpoints must be positive and used without other checkpoint options")
    args = _solver_args(parameters, Nx, Ny, Nz, Nn, Nm, Np, Ns)
    y0 = jnp.concatenate((parameters["Ck_0"].ravel(),
                          parameters["Fk_0"].ravel()))

    split = Ns * Np * Nm * Nn * Ny * (Nx // 2 + 1) * Nz
    state_size = y0.size

    def unpack(y):
        return (y[:split].reshape(Ns * Np * Nm * Nn, Ny, Nx // 2 + 1, Nz),
                y[split:state_size].reshape(6, Ny, Nx // 2 + 1, Nz))

    if integrand is not None:
        sample = jnp.asarray(integrand(0.0, *unpack(y0)))
        if jnp.iscomplexobj(sample):
            raise ValueError("integrand must return real values")
        integral_shape = sample.shape
        y0 = jnp.concatenate((y0, jnp.zeros(sample.size, dtype=y0.dtype)))

    def rhs(t, y):
        derivative = ode_system(Nx, Ny, Nz, Nn, Nm, Np, Ns, t, y[:state_size], args)
        if integrand is not None:
            rate = jnp.asarray(integrand(t, *unpack(y)))
            return jnp.concatenate((derivative, rate.ravel()))
        return derivative

    def advance(index, y):
        return _rk4_step(rhs, index * dt, dt, y)

    if checkpoints is not None:
        # Integrate on an integer clock to execute exactly `steps` equal updates.
        # This also keeps endpoint time derivatives in the physical RHS/step size.
        term = diffrax.ODETerm(lambda index, y, _: dt * rhs(index * dt, y))
        y = diffrax.diffeqsolve(
            term, _RK4(), t0=0, t1=steps, dt0=1,
            y0=y0, saveat=diffrax.SaveAt(t1=True), max_steps=steps,
            adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=checkpoints),
        ).ys[0]
    elif checkpointing:
        y = checkpointed_fori_loop(
            0, steps, jax.checkpoint(advance), y0,
            checkpoint_size=checkpoint_size,
        )
    else:
        y = jax.lax.fori_loop(0, steps, advance, y0)
    if integrand is not None:
        return (*unpack(y), jnp.real(y[state_size:]).reshape(integral_shape))
    return unpack(y)


def _rk4_step(rhs, t, dt, y):
    k1 = rhs(t, y)
    k2 = rhs(t + dt / 2, y + dt * k1 / 2)
    k3 = rhs(t + dt / 2, y + dt * k2 / 2)
    k4 = rhs(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class _RK4(diffrax.AbstractSolver):
    """Adapt the same RK4 step to Diffrax's fixed-budget reverse scheduler."""

    term_structure = diffrax.ODETerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def order(self, terms):
        return 4

    def init(self, terms, t0, t1, y0, args):
        return None

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        y1 = _rk4_step(lambda t, y: terms.vf(t, y, args), t0, t1 - t0, y0)
        return y1, None, dict(y0=y0, y1=y1), None, diffrax.RESULTS.successful
