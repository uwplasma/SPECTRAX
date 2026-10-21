"""Integrating-factor (Lawson) wrapper for an explicit Diffrax Runge-Kutta solver.

Motivation. In the Hermite-Fourier system the closure / hypercollision / hyper-diffusion terms are diagonal in the
state (a real, non-positive rate per coefficient). With a field-scaled closure rate (``field_scaled_closure_rate``)
that rate grows with the self-consistent field, and an explicit step must satisfy ``h * rate < ~5`` (Dopri8 real
axis). Lawson's method removes that limit for the diagonal part by treating it exactly.

Method. Write the right-hand side as ``f(t, y) = -r(y0) * y + N(t, y)`` where ``r >= 0`` is a diagonal rate
(a pytree like ``y``) *frozen at the step start* ``y0`` and ``N = f + r y0 * y`` is everything else, including the
variation of the true rate over the step. This is an exact identity for any ``r``. With ``w(tau) = exp(r (tau - t0)) y(tau)``,

    w' = exp(r (tau - t0)) * ( f(tau, exp(-r (tau - t0)) w) + r * exp(-r (tau - t0)) w )

and one step of the wrapped Runge-Kutta method is taken on ``w`` from ``w(t0) = y0``; then ``y1 = exp(-r h) w1``.

Order and consistency. The split is an identity for every ``r``, so the step is the wrapped explicit RK tableau
(order ``p``) applied to ``w`` with a parameter ``r = r(y0)``. Lawson (1967, SIAM J. Numer. Anal. 4, 372-380) shows that
the order ``p`` of the underlying RK method is kept for a fixed linear part as ``h -> 0``; here the rate is the
step-start rate (Lawson's re-evaluated-A form). This is a statement about ``h -> 0`` at fixed rate only. No stiff-order
claim is made when ``r h >> 1``: Hochbruck and Ostermann (2010, Acta Numerica 19, 209-286; Thm 2.17, Ex. 2.32,
Table 2.2) show that the stiff order of Lawson methods drops (to 1 for Lawson-Runge) when ``||h A||`` is large, because the
classical order conditions do not carry over. The empirical support here is measured agreement with plain Dopri8
records (5e-10 in the H05 c=64 case), not a theorem. The embedded error estimate is
the wrapped one, mapped back with ``exp(-r h)``. The stability function equals that of the wrapped tableau applied to
``N`` (Crouseilles-Einkemmer-Massot, arXiv:1910.12720). Dense output is ``exp(-r (t - t0))`` times the wrapped
interpolant of ``w``, so saves keep the wrapped interpolation order.

What it does not do. It removes the stiffness of the diagonal rate only; it neither removes nor adds damping to a
truncation instability (the exponential is exact), and it does not conserve any quadratic invariant beyond the
order of the scheme (see above for the order statement). Rows that must not be damped (density, momentum and energy rows, field, ledger/work
states) get rate zero; for those ``exp(0) = 1`` and the arithmetic is bitwise that of the wrapped solver.

Dense output. The interpolant of ``w`` is a polynomial, so ``y(t)`` between step ends is accurate only for ``r h`` of order
one or less; saves on step ends are exact. Use ``diffrax.ClipStepSizeController(controller, step_ts=save_times)`` (the
``integrating_factor`` options of ``simulation`` and Dark-SPECTRAX ``run`` do) whenever a stiff rate is present.

Limits. ``r h`` must stay below about 700 (``exp`` overflow in float64); a step that overflows is rejected and shrunk
(see ``_guard``), it is never accepted. The wrapped solver may be FSAL (e.g.
``Dopri8``); the first-same-as-last stage is then recomputed each step (one extra right-hand-side evaluation per
step) because the stage derivative lives in the transformed variable. Only ``ODETerm`` problems are supported.
"""
from __future__ import annotations

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from diffrax import ODETerm
from diffrax._local_interpolation import AbstractLocalInterpolation
from diffrax._solver.base import AbstractAdaptiveSolver, AbstractWrappedSolver

__all__ = ["LawsonSolver", "hypercollision_rate", "simulation_rate"]


def _scale(rate, y, tau):
    """Pytree of ``exp(rate * tau)`` factors (broadcast against ``y``)."""
    return jtu.tree_map(lambda r, v: jnp.exp(jnp.asarray(r) * tau).astype(jnp.real(v).dtype), rate, y)


def _mul(a, b):
    return jtu.tree_map(lambda x, y: x * y, a, b)


def _guard(y0, y1, y_err):
    """Overflow guard: if ``exp(r h)`` overflowed (non-finite candidate or error), return ``y0`` with a huge finite error
    estimate (1e30; an infinite one poisons Diffrax's PID memory with a zero inverse error and the next step becomes inf)
    so the step controller rejects and shrinks the step. Without it a non-finite candidate gives
    ``scale = inf`` and a zero scaled error, i.e. an accepted garbage step. Finite steps pass through bitwise."""
    if y_err is None:
        return y1, y_err
    leaves = jtu.tree_leaves((y1, y_err))
    ok = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))
    y1 = jtu.tree_map(lambda a, b: jnp.where(ok, a, b), y1, y0)
    y_err = jtu.tree_map(lambda e: jnp.where(ok, e, jnp.full_like(e, 1e30)), y_err)
    return y1, y_err


class _LawsonInterpolation(AbstractLocalInterpolation):
    t0: jax.Array
    t1: jax.Array
    base: AbstractLocalInterpolation
    rate: object

    def evaluate(self, t0, t1=None, left=True):
        if t1 is not None:
            return self.evaluate(t1, left=left) - self.evaluate(t0, left=left)
        w = self.base.evaluate(t0, left=left)
        if hasattr(self.rate, "phi"):  # general integrating factor: y = Phi(t - t0) w
            return self.rate.phi(t0 - self.t0, w)
        return _mul(_scale(self.rate, w, -(t0 - self.t0)), w)


class LawsonSolver(AbstractAdaptiveSolver, AbstractWrappedSolver):
    """Lawson (integrating-factor) version of an explicit Runge-Kutta solver with a step-start diagonal rate.

    Parameters
    ----------
    solver : diffrax.AbstractSolver
        Explicit Runge-Kutta solver (e.g. ``diffrax.Dopri8()``) with error estimate.
    factor_fn : callable ``(t, y, args) -> factor``, optional
        General integrating factor (instead of ``rate_fn``): an object with ``phi(tau, w) = Phi(tau) w`` and
        ``rhs(vf, t, tau, w, args) = Phi^{-1}(f(t, Phi w) - Phi' w)`` (``tau = t - t0``); see :func:`spectrax.hermite_streaming_factor`
        (exact Hermite free streaming, optionally with the diagonal rate inside).
    rate_fn : callable ``(t, y, args) -> rate``
        Diagonal damping rates ``r >= 0`` (the term ``-r * y`` is treated exactly), a pytree with the structure
        of ``y`` whose leaves broadcast against the leaves of ``y`` (use ``0.0`` for states outside the
        exponent). It is evaluated once per step at the step start. The right-hand side of the ``ODETerm`` must
        contain the same damping (``f = -r y + N``); the solver does not add it.
    """

    solver: diffrax.AbstractSolver
    rate_fn: object = eqx.field(static=True, default=None)
    factor_fn: object = eqx.field(static=True, default=None)

    term_structure = ODETerm

    @property
    def interpolation_cls(self):  # pyright: ignore
        base_cls = self.solver.interpolation_cls

        def make(*, t0, t1, base, rate):
            return _LawsonInterpolation(t0=t0, t1=t1, base=base_cls(t0=t0, t1=t1, **base), rate=rate)

        return make

    @property
    def term_compatible_contr_kwargs(self):
        return self.solver.term_compatible_contr_kwargs

    def order(self, terms):
        return self.solver.order(terms)

    def error_order(self, terms):
        return self.solver.error_order(terms)

    def init(self, terms, t0, t1, y0, args):
        return self.solver.init(terms, t0, t1, y0, args)

    def func(self, terms, t0, y0, args):
        return self.solver.func(terms, t0, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del made_jump  # always True for the wrapped solver: its cached first stage is in another variable
        vf = terms.vf
        if self.factor_fn is not None:  # general integrating factor Phi(tau) (e.g. exact Hermite streaming)
            factor = self.factor_fn(t0, y0, args)
            vf_f = lambda tau, w, a: factor.rhs(vf, tau, tau - t0, w, a)
            w1, w_err, dense, state, result = self.solver.step(ODETerm(vf_f), t0, t1, y0, args, solver_state, True)
            y_err = None if w_err is None else factor.phi(t1 - t0, w_err)
            return factor.phi(t1 - t0, w1), y_err, {"base": dense, "rate": factor}, state, result
        rate = self.rate_fn(t0, y0, args)

        def vf_w(tau, w, a):
            grow = _scale(rate, w, tau - t0)
            decay = _scale(rate, w, -(tau - t0))
            y = _mul(decay, w)
            f = vf(tau, y, a)
            fplus = jtu.tree_map(lambda fv, r, yv: fv + jnp.asarray(r) * yv, f, rate, y)
            return _mul(grow, fplus)

        w1, w_err, dense, state, result = self.solver.step(ODETerm(vf_w), t0, t1, y0, args, solver_state, True)
        back = _scale(rate, w1, -(t1 - t0))
        y1 = _mul(back, w1)
        y_err = None if w_err is None else _mul(back, w_err)
        y1, y_err = _guard(y0, y1, y_err)
        return y1, y_err, {"base": dense, "rate": rate}, state, result


def hypercollision_rate(nu, col, D=0.0, k2_grid=None):
    """Diagonal rate ``nu * col + D * k^2`` of the SPECTRAX kinetic right-hand side (not broadcast to the full state).

    ``nu`` is a scalar or per-species ``(Ns, 1, 1, 1, 1, 1, 1)``, ``col`` the ``(Np, Nm, Nn)`` collision spectrum
    (``parameters["collision_matrix"]``), as in ``Hermite_Fourier_system`` (``Col = -nu col Ck``,
    ``Diff = -D k2 Ck``). The result broadcasts against ``Ck`` of shape ``(Ns, Np, Nm, Nn, Ny, Nx//2+1, Nz)``.
    """
    rate = jnp.asarray(nu) * col[None, :, :, :, None, None, None]
    return rate if k2_grid is None else rate + D * k2_grid


def simulation_rate(args, Ns, Nn, Nm, Np, Ny, Nx, Nz):
    """Rate for the flat ``[Ck, Fk]`` state of :func:`spectrax.simulation`: hypercollision plus hyper-diffusion
    on every Hermite coefficient, zero on the six field components (they are never damped)."""
    nkx = Nx // 2 + 1
    r = hypercollision_rate(args[8], args[21], args[9], args[19])
    r = jnp.broadcast_to(r, (Ns, Np, Nm, Nn, Ny, nkx, Nz)).reshape(-1)
    return jnp.concatenate([r, jnp.zeros(6 * Ny * nkx * Nz, dtype=r.dtype)])
