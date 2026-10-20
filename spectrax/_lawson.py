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

Order and consistency. For a fixed ``r`` this is the classical Lawson-RK method built on the wrapped tableau, which has
the order ``p`` of that tableau for the problem ``y' = -r y + N(y)``. Because the split is an identity for every
``r``, taking ``r = r(y0)`` (a smooth function of the step-start state, chosen before the stages) does not change
the order: the step map is a ``p``-th order RK map in ``w`` with a parameter. Constants contain ``|r| h`` through
the exponentials, which is the point (the stiff part is not in the stability condition). The embedded error estimate is
the wrapped one, mapped back with ``exp(-r h)``. The stability function equals that of the wrapped tableau applied to
``N`` (Crouseilles-Einkemmer-Massot, arXiv:1910.12720). Dense output is ``exp(-r (t - t0))`` times the wrapped
interpolant of ``w``, so saves keep the wrapped interpolation order.

What it does not do. It removes the stiffness of the diagonal rate only; it neither removes nor adds damping to a
truncation instability (the exponential is exact), and it does not conserve any quadratic invariant beyond the
order of the scheme. Rows that must not be damped (density, momentum and energy rows, field, ledger/work
states) get rate zero; for those ``exp(0) = 1`` and the arithmetic is bitwise that of the wrapped solver.

Limits. ``r h`` must stay below about 700 (``exp`` overflow in float64). The wrapped solver may be FSAL (e.g.
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

__all__ = ["LawsonSolver", "hypercollision_rate"]


def _scale(rate, y, tau):
    """Pytree of ``exp(rate * tau)`` factors (broadcast against ``y``)."""
    return jtu.tree_map(lambda r, v: jnp.exp(jnp.asarray(r) * tau).astype(jnp.real(v).dtype), rate, y)


def _mul(a, b):
    return jtu.tree_map(lambda x, y: x * y, a, b)


class _LawsonInterpolation(AbstractLocalInterpolation):
    t0: jax.Array
    t1: jax.Array
    base: AbstractLocalInterpolation
    rate: object

    def evaluate(self, t0, t1=None, left=True):
        if t1 is not None:
            return self.evaluate(t1, left=left) - self.evaluate(t0, left=left)
        w = self.base.evaluate(t0, left=left)
        return _mul(_scale(self.rate, w, -(t0 - self.t0)), w)


class LawsonSolver(AbstractAdaptiveSolver, AbstractWrappedSolver):
    """Lawson (integrating-factor) version of an explicit Runge-Kutta solver with a step-start diagonal rate.

    Parameters
    ----------
    solver : diffrax.AbstractSolver
        Explicit Runge-Kutta solver (e.g. ``diffrax.Dopri8()``) with error estimate.
    rate_fn : callable ``(t, y, args) -> rate``
        Diagonal damping rates ``r >= 0`` (the term ``-r * y`` is treated exactly), a pytree with the structure
        of ``y`` whose leaves broadcast against the leaves of ``y`` (use ``0.0`` for states outside the
        exponent). It is evaluated once per step at the step start. The right-hand side of the ``ODETerm`` must
        contain the same damping (``f = -r y + N``); the solver does not add it.
    """

    solver: diffrax.AbstractSolver
    rate_fn: object = eqx.field(static=True)

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
        rate = self.rate_fn(t0, y0, args)
        vf = terms.vf

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
        return y1, y_err, {"base": dense, "rate": rate}, state, result


def hypercollision_rate(nu, col, D=0.0, k2_grid=None):
    """Diagonal rate ``nu * col + D * k^2`` of the SPECTRAX kinetic right-hand side (not broadcast to the full state).

    ``nu`` is a scalar or per-species ``(Ns, 1, 1, 1, 1, 1, 1)``, ``col`` the ``(Np, Nm, Nn)`` spectrum from
    :func:`hypercollision_spectrum`, as in ``Hermite_Fourier_system`` (``Col = -nu col Ck``, ``Diff = -D k2 Ck``).
    """
    rate = jnp.asarray(nu) * col[None, :, :, :, None, None, None]
    return rate if k2_grid is None else rate + D * k2_grid
