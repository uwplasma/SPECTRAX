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
import numpy as np
from diffrax import ODETerm
from diffrax._local_interpolation import AbstractLocalInterpolation
from diffrax._solver.base import AbstractAdaptiveSolver, AbstractWrappedSolver

__all__ = ["LawsonSolver", "hypercollision_rate", "simulation_rate", "StreamingFactor", "hermite_streaming_factor"]


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
    """Rate for the ``(Ck, Fk)`` pytree state of :func:`spectrax.simulation`: hypercollision plus hyper-diffusion
    on every Hermite coefficient, zero on the six field components (they are never damped)."""
    nkx = Nx // 2 + 1
    r = hypercollision_rate(args[8], args[21], args[9], args[19])
    return jnp.broadcast_to(r, (Ns, Np, Nm, Nn, Ny, nkx, Nz)), 0.0


# --- Exact Hermite free streaming: the factor of ``LawsonSolver(solver, factor_fn=...)`` ---------------------------
# Per Fourier mode and species the free-streaming operator acts along each Hermite axis as G_ax = -i k (a T + u I),
# T the symmetric tridiagonal recurrence matrix T[j, j+1] = T[j+1, j] = sqrt((j+1)/2), k = 2 pi index / L. The three
# axes commute, and T = V diag(lam) V^T gives exp(tau G_ax) = exp(-i tau k u) V diag(exp(-i tau k a lam)) V^T: unitary,
# exact for any step (the explicit limit h k a sqrt(2 Nn) < ~3.7 disappears), one eigh shared by all k and species.
# A diagonal rate r >= 0 (closure, ``-r y`` in the right-hand side) that does not commute with G can be put inside the
# Strang product Phi(tau) = A(tau/2) B(tau) A(tau/2), A(s) = exp(-r s), B(tau) = exp(tau G). For any smooth Phi with
# Phi(0) = I the substitution y = Phi w is an exact identity, w' = Phi^-1 (f(Phi w) - Phi' w); only the size of what is
# left in w' depends on how well Phi approximates the flow of G - r (leftover O((tau |G|)^2 |r|)). With z = A w,
# u = B z, y = A u this is w' = A^-1 B^-1 [A^-1 f(y) + (r/2) u] - A^-1 G z + (r/2) w. Order and stiff-order statements
# are those of the module docstring. The unitary B conserves the norm of every k block and leaves k = 0 untouched.
# Fields and other state leaves are left alone (Phi = I on them).

_AXES = (3, 2, 1)  # Hermite axes (n, m, p) of Ck (Ns, Np, Nm, Nn, Ny, Nkx, Nz), paired with k_x, k_y, k_z


def _apply(M, z, axis):
    return jnp.moveaxis(jnp.tensordot(M, z, axes=([1], [axis])), 0, axis)


class StreamingFactor(eqx.Module):
    """Integrating factor ``Phi(tau) = A(tau/2) B(tau) A(tau/2)`` acting on ``Ck = y[0]``.

    ``kappa``: three arrays ``k_ax / L_ax`` broadcastable to ``(Ny, Nkx, Nz)``; ``a``, ``u``: ``(Ns, 3)`` widths and
    drifts; ``rate``: None or a diagonal rate broadcastable to ``Ck``; ``T``, ``lam``, ``V``: per-axis Hermite matrix
    and its eigendecomposition.
    """

    kappa: tuple
    a: jax.Array
    u: jax.Array
    rate: object
    T: tuple
    lam: tuple
    V: tuple

    def _kau(self, i):
        s = (slice(None),) + (None,) * 6
        return self.kappa[i][None, None, None, None], self.a[:, i][s], self.u[:, i][s]

    def streaming(self, z):
        """``G z`` (generator of the streaming flow)."""
        out = 0.0
        for i, ax in enumerate(_AXES):
            k, a, u = self._kau(i)
            out = out - 1j * k * (a * _apply(self.T[i], z, ax) + u * z)
        return out

    def B(self, tau, z):
        """``exp(tau G) z`` (exact; ``tau`` may be negative)."""
        for i, ax in enumerate(_AXES):
            k, a, u = self._kau(i)
            lam = jnp.reshape(self.lam[i], [-1 if j == ax else 1 for j in range(7)])
            z = _apply(self.V[i], _apply(self.V[i].T, z, ax) * jnp.exp(-1j * tau * k * (a * lam + u)), ax)
        return z

    def _A(self, s, ck):
        return ck if self.rate is None else ck * jnp.exp(-self.rate * s).astype(jnp.real(ck).dtype)

    @staticmethod
    def _put(y, ck):
        return type(y)((ck,) + tuple(y[1:]))

    def phi(self, tau, w):
        """``Phi(tau) w`` for the state pytree (identity off ``Ck``)."""
        return self._put(w, self._A(tau / 2, self.B(tau, self._A(tau / 2, w[0]))))

    def rhs(self, vf, t, tau, w, args):
        """Transformed field ``w' = Phi^-1 (f(t, Phi w) - Phi' w)`` at time ``t``, ``tau`` after the step start."""
        z = self._A(tau / 2, w[0])
        u = self.B(tau, z)
        f = vf(t, self._put(w, self._A(tau / 2, u)), args)
        if self.rate is None:
            out = self.B(-tau, f[0]) - self.streaming(z)
        else:
            r = self.rate
            iA = jnp.exp(r * (tau / 2)).astype(jnp.real(w[0]).dtype)
            out = iA * self.B(-tau, iA * f[0] + 0.5 * r * u) - iA * self.streaming(z) + 0.5 * r * w[0]
        return self._put(f, out)


def hermite_streaming_factor(rate_fn=None, basis_fn=None):
    """``factor_fn(t, y, args) -> StreamingFactor`` for ``LawsonSolver(solver, factor_fn=...)``; ``y[0]`` is the 7-D ``Ck``.

    ``rate_fn(t, y, args) -> rate`` (>= 0, broadcastable to ``Ck``, already part of the right-hand side as ``-r y``)
    puts a diagonal closure inside the factor; None: streaming only, damping stays explicit in the stages.
    ``basis_fn(t, y, args) -> (a, u)``, each ``(3 * Ns,)``, gives step-start widths and drifts of a moving basis;
    None: ``args[11]``, ``args[12]``. ``args[20]`` is ``nabla``.
    """
    def factor_fn(t, y, args):
        Ns = y[0].shape[0]
        a_s, u_s = (args[11], args[12]) if basis_fn is None else basis_fn(t, y, args)
        parts = []
        for ax in _AXES:
            N = y[0].shape[ax]
            off = np.sqrt(np.arange(1, N) / 2.0)
            T = np.diag(off, 1) + np.diag(off, -1)
            parts.append((T, *np.linalg.eigh(T)))
        T, lam, V = (tuple(jnp.asarray(p[i]) for p in parts) for i in range(3))
        return StreamingFactor(
            kappa=tuple(args[20][:3]), a=jnp.asarray(a_s).reshape(Ns, 3), u=jnp.asarray(u_s).reshape(Ns, 3),
            rate=None if rate_fn is None else jnp.asarray(rate_fn(t, y, args)), T=T, lam=lam, V=V)

    return factor_fn
