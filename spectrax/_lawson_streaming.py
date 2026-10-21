"""Exact Hermite-streaming integrating factor (with an optional diagonal closure rate) for :class:`LawsonSolver`.

The linear part of the Hermite-Fourier system that is *not* diagonal is the free-streaming operator. For every
Fourier mode and species it acts along each Hermite axis as ``G_ax = -i k_ax (a_ax T + u_ax I)`` with ``T`` the
symmetric tridiagonal matrix ``T[j, j+1] = T[j+1, j] = sqrt((j+1)/2)`` (the Hermite recurrence) and ``k_ax = 2 pi
index / L`` (``a`` the species thermal width, ``u`` its drift). The three axis operators act on different tensor
factors, so they commute and ``exp(tau G) = prod_ax exp(tau G_ax)``. ``T`` is real symmetric, so one ``eigh``
``T = V diag(lam) V^T`` (``lam`` are the Gauss-Hermite nodes over sqrt(2); ``V`` the nodal transform) gives

    exp(tau G_ax) = exp(-i tau k u) V diag(exp(-i tau k a lam)) V^T,

unitary, shared by every ``k`` and species (only the diagonal phases depend on ``k, a, u``), and exact: free
streaming is integrated for any step size, so the explicit step is no longer limited by ``h k a sqrt(2 Nn) < ~3.7``
(Dopri8 imaginary axis). Cost: two ``Nn x Nn`` matrix products per Hermite axis of size > 1 per application.

Combining with a diagonal rate ``r >= 0`` (hypercollision, field-scaled closure; ``-r y`` on the right-hand side) that
does *not* commute with ``G``: the integrating factor used is the Strang product

    Phi(tau) = A(tau/2) B(tau) A(tau/2),   A(s) = exp(-r s),   B(tau) = exp(tau G),

with ``w = Phi(tau)^{-1} y``. The transformation is *exact* for any smooth ``Phi`` with ``Phi(0) = I``:

    w' = Phi^{-1} ( f(Phi w) - Phi' w ),

and ``Phi' w`` is evaluated analytically, ``Phi' w = -(r/2) y + A G B A w - Phi (r/2) w``, so the splitting error is
not an error of the scheme: the identity is exact and only the *size* of what is left in ``w'`` depends on how well
``Phi`` approximates the true flow of ``G - r``. Collecting terms (``z = A w``, ``u = B z``, ``y = A u``)

    w' = A^{-1} B^{-1} [ A^{-1} f(y) + (r/2) u ] - A^{-1} G z + (r/2) w,

which needs one ``B`` and one ``B^{-1}`` application per Runge-Kutta stage. For ``G = 0`` it reduces to the diagonal
Lawson of :mod:`spectrax._lawson`; for ``r = 0`` to ``B^{-1} f(B w) - G w``. The Strang flow equals
``exp(tau (G - r) + O(tau^3 [G, [G, r]]))`` (symmetric BCH), so the leftover is ``O((tau |G|)^2 |r|)``: a stiff closure is
removed completely only while ``h |G|`` is not large; for ``h |G| >~ 1`` the closure part acts as in a Strang split.
Order: the step is the wrapped RK tableau applied in transformed variables, so the order of the underlying RK method is
retained for ``h -> 0`` at a fixed linear part (Lawson 1967, SIAM J. Numer. Anal. 4, 372-380); the rate and ``B`` are
step-start (Lawson's re-evaluated-A form). No stiff-order claim is made when ``r h >> 1``: the stiff order of Lawson
methods drops (to 1 for Lawson-Runge; Hochbruck and Ostermann 2010, Acta Numerica 19, 209-286, Thm 2.17, Ex. 2.32,
Table 2.2). Empirical support is measured agreement with plain Dopri8 records (5e-10 in H05 c=64). The stability function for the streaming/closure part is
that of the Lawson test equation of the wrapped tableau (arXiv:1910.12720). The unitary ``B`` conserves the
Euclidean norm of every ``k`` block; the density row at ``k = 0`` (``G = 0``) is untouched.

Fields and any other state leaves are left alone (``Phi = I`` on them).
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["StreamingFactor", "hermite_streaming_factor", "hermite_axis_eig"]

_AXES = (3, 2, 1)  # Hermite axes (n, m, p) of Ck with shape (Ns, Np, Nm, Nn, Ny, Nkx, Nz), paired with k_x, k_y, k_z


def hermite_axis_eig(N):
    """``(lam, V)`` of the symmetric tridiagonal Hermite recurrence matrix ``T`` (size N): ``T = V diag(lam) V^T``."""
    off = np.sqrt(np.arange(1, N) / 2.0)
    T = np.diag(off, 1) + np.diag(off, -1) if N > 1 else np.zeros((1, 1))
    lam, V = np.linalg.eigh(T)
    return lam, V


def _bc(x, axis):
    """Reshape a length-N vector to broadcast along ``axis`` of a 7-D ``Ck``."""
    shape = [1] * 7
    shape[axis] = -1
    return jnp.reshape(x, shape)


def _tri(z, axis):
    """``T z`` along ``axis`` using the recurrence coefficients (no dense matrix)."""
    N = z.shape[axis]
    if N == 1:
        return jnp.zeros_like(z)
    j = jnp.arange(N)
    up = _bc(jnp.sqrt((j + 1) / 2.0) * (j < N - 1), axis)  # coefficient of z_{j+1} in row j
    dn = _bc(jnp.sqrt(j / 2.0), axis)  # coefficient of z_{j-1}
    zp = jnp.concatenate([jax.lax.slice_in_dim(z, 1, N, axis=axis), jnp.zeros_like(jax.lax.slice_in_dim(z, 0, 1, axis=axis))], axis=axis)
    zm = jnp.concatenate([jnp.zeros_like(jax.lax.slice_in_dim(z, 0, 1, axis=axis)), jax.lax.slice_in_dim(z, 0, N - 1, axis=axis)], axis=axis)
    return up * zp + dn * zm


class StreamingFactor(eqx.Module):
    """Integrating factor ``Phi(tau) = A(tau/2) B(tau) A(tau/2)`` on the Hermite-Fourier block of the state.

    ``kappa`` is a tuple of three arrays ``k_ax / L_ax`` broadcastable to ``(Ny, Nkx, Nz)``; ``a`` and ``u`` are
    ``(Ns, 3)`` widths and drifts; ``rate`` is None or a diagonal rate broadcastable to ``Ck``. ``to_ck`` / ``from_ck``
    map the full state pytree to / from the 7-D ``Ck`` (static callables).
    """

    kappa: tuple
    a: jax.Array
    u: jax.Array
    rate: object
    lam: tuple = eqx.field(static=True)
    V: tuple = eqx.field(static=True)
    to_ck: object = eqx.field(static=True)
    from_ck: object = eqx.field(static=True)

    def _k(self, i):
        return self.kappa[i][None, None, None, None, :, :, :]

    def _au(self, i):
        return (self.a[:, i][:, None, None, None, None, None, None], self.u[:, i][:, None, None, None, None, None, None])

    def streaming(self, z):
        """``G z`` (generator of the streaming flow)."""
        out = 0.0
        for i, ax in enumerate(_AXES):
            a, u = self._au(i)
            out = out - 1j * self._k(i) * (a * _tri(z, ax) + u * z)
        return out

    def B(self, tau, z):
        """``exp(tau G) z`` (exact; ``tau`` may be negative)."""
        for i, ax in enumerate(_AXES):
            a, u = self._au(i)
            k = self._k(i)
            lam = _bc(jnp.asarray(self.lam[i]), ax)
            N = z.shape[ax]
            if N > 1:
                V = jnp.asarray(self.V[i])
                z = jnp.moveaxis(jnp.tensordot(V.T, z, axes=([1], [ax])), 0, ax)
                z = z * jnp.exp(-1j * tau * k * (a * lam + u))
                z = jnp.moveaxis(jnp.tensordot(V, z, axes=([1], [ax])), 0, ax)
            else:
                z = z * jnp.exp(-1j * tau * k * u)
        return z

    def _A(self, s, ck):
        if self.rate is None:
            return ck
        return ck * jnp.exp(-self.rate * s).astype(jnp.real(ck).dtype)

    def phi(self, tau, w):
        """``Phi(tau) w`` for the full state pytree (identity off the Hermite-Fourier block)."""
        ck = self._A(tau / 2, self.B(tau, self._A(tau / 2, self.to_ck(w))))
        return self.from_ck(w, ck)

    def rhs(self, vf, t, tau, w, args):
        """Transformed vector field ``w' = Phi^{-1}(f(t, Phi w) - Phi' w)`` at absolute time ``t`` and time since step start ``tau``."""
        ck = self.to_ck(w)
        z = self._A(tau / 2, ck)
        u = self.B(tau, z)
        y = self.from_ck(w, self._A(tau / 2, u))
        f = vf(t, y, args)
        fck = self.to_ck(f)
        if self.rate is None:
            out = self.B(-tau, fck) - self.streaming(z)
        else:
            r = self.rate
            iA = jnp.exp(r * (tau / 2)).astype(jnp.real(ck).dtype)
            out = iA * self.B(-tau, iA * fck + 0.5 * r * u) - iA * self.streaming(z) + 0.5 * r * ck
        return self.from_ck(f, out)


def hermite_streaming_factor(layout="flat", shape=None, rate_fn=None, basis_fn=None):
    """Return a ``factor_fn(t, y, args) -> StreamingFactor`` for ``LawsonSolver(solver, factor_fn=...)``.

    Parameters
    ----------
    layout : {"flat", "first"}
        ``"flat"``: the state of :func:`spectrax.simulation` (flat ``[Ck, Fk]``; needs ``shape``). ``"first"``: a
        tuple / list whose first entry is the 7-D ``Ck`` (Dark-SPECTRAX style pytree states).
    shape : tuple, optional
        ``(Ns, Np, Nm, Nn, Ny, Nkx, Nz)``, required for ``"flat"``.
    rate_fn : callable ``(t, y, args) -> rate`` or None
        Diagonal closure rate (>= 0, broadcastable to ``Ck``, already part of the right-hand side as ``-r y``) put
        inside the factor. None: streaming only; the damping then stays explicit in the Runge-Kutta stages.
    basis_fn : callable ``(t, y, args) -> (a, u)`` or None
        Hermite widths and drifts, each ``(3 * Ns,)``, taken at the step start (a moving / pump-frame basis stored
        in the state). None: ``args[11]`` and ``args[12]`` (fixed basis). The basis change over the step stays
        in the explicit part (the factor is an exact identity for any frozen ``a, u``).

    ``args`` is the SPECTRAX argument tuple (``nabla = args[20]``, ``alpha_s = args[11]``, ``u_s = args[12]``).
    """
    if layout == "flat":
        nck = int(np.prod(shape))

        def to_ck(y):
            return y[:nck].reshape(shape)

        def from_ck(y, ck):
            return jnp.concatenate([ck.reshape(-1).astype(y.dtype), y[nck:]])
    elif layout == "first":
        def to_ck(y):
            return y[0]

        def from_ck(y, ck):
            return type(y)((ck,) + tuple(y[1:]))
    else:
        raise ValueError("layout must be 'flat' or 'first'")

    def factor_fn(t, y, args):
        ck = to_ck(y)
        Ns = ck.shape[0]
        eig = [hermite_axis_eig(ck.shape[ax]) for ax in _AXES]
        nabla = args[20]
        a_s, u_s = (args[11], args[12]) if basis_fn is None else basis_fn(t, y, args)
        return StreamingFactor(
            kappa=(nabla[0], nabla[1], nabla[2]),
            a=jnp.asarray(a_s).reshape(Ns, 3), u=jnp.asarray(u_s).reshape(Ns, 3),
            rate=None if rate_fn is None else jnp.asarray(rate_fn(t, y, args)),
            lam=tuple(tuple(float(x) for x in e[0]) for e in eig),
            V=tuple(tuple(tuple(float(x) for x in row) for row in e[1]) for e in eig),
            to_ck=to_ck, from_ck=from_ck)

    return factor_fn
