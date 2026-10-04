"""Exact segment remap of the Hermite basis: (u, a) -> (u', a') per species and velocity axis.

With ``xi = (v - u)/a`` and ``xi' = (v - u')/a'``, the new coefficients of the same distribution are
``g'_m = sum_{n<=m} T_mn g_n`` with ``T_mn = integral psi_n(xi) h_m(A xi + B) dxi``, ``A = a/a'``,
``B = (u - u')/a'`` (Pagliantini et al., arXiv 2208.14373, Definition 1). ``h_m(A xi + B)`` is a
degree-m polynomial, so ``T`` is lower-triangular with ``T_mm = A^m``: every velocity moment of
order below the truncation is preserved exactly, and only the unresolved tail beyond order N
changes. In SPECTRAX coefficients ``C = n* g / (a_x a_y a_z)`` the map is
``C' = (a_x a_y a_z)/(a'_x a'_y a'_z) T_x T_y T_z C``.

Conditioning grows like ``(a/a')^N`` when the basis narrows and like ``(du/a)^N / sqrt(N!)`` with the
shift, so :func:`remap_event` refuses moves outside caps; :func:`moment_target` gives the moment-
matched basis and :func:`cap_target` clips it into the caps. Events are recorded so that a run can
be replayed with a frozen schedule, which is what gradients should differentiate.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

__all__ = ["remap_matrix", "remap", "moment_target", "cap_target", "remap_event", "low_moments",
           "tail_fraction", "remap_trigger"]

_AXES = ((3, "n"), (2, "m"), (1, "p"))  # storage axis of the x, y, z Hermite index in (Ns, Np, Nm, Nn, ...)


def remap_matrix(N, A, B):
    """Lower-triangular ``T[m, n]`` (N x N) with ``h_m(A xi + B) = sum_n T[m, n] h_n(xi)``.

    Built by the three-term recurrence ``h_{m+1}(y) = sqrt(2/(m+1)) y h_m(y) - sqrt(m/(m+1)) h_{m-1}(y)``
    applied to coefficient vectors, with ``xi h_k = sqrt((k+1)/2) h_{k+1} + sqrt(k/2) h_{k-1}``.
    O(N^2), exact on the truncated space (row m only involves columns n <= m).
    """
    k = jnp.sqrt(jnp.arange(1, N, dtype=float) / 2)
    zero = jnp.zeros(1)

    def step(carry, m):
        prev, cur = carry
        xi = jnp.concatenate([zero, k * cur[:-1]]) + jnp.concatenate([k * cur[1:], zero])
        nxt = jnp.sqrt(2 / (m + 1)) * (A * xi + B * cur) - jnp.sqrt(m / (m + 1)) * prev
        return (cur, nxt), nxt

    first = jnp.zeros(N).at[0].set(1.0)
    if N == 1:
        return first[None, :]
    _, rows = jax.lax.scan(step, (jnp.zeros(N), first), jnp.arange(N - 1, dtype=float))
    return jnp.concatenate([first[None, :], rows])


@partial(jit, static_argnames=["Nn", "Nm", "Np", "Ns"])
def remap(Ck, basis, new_basis, Nn, Nm, Np, Ns):
    """Coefficients of the same distribution in the basis ``new_basis``.

    ``basis`` and ``new_basis`` are ``stack([u_s, alpha_s])``, shape ``(2, 3*Ns)``. ``Ck`` has shape
    ``(Ns*Np*Nm*Nn, ...)`` or ``(Ns, Np, Nm, Nn, ...)``; the result has the same shape. Pure and
    differentiable; it does not check caps (see :func:`remap_event`).
    """
    shape = Ck.shape
    C = Ck.reshape(Ns, Np, Nm, Nn, -1)
    u, a = jnp.real(basis[0]).reshape(Ns, 3), jnp.real(basis[1]).reshape(Ns, 3)
    u2, a2 = jnp.real(new_basis[0]).reshape(Ns, 3), jnp.real(new_basis[1]).reshape(Ns, 3)
    A, B = a / a2, (u - u2) / a2
    for i, N in enumerate((Nn, Nm, Np)):
        if N == 1:
            continue
        T = jax.vmap(partial(remap_matrix, N))(A[:, i], B[:, i])  # (Ns, N, N)
        C = {0: lambda T, C: jnp.einsum("smn,spqnk->spqmk", T, C),
             1: lambda T, C: jnp.einsum("smn,spnqk->spmqk", T, C),
             2: lambda T, C: jnp.einsum("smn,snpqk->smpqk", T, C)}[i](T, C)
    scale = jnp.prod(A, axis=1)[:, None, None, None, None]  # (a_x a_y a_z)/(a'_x a'_y a'_z)
    return (scale * C).reshape(shape)


def _k0(Ck, Nn, Nm, Np, Ns):
    """k = 0 coefficients as (Ns, Np, Nm, Nn) real array."""
    C = Ck.reshape(Ns, Np, Nm, Nn, -1)
    return jnp.real(C[..., 0])


def moment_target(Ck, basis, Nn, Nm, Np, Ns, width_ratio=np.sqrt(2)):
    """Moment-matched basis from the box-averaged (k = 0) coefficients.

    Per species and axis, with ``r1 = g_1/g_0`` and ``r2 = g_2/g_0``: mean velocity
    ``U = u + a r1/sqrt(2)`` and variance ``sigma^2 = (a^2/2)(1 + sqrt(2) r2 - r1^2)``. Returns
    ``(target, sigma)`` with ``target = stack([U, width_ratio * sigma])``; ``width_ratio = sqrt(2)``
    makes a Maxwellian a single mode. Axes with fewer than 2 (3) modes keep their centre (width).
    A non-positive variance (possible for a badly resolved state) keeps the old width.
    """
    C = _k0(Ck, Nn, Nm, Np, Ns)
    u, a = jnp.real(basis[0]).reshape(Ns, 3), jnp.real(basis[1]).reshape(Ns, 3)
    C0 = C[:, 0, 0, 0]
    picks = [(lambda k: C[:, 0, 0, k] if k < Nn else jnp.zeros(Ns)),
             (lambda k: C[:, 0, k, 0] if k < Nm else jnp.zeros(Ns)),
             (lambda k: C[:, k, 0, 0] if k < Np else jnp.zeros(Ns))]
    r1 = jnp.stack([picks[i](1) / C0 for i in range(3)], axis=1)
    r2 = jnp.stack([picks[i](2) / C0 for i in range(3)], axis=1)
    var = a ** 2 / 2 * (1 + np.sqrt(2) * r2 - r1 ** 2)
    has3 = jnp.array([Nn, Nm, Np]) >= 3
    sigma = jnp.where(has3 & (var > 0), jnp.sqrt(jnp.where(var > 0, var, 1.0)), a / np.sqrt(2))
    U = u + a * r1 / np.sqrt(2)
    a_new = jnp.where(has3, width_ratio * sigma, a)
    return jnp.stack([U.reshape(-1), a_new.reshape(-1)]), sigma.reshape(-1)


def cap_target(basis, target, sigma, max_shift=1.0, max_narrow=1.1, min_width=1.1):
    """Clip a target basis into the caps: ``|u' - u| <= max_shift * a'``, ``a/a' <= max_narrow`` and
    ``a' >= min_width * sigma`` (the AW weighted norm needs a' > sigma). Widening is not capped."""
    u, a = jnp.real(basis[0]), jnp.real(basis[1])
    a2 = jnp.maximum(jnp.maximum(target[1], a / max_narrow), min_width * sigma)
    u2 = u + jnp.clip(target[0] - u, -max_shift * a2, max_shift * a2)
    return jnp.stack([u2, a2])


def low_moments(Ck, basis, Nn, Nm, Np, Ns):
    """Density, flux ``M_i`` and diagonal second moments ``M_ii`` (per species, every Fourier mode).

    ``n = A C_0``, ``M_i = A (u_i C_0 + a_i C_{e_i}/sqrt 2)``,
    ``M_ii = A [(u_i^2 + a_i^2/2) C_0 + sqrt 2 u_i a_i C_{e_i} + a_i^2 C_{2e_i}/sqrt 2]``, ``A = a_x a_y a_z``.
    Only moments the truncation resolves (and the remap preserves) are returned: ``M_i`` for axes with
    at least 2 modes, ``M_ii`` for axes with at least 3. Shape ``(Ns, n_moments, n_fourier)``.
    """
    C = Ck.reshape(Ns, Np, Nm, Nn, -1)
    u, a = jnp.real(basis[0]).reshape(Ns, 3), jnp.real(basis[1]).reshape(Ns, 3)
    A = jnp.prod(a, axis=1)[:, None]
    get = [lambda k: C[:, 0, 0, k], lambda k: C[:, 0, k, 0], lambda k: C[:, k, 0, 0]]
    C0 = C[:, 0, 0, 0]
    out = [A * C0]
    for i, N in enumerate((Nn, Nm, Np)):
        ui, ai = u[:, i, None], a[:, i, None]
        if N >= 2:
            out.append(A * (ui * C0 + ai / np.sqrt(2) * get[i](1)))
        if N >= 3:
            out.append(A * ((ui ** 2 + ai ** 2 / 2) * C0 + np.sqrt(2) * ui * ai * get[i](1)
                            + ai ** 2 / np.sqrt(2) * get[i](2)))
    return jnp.stack(out, axis=1)


def tail_fraction(Ck, Nn, Nm, Np, Ns):
    """Per-species fraction of sum |C|^2 (over every Fourier mode) held by any Hermite index above
    2/3 of its axis order (axes with more than 3 modes)."""
    C = jnp.abs(Ck.reshape(Ns, Np, Nm, Nn, -1)) ** 2
    tail = jnp.zeros((Np, Nm, Nn), bool)
    for N, idx in ((Nn, (None, None, slice(None))), (Nm, (None, slice(None), None)), (Np, (slice(None), None, None))):
        if N > 3:
            tail = tail | (jnp.arange(N) > 2 * N // 3)[idx]
    total = jnp.sum(C, axis=(1, 2, 3, 4))
    return jnp.sum(C * tail[None, :, :, :, None], axis=(1, 2, 3, 4)) / jnp.where(total > 0, total, 1.0)


def remap_event(Ck, basis, new_basis, Nn, Nm, Np, Ns, t=None, max_shift=1.0, max_narrow=1.1):
    """Remap with cap checks and a record: refuses (ValueError) a move outside the caps.

    Returns ``(Ck_new, record)``; ``record`` holds t, old/new basis, the per-species tail fraction
    before and after, and the maximum relative change of the low moments (round-off for an exact map).
    Call it between solver steps with concrete arrays, never inside a stage.
    """
    u, a = np.real(np.asarray(basis))
    u2, a2 = np.real(np.asarray(new_basis))
    if np.any(a2 <= 0) or np.any(np.abs(u2 - u) > max_shift * a2 * (1 + 1e-12)) or np.any(a / a2 > max_narrow * (1 + 1e-12)):
        raise ValueError(f"remap outside caps: max |du|/a' = {np.max(np.abs(u2 - u) / a2):.3g} (cap {max_shift}), "
                         f"max a/a' = {np.max(a / a2):.3g} (cap {max_narrow})")
    Ck_new = remap(Ck, basis, new_basis, Nn, Nm, Np, Ns)
    m0, m1 = low_moments(Ck, basis, Nn, Nm, Np, Ns), low_moments(Ck_new, new_basis, Nn, Nm, Np, Ns)
    record = {"t": None if t is None else float(t), "basis": np.asarray(basis).real.tolist(),
              "new_basis": np.asarray(new_basis).real.tolist(),
              "tail_before": np.asarray(tail_fraction(Ck, Nn, Nm, Np, Ns)).tolist(),
              "tail_after": np.asarray(tail_fraction(Ck_new, Nn, Nm, Np, Ns)).tolist(),
              "moment_defect": float(jnp.max(jnp.abs(m1 - m0)) / jnp.max(jnp.abs(m0)))}
    return Ck_new, record


def remap_trigger(Ck, basis, Nn, Nm, Np, Ns, shift_on=0.3, width_on=0.05, tail_on=None, last_tail=None,
                  width_ratio=np.sqrt(2), max_shift=1.0, max_narrow=1.1, min_width=1.1):
    """Decide a remap from measured moments, with hysteresis. Call between steps with concrete arrays.

    Per species, the capped moment target (:func:`moment_target`, :func:`cap_target`) is compared
    with the current basis: it fires when ``max_i |U_i - u_i| / a_i > shift_on``, when
    ``max_i |a'_i/a_i - 1| > width_on``, or (if ``tail_on`` is set) when the tail fraction exceeds
    ``tail_on`` and has at least doubled since ``last_tail`` (the value at the previous event).
    A remap resets the moment measures to round-off, so the next event needs the same growth
    again: thresholds act with hysteresis and an unchanged state cannot fire twice. A tail event
    with a negligible move (below a tenth of both thresholds) is suppressed.

    Returns ``(fire, new_basis, info)``: ``fire`` is a Python bool, ``new_basis`` equals ``basis``
    for species that do not fire, ``info`` holds the per-species measures.
    """
    target, sigma = moment_target(Ck, basis, Nn, Nm, Np, Ns, width_ratio)
    capped = np.asarray(cap_target(basis, target, sigma, max_shift, max_narrow, min_width))
    b = np.real(np.asarray(basis))
    a = b[1].reshape(Ns, 3)
    shift = np.max(np.abs(np.asarray(target[0]) - b[0]).reshape(Ns, 3) / a, axis=1)
    width = np.max(np.abs(capped[1].reshape(Ns, 3) / a - 1), axis=1)
    tail = np.asarray(tail_fraction(Ck, Nn, Nm, Np, Ns))
    fire = (shift > shift_on) | (width > width_on)
    if tail_on is not None:
        ref = np.zeros(Ns) if last_tail is None else np.asarray(last_tail)
        moves = (shift > shift_on / 10) | (width > width_on / 10)
        fire |= (tail > tail_on) & (tail > 2 * ref) & moves
    species = np.repeat(fire, 3)
    new = np.where(species[None, :], capped, b)
    return bool(fire.any()), new, {"shift": shift.tolist(), "width": width.tolist(), "tail": tail.tolist(),
                                   "fire": fire.tolist()}
