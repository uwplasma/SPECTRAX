#!/usr/bin/env python3
"""
landau_hermite_jax.py

Single-file, standalone reproduction of the PRL-style Fig.1 2x2 panel
(landau_hermite/Fig1_panel.pdf) using ONLY the fast SOE→MPO/TT machinery,
implemented JAX-first (with a NumPy backend for debugging).

By default (no flags) this script writes:
  - Fig1_panel.pdf / Fig1_panel.png  (collisional relaxation from the manuscript ICs)
with `--nmax 4` by default.

--------------------------------------------------------------------------------
What is being simulated?
--------------------------------------------------------------------------------
This script evolves Hermite expansion coefficients of a velocity-space distribution
under the (spatially homogeneous) Landau / Fokker–Planck collision operator.

We represent the distribution of each species in a tensor-product Hermite basis:

  f(v) = Σ_{n_x,n_y,n_z=0..nmax} f[nx,ny,nz] ψ_{nx}(x) ψ_{ny}(y) ψ_{nz}(z),
  x = v_x / v_th,  (and similarly y,z)

where the 1D basis ψ_n(x) is the *Hermite-Gaussian* used by the coefficient-space
collision algorithm (see `psi_1d`).

--------------------------------------------------------------------------------
Nonlinear vs linearized in this script
--------------------------------------------------------------------------------
Yes: the *main time evolution* uses the **full nonlinear** collision operator.
In coefficient form, the Landau collision operator is bilinear:

  Q_ab(f_a, f_b)   (quadratic in the distributions)

and the self-collision term is Q_aa(f, f), which is nonlinear in f. In the code,
this is exactly what the nonlinear RHS does:

  - 1sp: rhs_self(f) = Q_11(f, f)
  - 2sp: rhs_pair(fa, fb) = (Q_ab(fa, fb), Q_ba(fb, fa))   (cross-only by default)

The script also optionally overlays a **linearized** evolution in the panel.
“Linearized” here means: expand around a Maxwellian background M and keep only
the first-order (tangent) terms in a perturbation h:

  f = M + h,   with ||h|| small
  Q(M+h, M+h) = Q(M,M) + [Q(h,M) + Q(M,h)] + O(h^2)

For a Maxwellian, Q(M,M)=0, so the linearized operator is:

  L(h) = Q(h,M) + Q(M,h)

For 2 species (cross-only), the linearized Jacobian apply is:

  δrhs_a = Q_ab(δf_a, M_b) + Q_ab(M_a, δf_b)
  δrhs_b = Q_ba(δf_b, M_a) + Q_ba(M_b, δf_a)

The CLI flag `--linearized_method` only affects *how we compute those overlays*:

  - `--linearized_method tangent` (default, scalable):
      uses JAX JVPs (Jacobian-vector products) to apply the linearization without
      forming explicit dense matrices. This is “matrix-free tangent linear”.
  - `--linearized_method matrix` (convenient, slower):
      constructs explicit dense Jacobian matrices (via jacfwd or NumPy assembly)
      and uses expm-based propagation.

In all cases, the nonlinear run remains the full nonlinear operator.

This file intentionally does NOT import any internal project modules.

Key options:
  - `--backend {jax,numpy}`: JAX is default; NumPy is for debugging/validation.
  - `--linearized_method {tangent,matrix}`: default `tangent` uses matrix-free JVPs (scales better).
  - `--progress_chunks N`: prints progress during JAX stepping (optional; default keeps fastest single `lax.scan`).
  - `--run_tests`: writes correctness/performance plots to `tests_landau_hermite/`.
  - `--skip_fig1`: skip producing the Fig1 panel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

# Matplotlib cache/config to a local writable dir.
os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))


# ----------------------------
# Backend selection
# ----------------------------

BackendName = Literal["jax", "numpy"]


def _maybe_import_jax(backend: BackendName):
    if backend != "jax":
        return None, None
    import jax  # type: ignore

    # Must be set before heavy JAX work.
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp  # type: ignore

    return jax, jnp


# ----------------------------
# Numerics utilities (NumPy precompute)
# ----------------------------

try:
    from scipy.special import gammaln as _gammaln  # type: ignore
except Exception:  # pragma: no cover
    _gammaln = None


def _logfac_np(n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    if _gammaln is not None:
        return _gammaln(n + 1.0)
    # fallback
    import math as _math

    out = np.empty_like(n)
    it = np.nditer(n, flags=["multi_index"])
    while not it.finished:
        out[it.multi_index] = _math.lgamma(float(it[0]) + 1.0)
        it.iternext()
    return out


def leggauss_01_np(Q: int) -> Tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(Q)
    s = 0.5 * (x + 1.0)
    w01 = 0.5 * w
    return s.astype(np.float64), w01.astype(np.float64)


def t_factor_np(max_n: int) -> np.ndarray:
    """
    t_n = sqrt((2n)!)/(2^n n!), returned for n=0..max_n as float64.
    """
    n = np.arange(max_n + 1, dtype=np.float64)
    return np.exp(0.5 * _logfac_np(2 * n) - n * np.log(2.0) - _logfac_np(n))


def build_a1d_np(kp_max: int, kb_max: int, cos_t: float, sin_t: float) -> np.ndarray:
    """
    a1d(kp,kb) = (-1)^kb cos^kp sin^kb sqrt((kp+kb)!/(kp! kb!))
    shape (kp_max+1, kb_max+1)
    """
    kp = np.arange(kp_max + 1, dtype=np.int64)[:, None]
    kb = np.arange(kb_max + 1, dtype=np.int64)[None, :]
    kp_f = kp.astype(np.float64)
    kb_f = kb.astype(np.float64)
    loga = 0.5 * (_logfac_np(kp_f + kb_f) - _logfac_np(kp_f) - _logfac_np(kb_f))
    base = np.exp(loga)
    sign = np.where((kb % 2) == 0, 1.0, -1.0)
    return sign * (cos_t ** kp_f) * (sin_t ** kb_f) * base


def build_P1D_np(n_dual_max: int, n_prim_max: int, kp_max: int) -> np.ndarray:
    """
    Dense P1D table:
      P[n_dual, n_prim, nprime], with nprime=0..kp_max.
    Matches the reference code's coefficient formula.
    """
    nd = np.arange(n_dual_max + 1, dtype=np.int64)[:, None, None]
    npm = np.arange(n_prim_max + 1, dtype=np.int64)[None, :, None]
    npr = np.arange(kp_max + 1, dtype=np.int64)[None, None, :]

    max_npr = nd + npm
    parity_ok = ((npr - max_npr) % 2) == 0
    in_range = (npr <= max_npr)
    ok = parity_ok & in_range

    r = (max_npr - npr) // 2
    # valid where ok and 0<=r<=min(nd,npm)
    ok = ok & (r >= 0) & (r <= nd) & (r <= npm)

    nd_f = nd.astype(np.float64)
    npm_f = npm.astype(np.float64)
    npr_f = npr.astype(np.float64)
    r_f = r.astype(np.float64)

    # Guard log-factorials against masked-out negatives to avoid spurious NaNs/warnings.
    lf = lambda x: _logfac_np(np.maximum(x, 0.0))

    # logc = log(r!) + log(C(nd,r)) + log(C(npm,r)) + 0.5*log(npr!/(nd! npm!))
    logc = (
        lf(r_f)
        + (lf(nd_f) - lf(r_f) - lf(nd_f - r_f))
        + (lf(npm_f) - lf(r_f) - lf(npm_f - r_f))
        + 0.5 * (lf(npr_f) - lf(nd_f) - lf(npm_f))
    )

    P = np.zeros((n_dual_max + 1, n_prim_max + 1, kp_max + 1), dtype=np.float64)
    P[ok] = np.exp(logc[ok])
    return P


def Fq_table_np(t: np.ndarray, s_nodes: np.ndarray, maxK: int) -> np.ndarray:
    """
    F[q,n] for q=0..Q-1, n=0..maxK:
      F = 0 if n odd else (-1)^{n/2} t_{n/2} s^n.
    """
    Q = int(s_nodes.size)
    n = np.arange(maxK + 1, dtype=np.int64)[None, :]
    even = (n % 2) == 0
    half = n // 2
    base = np.zeros((1, maxK + 1), dtype=np.float64)
    base[0, even[0]] = t[half[0, even[0]]]
    sign = np.where(((half % 2) == 0), 1.0, -1.0).astype(np.float64)
    s = s_nodes.astype(np.float64)[:, None]
    return (sign * base) * (s ** n.astype(np.float64))


def hankel_indices_np(kp_max: int, kb_max: int, shift: int) -> np.ndarray:
    kp = np.arange(kp_max + 1, dtype=np.int64)[:, None]
    kb = np.arange(kb_max + 1, dtype=np.int64)[None, :]
    return kp + kb + int(shift)


def build_M_dim_np(
    *,
    a1d: np.ndarray,
    Fq: np.ndarray,
    idx_hankel: np.ndarray,
    sqrt_kind: str,
    baseK_offset: int,
    kb_col_offset: int,
    extra_kb_sqrt: bool,
) -> np.ndarray:
    """
    Build one 1D matrix M[kp,kb] = a1d[kp,kb+off] * Fq[kp+kb+shift] * S(...)
    with safe handling for the single negative-index corner.
    """
    kp_max = a1d.shape[0] - 1
    kb_max = idx_hankel.shape[1] - 1
    if kb_col_offset != 0:
        cols = np.arange(kb_max + 1, dtype=np.int64) + int(kb_col_offset)
        a_use = a1d[:, cols]
    else:
        a_use = a1d[:, : (kb_max + 1)]

    # Safe Hankel gather: only idx=-1 can occur, and that corner is zeroed by sqrt anyway,
    # but we guard for correctness.
    idx = idx_hankel
    mask = (idx >= 0) & (idx < Fq.size)
    idxc = np.clip(idx, 0, Fq.size - 1)
    H = np.where(mask, Fq[idxc], 0.0)

    kp = np.arange(kp_max + 1, dtype=np.float64)[:, None]
    kb = np.arange(kb_max + 1, dtype=np.float64)[None, :]
    K = kp + kb + float(baseK_offset)

    if sqrt_kind == "none":
        S = 1.0
    elif sqrt_kind == "sqrt_kplus1":
        S = np.sqrt(K + 1.0)
    elif sqrt_kind == "sqrt_kplus2":
        S = np.sqrt(K + 2.0)
    elif sqrt_kind == "sqrt_k":
        S = np.sqrt(np.maximum(K, 0.0))
    elif sqrt_kind == "kplus1":
        S = (K + 1.0)
    elif sqrt_kind == "sqrt_kplus1_times_sqrt_kplus2":
        S = np.sqrt(K + 1.0) * np.sqrt(K + 2.0)
    else:
        raise ValueError(f"unknown sqrt_kind={sqrt_kind}")

    if extra_kb_sqrt:
        S = S * np.sqrt(kb + 1.0)

    return a_use * H * S


def apply_kronecker_3_np(Mx: np.ndarray, My: np.ndarray, Mz: np.ndarray, f: np.ndarray) -> np.ndarray:
    tmp = np.tensordot(Mx, f, axes=(1, 0))  # (kx, py, pz)
    tmp = np.tensordot(My, tmp, axes=(1, 1))  # (ky, kx, pz)
    tmp = np.transpose(tmp, (1, 0, 2))  # (kx, ky, pz)
    out = np.tensordot(Mz, tmp, axes=(1, 2))  # (kz, kx, ky)
    return np.transpose(out, (1, 2, 0))


def einsum_mpo_dot_np(g: np.ndarray, S: np.ndarray, Px: np.ndarray, Py: np.ndarray, Pz: np.ndarray) -> np.ndarray:
    return np.einsum("abc,xad,ybe,zcf,def->xyz", g, Px, Py, Pz, S, optimize=True)


# ----------------------------
# Optional TT helpers (NumPy)
# ----------------------------


@dataclass
class TT3NP:
    """3-core tensor train for a 3D tensor (NumPy)."""

    G1: np.ndarray  # (1, n1, r1)
    G2: np.ndarray  # (r1, n2, r2)
    G3: np.ndarray  # (r2, n3, 1)


def _svd_trunc_np(M: np.ndarray, tol: float, rmax: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    if tol <= 0:
        r = min(int(rmax), int(s.size))
        return U[:, :r], s[:r], Vt[:r, :]
    s2 = s * s
    tot = float(np.sum(s2))
    if tot <= 0:
        r = 1
    else:
        disc = np.cumsum(s2[::-1])[::-1]
        thresh = (tol * tol) * tot
        r = int(s.size)
        for k in range(int(s.size)):
            if float(disc[k]) <= thresh:
                r = int(k)
                break
        r = max(1, r)
    r = min(int(r), int(rmax), int(s.size))
    return U[:, :r], s[:r], Vt[:r, :]


def tt_svd_3d_np(A: np.ndarray, tol: float, rmax: int) -> TT3NP:
    n1, n2, n3 = A.shape
    M = A.reshape(n1, n2 * n3)
    U1, s1, V1t = _svd_trunc_np(M, tol, rmax)
    r1 = int(U1.shape[1])
    M2 = (s1[:, None] * V1t).reshape(r1 * n2, n3)
    U2, s2, V2t = _svd_trunc_np(M2, tol, rmax)
    r2 = int(U2.shape[1])
    G1 = U1.reshape(1, n1, r1)
    G2 = U2.reshape(r1, n2, r2)
    G3 = (s2[:, None] * V2t).reshape(r2, n3, 1)
    return TT3NP(G1=G1, G2=G2, G3=G3)


def tt_to_dense_np(tt: TT3NP) -> np.ndarray:
    tmp = np.tensordot(tt.G1[0, :, :], tt.G2, axes=(1, 0))  # (n1,n2,r2)
    out = np.tensordot(tmp, tt.G3[:, :, 0], axes=(2, 0))  # (n1,n2,n3)
    return out


def _tt_contract_dim_np(G: np.ndarray, P: np.ndarray, S: np.ndarray) -> np.ndarray:
    T = np.einsum("amc,nmk,bkd->abncd", G, P, S, optimize=True)
    rL, sL, n, rR, sR = T.shape
    return T.reshape(rL * sL, n, rR * sR)


def mpo_dot_all_n_tt_np(
    g: np.ndarray,
    S: np.ndarray,
    Px: np.ndarray,
    Py: np.ndarray,
    Pz: np.ndarray,
    *,
    tol: float,
    rmax: int,
) -> np.ndarray:
    g_tt = tt_svd_3d_np(np.asarray(g, dtype=np.float64), tol=tol, rmax=rmax)
    S_tt = tt_svd_3d_np(np.asarray(S, dtype=np.float64), tol=tol, rmax=rmax)

    H1 = _tt_contract_dim_np(g_tt.G1, Px, S_tt.G1)  # (1, nx, r1*s1)
    H2 = _tt_contract_dim_np(g_tt.G2, Py, S_tt.G2)  # (r1*s1, ny, r2*s2)
    H3 = _tt_contract_dim_np(g_tt.G3, Pz, S_tt.G3)  # (r2*s2, nz, 1)
    h_tt = TT3NP(G1=H1.reshape(1, H1.shape[1], H1.shape[2]), G2=H2, G3=H3.reshape(H3.shape[0], H3.shape[1], 1))
    return tt_to_dense_np(h_tt)


# ----------------------------
# Model + tables
# ----------------------------


@dataclass(frozen=True)
class Species:
    m: float
    vth: float


@dataclass(frozen=True)
class ICPositivityInfo:
    f: np.ndarray
    gamma: float
    min_slice: float
    min_plane: float


@dataclass(frozen=True)
class ModelTablesNP:
    # sizes
    nmax: int
    p: int
    p_ext: int
    kp_max: int
    p_kp: int
    Q: int
    maxK: int
    # physics
    ma: float
    mb: float
    vtha: float
    vthb: float
    nu_ab: float
    cos_t: float
    sin_t: float
    ratio: float
    pref_cos: float
    # quadrature
    s_nodes: np.ndarray  # (Q,)
    w_nodes: np.ndarray  # (Q,)
    # coefficient tables
    P1D: np.ndarray  # (p, p_ext, p_kp)
    # build_S constants
    wq_prefI: np.ndarray  # (Q,)
    M_buildS: np.ndarray  # (Q,3,3,4,3,p_kp,p)


def build_model_tables_np(
    *,
    nmax: int,
    Q: int,
    maxK: int,
    ma: float,
    mb: float,
    vtha: float,
    vthb: float,
    nu_ab: float,
) -> ModelTablesNP:
    """
    Precompute all SOE→MPO coefficient tables needed to evaluate Q_ab(fa, fb)
    efficiently for a fixed (nmax, Q, maxK) and fixed species parameters.

    Notes on parameters:
      - `nmax` sets the Hermite truncation per dimension: p = nmax+1.
      - `Q` is the number of Gauss-Legendre nodes used in the SOE separation.
      - `maxK` must be large enough to cover all Hankel indices `kp+kb+shift`
        used by the SOE-based Coulomb moments; keep it comfortably above ~3*nmax.
      - `ma,mb,vtha,vthb,nu_ab` define the collision pair and prefactors.

    Output `ModelTablesNP` is a container of dense NumPy arrays; for JAX it is
    transferred once to device and treated as constants by JIT.
    """
    p = nmax + 1
    p_ext = nmax + 2
    kp_max = 2 * nmax + 1
    p_kp = kp_max + 1

    den = math.sqrt(vtha * vtha + vthb * vthb)
    cos_t = float(vtha / den)
    sin_t = float(vthb / den)
    ratio = float((ma / mb) * (vtha / vthb))
    pref_cos = float(cos_t * nu_ab)

    s_nodes, w_nodes = leggauss_01_np(Q)

    max_half = (maxK // 2) + 6
    t = t_factor_np(max_half)
    Fq = Fq_table_np(t, s_nodes, maxK=maxK)  # (Q, maxK+1)

    a1d_kb = build_a1d_np(kp_max, nmax, cos_t, sin_t)
    a1d_kbext = build_a1d_np(kp_max, nmax + 1, cos_t, sin_t)
    P1D = build_P1D_np(n_dual_max=nmax, n_prim_max=nmax + 1, kp_max=kp_max)

    # Precompute all needed Hankel indices for shifts in [-1..3]
    idx_shift = {sh: hankel_indices_np(kp_max, nmax, sh) for sh in (-1, 0, 1, 2, 3)}

    # Precompute M matrices used to build S1/S2 for all q,i,j,term,dim.
    # term: 0=A (S1), 1=B (S1), 2=A2 (S2), 3=B2 (S2)
    M = np.zeros((Q, 3, 3, 4, 3, p_kp, p), dtype=np.float64)

    def sqrt_kind_for(i: int, j: int, which: str) -> Tuple[str, str, str]:
        # which in {"A","B"} for both S1 and S2 (the sqrt kinds are identical; base offsets differ).
        out = ["none", "none", "none"]
        if which == "A":
            if i == j:
                out[i] = "kplus1"
            else:
                out[j] = "sqrt_kplus1"
                out[i] = "sqrt_k"
        elif which == "B":
            if i == j:
                out[i] = "sqrt_kplus1_times_sqrt_kplus2"
            else:
                out[j] = "sqrt_kplus1"
                out[i] = "sqrt_kplus1"
        else:
            raise ValueError(which)
        return (out[0], out[1], out[2])

    for i in range(3):
        for j in range(3):
            # ---- S1 shifts ----
            shiftA = [0, 0, 0]
            shiftA[j] += 1
            shiftA[i] -= 1

            shiftB = [0, 0, 0]
            shiftB[j] += 1
            shiftB[i] += 1

            sqrtA = sqrt_kind_for(i, j, "A")
            sqrtB = sqrt_kind_for(i, j, "B")

            # ---- S2 base offsets and shifts ----
            base_offset = [0, 0, 0]
            base_offset[j] += 1

            shiftA2 = [base_offset[0], base_offset[1], base_offset[2]]
            shiftA2[j] += 1
            shiftA2[i] -= 1

            shiftB2 = [base_offset[0], base_offset[1], base_offset[2]]
            shiftB2[j] += 1
            shiftB2[i] += 1

            sqrtA2 = sqrtA
            sqrtB2 = sqrtB

            for q in range(Q):
                Fq_q = Fq[q]

                # termA (S1) uses a1d_kb, baseK_offset=0
                for dim in range(3):
                    M[q, i, j, 0, dim] = build_M_dim_np(
                        a1d=a1d_kb,
                        Fq=Fq_q,
                        idx_hankel=idx_shift[int(shiftA[dim])],
                        sqrt_kind=sqrtA[dim],
                        baseK_offset=0,
                        kb_col_offset=0,
                        extra_kb_sqrt=False,
                    )

                # termB (S1)
                for dim in range(3):
                    M[q, i, j, 1, dim] = build_M_dim_np(
                        a1d=a1d_kb,
                        Fq=Fq_q,
                        idx_hankel=idx_shift[int(shiftB[dim])],
                        sqrt_kind=sqrtB[dim],
                        baseK_offset=0,
                        kb_col_offset=0,
                        extra_kb_sqrt=False,
                    )

                # termA2 (S2): use a1d_kbext with kb_col_offset=1 on dim=j, extra sqrt(kb+1) on dim=j
                for dim in range(3):
                    kb_off = 1 if dim == j else 0
                    extra = True if dim == j else False
                    M[q, i, j, 2, dim] = build_M_dim_np(
                        a1d=a1d_kbext,
                        Fq=Fq_q,
                        idx_hankel=idx_shift[int(shiftA2[dim])],
                        sqrt_kind=sqrtA2[dim],
                        baseK_offset=int(base_offset[dim]),
                        kb_col_offset=kb_off,
                        extra_kb_sqrt=extra,
                    )

                # termB2 (S2)
                for dim in range(3):
                    kb_off = 1 if dim == j else 0
                    extra = True if dim == j else False
                    M[q, i, j, 3, dim] = build_M_dim_np(
                        a1d=a1d_kbext,
                        Fq=Fq_q,
                        idx_hankel=idx_shift[int(shiftB2[dim])],
                        sqrt_kind=sqrtB2[dim],
                        baseK_offset=int(base_offset[dim]),
                        kb_col_offset=kb_off,
                        extra_kb_sqrt=extra,
                    )

    pref_I = 2.0 / math.sqrt(math.pi)
    wq_prefI = (pref_I * w_nodes).astype(np.float64)

    return ModelTablesNP(
        nmax=nmax,
        p=p,
        p_ext=p_ext,
        kp_max=kp_max,
        p_kp=p_kp,
        Q=Q,
        maxK=maxK,
        ma=float(ma),
        mb=float(mb),
        vtha=float(vtha),
        vthb=float(vthb),
        nu_ab=float(nu_ab),
        cos_t=cos_t,
        sin_t=sin_t,
        ratio=ratio,
        pref_cos=pref_cos,
        s_nodes=s_nodes,
        w_nodes=w_nodes,
        P1D=P1D.astype(np.float64),
        wq_prefI=wq_prefI,
        M_buildS=M,
    )


def build_ic_fig1_1sp(nmax: int, sp: Species, amp1: float) -> np.ndarray:
    """
    Fig1 1-species IC: a low-order temperature anisotropy (second moments only).

    This mirrors the baseline PRL panel logic:
      f000 = 1/vth^3, and we perturb only the (2,0,0),(0,2,0),(0,0,2) modes.
    """
    p = nmax + 1
    f0 = np.zeros((p, p, p), dtype=np.float64)
    f0[0, 0, 0] = 1.0 / (sp.vth**3)
    if nmax >= 2:
        a = float(amp1)
        f0[2, 0, 0] += 2.0 * a
        f0[0, 2, 0] += -a
        f0[0, 0, 2] += -a
    return f0


def _project_1d_to_psi_basis(
    *,
    nmax: int,
    xgrid: np.ndarray,
    w: np.ndarray,
    g: np.ndarray,
    ridge: float = 0.0,
) -> np.ndarray:
    """
    Project a 1D function g(x) onto span{psi_0..psi_nmax} in a weighted L2 sense:

        argmin_c  ∫ | Σ_n c_n psi_n(x) - g(x) |^2 dx

    This is used only for building optional "two-stream-like" initial conditions
    outside JIT, so a dense grid solve is fine.
    """
    p = int(nmax) + 1
    xgrid = np.asarray(xgrid, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    if xgrid.ndim != 1 or w.ndim != 1 or g.ndim != 1 or xgrid.size != w.size or xgrid.size != g.size:
        raise ValueError("xgrid,w,g must be 1D arrays of the same length")

    psi = np.stack([psi_1d(n, xgrid) for n in range(p)], axis=1)  # (nx,p)
    WT = (w[:, None] * psi)
    G = psi.T @ WT  # (p,p)
    if float(ridge) > 0.0:
        G = G + float(ridge) * np.eye(p, dtype=np.float64)
    b = psi.T @ (w * g)
    return np.linalg.solve(G, b)


def build_ic_fig1_1sp_twostream(
    nmax: int,
    sp: Species,
    u: float,
    *,
    xlim: float = 8.0,
    nx: int = 4001,
    enforce_nonneg: bool = True,
) -> ICPositivityInfo:
    """
    Fig1 alternative 1-species IC: a *two-stream-like* (double-peaked) distribution in vx
    with zero net momentum, constructed as the sum of two shifted Maxwellians:

        f(x,y,z) ∝ 0.5[ M(x-u) + M(x+u) ] M(y) M(z)

    where x=vx/vth and M(x) is the equilibrium Maxwellian used by this script.

    We build the Hermite tensor by projecting the 1D x-dependence onto the truncated
    ψ-basis and (optionally) scaling high modes to enforce nonnegativity on the
    diagnostic slice/plane grids.
    """
    p = nmax + 1
    u = float(u)
    x = np.linspace(-float(xlim), float(xlim), int(nx), dtype=np.float64)
    dx = float(x[1] - x[0])
    w = np.full_like(x, dx, dtype=np.float64)
    w[0] *= 0.5
    w[-1] *= 0.5

    g = 0.5 * (psi_1d(0, x - u) + psi_1d(0, x + u))
    c = _project_1d_to_psi_basis(nmax=nmax, xgrid=x, w=w, g=g, ridge=0.0)  # (p,)
    c0 = float(c[0])
    if not np.isfinite(c0) or abs(c0) < 1e-300:
        raise ValueError("two-stream projection produced invalid c0")

    f000_target = 1.0 / (sp.vth**3)
    scale = f000_target / c0

    f = np.zeros((p, p, p), dtype=np.float64)
    f[:, 0, 0] = scale * c

    # Enforce nonnegativity by scaling the non-Maxwellian part (on diagnostic grids).
    gamma = 1.0
    ms, mp = _min_f_checks_tensor(f, nmax=nmax, xlim=6.0, nx=161)
    if enforce_nonneg:
        fM = np.zeros_like(f); fM[0, 0, 0] = f000_target
        high = f - fM
        f, gamma, ms, mp = _enforce_nonnegativity_by_scaling_high_modes(fM, high, nmax=nmax, xlim=6.0, nx=161, tol=0.0)

    return ICPositivityInfo(f=f, gamma=float(gamma), min_slice=float(ms), min_plane=float(mp))


def build_ic_fig1_1sp_poly4(nmax: int, sp: Species, amp1: float) -> ICPositivityInfo:
    """
    Fig1 1-species IC (default): a strongly non-Maxwellian but positivity-safe distribution built as
    an equilibrium Maxwellian times a positive even polynomial (up to quartic content when nmax>=4).

    This IC is designed to be:
      - farther from Maxwellian than the legacy `prl_m2` second-moment perturbation,
      - robustly nonnegative (the physical-space target is >=0 everywhere),
      - exactly representable for nmax>=4 (quartic terms live in modes up to n=4).

    `amp1` only controls anisotropy between x and y/z; the overall strength is fixed by constants
    chosen to give an obviously far-from-equilibrium shape while remaining well behaved at nmax=4.
    """
    p = nmax + 1
    f000_target = 1.0 / (sp.vth**3)
    if nmax < 2:
        f = np.zeros((p, p, p), dtype=np.float64)
        f[0, 0, 0] = f000_target
        ms, mp = _min_f_checks_tensor(f, nmax=nmax, xlim=6.0, nx=161)
        return ICPositivityInfo(f=f, gamma=1.0, min_slice=ms, min_plane=mp)

    a = float(amp1)
    # Far-from-equilibrium strength: positive coefficients guarantee global nonnegativity in physical space.
    base2 = 0.55
    base4 = 0.32 if nmax >= 4 else 0.0

    a2x = base2 * (1.0 + 0.60 * a)
    a2y = base2 * (1.0 - 0.30 * a)
    a2z = base2 * (1.0 - 0.30 * a)
    a4x = base4 * (1.0 + 0.60 * a)
    a4y = base4 * (1.0 - 0.30 * a)
    a4z = base4 * (1.0 - 0.30 * a)

    f = _poly_ic_tensor_from_coeffs(
        f000_target=f000_target,
        a2x=a2x,
        a2y=a2y,
        a2z=a2z,
        a4x=a4x,
        a4y=a4y,
        a4z=a4z,
        nmax=nmax,
    )
    ms, mp = _min_f_checks_tensor(f, nmax=nmax, xlim=6.0, nx=161)
    return ICPositivityInfo(f=f, gamma=1.0, min_slice=ms, min_plane=mp)


def build_separable_tensor_from_1d_coeffs(c_x: np.ndarray, c_y: np.ndarray, c_z: np.ndarray) -> np.ndarray:
    """
    Given 1D coefficient vectors c_x,c_y,c_z (each length p=nmax+1) in the 1D ψ-basis, build the 3D
    tensor-product coefficient array f[p,p,p] corresponding to the separable function

        f(x,y,z) = (Σ_a c_x[a] ψ_a(x)) (Σ_b c_y[b] ψ_b(y)) (Σ_c c_z[c] ψ_c(z)).
    """
    c_x = np.asarray(c_x, dtype=np.float64)
    c_y = np.asarray(c_y, dtype=np.float64)
    c_z = np.asarray(c_z, dtype=np.float64)
    if c_x.ndim != 1 or c_y.ndim != 1 or c_z.ndim != 1 or c_x.size != c_y.size or c_x.size != c_z.size:
        raise ValueError("c_x,c_y,c_z must be 1D vectors of the same length")
    return np.einsum("a,b,c->abc", c_x, c_y, c_z, optimize=True)


def build_maxwellian_like_tensor_via_projection(
    *,
    nmax: int,
    sp: Species,
    ux_hat: float = 0.0,
    uy_hat: float = 0.0,
    uz_hat: float = 0.0,
    alpha: float = 1.0,
    density: float = 1.0,
    xlim: float = 10.0,
    nx: int = 4001,
) -> np.ndarray:
    """
    Build coefficients for a (possibly drifted / temperature-scaled) Maxwellian-like separable state
    by 1D projection on a dense grid. This is used only in tests (outside JIT).

    In the normalized coordinate x=v/v_th, a Maxwellian with drift u_hat and temperature scaling alpha
    has 1D shape:

        g(x) = (1/alpha) ψ_0((x-u_hat)/alpha),

    where ψ_0 is the equilibrium 1D Maxwellian used by this script (psi_1d(0,x)).
    The 3D distribution is g_x(x) g_y(y) g_z(z), and we rescale so that f000 = density / vth^3.
    """
    p = int(nmax) + 1
    x = np.linspace(-float(xlim), float(xlim), int(nx), dtype=np.float64)
    dx = float(x[1] - x[0])
    w = np.full_like(x, dx, dtype=np.float64)
    w[0] *= 0.5
    w[-1] *= 0.5

    a = float(alpha)
    if not np.isfinite(a) or a <= 0:
        raise ValueError("alpha must be positive")

    def g1(u):
        return (1.0 / a) * psi_1d(0, (x - float(u)) / a)

    cx = _project_1d_to_psi_basis(nmax=nmax, xgrid=x, w=w, g=g1(ux_hat), ridge=0.0)
    cy = _project_1d_to_psi_basis(nmax=nmax, xgrid=x, w=w, g=g1(uy_hat), ridge=0.0)
    cz = _project_1d_to_psi_basis(nmax=nmax, xgrid=x, w=w, g=g1(uz_hat), ridge=0.0)

    c0 = float(cx[0] * cy[0] * cz[0])
    if (not np.isfinite(c0)) or abs(c0) < 1e-300:
        raise ValueError("projection produced invalid c0")

    f000_target = float(density) / (sp.vth**3)
    scale = f000_target / c0
    return scale * build_separable_tensor_from_1d_coeffs(cx, cy, cz)


def _thermal_temperature_from_invariants(inv: np.ndarray, sp: Species) -> float:
    """
    Thermal temperature inferred from invariants by subtracting drift kinetic energy.

    Our invariant vector is inv = [n, Px, Py, Pz, W], where W is the total kinetic energy density:

        W = ∫ (1/2) m |v|^2 f(v) dv.

    For a drifting Maxwellian, W = (1/2) m n |u|^2 + (3/2) n T, so:

        T = (2/3) * (W/n - (1/2) m |u|^2).
    """
    inv = np.asarray(inv, dtype=np.float64)
    n = float(inv[0])
    if not np.isfinite(n) or abs(n) < 1e-300:
        return float("nan")
    P = inv[1:4].astype(np.float64)
    u = P / (float(sp.m) * n)
    W = float(inv[4])
    Wth_per_n = (W / n) - 0.5 * float(sp.m) * float(np.dot(u, u))
    return (2.0 / 3.0) * Wth_per_n


def build_maxwellian_tensor_from_invariants(
    *,
    nmax: int,
    sp: Species,
    inv: np.ndarray,
    xlim: float = 10.0,
    nx: int = 3001,
) -> np.ndarray:
    """
    Build the truncated Hermite coefficient tensor for the (drifting, isotropic) Maxwellian with
    the *same invariants* (n,P,W) as `inv`.

    This is used to define the correct equilibrium background for the linearized operator:
    linearized evolution should relax toward that Maxwellian, not necessarily toward the
    reference equilibrium associated with `sp.vth`.

    Implementation: we use a robust 1D projection on a dense grid (outside JIT).
    """
    inv = np.asarray(inv, dtype=np.float64)
    n = float(inv[0])
    P = inv[1:4].astype(np.float64)
    if not np.isfinite(n) or abs(n) < 1e-300:
        raise ValueError("invalid density in invariants")
    u = P / (float(sp.m) * n)
    ux_hat, uy_hat, uz_hat = (float(u[0] / sp.vth), float(u[1] / sp.vth), float(u[2] / sp.vth))

    T = float(_thermal_temperature_from_invariants(inv, sp))
    if not np.isfinite(T) or T <= 0:
        # Fall back to the raw definition used elsewhere in the script; still yields a sensible scale
        # when drift is essentially zero.
        T = float(temperature_from_invariants(inv))
        T = max(T, 1e-12)

    Teq = _Teq_from_species(sp)
    That = T / max(1e-300, Teq)
    alpha = math.sqrt(max(That, 1e-12))

    return build_maxwellian_like_tensor_via_projection(
        nmax=nmax,
        sp=sp,
        ux_hat=ux_hat,
        uy_hat=uy_hat,
        uz_hat=uz_hat,
        alpha=alpha,
        density=n,
        xlim=float(xlim),
        nx=int(nx),
    )


def build_common_equilibrium_maxwellians_2sp_from_invariants(
    *,
    nmax: int,
    spa: Species,
    spb: Species,
    inva: np.ndarray,
    invb: np.ndarray,
    xlim: float = 10.0,
    nx: int = 3001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Build (Ma, Mb) for the *common* two-species equilibrium Maxwellian implied by the total conserved
    invariants of a cross-collision system:

      - each species keeps its own density n_a, n_b,
      - total momentum is shared via a common drift velocity u_eq,
      - total thermal energy sets a common temperature T_eq.

    Returns (faM_eq, fbM_eq, u_eq_phys, T_eq_phys).
    """
    inva = np.asarray(inva, dtype=np.float64)
    invb = np.asarray(invb, dtype=np.float64)
    na = float(inva[0])
    nb = float(invb[0])
    if (not np.isfinite(na)) or (not np.isfinite(nb)) or abs(na) < 1e-300 or abs(nb) < 1e-300:
        raise ValueError("invalid species density in invariants")

    Ptot = (inva[1:4] + invb[1:4]).astype(np.float64)
    Wtot = float(inva[4] + invb[4])

    Mtot = float(spa.m) * na + float(spb.m) * nb
    u_eq = Ptot / max(1e-300, Mtot)
    # Total thermal energy after removing drift kinetic energy of the common flow.
    Wth_tot = Wtot - 0.5 * Mtot * float(np.dot(u_eq, u_eq))
    T_eq = (2.0 / 3.0) * (Wth_tot / max(1e-300, (na + nb)))
    if not np.isfinite(T_eq) or T_eq <= 0:
        # Conservative fallback: use the average of raw per-species temperatures.
        Ta = float(temperature_from_invariants(inva))
        Tb = float(temperature_from_invariants(invb))
        T_eq = max(0.5 * (Ta + Tb), 1e-12)

    def _alpha_for(sp: Species) -> float:
        Teq = _Teq_from_species(sp)
        That = T_eq / max(1e-300, Teq)
        return math.sqrt(max(That, 1e-12))

    ua_hat = u_eq / float(spa.vth)
    ub_hat = u_eq / float(spb.vth)

    faM_eq = build_maxwellian_like_tensor_via_projection(
        nmax=nmax,
        sp=spa,
        ux_hat=float(ua_hat[0]),
        uy_hat=float(ua_hat[1]),
        uz_hat=float(ua_hat[2]),
        alpha=_alpha_for(spa),
        density=na,
        xlim=float(xlim),
        nx=int(nx),
    )
    fbM_eq = build_maxwellian_like_tensor_via_projection(
        nmax=nmax,
        sp=spb,
        ux_hat=float(ub_hat[0]),
        uy_hat=float(ub_hat[1]),
        uz_hat=float(ub_hat[2]),
        alpha=_alpha_for(spb),
        density=nb,
        xlim=float(xlim),
        nx=int(nx),
    )
    return faM_eq, fbM_eq, u_eq, float(T_eq)


def build_ic_fig1_2sp(nmax: int, spa: Species, spb: Species, Teq: float, dT2: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fig1 2-species IC (positivity-safe and strongly non-Maxwellian):
      - one species is "hotter" than equilibrium (target Teq+|dT2|),
      - both species are non-Maxwellian via a positive isotropic polynomial factor,
      - densities are fixed by f000 = 1/vth^3.

    Construction in physical space (normalized coordinates) is:

        f(x,y,z) = Z * M0(x,y,z) * (1 + a2 r^2 + a4 r^4),   r^2=x^2+y^2+z^2,

    where M0 is the equilibrium Maxwellian (psi0 product) and Z rescales so that f[0,0,0]=1/vth^3.
    For nmax>=4, quartic content (a4>0) is included to excite higher Hermite modes while keeping the
    target shape globally nonnegative.
    """
    p = nmax + 1
    f000a = 1.0 / (spa.vth**3)
    f000b = 1.0 / (spb.vth**3)

    # Base Maxwellians.
    faM = np.zeros((p, p, p), dtype=np.float64); faM[0, 0, 0] = f000a
    fbM = np.zeros((p, p, p), dtype=np.float64); fbM[0, 0, 0] = f000b
    if nmax < 2:
        return faM, fbM

    # Choose which species is heated based on the sign of dT2.
    dT = float(dT2)
    heat_a = (dT >= 0.0)
    Thot = float(Teq) + abs(dT)
    if Thot <= float(Teq) + 1e-15:
        return faM, fbM

    # Choose fixed quartic strengths (bigger => farther from Maxwellian but still positivity-safe).
    a4_hot = 0.14 if nmax >= 4 else 0.0
    a4_cool = 0.08 if nmax >= 4 else 0.0

    # Helper: temperature for an isotropic (a2,a4) polynomial state.
    def _temp_for_a2_a4(a2: float, a4: float, sp: Species, f000: float) -> float:
        f = _poly_ic_tensor_from_coeffs(
            f000_target=f000,
            a2x=a2,
            a2y=a2,
            a2z=a2,
            a4x=a4,
            a4y=a4,
            a4z=a4,
            nmax=nmax,
        )
        return float(temperature_from_invariants(invariants_from_tensor(f, sp)))

    # ----------------------------
    # Hot species: tune a2>=0 to hit Thot (positivity is automatic for a2>=0,a4>=0).
    # ----------------------------
    a2_lo = 0.0
    a2_hi = 5.0
    if heat_a:
        T_lo = _temp_for_a2_a4(a2_lo, a4_hot, spa, f000a)
        T_hi = _temp_for_a2_a4(a2_hi, a4_hot, spa, f000a)
    else:
        T_lo = _temp_for_a2_a4(a2_lo, a4_hot, spb, f000b)
        T_hi = _temp_for_a2_a4(a2_hi, a4_hot, spb, f000b)
    for _ in range(12):
        if T_hi >= Thot:
            break
        a2_hi *= 2.0
        if heat_a:
            T_hi = _temp_for_a2_a4(a2_hi, a4_hot, spa, f000a)
        else:
            T_hi = _temp_for_a2_a4(a2_hi, a4_hot, spb, f000b)

    if T_hi < Thot:
        # Saturate rather than fail.
        a2_star = a2_hi
    else:
        # Bisection for a2 in [lo,hi].
        a2_star = a2_hi
        for _ in range(40):
            mid = 0.5 * (a2_lo + a2_hi)
            if heat_a:
                T_mid = _temp_for_a2_a4(mid, a4_hot, spa, f000a)
            else:
                T_mid = _temp_for_a2_a4(mid, a4_hot, spb, f000b)
            if T_mid >= Thot:
                a2_star = mid
                a2_hi = mid
            else:
                a2_lo = mid

    f_hot = _poly_ic_tensor_from_coeffs(
        f000_target=(f000a if heat_a else f000b),
        a2x=a2_star,
        a2y=a2_star,
        a2z=a2_star,
        a4x=a4_hot,
        a4y=a4_hot,
        a4z=a4_hot,
        nmax=nmax,
    )

    # ----------------------------
    # Cooler species: add non-Maxwellian quartic content but tune a2 (possibly negative) so T≈Teq.
    # Keep global positivity by restricting a2 to satisfy 1 + a2 t + a4 t^2 >= 0 for all t>=0, i.e.
    # a2^2 <= 4 a4 when a4>0.
    # ----------------------------
    def _tuned_cooler(sp: Species, f000: float) -> np.ndarray:
        if a4_cool <= 0.0:
            fM = np.zeros((p, p, p), dtype=np.float64)
            fM[0, 0, 0] = f000
            return fM

        # Bracket a2 in [a2_lo, 0] where a2_lo is the global-positivity limit (slightly inside).
        a2_lim = 0.98 * math.sqrt(max(0.0, 4.0 * a4_cool))
        a2_lo = -a2_lim
        a2_hi = 0.0
        T_lo = _temp_for_a2_a4(a2_lo, a4_cool, sp, f000)
        T_hi = _temp_for_a2_a4(a2_hi, a4_cool, sp, f000)

        # If Teq is outside the bracket, reduce a4 and fall back if needed.
        a4 = float(a4_cool)
        for _ in range(6):
            if min(T_lo, T_hi) <= float(Teq) <= max(T_lo, T_hi):
                break
            a4 *= 0.5
            if a4 < 1e-10:
                fM = np.zeros((p, p, p), dtype=np.float64)
                fM[0, 0, 0] = f000
                return fM
            a2_lim = 0.98 * math.sqrt(max(0.0, 4.0 * a4))
            a2_lo = -a2_lim
            T_lo = _temp_for_a2_a4(a2_lo, a4, sp, f000)
            T_hi = _temp_for_a2_a4(0.0, a4, sp, f000)

        # Bisection to solve T(a2)=Teq on the bracket.
        lo, hi = float(a2_lo), 0.0
        a2_star = 0.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            T_mid = _temp_for_a2_a4(mid, a4, sp, f000)
            if (T_mid >= float(Teq)) == (T_hi >= float(Teq)):
                hi = mid
            else:
                lo = mid
            a2_star = mid
            if abs(hi - lo) < 1e-10:
                break

        return _poly_ic_tensor_from_coeffs(
            f000_target=f000,
            a2x=a2_star,
            a2y=a2_star,
            a2z=a2_star,
            a4x=a4,
            a4y=a4,
            a4z=a4,
            nmax=nmax,
        )

    f_cool = _tuned_cooler(spb if heat_a else spa, f000b if heat_a else f000a)

    if heat_a:
        return f_hot, f_cool
    return f_cool, f_hot


def reconstruct_plane_vx_vy_tensor(f: np.ndarray, nmax: int, xgrid: np.ndarray, ygrid: np.ndarray) -> np.ndarray:
    """
    Reconstruct f(vx,vy,0) on a tensor-product grid in normalized coordinates.
    """
    p = nmax + 1
    psi_x = np.stack([psi_1d(n, xgrid) for n in range(p)], axis=0)  # (p,nx)
    psi_y = np.stack([psi_1d(n, ygrid) for n in range(p)], axis=0)  # (p,ny)
    psi_z0 = np.array([psi_1d(n, np.array([0.0]))[0] for n in range(p)], dtype=np.float64)  # (p,)
    return np.einsum("ijk,ix,jy,k->xy", f[:p, :p, :p], psi_x, psi_y, psi_z0, optimize=True)


def _min_f_checks_tensor(f: np.ndarray, nmax: int, xlim: float = 6.0, nx: int = 161) -> Tuple[float, float]:
    """
    Cheap non-negativity checks for a Hermite-tensor distribution:
      - min on the 1D slice f(vx,0,0) over vx in [-xlim,xlim]
      - min on the 2D plane f(vx,vy,0) over vx,vy in [-xlim,xlim]

    These checks are used only for IC construction / reporting (outside jit).
    """
    x = np.linspace(-xlim, xlim, nx, dtype=np.float64)
    s = reconstruct_slice_vx_tensor(f, nmax=nmax, xgrid=x)
    fmin_slice = float(np.min(s))
    xy = reconstruct_plane_vx_vy_tensor(f, nmax=nmax, xgrid=x, ygrid=x)
    fmin_plane = float(np.min(xy))
    return fmin_slice, fmin_plane


def _enforce_nonnegativity_by_scaling_high_modes(
    base: np.ndarray,
    high: np.ndarray,
    *,
    nmax: int,
    xlim: float = 6.0,
    nx: int = 161,
    tol: float = 0.0,
    max_iter: int = 40,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Find the largest gamma∈[0,1] such that f = base + gamma*high passes the non-negativity
    checks in `_min_f_checks_tensor` (within tolerance `tol`).

    Returns (f, gamma, min_slice, min_plane).
    """
    base = np.asarray(base, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    f0 = base + 0.0 * high
    ms0, mp0 = _min_f_checks_tensor(f0, nmax=nmax, xlim=xlim, nx=nx)
    if (ms0 < -abs(tol)) or (mp0 < -abs(tol)):
        # Baseline itself fails: do not attempt to "fix" it here.
        return f0, 0.0, ms0, mp0

    # Try full strength first.
    f1 = base + 1.0 * high
    ms1, mp1 = _min_f_checks_tensor(f1, nmax=nmax, xlim=xlim, nx=nx)
    if (ms1 >= -abs(tol)) and (mp1 >= -abs(tol)):
        return f1, 1.0, ms1, mp1

    # Bracket gamma in [lo,hi] with hi failing.
    lo = 0.0
    hi = 1.0
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        fm = base + mid * high
        ms, mp = _min_f_checks_tensor(fm, nmax=nmax, xlim=xlim, nx=nx)
        if (ms >= -abs(tol)) and (mp >= -abs(tol)):
            lo = mid
        else:
            hi = mid
        if (hi - lo) < 1e-6:
            break

    gamma = lo
    fout = base + gamma * high
    ms, mp = _min_f_checks_tensor(fout, nmax=nmax, xlim=xlim, nx=nx)
    return fout, float(gamma), float(ms), float(mp)


def _x_operator_matrix_1d(p: int) -> np.ndarray:
    """
    Dense (p x p) matrix representation of multiplication by x in the orthonormal
    Hermite-function basis {psi_n}, using:

      x psi_n = sqrt((n+1)/2) psi_{n+1} + sqrt(n/2) psi_{n-1}.
    """
    X = np.zeros((p, p), dtype=np.float64)
    for n in range(p - 1):
        X[n, n + 1] = math.sqrt((n + 1) / 2.0)
    for n in range(1, p):
        X[n, n - 1] = math.sqrt(n / 2.0)
    return X


def _poly_ic_tensor_from_coeffs(
    *,
    f000_target: float,
    a2x: float,
    a2y: float,
    a2z: float,
    a4x: float,
    a4y: float,
    a4z: float,
    nmax: int,
) -> np.ndarray:
    """
    Build a coefficient tensor for the physical-space distribution:

      f(x,y,z) ∝ psi0(x)psi0(y)psi0(z) * (1 + a2x x^2 + a2y y^2 + a2z z^2 + a4x x^4 + a4y y^4 + a4z z^4)

    If the polynomial factor is positive for all (x,y,z), this distribution is
    nonnegative everywhere at t=0. The script verifies nonnegativity on diagnostic
    grids via `_min_f_checks_tensor` for reporting/guardrails.

    We then scale the whole tensor so that coefficient f[0,0,0] equals f000_target
    (which is the density normalization used by this script).
    """
    p = nmax + 1
    X = _x_operator_matrix_1d(p)
    e0 = np.zeros((p,), dtype=np.float64)
    e0[0] = 1.0
    c0 = e0
    c2 = X @ (X @ e0)
    c4 = X @ (X @ (X @ (X @ e0)))

    def _outer(ax, ay, az):
        return np.einsum("i,j,k->ijk", ax, ay, az, optimize=True)

    f = _outer(c0, c0, c0)
    f = f + float(a2x) * _outer(c2, c0, c0)
    f = f + float(a2y) * _outer(c0, c2, c0)
    f = f + float(a2z) * _outer(c0, c0, c2)
    f = f + float(a4x) * _outer(c4, c0, c0)
    f = f + float(a4y) * _outer(c0, c4, c0)
    f = f + float(a4z) * _outer(c0, c0, c4)

    # Scale so f000 matches target density normalization.
    f000 = float(f[0, 0, 0])
    if not np.isfinite(f000) or abs(f000) < 1e-300:
        raise ValueError("constructed IC has invalid f000")
    f = (float(f000_target) / f000) * f
    return f


# ----------------------------
# Fast RHS (NumPy backend)
# ----------------------------


def build_S_np(fb: np.ndarray, T: ModelTablesNP) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build S1,S2 of shape (3,3,p_kp,p_kp,p_kp) from fb (p,p,p).
    """
    fb = np.asarray(fb, dtype=np.float64)
    S1 = np.zeros((3, 3, T.p_kp, T.p_kp, T.p_kp), dtype=np.float64)
    S2 = np.zeros_like(S1)

    for i in range(3):
        for j in range(3):
            acc1 = np.zeros((T.p_kp, T.p_kp, T.p_kp), dtype=np.float64)
            acc2 = np.zeros_like(acc1)
            for q in range(T.Q):
                wq = float(T.wq_prefI[q])
                # term A,B for S1
                tA = apply_kronecker_3_np(
                    T.M_buildS[q, i, j, 0, 0],
                    T.M_buildS[q, i, j, 0, 1],
                    T.M_buildS[q, i, j, 0, 2],
                    fb,
                )
                tB = apply_kronecker_3_np(
                    T.M_buildS[q, i, j, 1, 0],
                    T.M_buildS[q, i, j, 1, 1],
                    T.M_buildS[q, i, j, 1, 2],
                    fb,
                )
                acc1 += wq * (tA + tB)
                # term A2,B2 for S2
                tA2 = apply_kronecker_3_np(
                    T.M_buildS[q, i, j, 2, 0],
                    T.M_buildS[q, i, j, 2, 1],
                    T.M_buildS[q, i, j, 2, 2],
                    fb,
                )
                tB2 = apply_kronecker_3_np(
                    T.M_buildS[q, i, j, 3, 0],
                    T.M_buildS[q, i, j, 3, 1],
                    T.M_buildS[q, i, j, 3, 2],
                    fb,
                )
                acc2 += wq * (tA2 + tB2)
            S1[i, j] = acc1
            S2[i, j] = acc2
    return S1, S2


def shift_mul_sqrt_np(fa_ext: np.ndarray, j: int) -> np.ndarray:
    p_ext = fa_ext.shape[0]
    sqrt_m = np.sqrt(np.arange(p_ext, dtype=np.float64))
    g = np.zeros_like(fa_ext, dtype=np.float64)
    if j == 0:
        g[1:, :, :] = sqrt_m[1:, None, None] * fa_ext[:-1, :, :]
    elif j == 1:
        g[:, 1:, :] = sqrt_m[None, 1:, None] * fa_ext[:, :-1, :]
    else:
        g[:, :, 1:] = sqrt_m[None, None, 1:] * fa_ext[:, :, :-1]
    return g


def rhs_ab_with_S_np(
    fa: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
    T: ModelTablesNP,
    *,
    use_tt: bool,
    tt_tol: float,
    tt_rmax: int,
) -> np.ndarray:
    """
    Apply rhs_ab using precomputed S1,S2 (already built from fb).
    fa: (p,p,p) -> rhs: (p,p,p)
    """
    fa = np.asarray(fa, dtype=np.float64)
    fa_ext = np.zeros((T.p_ext, T.p_ext, T.p_ext), dtype=np.float64)
    fa_ext[: T.p, : T.p, : T.p] = fa

    Px_ext = T.P1D[:, : T.p_ext, : T.p_kp]
    Py_ext = Px_ext
    Pz_ext = Px_ext
    Px = T.P1D[:, : T.p, : T.p_kp]
    Py = Px
    Pz = Px

    rhs = np.zeros((T.p, T.p, T.p), dtype=np.float64)
    sqrt_k = np.sqrt(np.arange(T.p, dtype=np.float64))

    def add_shifted_i(rhs_acc: np.ndarray, h: np.ndarray, i: int, scale: float) -> None:
        if i == 0:
            rhs_acc[1:, :, :] += scale * (sqrt_k[1:, None, None] * h[:-1, :, :])
        elif i == 1:
            rhs_acc[:, 1:, :] += scale * (sqrt_k[None, 1:, None] * h[:, :-1, :])
        else:
            rhs_acc[:, :, 1:] += scale * (sqrt_k[None, None, 1:] * h[:, :, :-1])

    for i in range(3):
        for j in range(3):
            g_j = shift_mul_sqrt_np(fa_ext, j=j)
            if use_tt:
                h1 = mpo_dot_all_n_tt_np(g_j, S1[i, j], Px_ext, Py_ext, Pz_ext, tol=float(tt_tol), rmax=int(tt_rmax))
                h2 = mpo_dot_all_n_tt_np(fa, S2[i, j], Px, Py, Pz, tol=float(tt_tol), rmax=int(tt_rmax))
            else:
                h1 = einsum_mpo_dot_np(g_j, S1[i, j], Px_ext, Py_ext, Pz_ext)
                h2 = einsum_mpo_dot_np(fa, S2[i, j], Px, Py, Pz)
            add_shifted_i(rhs, h1, i=i, scale=+1.0)
            add_shifted_i(rhs, h2, i=i, scale=-T.ratio)

    return T.pref_cos * rhs


def rhs_ab_np(
    fa: np.ndarray,
    fb: np.ndarray,
    T: ModelTablesNP,
    *,
    use_tt: bool,
    tt_tol: float,
    tt_rmax: int,
) -> np.ndarray:
    """
    Nonlinear Landau RHS for species a due to b:
      rhs_a = Q_ab(fa, fb)
    where Q_ab is bilinear in (fa, fb).

    This is the “full nonlinear operator” building block: self-collisions use
    Q_aa(f, f) and are therefore nonlinear in f.
    """
    S1, S2 = build_S_np(fb, T)
    return rhs_ab_with_S_np(fa, S1, S2, T, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax)


# ----------------------------
# Fast RHS (JAX backend)
# ----------------------------


def build_jax_functions(T: ModelTablesNP):
    """
    Return (rhs_ab_jit, rhs_pair_ab_ba_jit, integrate_1sp, integrate_2sp) for JAX backend.
    """
    jax, jnp = _maybe_import_jax("jax")
    assert jax is not None and jnp is not None

    # Transfer tables to device.
    Tj = {
        "p": T.p,
        "p_ext": T.p_ext,
        "p_kp": T.p_kp,
        "Q": T.Q,
        "ratio": jnp.array(T.ratio, dtype=jnp.float64),
        "pref_cos": jnp.array(T.pref_cos, dtype=jnp.float64),
        "P1D": jnp.asarray(T.P1D, dtype=jnp.float64),
        "wq_prefI": jnp.asarray(T.wq_prefI, dtype=jnp.float64),
        "M_buildS": jnp.asarray(T.M_buildS, dtype=jnp.float64),
    }

    # Precompute sqrt tables (avoid per-call recomputation + helps XLA hoist constants).
    sqrt_p = jnp.sqrt(jnp.arange(Tj["p"], dtype=jnp.float64))
    sqrt_p_ext = jnp.sqrt(jnp.arange(Tj["p_ext"], dtype=jnp.float64))

    def build_S_jax(fb):
        fb = jnp.asarray(fb, dtype=jnp.float64)
        p_kp = Tj["p_kp"]
        p = Tj["p"]

        def apply_kronecker_3_jax(Mx, My, Mz, f):
            tmp = jnp.tensordot(Mx, f, axes=(1, 0))  # (kx, py, pz)
            tmp = jnp.tensordot(My, tmp, axes=(1, 1))  # (ky, kx, pz)
            tmp = jnp.transpose(tmp, (1, 0, 2))  # (kx, ky, pz)
            out = jnp.tensordot(Mz, tmp, axes=(1, 2))  # (kz, kx, ky)
            return jnp.transpose(out, (1, 2, 0))

        # Mq: (3,3,4,3,p_kp,p). Flatten (i,j,term) => B=36 to avoid Python loops in jit.
        def terms_for_q(Mq):
            Mq_flat = Mq.reshape((36, 3, p_kp, p))
            Mx = Mq_flat[:, 0, :, :]
            My = Mq_flat[:, 1, :, :]
            Mz = Mq_flat[:, 2, :, :]
            # terms_flat: (36,p_kp,p_kp,p_kp)
            terms_flat = jax.vmap(apply_kronecker_3_jax, in_axes=(0, 0, 0, None))(Mx, My, Mz, fb)
            return terms_flat.reshape((3, 3, 4, p_kp, p_kp, p_kp))

        terms_q = jax.vmap(terms_for_q, in_axes=0)(Tj["M_buildS"])  # (Q,3,3,4,p_kp,p_kp,p_kp)
        summed = jnp.sum(terms_q * Tj["wq_prefI"][:, None, None, None, None, None, None], axis=0)
        S1 = summed[:, :, 0] + summed[:, :, 1]
        S2 = summed[:, :, 2] + summed[:, :, 3]
        return S1, S2

    def shift_mul_sqrt_jax(fa_ext, j: int):
        p_ext = Tj["p_ext"]
        g = jnp.zeros_like(fa_ext)
        if j == 0:
            g = g.at[1:, :, :].set(sqrt_p_ext[1:, None, None] * fa_ext[:-1, :, :])
        elif j == 1:
            g = g.at[:, 1:, :].set(sqrt_p_ext[None, 1:, None] * fa_ext[:, :-1, :])
        else:
            g = g.at[:, :, 1:].set(sqrt_p_ext[None, None, 1:] * fa_ext[:, :, :-1])
        return g

    def mpo_dot_all_n_jax(g, S, Px, Py, Pz):
        return jnp.einsum("abc,xad,ybe,zcf,def->xyz", g, Px, Py, Pz, S)

    def mpo_dot_all_n_batched_jax(g, S_batched, Px, Py, Pz):
        # g: (m,m,m), S_batched: (B,kp,kp,kp) -> (B,p,p,p)
        return jnp.einsum("abc,xad,ybe,zcf,udef->uxyz", g, Px, Py, Pz, S_batched)

    def rhs_ab_jax(fa, fb):
        p = Tj["p"]
        p_ext = Tj["p_ext"]
        p_kp = Tj["p_kp"]

        fa = jnp.asarray(fa, dtype=jnp.float64)
        fb = jnp.asarray(fb, dtype=jnp.float64)

        S1, S2 = build_S_jax(fb)

        fa_ext = jnp.zeros((p_ext, p_ext, p_ext), dtype=jnp.float64)
        fa_ext = fa_ext.at[:p, :p, :p].set(fa)

        P1D = Tj["P1D"]
        Px_ext = P1D[:, :p_ext, :p_kp]
        Py_ext = Px_ext
        Pz_ext = Px_ext
        Px = P1D[:, :p, :p_kp]
        Py = Px
        Pz = Px

        rhs = jnp.zeros((p, p, p), dtype=jnp.float64)

        def add_shifted_i(rhs_acc, h, i: int, scale):
            if i == 0:
                return rhs_acc.at[1:, :, :].add(scale * (sqrt_p[1:, None, None] * h[:-1, :, :]))
            if i == 1:
                return rhs_acc.at[:, 1:, :].add(scale * (sqrt_p[None, 1:, None] * h[:, :-1, :]))
            return rhs_acc.at[:, :, 1:].add(scale * (sqrt_p[None, None, 1:] * h[:, :, :-1]))

        # Compute all h1_{ij} and h2_{ij} with batching to reduce the number of large einsums.
        # h1 depends on (i,j) through S1[i,j] and on j through the shifted tensor g_j.
        g_stack = jnp.stack([shift_mul_sqrt_jax(fa_ext, j=j) for j in range(3)], axis=0)  # (3,p_ext,p_ext,p_ext)

        h1_list = []
        for j in range(3):
            hij = mpo_dot_all_n_batched_jax(g_stack[j], S1[:, j, :, :, :], Px_ext, Py_ext, Pz_ext)  # (3,p,p,p)
            h1_list.append(hij)
        h1 = jnp.transpose(jnp.stack(h1_list, axis=0), (1, 0, 2, 3, 4))  # (i,j,p,p,p)

        # h2 uses the same g=fa for all (i,j), so batch over all 9 in one contraction.
        S2_flat = S2.reshape((9, p_kp, p_kp, p_kp))
        h2_flat = mpo_dot_all_n_batched_jax(fa, S2_flat, Px, Py, Pz)  # (9,p,p,p)
        h2 = h2_flat.reshape((3, 3, p, p, p))

        # Sum over j before shifting (linearity).
        h1_i = jnp.sum(h1, axis=1)  # (3,p,p,p)
        h2_i = jnp.sum(h2, axis=1)  # (3,p,p,p)
        for i in range(3):
            rhs = add_shifted_i(rhs, h1_i[i], i=i, scale=+1.0)
            rhs = add_shifted_i(rhs, h2_i[i], i=i, scale=-Tj["ratio"])

        return Tj["pref_cos"] * rhs

    rhs_ab_jit = jax.jit(rhs_ab_jax)

    return rhs_ab_jit


# ----------------------------
# Integrators (JAX scan)
# ----------------------------


def build_integrators_jax(rhs_self, rhs_pair, integrator: str):
    jax, jnp = _maybe_import_jax("jax")
    assert jax is not None and jnp is not None

    def step_1sp(f, dt):
        if integrator == "rk2":
            k1 = rhs_self(f)
            k2 = rhs_self(f + 0.5 * dt * k1)
            f_new = f + dt * k2
        elif integrator == "ssprk3":
            u1 = f + dt * rhs_self(f)
            u2 = 0.75 * f + 0.25 * (u1 + dt * rhs_self(u1))
            f_new = (1.0 / 3.0) * f + (2.0 / 3.0) * (u2 + dt * rhs_self(u2))
        elif integrator == "rk4":
            k1 = rhs_self(f)
            k2 = rhs_self(f + 0.5 * dt * k1)
            k3 = rhs_self(f + 0.5 * dt * k2)
            k4 = rhs_self(f + dt * k3)
            f_new = f + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(integrator)
        return f_new, f_new

    def integrate_1sp(f0, dt: float, steps: int):
        dtj = jnp.asarray(dt, dtype=jnp.float64)
        def scan_step(f, _):
            f_new, out = step_1sp(f, dtj)
            return f_new, out
        fs = jax.lax.scan(scan_step, f0, xs=None, length=steps)[1]
        return jnp.concatenate([f0[None, ...], fs], axis=0)

    def step_2sp(state, dt):
        fa, fb = state
        if integrator == "rk2":
            k1a, k1b = rhs_pair(fa, fb)
            k2a, k2b = rhs_pair(fa + 0.5 * dt * k1a, fb + 0.5 * dt * k1b)
            fa_new = fa + dt * k2a
            fb_new = fb + dt * k2b
        elif integrator == "ssprk3":
            k1a, k1b = rhs_pair(fa, fb)
            u1a = fa + dt * k1a
            u1b = fb + dt * k1b
            k2a, k2b = rhs_pair(u1a, u1b)
            u2a = 0.75 * fa + 0.25 * (u1a + dt * k2a)
            u2b = 0.75 * fb + 0.25 * (u1b + dt * k2b)
            k3a, k3b = rhs_pair(u2a, u2b)
            fa_new = (1.0 / 3.0) * fa + (2.0 / 3.0) * (u2a + dt * k3a)
            fb_new = (1.0 / 3.0) * fb + (2.0 / 3.0) * (u2b + dt * k3b)
        elif integrator == "rk4":
            k1a, k1b = rhs_pair(fa, fb)
            k2a, k2b = rhs_pair(fa + 0.5 * dt * k1a, fb + 0.5 * dt * k1b)
            k3a, k3b = rhs_pair(fa + 0.5 * dt * k2a, fb + 0.5 * dt * k2b)
            k4a, k4b = rhs_pair(fa + dt * k3a, fb + dt * k3b)
            fa_new = fa + (dt / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
            fb_new = fb + (dt / 6.0) * (k1b + 2 * k2b + 2 * k3b + k4b)
        else:
            raise ValueError(integrator)
        return (fa_new, fb_new), (fa_new, fb_new)

    def integrate_2sp(fa0, fb0, dt: float, steps: int):
        dtj = jnp.asarray(dt, dtype=jnp.float64)
        def scan_step(state, _):
            return step_2sp(state, dtj)
        (_faT, _fbT), outs = jax.lax.scan(scan_step, (fa0, fb0), xs=None, length=steps)
        fa_hist, fb_hist = outs
        fa_hist = jnp.concatenate([fa0[None, ...], fa_hist], axis=0)
        fb_hist = jnp.concatenate([fb0[None, ...], fb_hist], axis=0)
        return fa_hist, fb_hist

    return integrate_1sp, integrate_2sp


# ----------------------------
# Time stepping helpers (NumPy + JAX progress mode)
# ----------------------------


def integrate_1sp_numpy(rhs_self, f0: np.ndarray, dt: float, steps: int, integrator: str) -> np.ndarray:
    f0 = np.asarray(f0, dtype=np.float64)
    hist = np.zeros((steps + 1,) + f0.shape, dtype=np.float64)
    hist[0] = f0
    f = f0
    for _ in range(int(steps)):
        if integrator == "rk2":
            k1 = rhs_self(f)
            k2 = rhs_self(f + 0.5 * dt * k1)
            f = f + dt * k2
        elif integrator == "ssprk3":
            u1 = f + dt * rhs_self(f)
            u2 = 0.75 * f + 0.25 * (u1 + dt * rhs_self(u1))
            f = (1.0 / 3.0) * f + (2.0 / 3.0) * (u2 + dt * rhs_self(u2))
        elif integrator == "rk4":
            k1 = rhs_self(f)
            k2 = rhs_self(f + 0.5 * dt * k1)
            k3 = rhs_self(f + 0.5 * dt * k2)
            k4 = rhs_self(f + dt * k3)
            f = f + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(integrator)
        hist[_ + 1] = f
    return hist


def integrate_2sp_numpy(rhs_pair, fa0: np.ndarray, fb0: np.ndarray, dt: float, steps: int, integrator: str) -> Tuple[np.ndarray, np.ndarray]:
    fa0 = np.asarray(fa0, dtype=np.float64)
    fb0 = np.asarray(fb0, dtype=np.float64)
    fa_hist = np.zeros((steps + 1,) + fa0.shape, dtype=np.float64)
    fb_hist = np.zeros((steps + 1,) + fb0.shape, dtype=np.float64)
    fa_hist[0] = fa0
    fb_hist[0] = fb0
    fa, fb = fa0, fb0
    for _ in range(int(steps)):
        if integrator == "rk2":
            k1a, k1b = rhs_pair(fa, fb)
            k2a, k2b = rhs_pair(fa + 0.5 * dt * k1a, fb + 0.5 * dt * k1b)
            fa = fa + dt * k2a
            fb = fb + dt * k2b
        elif integrator == "ssprk3":
            k1a, k1b = rhs_pair(fa, fb)
            u1a, u1b = fa + dt * k1a, fb + dt * k1b
            k2a, k2b = rhs_pair(u1a, u1b)
            u2a = 0.75 * fa + 0.25 * (u1a + dt * k2a)
            u2b = 0.75 * fb + 0.25 * (u1b + dt * k2b)
            k3a, k3b = rhs_pair(u2a, u2b)
            fa = (1.0 / 3.0) * fa + (2.0 / 3.0) * (u2a + dt * k3a)
            fb = (1.0 / 3.0) * fb + (2.0 / 3.0) * (u2b + dt * k3b)
        elif integrator == "rk4":
            k1a, k1b = rhs_pair(fa, fb)
            k2a, k2b = rhs_pair(fa + 0.5 * dt * k1a, fb + 0.5 * dt * k1b)
            k3a, k3b = rhs_pair(fa + 0.5 * dt * k2a, fb + 0.5 * dt * k2b)
            k4a, k4b = rhs_pair(fa + dt * k3a, fb + dt * k3b)
            fa = fa + (dt / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
            fb = fb + (dt / 6.0) * (k1b + 2 * k2b + 2 * k3b + k4b)
        else:
            raise ValueError(integrator)
        fa_hist[_ + 1] = fa
        fb_hist[_ + 1] = fb
    return fa_hist, fb_hist


def integrate_1sp_jax_chunked(rhs_self, f0, dt: float, steps: int, integrator: str, chunk: int, *, label: str = "1sp", quiet: bool = False):
    """
    JAX progress mode: run scan in fixed-size chunks, with correctness-preserving
    gating inside the last chunk (no extra physical steps).
    Returns a device array history of shape (steps+1,p,p,p).
    """
    jax, jnp = _maybe_import_jax("jax")
    assert jax is not None and jnp is not None
    chunk = int(chunk)
    if chunk <= 0:
        raise ValueError("chunk must be > 0")

    dtj = jnp.asarray(dt, dtype=jnp.float64)

    def one_step(f):
        if integrator == "rk2":
            k1 = rhs_self(f)
            k2 = rhs_self(f + 0.5 * dtj * k1)
            return f + dtj * k2
        if integrator == "ssprk3":
            u1 = f + dtj * rhs_self(f)
            u2 = 0.75 * f + 0.25 * (u1 + dtj * rhs_self(u1))
            return (1.0 / 3.0) * f + (2.0 / 3.0) * (u2 + dtj * rhs_self(u2))
        if integrator == "rk4":
            k1 = rhs_self(f)
            k2 = rhs_self(f + 0.5 * dtj * k1)
            k3 = rhs_self(f + 0.5 * dtj * k2)
            k4 = rhs_self(f + dtj * k3)
            return f + (dtj / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        raise ValueError(integrator)

    def scan_chunk(carry, _):
        f, remaining = carry
        f_new = jax.lax.cond(remaining > 0, one_step, lambda x: x, f)
        remaining_new = jnp.maximum(remaining - 1, 0)
        return (f_new, remaining_new), f_new

    scan_chunk_jit = jax.jit(lambda f_init, rem: jax.lax.scan(scan_chunk, (f_init, rem), xs=None, length=chunk))

    f = jnp.asarray(f0, dtype=jnp.float64)
    outs = []
    done = 0
    t0 = time.perf_counter()
    while done < steps:
        rem = min(chunk, steps - done)
        (f, _rem2), y = scan_chunk_jit(f, jnp.asarray(rem, dtype=jnp.int32))
        outs.append(y[:rem])
        done += rem
        if not quiet:
            t1 = time.perf_counter()
            print(f"[run:{label}] step={done:6d}/{steps}  elapsed={t1 - t0:.2f}s", flush=True)
    return jnp.concatenate([jnp.asarray(f0, dtype=jnp.float64)[None, ...]] + outs, axis=0)


def integrate_2sp_jax_chunked(rhs_pair, fa0, fb0, dt: float, steps: int, integrator: str, chunk: int, *, label: str = "2sp", quiet: bool = False):
    jax, jnp = _maybe_import_jax("jax")
    assert jax is not None and jnp is not None
    chunk = int(chunk)
    if chunk <= 0:
        raise ValueError("chunk must be > 0")

    dtj = jnp.asarray(dt, dtype=jnp.float64)

    def one_step(state):
        fa, fb = state
        if integrator == "rk2":
            k1a, k1b = rhs_pair(fa, fb)
            k2a, k2b = rhs_pair(fa + 0.5 * dtj * k1a, fb + 0.5 * dtj * k1b)
            return (fa + dtj * k2a, fb + dtj * k2b)
        if integrator == "ssprk3":
            k1a, k1b = rhs_pair(fa, fb)
            u1a, u1b = fa + dtj * k1a, fb + dtj * k1b
            k2a, k2b = rhs_pair(u1a, u1b)
            u2a = 0.75 * fa + 0.25 * (u1a + dtj * k2a)
            u2b = 0.75 * fb + 0.25 * (u1b + dtj * k2b)
            k3a, k3b = rhs_pair(u2a, u2b)
            fa_new = (1.0 / 3.0) * fa + (2.0 / 3.0) * (u2a + dtj * k3a)
            fb_new = (1.0 / 3.0) * fb + (2.0 / 3.0) * (u2b + dtj * k3b)
            return (fa_new, fb_new)
        if integrator == "rk4":
            k1a, k1b = rhs_pair(fa, fb)
            k2a, k2b = rhs_pair(fa + 0.5 * dtj * k1a, fb + 0.5 * dtj * k1b)
            k3a, k3b = rhs_pair(fa + 0.5 * dtj * k2a, fb + 0.5 * dtj * k2b)
            k4a, k4b = rhs_pair(fa + dtj * k3a, fb + dtj * k3b)
            fa_new = fa + (dtj / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
            fb_new = fb + (dtj / 6.0) * (k1b + 2 * k2b + 2 * k3b + k4b)
            return (fa_new, fb_new)
        raise ValueError(integrator)

    def scan_chunk(carry, _):
        (fa, fb), remaining = carry
        (fa_new, fb_new) = jax.lax.cond(remaining > 0, one_step, lambda s: s, (fa, fb))
        remaining_new = jnp.maximum(remaining - 1, 0)
        return ((fa_new, fb_new), remaining_new), (fa_new, fb_new)

    scan_chunk_jit = jax.jit(lambda fa_init, fb_init, rem: jax.lax.scan(scan_chunk, ((fa_init, fb_init), rem), xs=None, length=chunk))

    fa = jnp.asarray(fa0, dtype=jnp.float64)
    fb = jnp.asarray(fb0, dtype=jnp.float64)
    out_a = []
    out_b = []
    done = 0
    t0 = time.perf_counter()
    while done < steps:
        rem = min(chunk, steps - done)
        ((fa, fb), _rem2), y = scan_chunk_jit(fa, fb, jnp.asarray(rem, dtype=jnp.int32))
        ya, yb = y
        out_a.append(ya[:rem])
        out_b.append(yb[:rem])
        done += rem
        if not quiet:
            t1 = time.perf_counter()
            print(f"[run:{label}] step={done:6d}/{steps}  elapsed={t1 - t0:.2f}s", flush=True)
    fa_hist = jnp.concatenate([jnp.asarray(fa0, dtype=jnp.float64)[None, ...]] + out_a, axis=0)
    fb_hist = jnp.concatenate([jnp.asarray(fb0, dtype=jnp.float64)[None, ...]] + out_b, axis=0)
    return fa_hist, fb_hist


# ----------------------------
# Linearized (Maxwellian-background) RHS builders
# ----------------------------


def make_linearized_rhs_1sp_numpy(T: ModelTablesNP, fM: np.ndarray, *, use_tt: bool, tt_tol: float, tt_rmax: int):
    """
    Return L(h) = Q(h,fM) + Q(fM,h) using the same fast SOE→MPO machinery.
    """
    fM = np.asarray(fM, dtype=np.float64)
    S1M, S2M = build_S_np(fM, T)

    def L(h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=np.float64)
        term1 = rhs_ab_with_S_np(h, S1M, S2M, T, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax)
        S1h, S2h = build_S_np(h, T)
        term2 = rhs_ab_with_S_np(fM, S1h, S2h, T, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax)
        return term1 + term2

    return L


def make_linearized_rhs_2sp_numpy(
    Tab: ModelTablesNP,
    Tba: ModelTablesNP,
    faM: np.ndarray,
    fbM: np.ndarray,
    *,
    use_tt: bool,
    tt_tol: float,
    tt_rmax: int,
):
    """
    Cross-only 2sp Jacobian apply:
      δrhs_a = Q_ab(δfa, fbM) + Q_ab(faM, δfb)
      δrhs_b = Q_ba(δfb, faM) + Q_ba(fbM, δfa)
    """
    faM = np.asarray(faM, dtype=np.float64)
    fbM = np.asarray(fbM, dtype=np.float64)
    S1_bM, S2_bM = build_S_np(fbM, Tab)
    S1_aM, S2_aM = build_S_np(faM, Tba)

    def J_apply(dfa: np.ndarray, dfb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dfa = np.asarray(dfa, dtype=np.float64)
        dfb = np.asarray(dfb, dtype=np.float64)

        # δrhs_a
        term_a1 = rhs_ab_with_S_np(dfa, S1_bM, S2_bM, Tab, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax)
        S1_dfb, S2_dfb = build_S_np(dfb, Tab)
        term_a2 = rhs_ab_with_S_np(faM, S1_dfb, S2_dfb, Tab, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax)

        # δrhs_b
        term_b1 = rhs_ab_with_S_np(dfb, S1_aM, S2_aM, Tba, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax)
        S1_dfa, S2_dfa = build_S_np(dfa, Tba)
        term_b2 = rhs_ab_with_S_np(fbM, S1_dfa, S2_dfa, Tba, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax)

        return (term_a1 + term_a2), (term_b1 + term_b2)

    return J_apply


def make_linearized_rhs_1sp_jax(rhs11_jit, fM):
    jax, jnp = _maybe_import_jax("jax")
    assert jax is not None and jnp is not None

    fM = jnp.asarray(fM, dtype=jnp.float64)
    p = fM.shape[0]
    fM_flat = fM.reshape(-1)

    def rhs_self_flat(x_flat):
        x = x_flat.reshape(p, p, p)
        y = rhs11_jit(x, x)
        return y.reshape(-1)

    def L_apply_flat(h_flat):
        return jax.jvp(rhs_self_flat, (fM_flat,), (h_flat,))[1]

    L_apply_flat = jax.jit(L_apply_flat)

    def L(h):
        h = jnp.asarray(h, dtype=jnp.float64)
        return L_apply_flat(h.reshape(-1)).reshape(p, p, p)

    return jax.jit(L)


def make_linearized_rhs_2sp_jax(rhsab_jit, rhsba_jit, faM, fbM):
    jax, jnp = _maybe_import_jax("jax")
    assert jax is not None and jnp is not None

    faM = jnp.asarray(faM, dtype=jnp.float64)
    fbM = jnp.asarray(fbM, dtype=jnp.float64)
    p = faM.shape[0]
    Na = p**3
    y0 = jnp.concatenate([faM.reshape(-1), fbM.reshape(-1)], axis=0)

    def rhs_pair_flat(y_flat):
        ya = y_flat[:Na].reshape(p, p, p)
        yb = y_flat[Na:].reshape(p, p, p)
        dya = rhsab_jit(ya, yb)
        dyb = rhsba_jit(yb, ya)
        return jnp.concatenate([dya.reshape(-1), dyb.reshape(-1)], axis=0)

    def J_apply_flat(dy_flat):
        return jax.jvp(rhs_pair_flat, (y0,), (dy_flat,))[1]

    J_apply_flat = jax.jit(J_apply_flat)

    def J_apply(dfa, dfb):
        dy = jnp.concatenate([jnp.asarray(dfa, dtype=jnp.float64).reshape(-1), jnp.asarray(dfb, dtype=jnp.float64).reshape(-1)], axis=0)
        out = J_apply_flat(dy)
        dfa_out = out[:Na].reshape(p, p, p)
        dfb_out = out[Na:].reshape(p, p, p)
        return dfa_out, dfb_out

    return jax.jit(J_apply)


# ----------------------------
# Diagnostics + plotting (NumPy)
# ----------------------------


def invariants_from_tensor(f: np.ndarray, sp: Species) -> np.ndarray:
    vth = sp.vth
    m = sp.m
    f000 = float(f[0, 0, 0])
    n = (vth**3) * f000
    px = (m * vth**4) * (float(f[1, 0, 0]) / math.sqrt(2.0)) if f.shape[0] > 1 else 0.0
    py = (m * vth**4) * (float(f[0, 1, 0]) / math.sqrt(2.0)) if f.shape[1] > 1 else 0.0
    pz = (m * vth**4) * (float(f[0, 0, 1]) / math.sqrt(2.0)) if f.shape[2] > 1 else 0.0
    mom2 = 1.5 * f000
    if f.shape[0] > 2:
        mom2 += float(f[2, 0, 0]) / math.sqrt(2.0)
    if f.shape[1] > 2:
        mom2 += float(f[0, 2, 0]) / math.sqrt(2.0)
    if f.shape[2] > 2:
        mom2 += float(f[0, 0, 2]) / math.sqrt(2.0)
    W = 0.5 * m * (vth**5) * mom2
    return np.array([n, px, py, pz, W], dtype=np.float64)


def temperature_from_invariants(inv: np.ndarray) -> float:
    n = float(inv[0])
    W = float(inv[4])
    return (2.0 / 3.0) * (W / n)


def temperature_components_hat_from_tensor(f: np.ndarray) -> Tuple[float, float, float]:
    f000 = float(f[0, 0, 0])
    if not np.isfinite(f000) or abs(f000) < 1e-300:
        return (float("nan"), float("nan"), float("nan"))
    f100 = float(f[1, 0, 0]) if f.shape[0] > 1 else 0.0
    f010 = float(f[0, 1, 0]) if f.shape[1] > 1 else 0.0
    f001 = float(f[0, 0, 1]) if f.shape[2] > 1 else 0.0

    ux = f100 / (math.sqrt(2.0) * f000)
    uy = f010 / (math.sqrt(2.0) * f000)
    uz = f001 / (math.sqrt(2.0) * f000)

    f200 = float(f[2, 0, 0]) if f.shape[0] > 2 else 0.0
    f020 = float(f[0, 2, 0]) if f.shape[1] > 2 else 0.0
    f002 = float(f[0, 0, 2]) if f.shape[2] > 2 else 0.0

    x2 = (0.5 * f000 + f200 / math.sqrt(2.0)) / f000
    y2 = (0.5 * f000 + f020 / math.sqrt(2.0)) / f000
    z2 = (0.5 * f000 + f002 / math.sqrt(2.0)) / f000
    Tx = x2 - ux * ux
    Ty = y2 - uy * uy
    Tz = z2 - uz * uz
    return Tx, Ty, Tz


def anisotropy_measure_from_tensor(f: np.ndarray) -> float:
    Tx, Ty, Tz = temperature_components_hat_from_tensor(f)
    Tavg = (Tx + Ty + Tz) / 3.0
    if not np.isfinite(Tavg) or abs(Tavg) < 1e-300:
        return float("nan")
    return (Tz - Tx) / Tavg


def hermite_phys(n: int, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    Hnm1 = np.ones_like(x)
    Hn = 2.0 * x
    for k in range(1, n):
        Hnp1 = 2.0 * x * Hn - 2.0 * k * Hnm1
        Hnm1, Hn = Hn, Hnp1
    return Hn


def psi_1d(n: int, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (1.0 / math.sqrt(math.pi)) * hermite_phys(n, x) * np.exp(-(x**2)) / math.sqrt(
        (2.0**n) * math.factorial(n)
    )


def reconstruct_slice_vx_tensor(f: np.ndarray, nmax: int, xgrid: np.ndarray) -> np.ndarray:
    psi0 = np.array([psi_1d(m, np.array([0.0]))[0] for m in range(nmax + 1)], dtype=np.float64)
    f_eff = np.zeros(nmax + 1, dtype=np.float64)
    for n in range(nmax + 1):
        # sum_{m,p} f[n,m,p] psi_m(0) psi_p(0)
        f_eff[n] = float(np.sum(f[n, :, :] * (psi0[:, None] * psi0[None, :])))
    out = np.zeros_like(xgrid, dtype=np.float64)
    for n in range(nmax + 1):
        out += f_eff[n] * psi_1d(n, xgrid)
    return out


def local_maxwellian_slice_vx_from_tensor(f: np.ndarray, xgrid: np.ndarray) -> np.ndarray:
    """
    Value of the *local Maxwellian* M(x,y,z) evaluated on the slice (x,0,0),
    where x=vx/vth, using moments extracted from the Hermite tensor.
    """
    f000 = float(f[0, 0, 0])
    if not np.isfinite(f000) or abs(f000) < 1e-300:
        return np.full_like(xgrid, float("nan"), dtype=np.float64)

    ux = float(f[1, 0, 0]) / (math.sqrt(2.0) * f000) if f.shape[0] > 1 else 0.0
    uy = float(f[0, 1, 0]) / (math.sqrt(2.0) * f000) if f.shape[1] > 1 else 0.0
    uz = float(f[0, 0, 1]) / (math.sqrt(2.0) * f000) if f.shape[2] > 1 else 0.0

    Tx_hat, Ty_hat, Tz_hat = temperature_components_hat_from_tensor(f)
    Tiso = float((Tx_hat + Ty_hat + Tz_hat) / 3.0)
    Tiso = max(Tiso, 1e-12)

    pref = f000 / ((2.0 * math.pi * Tiso) ** 1.5)
    r2 = (xgrid - ux) ** 2 + (0.0 - uy) ** 2 + (0.0 - uz) ** 2
    return pref * np.exp(-0.5 * r2 / Tiso)


def _Teq_from_species(sp: Species) -> float:
    """
    Reference equilibrium temperature associated with `sp.vth` under the convention
    used throughout this script:

        v_th = sqrt(2 T_eq / m)  =>  T_eq = (m v_th^2)/2

    This is used to convert a *physical* temperature inferred from invariants into
    the dimensionless "hat" temperature that appears in the normalized coordinate
    x = v_x / v_th.
    """
    return 0.5 * float(sp.m) * float(sp.vth) * float(sp.vth)


def maxwellian_slice_vx_from_invariants(inv: np.ndarray, sp: Species, xgrid: np.ndarray) -> np.ndarray:
    """
    Maxwellian slice M(x,0,0) constructed from conserved moments (n, P, W):

      - density n
      - drift velocity u = P/(m n)
      - isotropic temperature T = (2/3) W/n

    evaluated in normalized coordinate x = v_x / v_th (i.e. xgrid is dimensionless).
    """
    inv = np.asarray(inv, dtype=np.float64)
    n = float(inv[0])
    if not np.isfinite(n) or abs(n) < 1e-300:
        return np.full_like(xgrid, float("nan"), dtype=np.float64)
    px, py, pz = (float(inv[1]), float(inv[2]), float(inv[3]))

    ux = px / (float(sp.m) * n)
    uy = py / (float(sp.m) * n)
    uz = pz / (float(sp.m) * n)
    ux_hat = ux / float(sp.vth)
    uy_hat = uy / float(sp.vth)
    uz_hat = uz / float(sp.vth)

    # Physical temperature T = (2/3) W/n.
    # In the normalized coordinate x=v/v_th with v_th = sqrt(2 T_eq / m),
    # the Maxwellian is:
    #   f(x) ∝ exp(-(x-u_hat)^2 / T_hat), where T_hat = T / T_eq,
    # equivalently exp(-0.5 (x-u_hat)^2 / Tvar) with Tvar = T_hat/2.
    T = float(temperature_from_invariants(inv))
    Teq = _Teq_from_species(sp)
    That = T / max(1e-300, Teq)
    Tvar = max(That / 2.0, 1e-12)

    f000 = n / (float(sp.vth) ** 3)
    pref = f000 / ((2.0 * math.pi * Tvar) ** 1.5)
    r2 = (xgrid - ux_hat) ** 2 + (0.0 - uy_hat) ** 2 + (0.0 - uz_hat) ** 2
    return pref * np.exp(-0.5 * r2 / Tvar)


def slice_deviation_timeseries(f_hist: np.ndarray, *, nmax: int, xgrid: np.ndarray) -> np.ndarray:
    """
    Scalar measure of how far the vx-slice is from a local Maxwellian over time:

        dev(t) = || f(vx,0,0,t) - M(vx,0,0,t) ||_2 / || M(vx,0,0,t) ||_2
    """
    dx = float(xgrid[1] - xgrid[0]) if xgrid.size > 1 else 1.0
    w = math.sqrt(dx)
    out = np.zeros((f_hist.shape[0],), dtype=np.float64)
    for i in range(f_hist.shape[0]):
        s = reconstruct_slice_vx_tensor(f_hist[i], nmax=nmax, xgrid=xgrid)
        m = local_maxwellian_slice_vx_from_tensor(f_hist[i], xgrid=xgrid)
        num = float(np.linalg.norm((s - m) * w))
        den = float(np.linalg.norm(m * w)) + 1e-300
        out[i] = num / den
    return out


def value_at_origin_tensor(f: np.ndarray, *, nmax: int) -> float:
    """
    Evaluate the reconstructed distribution at (x,y,z)=(0,0,0) in normalized coordinates:

        f(0,0,0) = Σ_{n,m,p} f[n,m,p] ψ_n(0) ψ_m(0) ψ_p(0).
    """
    p = nmax + 1
    psi0 = np.array([psi_1d(n, np.array([0.0]))[0] for n in range(p)], dtype=np.float64)
    return float(np.einsum("nmp,n,m,p->", f[:p, :p, :p], psi0, psi0, psi0, optimize=True))


def reconstruct_marginal_vx_tensor(f: np.ndarray, nmax: int, xgrid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the 1D marginal (in x=vx/vth units):

        g(x) = ∫∫ f(x,y,z) dy dz

    In the product basis ψ_n(x)ψ_m(y)ψ_p(z), the y,z integrals annihilate all
    modes with m>0 or p>0 (they integrate to 0), leaving:

        g(x) = Σ_n f[n,0,0] ψ_n(x).

    Therefore ∫ g(x) dx = f[0,0,0], i.e. the area under the marginal is the
    (dimensionless) density coefficient and should be conserved by collisions.
    """
    out = np.zeros_like(xgrid, dtype=np.float64)
    for n in range(nmax + 1):
        out += float(f[n, 0, 0]) * psi_1d(n, xgrid)
    return out


def rel_entropy_KL_local_maxwellian_grid_tensor(
    f: np.ndarray,
    sp: Species,
    nmax: int,
    xlim: float,
    nx: int,
    eps: float = 1e-30,
) -> float:
    grid = prepare_entropy_grid(nmax=nmax, xlim=xlim, nx=nx)
    return rel_entropy_KL_local_maxwellian_grid_tensor_precomp(f, sp, grid, eps=eps)


@dataclass(frozen=True)
class EntropyGrid:
    nmax: int
    x: np.ndarray          # (nx,)
    dx: float
    psi: np.ndarray        # (p,nx)
    X: np.ndarray          # (nx,nx,nx)
    Y: np.ndarray          # (nx,nx,nx)
    Z: np.ndarray          # (nx,nx,nx)


def prepare_entropy_grid(*, nmax: int, xlim: float, nx: int) -> EntropyGrid:
    x = np.linspace(-float(xlim), float(xlim), int(nx))
    dx = float(x[1] - x[0])
    psi = np.zeros((nmax + 1, x.size), dtype=np.float64)
    for n in range(nmax + 1):
        psi[n, :] = psi_1d(n, x)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return EntropyGrid(nmax=int(nmax), x=x, dx=dx, psi=psi, X=X, Y=Y, Z=Z)


def rel_entropy_KL_local_maxwellian_grid_tensor_precomp(
    f: np.ndarray,
    sp: Species,
    grid: EntropyGrid,
    eps: float = 1e-30,
) -> float:
    nmax = int(grid.nmax)

    # reconstruct f(x,y,z)
    fxyz = np.einsum("nmp,nx,my,pz->xyz", f, grid.psi, grid.psi, grid.psi, optimize=True)
    fxyz = np.maximum(fxyz, eps)

    f000 = float(f[0, 0, 0])
    if not np.isfinite(f000) or abs(f000) < 1e-300:
        return float("nan")

    ux = float(f[1, 0, 0]) / (math.sqrt(2.0) * f000) if nmax >= 1 else 0.0
    uy = float(f[0, 1, 0]) / (math.sqrt(2.0) * f000) if nmax >= 1 else 0.0
    uz = float(f[0, 0, 1]) / (math.sqrt(2.0) * f000) if nmax >= 1 else 0.0

    Tx_hat, Ty_hat, Tz_hat = temperature_components_hat_from_tensor(f)
    Tiso = float((Tx_hat + Ty_hat + Tz_hat) / 3.0)
    Tiso = max(Tiso, 1e-12)

    r2 = (grid.X - ux) ** 2 + (grid.Y - uy) ** 2 + (grid.Z - uz) ** 2
    pref = f000 / ((2.0 * math.pi * Tiso) ** 1.5)
    Mxyz = pref * np.exp(-0.5 * r2 / Tiso)
    Mxyz = np.maximum(Mxyz, eps)

    Dhat = float(np.sum(fxyz * (np.log(fxyz) - np.log(Mxyz))) * (grid.dx**3))
    return (sp.vth**3) * Dhat


def linear_free_energy_grid_tensor_precomp(
    f: np.ndarray,
    fM: np.ndarray,
    sp: Species,
    grid: EntropyGrid,
    eps: float = 1e-30,
) -> float:
    """
    Quadratic "free energy" used as a Lyapunov functional for *linearized* evolution about a
    fixed Maxwellian background fM:

        F(f; fM) = ∫ (f - fM)^2 / fM dv

    Notes:
      - KL divergence D(f||M_local) is the right entropy diagnostic for the *nonlinear* operator,
        but it is not expected to be monotone along tangent-linear dynamics (and can even be
        undefined if the linearized evolution produces small negative f).
      - This quadratic functional is standard in linearized kinetic theory (it corresponds to
        the 2nd variation of entropy at a Maxwellian).
    """
    # reconstruct f(x,y,z)
    fxyz = np.einsum("nmp,nx,my,pz->xyz", f, grid.psi, grid.psi, grid.psi, optimize=True)

    # reconstruct background Maxwellian M(x,y,z) from fM moments (should be u=0, Tvar=0.5).
    fM000 = float(fM[0, 0, 0])
    if not np.isfinite(fM000) or abs(fM000) < 1e-300:
        return float("nan")
    ux = float(fM[1, 0, 0]) / (math.sqrt(2.0) * fM000) if grid.nmax >= 1 else 0.0
    uy = float(fM[0, 1, 0]) / (math.sqrt(2.0) * fM000) if grid.nmax >= 1 else 0.0
    uz = float(fM[0, 0, 1]) / (math.sqrt(2.0) * fM000) if grid.nmax >= 1 else 0.0
    Tx_hat, Ty_hat, Tz_hat = temperature_components_hat_from_tensor(fM)
    Tvar = float((Tx_hat + Ty_hat + Tz_hat) / 3.0)
    Tvar = max(Tvar, 1e-12)
    r2 = (grid.X - ux) ** 2 + (grid.Y - uy) ** 2 + (grid.Z - uz) ** 2
    pref = fM000 / ((2.0 * math.pi * Tvar) ** 1.5)
    Mxyz = pref * np.exp(-0.5 * r2 / Tvar)
    Mxyz = np.maximum(Mxyz, eps)

    h = fxyz - Mxyz
    Fhat = float(np.sum((h * h) / Mxyz) * (grid.dx**3))
    return (sp.vth**3) * Fhat


def make_fig1_panel(
    *,
    outprefix: str,
    # 1sp dynamics
    sp1: Species,
    t1: np.ndarray,
    f_hist1: np.ndarray,
    f_hist1_lin: Optional[np.ndarray],
    inv1: np.ndarray,
    inv1_lin: Optional[np.ndarray],
    D1_ratio: np.ndarray,
    D1_ratio_lin: Optional[np.ndarray],
    # 2sp dynamics
    spa: Species,
    spb: Species,
    t2: np.ndarray,
    fa_hist: np.ndarray,
    fb_hist: np.ndarray,
    fa_hist_lin: Optional[np.ndarray],
    fb_hist_lin: Optional[np.ndarray],
    inv2a: np.ndarray,
    inv2b: np.ndarray,
    inv2a_lin: Optional[np.ndarray],
    inv2b_lin: Optional[np.ndarray],
    D2tot_ratio: np.ndarray,
    D2tot_ratio_lin: Optional[np.ndarray],
    nmax: int,
) -> None:
    """
    Create a 2x2 panel used for *both* Fig.1 and Fig.2 runs.

    Panels (user-facing meaning):
      (a) Shared relaxation metrics (no log scale): A(t)/A0 (1sp), (Ta-Tb)/(Ta-Tb)0 (2sp),
          and entropy ratios for all cases (1sp/2sp, nonlinear/linearized).
      (b) Conserved-quantity drifts (|Δn|, |ΔP|, |ΔW|) for all cases (1sp/2sp, nonlinear/linearized).
      (c) 1sp vx-slice f(vx,0,0): IC, snapshot at A/A0≈0.3 (nonlinear + linearized), and the local Maxwellian.
      (d) 2sp vx-slices fa(vx,0,0) and fb(vx,0,0): IC, snapshot at (Ta-Tb)/(Ta-Tb)0≈0.3
          (nonlinear + linearized), and local Maxwellians.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.35,
            "axes.linewidth": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    def _tidy(ax):
        ax.tick_params(top=True, right=True)

    fig = plt.figure(figsize=(4.6, 5.2), dpi=300)
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.995, bottom=0.05, top=0.95, wspace=0.25, hspace=0.25)

    def _drifts_1sp(inv: np.ndarray, *, mass: float, vth: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n0 = float(inv[0, 0])
        P0 = inv[0, 1:4].astype(np.float64)
        W0 = float(inv[0, 4])
        dn = np.abs(inv[:, 0] - n0) / max(1e-30, abs(n0))
        Pscale = max(1e-30, abs(mass * n0 * vth))
        dP = np.linalg.norm(inv[:, 1:4] - P0[None, :], axis=1) / Pscale
        dW = np.abs(inv[:, 4] - W0) / max(1e-30, abs(W0))
        return dn, dP, dW

    def _drifts_2sp_tot(inv2a_: np.ndarray, inv2b_: np.ndarray, *, PscaleT: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        invT = inv2a_ + inv2b_
        dn = np.abs(invT[:, 0] - invT[0, 0]) / max(1e-30, abs(invT[0, 0]))
        dP = np.linalg.norm(invT[:, 1:4] - invT[0, 1:4][None, :], axis=1) / max(1e-30, PscaleT)
        dW = np.abs(invT[:, 4] - invT[0, 4]) / max(1e-30, abs(invT[0, 4]))
        return dn, dP, dW

    def _idx_near(y: np.ndarray, target: float) -> int:
        y = np.asarray(y, dtype=np.float64)
        good = np.isfinite(y)
        if not np.any(good):
            return 0
        j = int(np.argmin(np.abs(y[good] - float(target))))
        return int(np.flatnonzero(good)[j])

    # 1sp anisotropy ratio
    A_nl = np.array([anisotropy_measure_from_tensor(f_hist1[i]) for i in range(len(t1))], dtype=np.float64)
    A0 = A_nl[0] if (np.isfinite(A_nl[0]) and abs(A_nl[0]) > 1e-30) else 1.0
    A_ratio_nl = A_nl / A0
    if f_hist1_lin is not None:
        A_lin = np.array([anisotropy_measure_from_tensor(f_hist1_lin[i]) for i in range(len(t1))], dtype=np.float64)
        A0_lin = A_lin[0] if (np.isfinite(A_lin[0]) and abs(A_lin[0]) > 1e-30) else 1.0
        A_ratio_lin = A_lin / A0_lin
    else:
        A_ratio_lin = None

    # 2sp temperature exchange
    Ta_nl = np.array([temperature_from_invariants(inv2a[i]) for i in range(inv2a.shape[0])], dtype=np.float64)
    Tb_nl = np.array([temperature_from_invariants(inv2b[i]) for i in range(inv2b.shape[0])], dtype=np.float64)
    dT_nl = Ta_nl - Tb_nl
    dT0 = dT_nl[0] if (np.isfinite(dT_nl[0]) and abs(dT_nl[0]) > 1e-30) else 1.0
    dT_ratio_nl = dT_nl / dT0
    if (inv2a_lin is not None) and (inv2b_lin is not None):
        Ta_lin = np.array([temperature_from_invariants(inv2a_lin[i]) for i in range(inv2a_lin.shape[0])], dtype=np.float64)
        Tb_lin = np.array([temperature_from_invariants(inv2b_lin[i]) for i in range(inv2b_lin.shape[0])], dtype=np.float64)
        dT_lin = Ta_lin - Tb_lin
        dT0_lin = dT_lin[0] if (np.isfinite(dT_lin[0]) and abs(dT_lin[0]) > 1e-30) else 1.0
        dT_ratio_lin = dT_lin / dT0_lin
    else:
        dT_ratio_lin = None

    # (a) Shared relaxation metrics: A/A0, dT/dT0, and entropy ratios for all cases.
    axa = fig.add_subplot(gs[0, 0]); _tidy(axa)
    axa.plot(t1, A_ratio_nl, color="C0", label=r"$A/A_0$ (1sp, nl)")
    if A_ratio_lin is not None:
        axa.plot(t1, A_ratio_lin, color="C0", ls="--", label=r"$A/A_0$ (1sp, lin)")
    axa.plot(t2, dT_ratio_nl, color="C1", label=r"$(T_a-T_b)/(T_a-T_b)_0$ (2sp, nl)")
    if dT_ratio_lin is not None:
        axa.plot(t2, dT_ratio_lin, color="C1", ls="--", label=r"$(T_a-T_b)/(T_a-T_b)_0$ (2sp, lin)")
    axa.plot(t1, D1_ratio, color="C2", label=r"$\mathcal{D}/\mathcal{D}_0$ (1sp, nl)")
    if D1_ratio_lin is not None:
        axa.plot(t1, D1_ratio_lin, color="C2", ls="--", label=r"$\mathcal{F}/\mathcal{F}_0$ (1sp, lin)")
    axa.plot(t2, D2tot_ratio, color="C3", label=r"$\mathcal{D}_{\rm tot}/\mathcal{D}_0$ (2sp, nl)")
    if D2tot_ratio_lin is not None:
        axa.plot(t2, D2tot_ratio_lin, color="C3", ls="--", label=r"$\mathcal{F}_{\rm tot}/\mathcal{F}_0$ (2sp, lin)")
    axa.set_title("(a) Relaxation metrics")
    axa.set_xlabel("t")
    axa.set_ylabel("normalized (0 to 1)")
    # Use a shared axis for A/A0 and dT/dT0 as requested (typically both ∈[0,1]).
    y_all = [A_ratio_nl, dT_ratio_nl, D1_ratio, D2tot_ratio]
    if A_ratio_lin is not None:
        y_all.append(A_ratio_lin)
    if dT_ratio_lin is not None:
        y_all.append(dT_ratio_lin)
    if D1_ratio_lin is not None:
        y_all.append(D1_ratio_lin)
    if D2tot_ratio_lin is not None:
        y_all.append(D2tot_ratio_lin)
    y_min = float(np.min([np.min(np.asarray(y)) for y in y_all if y is not None]))
    y_max = float(np.max([np.max(np.asarray(y)) for y in y_all if y is not None]))
    axa.set_ylim(0.0, 1.05)
    if (y_min < -1e-6) or (y_max > 1.05 + 1e-6):
        axa.text(
            0.02,
            0.02,
            rf"out of range: [{y_min:.2g},{y_max:.2g}]",
            transform=axa.transAxes,
            fontsize=6,
            alpha=0.85,
        )
    axa.grid(True, which="both", alpha=0.25)
    axa.legend(frameon=False, loc="upper right", borderpad=0.2, labelspacing=0.16, handlelength=1.7)

    # (b) Conserved drifts for all cases.
    axb = fig.add_subplot(gs[0, 1]); _tidy(axb)
    eps = 1e-30
    dn1, dP1, dW1 = _drifts_1sp(inv1, mass=sp1.m, vth=sp1.vth)
    # 2sp totals momentum scale: sum of typical species momenta.
    PscaleT = max(1e-30, abs(spa.m * inv2a[0, 0] * spa.vth) + abs(spb.m * inv2b[0, 0] * spb.vth))
    dn2, dP2, dW2 = _drifts_2sp_tot(inv2a, inv2b, PscaleT=PscaleT)
    axb.plot(t1, dn1 + eps, color="C0", lw=1.35, label="|Δn| 1sp nl")
    axb.plot(t1, dP1 + eps, color="C1", lw=1.35, label="|ΔP| 1sp nl")
    axb.plot(t1, dW1 + eps, color="C2", lw=1.35, label="|ΔW| 1sp nl")
    axb.plot(t2, dn2 + eps, color="C0", lw=1.05, alpha=0.65, label="|Δn| 2sp nl")
    axb.plot(t2, dP2 + eps, color="C1", lw=1.05, alpha=0.65, label="|ΔP| 2sp nl")
    axb.plot(t2, dW2 + eps, color="C2", lw=1.05, alpha=0.65, label="|ΔW| 2sp nl")
    if inv1_lin is not None:
        dn1l, dP1l, dW1l = _drifts_1sp(inv1_lin, mass=sp1.m, vth=sp1.vth)
        axb.plot(t1, dn1l + eps, color="C0", ls="--", lw=1.35, label="|Δn| 1sp lin")
        axb.plot(t1, dP1l + eps, color="C1", ls="--", lw=1.35, label="|ΔP| 1sp lin")
        axb.plot(t1, dW1l + eps, color="C2", ls="--", lw=1.35, label="|ΔW| 1sp lin")
    if (inv2a_lin is not None) and (inv2b_lin is not None):
        dn2l, dP2l, dW2l = _drifts_2sp_tot(inv2a_lin, inv2b_lin, PscaleT=PscaleT)
        axb.plot(t2, dn2l + eps, color="C0", ls="--", lw=1.05, alpha=0.65, label="|Δn| 2sp lin")
        axb.plot(t2, dP2l + eps, color="C1", ls="--", lw=1.05, alpha=0.65, label="|ΔP| 2sp lin")
        axb.plot(t2, dW2l + eps, color="C2", ls="--", lw=1.05, alpha=0.65, label="|ΔW| 2sp lin")
    axb.set_yscale("log")
    axb.set_title("(b) Conserved drifts")
    axb.set_xlabel("t")
    axb.set_ylabel("absolute relative drift (log)")
    axb.grid(True, which="both", alpha=0.25)
    # Two compact legends: (i) quantity colors, (ii) case/style meaning.
    qty_leg = [
        Line2D([0], [0], color="C0", lw=1.5, label=r"$|\Delta n|$"),
        Line2D([0], [0], color="C1", lw=1.5, label=r"$|\Delta P|$"),
        Line2D([0], [0], color="C2", lw=1.5, label=r"$|\Delta W|$"),
    ]
    case_leg = [
        Line2D([0], [0], color="k", lw=1.5, ls="-", label="1sp nl"),
        Line2D([0], [0], color="k", lw=1.5, ls="--", label="1sp lin"),
        Line2D([0], [0], color="k", lw=1.2, ls="-", alpha=0.65, label="2sp nl"),
        Line2D([0], [0], color="k", lw=1.2, ls="--", alpha=0.65, label="2sp lin"),
    ]
    leg1 = axb.legend(handles=qty_leg, frameon=False, loc="upper right", borderpad=0.2, labelspacing=0.18, handlelength=1.8)
    axb.add_artist(leg1)
    axb.legend(handles=case_leg, frameon=False, loc="lower right", borderpad=0.2, labelspacing=0.18, handlelength=1.8)

    # (c) 1sp: vx-slice at IC, at A/A0≈0.3, and the local Maxwellian reference.
    axc = fig.add_subplot(gs[1, 0]); _tidy(axc)
    xgrid = np.linspace(-4.0, 4.0, 900)
    idxA_nl = _idx_near(A_ratio_nl, 0.3)
    tA_nl = float(t1[idxA_nl])
    s_ic = reconstruct_slice_vx_tensor(f_hist1[0], nmax=nmax, xgrid=xgrid)
    s_nl = reconstruct_slice_vx_tensor(f_hist1[idxA_nl], nmax=nmax, xgrid=xgrid)
    # Use a moment-constrained Maxwellian built from conserved invariants (t=0).
    s_M = maxwellian_slice_vx_from_invariants(inv1[0], sp1, xgrid=xgrid)
    axc.plot(xgrid, s_ic, color="k", lw=1.4, label="IC (t=0)")
    axc.plot(xgrid, s_nl, color="C0", label=rf"nl @ $A/A_0\approx0.3$ (t={tA_nl:.3g})")
    if f_hist1_lin is not None and A_ratio_lin is not None:
        idxA_l = _idx_near(A_ratio_lin, 0.3)
        tA_l = float(t1[idxA_l])
        s_l = reconstruct_slice_vx_tensor(f_hist1_lin[idxA_l], nmax=nmax, xgrid=xgrid)
        axc.plot(xgrid, s_l, color="C0", ls="--", label=rf"lin @ $A/A_0\approx0.3$ (t={tA_l:.3g})")
    axc.plot(xgrid, s_M, color="C3", ls=":", alpha=0.9, label="Maxwellian (from conserved moments)")
    axc.set_title("(c) 1sp vx-slice")
    axc.set_xlabel(r"$v_x/v_{th}$")
    axc.set_ylabel(r"$f(v_x,0,0)$")
    axc.grid(True, which="both", alpha=0.25)
    axc.legend(frameon=False, loc="upper left", borderpad=0.2, labelspacing=0.16, handlelength=1.7)

    # (d) 2sp: vx-slices at IC, at dT/dT0≈0.3, and Maxwellians at the final common equilibrium temperature.
    axd = fig.add_subplot(gs[1, 1]); _tidy(axd)
    idxT_nl = _idx_near(dT_ratio_nl, 0.3)
    tT_nl = float(t2[idxT_nl])
    # Plot vs a *common* velocity axis v_x/v_th,a (physical v normalized by species-a reference vth).
    xref = xgrid
    xB = xref * (float(spa.vth) / float(spb.vth))
    sa_ic = reconstruct_slice_vx_tensor(fa_hist[0], nmax=nmax, xgrid=xref)
    sb_ic = reconstruct_slice_vx_tensor(fb_hist[0], nmax=nmax, xgrid=xB)
    sa_nl = reconstruct_slice_vx_tensor(fa_hist[idxT_nl], nmax=nmax, xgrid=xref)
    sb_nl = reconstruct_slice_vx_tensor(fb_hist[idxT_nl], nmax=nmax, xgrid=xB)

    invTot0 = inv2a[0] + inv2b[0]
    Teq_star = float((2.0 / 3.0) * (float(invTot0[4]) / max(1e-300, float(invTot0[0]))))
    # Maxwellians for each species at the common equilibrium temperature Teq_star.
    inva_eq = inv2a[0].copy(); inva_eq[4] = 1.5 * float(inva_eq[0]) * Teq_star
    invb_eq = inv2b[0].copy(); invb_eq[4] = 1.5 * float(invb_eq[0]) * Teq_star
    Ma = maxwellian_slice_vx_from_invariants(inva_eq, spa, xgrid=xref)
    Mb = maxwellian_slice_vx_from_invariants(invb_eq, spb, xgrid=xB)

    axd.plot(xref, sa_ic, color="C0", lw=1.2, alpha=0.65, label="A IC (t=0)")
    axd.plot(xref, sb_ic, color="C1", lw=1.2, alpha=0.65, label="B IC (t=0)")
    axd.plot(xref, sa_nl, color="C0", lw=1.35, label=rf"A nl @ $dT/dT_0\approx0.3$ (t={tT_nl:.3g})")
    axd.plot(xref, sb_nl, color="C1", lw=1.35, label=rf"B nl @ $dT/dT_0\approx0.3$ (t={tT_nl:.3g})")
    if (fa_hist_lin is not None) and (fb_hist_lin is not None) and (dT_ratio_lin is not None):
        idxT_l = _idx_near(dT_ratio_lin, 0.3)
        tT_l = float(t2[idxT_l])
        sa_l = reconstruct_slice_vx_tensor(fa_hist_lin[idxT_l], nmax=nmax, xgrid=xref)
        sb_l = reconstruct_slice_vx_tensor(fb_hist_lin[idxT_l], nmax=nmax, xgrid=xB)
        axd.plot(xref, sa_l, color="C0", ls="--", lw=1.35, label=rf"A lin @ $dT/dT_0\approx0.3$ (t={tT_l:.3g})")
        axd.plot(xref, sb_l, color="C1", ls="--", lw=1.35, label=rf"B lin @ $dT/dT_0\approx0.3$ (t={tT_l:.3g})")
    axd.plot(xref, Ma, color="C0", ls=":", alpha=0.85, label=rf"A Maxwellian ($T={Teq_star:.3g}$)")
    axd.plot(xref, Mb, color="C1", ls=":", alpha=0.85, label=rf"B Maxwellian ($T={Teq_star:.3g}$)")
    axd.set_title("(d) 2sp vx-slices")
    axd.set_xlabel(r"$v_x/v_{th,a}$")
    axd.set_ylabel(r"$f(v_x,0,0)$")
    axd.grid(True, which="both", alpha=0.25)
    axd.legend(frameon=False, loc="upper left", borderpad=0.2, labelspacing=0.16, handlelength=1.7)

    base = outprefix
    fig.savefig(base + ".png", dpi=450, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Linearization (NumPy, fast machinery only)
# ----------------------------


def linearized_matrix_1sp_fast(T: ModelTablesNP, fM: np.ndarray, *, use_tt: bool, tt_tol: float, tt_rmax: int) -> np.ndarray:
    """
    Build L for 1sp about Maxwellian fM via L(h)=Q(h,fM)+Q(fM,h), using fast MPO RHS.
    """
    p = T.p
    N = p**3
    fM = np.asarray(fM, dtype=np.float64)
    S1M, S2M = build_S_np(fM, T)

    L = np.zeros((N, N), dtype=np.float64)
    # column-by-column
    for j in range(N):
        ej = np.zeros((p, p, p), dtype=np.float64)
        ej.reshape(-1)[j] = 1.0
        # Q(h,fM)
        q_hM = rhs_ab_with_S_np(ej, S1M, S2M, T, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax)
        # Q(fM,h): build S(h) then apply to fM
        S1h, S2h = build_S_np(ej, T)
        q_Mh = rhs_ab_with_S_np(fM, S1h, S2h, T, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax)
        L[:, j] = (q_hM + q_Mh).reshape(-1)
    return L


def linearized_matrix_2sp_fast(
    Tab: ModelTablesNP,
    Tba: ModelTablesNP,
    faM: np.ndarray,
    fbM: np.ndarray,
    *,
    use_tt: bool,
    tt_tol: float,
    tt_rmax: int,
) -> np.ndarray:
    """
    Build 2sp Jacobian J about Maxwellians (faM,fbM) for cross-only system:
      rhs_a = Q_ab(fa,fb)
      rhs_b = Q_ba(fb,fa)
    """
    p = Tab.p
    N = p**3
    faM = np.asarray(faM, dtype=np.float64)
    fbM = np.asarray(fbM, dtype=np.float64)

    # Precompute S for backgrounds
    S1_bM, S2_bM = build_S_np(fbM, Tab)
    S1_aM, S2_aM = build_S_np(faM, Tba)

    J = np.zeros((2 * N, 2 * N), dtype=np.float64)

    # δrhs_a = Q_ab(δfa, fbM) + Q_ab(faM, δfb)
    for j in range(N):
        ej = np.zeros((p, p, p), dtype=np.float64)
        ej.reshape(-1)[j] = 1.0
        # wrt fa
        J[:N, j] = rhs_ab_with_S_np(ej, S1_bM, S2_bM, Tab, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax).reshape(-1)
        # wrt fb
        S1ej, S2ej = build_S_np(ej, Tab)
        J[:N, N + j] = rhs_ab_with_S_np(faM, S1ej, S2ej, Tab, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax).reshape(-1)

    # δrhs_b = Q_ba(δfb, faM) + Q_ba(fbM, δfa)
    for j in range(N):
        ej = np.zeros((p, p, p), dtype=np.float64)
        ej.reshape(-1)[j] = 1.0
        # wrt fb
        J[N:, N + j] = rhs_ab_with_S_np(ej, S1_aM, S2_aM, Tba, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax).reshape(-1)
        # wrt fa
        S1ej, S2ej = build_S_np(ej, Tba)
        J[N:, j] = rhs_ab_with_S_np(fbM, S1ej, S2ej, Tba, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax).reshape(-1)

    return J


# ----------------------------
# Test/benchmark harness (--run_tests)
# ----------------------------


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.linalg.norm((a - b).ravel()))
    den = float(np.linalg.norm(b.ravel())) + 1e-300
    return num / den


def run_tests(args) -> None:
    """
    Opt-in internal verification + benchmarking suite.
    Writes plots and CSV/JSON summaries to tests_landau_hermite/run_YYYYmmdd_HHMMSS/.

    Design philosophy:
      - Prefer structural properties of the Landau operator that are baseline-independent:
        Maxwellian fixed points, conservation laws, and Lyapunov/monotonicity diagnostics.
      - For "linearized" evolution, use the correct quadratic free energy (2nd variation of entropy)
        rather than KL to a time-dependent Maxwellian.
      - For properties that are exact only in the infinite-dimensional setting (e.g. Galilean
        invariance / drifted Maxwellians), test *convergence with nmax* rather than exact zeros at
        fixed truncation.
    """
    out_root = Path(str(args.tests_outdir))
    out_root.mkdir(parents=True, exist_ok=True)

    # Optional cleanup: remove previous timestamped runs so results are easy to browse and LaTeX picks up the
    # most recent artifacts without leaving stale folders around.
    if bool(getattr(args, "tests_clean", True)):
        try:
            for p in out_root.iterdir():
                if p.is_dir() and p.name.startswith("run_"):
                    shutil.rmtree(p)
        except Exception:
            pass

    run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convenience pointer for documentation/inspection: best-effort update of tests_landau_hermite/latest -> out_dir
    # (symlink if supported; otherwise write latest_path.txt).
    latest = out_root / "latest"
    try:
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        elif latest.is_dir():
            shutil.rmtree(latest)
    except Exception:
        pass
    try:
        import os

        os.symlink(out_dir.name, latest)  # relative symlink to keep paths portable inside repo
    except Exception:
        try:
            (out_root / "latest_path.txt").write_text(str(out_dir) + "\n")
        except Exception:
            pass

    print("=" * 100, flush=True)
    print(f"[tests] writing to {out_dir}", flush=True)

    nmax_list = _parse_int_list(args.tests_nmax_list)
    if not nmax_list:
        raise ValueError("--tests_nmax_list is empty")
    nmax_list = sorted(set(int(x) for x in nmax_list))

    Q = int(args.Q)
    if not bool(getattr(args, "no_auto_Q", False)):
        nmax_max = max(nmax_list)
        if nmax_max <= 5:
            Q_rec = 8
        else:
            Q_rec = int(min(24, max(8, 2 * nmax_max)))
        if Q < Q_rec:
            print(f"[tests] auto_Q: overriding Q={Q} -> Q={Q_rec} (recommended for max nmax={nmax_max})", flush=True)
            Q = Q_rec
    maxK = int(args.maxK)
    reps_rhs = int(args.tests_reps_rhs)
    reps_bench = int(args.tests_reps_bench)
    reps_integrate = int(getattr(args, "tests_reps_integrate", 3))
    steps = int(args.tests_steps)
    dt = float(args.tests_dt)
    max_s_per_nmax = float(getattr(args, "tests_max_seconds_per_nmax", 180.0))

    # fixed species definitions (same as main)
    sp1 = Species(m=1.0, vth=1.0)
    Teq = 1.0
    mA, mB = 1.0, 4.0
    vA = math.sqrt(2.0 * Teq / mA)
    vB = math.sqrt(2.0 * Teq / mB)
    spa = Species(m=mA, vth=vA)
    spb = Species(m=mB, vth=vB)
    nu_ab = 1.0
    nu_ba = nu_ab * (spa.m / spb.m) ** 2 * (spa.vth / spb.vth) ** 6

    rng = np.random.default_rng(int(getattr(args, "seed", 0)))

    rows: list[dict] = []
    rhs_err_self = []
    rhs_err_ab = []
    rhs_err_ba = []
    maxw_res_11 = []
    maxw_res_2 = []
    physics_rows: list[dict] = []
    lin_op_rows: list[dict] = []

    # For JAX timing
    jax, jnp = _maybe_import_jax("jax")

    def _fmt_ms(x_s: float, *, enabled: bool = True) -> str:
        if (not enabled) or (not np.isfinite(x_s)):
            return "skipped"
        return f"{1e3 * x_s:.3f}ms"

    def _fmt_s(x_s: float) -> str:
        if not np.isfinite(x_s):
            return "n/a"
        return f"{x_s:.3f}s"

    def _fmt_e(x: float, *, enabled: bool = True) -> str:
        if (not enabled) or (not np.isfinite(x)):
            return "skipped"
        return f"{x:.2e}"

    for nmax in nmax_list:
        t_nmax0 = time.perf_counter()
        p = nmax + 1
        N = p**3
        print("-" * 100, flush=True)
        print(f"[tests] nmax={nmax}  p={p}  N={N}", flush=True)
        do_numpy = (nmax <= int(args.tests_max_numpy_nmax))

        # Build tables
        t0 = time.perf_counter()
        T11 = build_model_tables_np(nmax=nmax, Q=Q, maxK=maxK, ma=sp1.m, mb=sp1.m, vtha=sp1.vth, vthb=sp1.vth, nu_ab=nu_ab)
        Tab = build_model_tables_np(nmax=nmax, Q=Q, maxK=maxK, ma=spa.m, mb=spb.m, vtha=spa.vth, vthb=spb.vth, nu_ab=nu_ab)
        Tba = build_model_tables_np(nmax=nmax, Q=Q, maxK=maxK, ma=spb.m, mb=spa.m, vtha=spb.vth, vthb=spa.vth, nu_ab=nu_ba)
        t_setup = time.perf_counter() - t0

        # Random near-Maxwellian states (small amplitude helps highlight conservation accuracy)
        amp_rand = 1e-3
        f = amp_rand * rng.standard_normal((p, p, p))
        f[0, 0, 0] = 1.0 / (sp1.vth**3)
        fa = amp_rand * rng.standard_normal((p, p, p))
        fb = amp_rand * rng.standard_normal((p, p, p))
        fa[0, 0, 0] = 1.0 / (spa.vth**3)
        fb[0, 0, 0] = 1.0 / (spb.vth**3)

        # Maxwellians
        fM1 = np.zeros((p, p, p), dtype=np.float64); fM1[0, 0, 0] = 1.0 / (sp1.vth**3)
        faM = np.zeros((p, p, p), dtype=np.float64); faM[0, 0, 0] = 1.0 / (spa.vth**3)
        fbM = np.zeros((p, p, p), dtype=np.float64); fbM[0, 0, 0] = 1.0 / (spb.vth**3)

        # NumPy RHS (optional; gets slow at larger nmax)
        t_numpy_rhs_self = float("nan")
        t_numpy_rhs_cross = float("nan")
        out_self_np = None
        out_ab_np = None
        out_ba_np = None
        if do_numpy:
            t0 = time.perf_counter()
            for _ in range(reps_rhs):
                out_self_np = rhs_ab_np(f, f, T11, use_tt=False, tt_tol=float(args.tt_tol), tt_rmax=int(args.tt_rmax))
            t_numpy_rhs_self = (time.perf_counter() - t0) / max(1, reps_rhs)

            t0 = time.perf_counter()
            for _ in range(reps_rhs):
                out_ab_np = rhs_ab_np(fa, fb, Tab, use_tt=False, tt_tol=float(args.tt_tol), tt_rmax=int(args.tt_rmax))
                out_ba_np = rhs_ab_np(fb, fa, Tba, use_tt=False, tt_tol=float(args.tt_tol), tt_rmax=int(args.tt_rmax))
            t_numpy_rhs_cross = (time.perf_counter() - t0) / max(1, reps_rhs)

        # Maxwellian residuals (prefer JAX if available)
        r11 = float("nan")
        r2 = float("nan")

        # JAX RHS + timings (if available)
        jax_compile_self = float("nan")
        jax_compile_cross = float("nan")
        jax_rhs_self = float("nan")
        jax_rhs_cross = float("nan")
        jax_integrate_compile = float("nan")
        jax_integrate_steady = float("nan")
        err_self = float("nan")
        err_ab = float("nan")
        err_ba = float("nan")
        cons_rate_n = float("nan")
        cons_rate_P = float("nan")
        cons_rate_W = float("nan")

        if jax is not None and jnp is not None:
            rhs11_jit = build_jax_functions(T11)
            rhsab_jit = build_jax_functions(Tab)
            rhsba_jit = build_jax_functions(Tba)

            fj = jnp.asarray(f)
            faj = jnp.asarray(fa)
            fbj = jnp.asarray(fb)

            # Warm up / compile (separate self vs cross)
            t0 = time.perf_counter()
            out_self_j = rhs11_jit(fj, fj)
            jax.block_until_ready(out_self_j)
            jax_compile_self = time.perf_counter() - t0

            t0 = time.perf_counter()
            _out_ab_j = rhsab_jit(faj, fbj)
            _out_ba_j = rhsba_jit(fbj, faj)
            jax.block_until_ready(_out_ba_j)
            jax_compile_cross = time.perf_counter() - t0

            t0 = time.perf_counter()
            for _ in range(reps_bench):
                out_self_j = rhs11_jit(fj, fj)
                jax.block_until_ready(out_self_j)
            jax_rhs_self = (time.perf_counter() - t0) / max(1, reps_bench)

            t0 = time.perf_counter()
            for _ in range(reps_bench):
                out_ab_j = rhsab_jit(faj, fbj)
                out_ba_j = rhsba_jit(fbj, faj)
                jax.block_until_ready(out_ba_j)
            jax_rhs_cross = (time.perf_counter() - t0) / max(1, reps_bench)

            out_self_j_np = np.array(out_self_j)
            out_ab_j_np = np.array(rhsab_jit(faj, fbj))
            out_ba_j_np = np.array(rhsba_jit(fbj, faj))
            if do_numpy and (out_self_np is not None) and (out_ab_np is not None) and (out_ba_np is not None):
                err_self = _rel_l2(out_self_j_np, out_self_np)
                err_ab = _rel_l2(out_ab_j_np, out_ab_np)
                err_ba = _rel_l2(out_ba_j_np, out_ba_np)

            # Maxwellian residuals via JAX (faster than NumPy for large nmax)
            r11 = float(np.linalg.norm(np.array(rhs11_jit(jnp.asarray(fM1), jnp.asarray(fM1))).ravel()) / max(1e-300, np.linalg.norm(fM1.ravel())))
            rab = float(np.linalg.norm(np.array(rhsab_jit(jnp.asarray(faM), jnp.asarray(fbM))).ravel()))
            rba = float(np.linalg.norm(np.array(rhsba_jit(jnp.asarray(fbM), jnp.asarray(faM))).ravel()))
            r2 = float((rab + rba) / max(1e-300, (np.linalg.norm(faM.ravel()) + np.linalg.norm(fbM.ravel()))))

            # Total-invariant rate check (cross-only): invariants(rhs_ab) + invariants(rhs_ba) ≈ 0
            invdot_a = invariants_from_tensor(out_ab_j_np, spa)
            invdot_b = invariants_from_tensor(out_ba_j_np, spb)
            invdot_tot = invdot_a + invdot_b
            cons_rate_n = float(abs(invdot_tot[0]))
            cons_rate_P = float(np.linalg.norm(invdot_tot[1:4]))
            cons_rate_W = float(abs(invdot_tot[4]))

        if not np.isfinite(r11) or not np.isfinite(r2):
            # Fallback Maxwellian residuals via NumPy
            r11 = float(np.linalg.norm(rhs_ab_np(fM1, fM1, T11, use_tt=False, tt_tol=0.0, tt_rmax=1).ravel()) / max(1e-300, np.linalg.norm(fM1.ravel())))
            rab = float(np.linalg.norm(rhs_ab_np(faM, fbM, Tab, use_tt=False, tt_tol=0.0, tt_rmax=1).ravel()))
            rba = float(np.linalg.norm(rhs_ab_np(fbM, faM, Tba, use_tt=False, tt_tol=0.0, tt_rmax=1).ravel()))
            r2 = float((rab + rba) / max(1e-300, (np.linalg.norm(faM.ravel()) + np.linalg.norm(fbM.ravel()))))

        if not np.isfinite(cons_rate_n) and do_numpy and (out_ab_np is not None) and (out_ba_np is not None):
            invdot_a = invariants_from_tensor(out_ab_np, spa)
            invdot_b = invariants_from_tensor(out_ba_np, spb)
            invdot_tot = invdot_a + invdot_b
            cons_rate_n = float(abs(invdot_tot[0]))
            cons_rate_P = float(np.linalg.norm(invdot_tot[1:4]))
            cons_rate_W = float(abs(invdot_tot[4]))

        maxw_res_11.append((nmax, float(r11)))
        maxw_res_2.append((nmax, float(r2)))

        rhs_err_self.append((nmax, float(err_self)))
        rhs_err_ab.append((nmax, float(err_ab)))
        rhs_err_ba.append((nmax, float(err_ba)))

        # Linearization finite-difference check (NumPy only)
        # Compute for a single random direction h at this nmax.
        h = 0.05 * rng.standard_normal((p, p, p))
        h[0, 0, 0] = 0.0
        if do_numpy:
            L_np = make_linearized_rhs_1sp_numpy(T11, fM1, use_tt=False, tt_tol=float(args.tt_tol), tt_rmax=int(args.tt_rmax))
            Lh_np = L_np(h)
            eps_list = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3], dtype=np.float64)
            fd_errs = []
            for eps in eps_list:
                fd = (rhs_ab_np(fM1 + eps * h, fM1 + eps * h, T11, use_tt=False, tt_tol=0.0, tt_rmax=1) - rhs_ab_np(fM1, fM1, T11, use_tt=False, tt_tol=0.0, tt_rmax=1)) / eps
                fd_errs.append(_rel_l2(fd, Lh_np))
            fd_best = float(np.min(fd_errs))
        else:
            fd_best = float("nan")

        # Linearized operator backend check: JAX JVP vs NumPy explicit linearization (small nmax only).
        lin_backend_relerr = float("nan")
        if do_numpy and (jax is not None and jnp is not None):
            rhs11_jit_tmp = build_jax_functions(T11)
            L_jax = make_linearized_rhs_1sp_jax(rhs11_jit_tmp, jnp.asarray(fM1))
            lin_backend_relerr = float(_rel_l2(np.array(L_jax(jnp.asarray(h))), Lh_np))

        # Short integration compare (NumPy-vs-JAX) only when NumPy is enabled for this nmax.
        jax_int_s = float("nan")
        numpy_int_s = float("nan")
        if do_numpy:
            # Nonlinear 1sp + 2sp short run (NumPy)
            def rhs1_np(x):
                return rhs_ab_np(x, x, T11, use_tt=False, tt_tol=float(args.tt_tol), tt_rmax=int(args.tt_rmax))

            def rhs_pair_np(xa, xb):
                return (
                    rhs_ab_np(xa, xb, Tab, use_tt=False, tt_tol=float(args.tt_tol), tt_rmax=int(args.tt_rmax)),
                    rhs_ab_np(xb, xa, Tba, use_tt=False, tt_tol=float(args.tt_tol), tt_rmax=int(args.tt_rmax)),
                )

            t0 = time.perf_counter()
            _ = integrate_1sp_numpy(rhs1_np, f, dt, steps, "ssprk3")
            _ = integrate_2sp_numpy(rhs_pair_np, fa, fb, dt, steps, "ssprk3")
            numpy_int_s = time.perf_counter() - t0

        # JAX short-run integration benchmark (independent of NumPy; provides runtime scaling for nmax>tests_max_numpy_nmax).
        if jax is not None and jnp is not None:
            rhs11_jit = build_jax_functions(T11)
            rhsab_jit = build_jax_functions(Tab)
            rhsba_jit = build_jax_functions(Tba)

            def rhs_self_j(x):
                return rhs11_jit(x, x)

            def rhs_pair_j(xa, xb):
                return rhsab_jit(xa, xb), rhsba_jit(xb, xa)

            integrate_1sp_j, integrate_2sp_j = build_integrators_jax(rhs_self_j, rhs_pair_j, "ssprk3")
            integrate_1sp_j = jax.jit(integrate_1sp_j, static_argnums=(2,))
            integrate_2sp_j = jax.jit(integrate_2sp_j, static_argnums=(3,))

            # Compile timing (first call)
            t0 = time.perf_counter()
            _fh = integrate_1sp_j(jnp.asarray(f), dt, steps)
            _fah, _fbh = integrate_2sp_j(jnp.asarray(fa), jnp.asarray(fb), dt, steps)
            jax.block_until_ready(_fbh)
            jax_integrate_compile = time.perf_counter() - t0

            # Steady-state timing (exclude compilation; average over reps_integrate)
            t0 = time.perf_counter()
            for _ in range(max(1, reps_integrate)):
                _fh = integrate_1sp_j(jnp.asarray(f), dt, steps)
                _fah, _fbh = integrate_2sp_j(jnp.asarray(fa), jnp.asarray(fb), dt, steps)
                jax.block_until_ready(_fbh)
            jax_integrate_steady = (time.perf_counter() - t0) / max(1, reps_integrate)

            jax_int_s = jax_integrate_steady

        print(
            f"[tests] setup={t_setup:.3f}s  numpy_rhs_self={_fmt_ms(t_numpy_rhs_self, enabled=do_numpy)}  numpy_rhs_cross={_fmt_ms(t_numpy_rhs_cross, enabled=do_numpy)}"
            f"  jax_compile(self,cross)=({_fmt_s(jax_compile_self)},{_fmt_s(jax_compile_cross)})  jax_rhs_self={_fmt_ms(jax_rhs_self)}  jax_rhs_cross={_fmt_ms(jax_rhs_cross)}"
            f"  err(self,ab,ba)=({_fmt_e(err_self, enabled=do_numpy)},{_fmt_e(err_ab, enabled=do_numpy)},{_fmt_e(err_ba, enabled=do_numpy)})  fd_best={_fmt_e(fd_best, enabled=do_numpy)}"
            f"  |dInv_tot|=(n:{cons_rate_n:.2e}, P:{cons_rate_P:.2e}, W:{cons_rate_W:.2e})",
            flush=True,
        )

        # Per-nmax "expected result" flags (used in summary + LaTeX companion).
        ok_maxw = bool((r11 < 1e-12) and (r2 < 1e-12))
        ok_cons = bool((cons_rate_n < 1e-10) and (cons_rate_P < 1e-10) and (cons_rate_W < 1e-8))
        ok_backend = True
        if do_numpy:
            ok_backend = bool((err_self < 1e-10) and (err_ab < 1e-10) and (err_ba < 1e-10))

        wall_nmax = time.perf_counter() - t_nmax0
        print(f"[tests] status: ok_maxwell={ok_maxw} ok_conservation={ok_cons} ok_backend={ok_backend}  wall={wall_nmax:.2f}s", flush=True)

        rows.append(
            {
                "nmax": nmax,
                "p": p,
                "N": N,
                "setup_s": t_setup,
                "numpy_rhs_self_s": t_numpy_rhs_self,
                "numpy_rhs_cross_s": t_numpy_rhs_cross,
                "jax_compile_self_s": jax_compile_self,
                "jax_compile_cross_s": jax_compile_cross,
                "jax_rhs_self_s": jax_rhs_self,
                "jax_rhs_cross_s": jax_rhs_cross,
                "jax_integrate_compile_s": jax_integrate_compile,
                "jax_integrate_steady_s": jax_integrate_steady,
                "rhs_relerr_self": err_self,
                "rhs_relerr_ab": err_ab,
                "rhs_relerr_ba": err_ba,
                "maxwell_relres_self": float(r11),
                "maxwell_relres_cross": float(r2),
                "cons_rate_n": cons_rate_n,
                "cons_rate_P": cons_rate_P,
                "cons_rate_W": cons_rate_W,
                "lin_fd_best_relerr": fd_best,
                "lin_backend_relerr": lin_backend_relerr,
                "numpy_integrate_s": numpy_int_s,
                "jax_integrate_s": jax_int_s,
                "ok_maxwell": ok_maxw,
                "ok_conservation": ok_cons,
                "ok_backend_consistency": ok_backend,
                "wall_s": wall_nmax,
            }
        )

        if (max_s_per_nmax > 0) and (wall_nmax > max_s_per_nmax):
            print(f"[tests] stopping early: nmax={nmax} exceeded max {max_s_per_nmax:.1f}s", flush=True)
            break

    # Extra physics checks (H-theorem style diagnostics) at a representative nmax.
    nmax_phys = max(nmax_list)
    try:
        nmax_phys = int(nmax_phys)
    except Exception:
        nmax_phys = int(nmax_list[-1])

    if jax is not None and jnp is not None:
        print(f"[tests] physics checks (entropy/positivity) at nmax={nmax_phys} ...", flush=True)
        p = nmax_phys + 1
        Q_phys = int(args.Q)
        maxK_phys = int(args.maxK)
        T11p = build_model_tables_np(nmax=nmax_phys, Q=Q_phys, maxK=maxK_phys, ma=sp1.m, mb=sp1.m, vtha=sp1.vth, vthb=sp1.vth, nu_ab=nu_ab)
        Tabp = build_model_tables_np(nmax=nmax_phys, Q=Q_phys, maxK=maxK_phys, ma=spa.m, mb=spb.m, vtha=spa.vth, vthb=spb.vth, nu_ab=nu_ab)
        Tbap = build_model_tables_np(nmax=nmax_phys, Q=Q_phys, maxK=maxK_phys, ma=spb.m, mb=spa.m, vtha=spb.vth, vthb=spa.vth, nu_ab=nu_ba)

        rhs11p = build_jax_functions(T11p)
        rhsabp = build_jax_functions(Tabp)
        rhsbap = build_jax_functions(Tbap)

        def rhs_self_j(x):
            return rhs11p(x, x)

        def rhs_pair_j(xa, xb):
            return rhsabp(xa, xb), rhsbap(xb, xa)

        integrate_1sp_nl, integrate_2sp_nl = build_integrators_jax(rhs_self_j, rhs_pair_j, "ssprk3")
        integrate_1sp_nl = jax.jit(integrate_1sp_nl, static_argnums=(2,))
        integrate_2sp_nl = jax.jit(integrate_2sp_nl, static_argnums=(3,))

        # Background Maxwellians for linearized/free-energy diagnostics.
        fM1 = np.zeros((p, p, p), dtype=np.float64); fM1[0, 0, 0] = 1.0 / (sp1.vth**3)
        faM = np.zeros((p, p, p), dtype=np.float64); faM[0, 0, 0] = 1.0 / (spa.vth**3)
        fbM = np.zeros((p, p, p), dtype=np.float64); fbM[0, 0, 0] = 1.0 / (spb.vth**3)

        L_apply = make_linearized_rhs_1sp_jax(rhs11p, jnp.asarray(fM1))
        J_apply = make_linearized_rhs_2sp_jax(rhsabp, rhsbap, jnp.asarray(faM), jnp.asarray(fbM))
        integrate_1sp_lin, integrate_2sp_lin = build_integrators_jax(lambda x: L_apply(x), lambda xa, xb: J_apply(xa, xb), "ssprk3")
        integrate_1sp_lin = jax.jit(integrate_1sp_lin, static_argnums=(2,))
        integrate_2sp_lin = jax.jit(integrate_2sp_lin, static_argnums=(3,))

        # Use the same IC family as the main script defaults (positivity-safe).
        ic1 = build_ic_fig1_1sp_twostream(nmax=nmax_phys, sp=sp1, u=0.95, enforce_nonneg=True)
        fa0, fb0 = build_ic_fig1_2sp(nmax=nmax_phys, spa=spa, spb=spb, Teq=Teq, dT2=float(getattr(args, "dT2", 0.85)))

        steps_phys = int(getattr(args, "tests_steps", 20))
        dt_phys = float(getattr(args, "tests_dt", 0.05))
        idxs = np.unique(np.round(np.linspace(0, steps_phys, min(steps_phys + 1, 11))).astype(int))

        fh = np.array(integrate_1sp_nl(jnp.asarray(ic1.f), dt_phys, steps_phys))
        fah, fbh = integrate_2sp_nl(jnp.asarray(fa0), jnp.asarray(fb0), dt_phys, steps_phys)
        fah = np.array(fah); fbh = np.array(fbh)

        # Linearized evolution about Maxwellians (perturbation form).
        df0 = jnp.asarray(ic1.f - fM1)
        dfa0 = jnp.asarray(fa0 - faM)
        dfb0 = jnp.asarray(fb0 - fbM)
        dfh = np.array(integrate_1sp_lin(df0, dt_phys, steps_phys)) + fM1[None, ...]
        dfa_hist, dfb_hist = integrate_2sp_lin(dfa0, dfb0, dt_phys, steps_phys)
        dfa_h = np.array(dfa_hist) + faM[None, ...]
        dfb_h = np.array(dfb_hist) + fbM[None, ...]

        grid_phys = prepare_entropy_grid(nmax=nmax_phys, xlim=float(getattr(args, "tests_entropy_xlim", 5.0)), nx=int(getattr(args, "tests_entropy_nx", 18)))

        # Nonlinear: KL entropy to local Maxwellians.
        D1 = np.array([rel_entropy_KL_local_maxwellian_grid_tensor_precomp(fh[i], sp1, grid_phys) for i in idxs], dtype=np.float64)
        D2 = np.array(
            [
                rel_entropy_KL_local_maxwellian_grid_tensor_precomp(fah[i], spa, grid_phys)
                + rel_entropy_KL_local_maxwellian_grid_tensor_precomp(fbh[i], spb, grid_phys)
                for i in idxs
            ],
            dtype=np.float64,
        )
        D1r = D1 / (D1[0] if (np.isfinite(D1[0]) and D1[0] > 1e-300) else 1.0)
        D2r = D2 / (D2[0] if (np.isfinite(D2[0]) and D2[0] > 1e-300) else 1.0)
        max_inc_D1 = float(np.max(np.maximum(0.0, np.diff(D1r)))) if D1r.size > 1 else 0.0
        max_inc_D2 = float(np.max(np.maximum(0.0, np.diff(D2r)))) if D2r.size > 1 else 0.0

        # Linearized: quadratic free energy about fixed Maxwellians.
        F1 = np.array([linear_free_energy_grid_tensor_precomp(dfh[i], fM1, sp1, grid_phys) for i in idxs], dtype=np.float64)
        F2 = np.array(
            [
                linear_free_energy_grid_tensor_precomp(dfa_h[i], faM, spa, grid_phys)
                + linear_free_energy_grid_tensor_precomp(dfb_h[i], fbM, spb, grid_phys)
                for i in idxs
            ],
            dtype=np.float64,
        )
        F1r = F1 / (F1[0] if (np.isfinite(F1[0]) and F1[0] > 1e-300) else 1.0)
        F2r = F2 / (F2[0] if (np.isfinite(F2[0]) and F2[0] > 1e-300) else 1.0)
        max_inc_F1 = float(np.max(np.maximum(0.0, np.diff(F1r)))) if F1r.size > 1 else 0.0
        max_inc_F2 = float(np.max(np.maximum(0.0, np.diff(F2r)))) if F2r.size > 1 else 0.0

        # Positivity on diagnostic slice at sampled times.
        xchk = np.linspace(-6.0, 6.0, 241)
        min_f1 = float(np.min([np.min(reconstruct_slice_vx_tensor(fh[i], nmax=nmax_phys, xgrid=xchk)) for i in idxs]))
        min_f1_lin = float(np.min([np.min(reconstruct_slice_vx_tensor(dfh[i], nmax=nmax_phys, xgrid=xchk)) for i in idxs]))

        physics_rows.append(
            {
                "nmax": nmax_phys,
                "Q": Q_phys,
                "maxK": maxK_phys,
                "steps": steps_phys,
                "dt": dt_phys,
                "max_inc_D1_ratio": max_inc_D1,
                "max_inc_D2_ratio": max_inc_D2,
                "max_inc_F1_ratio": max_inc_F1,
                "max_inc_F2_ratio": max_inc_F2,
                "min_slice_1sp_nl": min_f1,
                "min_slice_1sp_lin": min_f1_lin,
            }
        )

        # Plot (publication-ready diagnostic): entropy/free-energy monotonicity.
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6.4, 3.6), dpi=180)
        ax = fig.add_subplot(111)
        t_plot = idxs * dt_phys
        ax.set_title("Entropy / free-energy monotonicity check (sampled)")
        ax.set_xlabel("t")
        ax.set_ylabel("normalized value")
        ax.grid(True, which="both", alpha=0.3)
        ax.plot(t_plot, D1r, marker="o", ms=3, label=r"$\mathcal{D}/\mathcal{D}_0$ (1sp nl, KL)")
        ax.plot(t_plot, D2r, marker="s", ms=3, label=r"$\mathcal{D}_{\rm tot}/\mathcal{D}_0$ (2sp nl, KL)")
        ax.plot(t_plot, F1r, marker="^", ms=3, ls="--", label=r"$\mathcal{F}/\mathcal{F}_0$ (1sp lin, quadratic)")
        ax.plot(t_plot, F2r, marker="v", ms=3, ls="--", label=r"$\mathcal{F}_{\rm tot}/\mathcal{F}_0$ (2sp lin, quadratic)")
        ax.legend(frameon=False, loc="best")
        fig.savefig(out_dir / "physics_entropy_free_energy.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        print(
            f"[tests] physics: max Δ(D1/D10)+={max_inc_D1:.2e}  max Δ(D2/D20)+={max_inc_D2:.2e}  "
            f"max Δ(F1/F10)+={max_inc_F1:.2e}  max Δ(F2/F20)+={max_inc_F2:.2e}",
            flush=True,
        )
        print(f"[tests] physics: min slice f(vx,0,0) nl={min_f1:.2e}  lin={min_f1_lin:.2e}", flush=True)

        # Additional physics tests anchored in the Landau literature:
        #  - Maxwellians with drift (common velocity) and/or temperature are equilibria.
        #  - Linearized operator about a Maxwellian is (formally) self-adjoint and negative semidefinite
        #    in the weighted inner product ⟨a,b⟩_M = ∫ a b / M dv (restricted away from the nullspace).
        try:
            # Galilean invariance is exact in the continuous equation, but a fixed truncated Hermite basis
            # does *not* represent drifted Maxwellians exactly. We therefore test *convergence* in nmax
            # for a small drift u: residual should decrease as nmax increases.
            u_drift = 0.15
            drift_res = []
            drift_nmax = []
            for nmx in sorted(set(int(x) for x in nmax_list)):
                fU = build_maxwellian_like_tensor_via_projection(nmax=nmx, sp=sp1, ux_hat=float(u_drift), alpha=1.0, density=1.0, xlim=10.0, nx=2001)
                TT = build_model_tables_np(nmax=nmx, Q=Q_phys, maxK=maxK_phys, ma=sp1.m, mb=sp1.m, vtha=sp1.vth, vthb=sp1.vth, nu_ab=nu_ab)
                rhs11_tmp = build_jax_functions(TT)
                rU = float(np.linalg.norm(np.array(rhs11_tmp(jnp.asarray(fU), jnp.asarray(fU))).ravel()) / max(1e-300, np.linalg.norm(fU.ravel())))
                drift_nmax.append(int(nmx))
                drift_res.append(float(rU))
            drift_nmax = np.array(drift_nmax, dtype=int)
            drift_res = np.array(drift_res, dtype=np.float64)

            # Cross-species equilibrium: densities may differ but Maxwellians are exact fixed points.
            # (Here "Maxwellian" means the equilibrium Gaussian in each species' normalized coordinate.)
            faMd = np.zeros_like(faM); faMd[0, 0, 0] = 1.7 * faM[0, 0, 0]
            fbMd = np.zeros_like(fbM); fbMd[0, 0, 0] = 0.4 * fbM[0, 0, 0]
            rab = float(np.linalg.norm(np.array(rhsabp(jnp.asarray(faMd), jnp.asarray(fbMd))).ravel()) / max(1e-300, np.linalg.norm(faMd.ravel())))
            rba = float(np.linalg.norm(np.array(rhsbap(jnp.asarray(fbMd), jnp.asarray(faMd))).ravel()) / max(1e-300, np.linalg.norm(fbMd.ravel())))

            # Weighted inner-product checks for the linearized operator (1sp).
            # Use the fixed background Maxwellian fM1 (u=0, Tvar=0.5 in these normalized coords).
            def _Mxyz_from_fM(fM, sp: Species, grid: EntropyGrid):
                fM000 = float(fM[0, 0, 0])
                Tx_hat, Ty_hat, Tz_hat = temperature_components_hat_from_tensor(fM)
                Tvar = float((Tx_hat + Ty_hat + Tz_hat) / 3.0)
                r2 = grid.X**2 + grid.Y**2 + grid.Z**2
                pref = fM000 / ((2.0 * math.pi * Tvar) ** 1.5)
                return pref * np.exp(-0.5 * r2 / max(Tvar, 1e-12))

            Mxyz = _Mxyz_from_fM(fM1, sp1, grid_phys)
            dx3 = grid_phys.dx**3
            rng2 = np.random.default_rng(1234)
            h1 = 1e-3 * rng2.standard_normal((p, p, p))
            h2 = 1e-3 * rng2.standard_normal((p, p, p))
            # Remove the nullspace components (density, momentum, energy) to avoid degeneracy.
            for arr in (h1, h2):
                arr[0, 0, 0] = 0.0
                if nmax_phys >= 1:
                    arr[1, 0, 0] = 0.0; arr[0, 1, 0] = 0.0; arr[0, 0, 1] = 0.0
                if nmax_phys >= 2:
                    arr[2, 0, 0] = 0.0; arr[0, 2, 0] = 0.0; arr[0, 0, 2] = 0.0

            Lh1 = np.array(L_apply(jnp.asarray(h1)))
            Lh2 = np.array(L_apply(jnp.asarray(h2)))
            h1xyz = np.einsum("nmp,nx,my,pz->xyz", h1, grid_phys.psi, grid_phys.psi, grid_phys.psi, optimize=True)
            h2xyz = np.einsum("nmp,nx,my,pz->xyz", h2, grid_phys.psi, grid_phys.psi, grid_phys.psi, optimize=True)
            Lh1xyz = np.einsum("nmp,nx,my,pz->xyz", Lh1, grid_phys.psi, grid_phys.psi, grid_phys.psi, optimize=True)
            Lh2xyz = np.einsum("nmp,nx,my,pz->xyz", Lh2, grid_phys.psi, grid_phys.psi, grid_phys.psi, optimize=True)

            def ip(a, b):
                return float(np.sum(a * b / np.maximum(Mxyz, 1e-300)) * dx3)

            sym_def = abs(ip(h1xyz, Lh2xyz) - ip(Lh1xyz, h2xyz)) / max(1e-300, abs(ip(h1xyz, h1xyz)) + abs(ip(h2xyz, h2xyz)))
            diss1 = ip(h1xyz, Lh1xyz)
            diss2 = ip(h2xyz, Lh2xyz)

            physics_rows[-1].update(
                {
                    "drift_convergence_u": float(u_drift),
                    "drift_convergence_nmax": drift_nmax.tolist(),
                    "drift_convergence_relres": drift_res.tolist(),
                    "cross_equilibrium_density_relres_ab": rab,
                    "cross_equilibrium_density_relres_ba": rba,
                    "lin_selfadjoint_sym_def": float(sym_def),
                    "lin_weighted_diss_h1": float(diss1),
                    "lin_weighted_diss_h2": float(diss2),
                }
            )

            # Plots.
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(6.2, 3.4), dpi=180)
            ax = fig.add_subplot(111)
            ax.set_title(f"Drifted Maxwellian convergence (u={u_drift}, 1sp)")
            ax.set_xlabel("nmax")
            ax.set_ylabel(r"$\Vert RHS(M_u)\Vert/\Vert M_u\Vert$")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.plot(drift_nmax, drift_res, marker="o")
            fig.savefig(out_dir / "drifted_maxwellian_convergence.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(6.2, 3.4), dpi=180)
            ax = fig.add_subplot(111)
            ax.set_title(f"Linearized operator: symmetry + dissipation (nmax={nmax_phys})")
            ax.set_xlabel("metric")
            ax.set_ylabel("value")
            ax.set_yscale("symlog", linthresh=1e-18)
            ax.grid(True, which="both", alpha=0.3)
            ax.bar(["sym_def", "<h1,Lh1>_M", "<h2,Lh2>_M"], [sym_def, diss1, diss2])
            fig.savefig(out_dir / "linearized_symmetry_dissipation.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(6.2, 2.8), dpi=180)
            ax = fig.add_subplot(111)
            ax.set_title(f"Cross Maxwellian equilibrium (densities) residuals (nmax={nmax_phys})")
            ax.set_ylabel("relative residual")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.bar(["ab", "ba"], [rab, rba])
            fig.savefig(out_dir / "cross_equilibrium_residuals.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

            print(f"[tests] drift convergence (u={u_drift}): {list(zip(drift_nmax.tolist(), drift_res.tolist()))}", flush=True)
            print(f"[tests] cross equilibrium (densities) relres: ab={rab:.2e} ba={rba:.2e}", flush=True)
            print(f"[tests] linearized symmetry defect={sym_def:.2e}  diss(h1)={diss1:.2e} diss(h2)={diss2:.2e}", flush=True)
        except Exception as e:
            print(f"[tests] extra physics tests failed: {type(e).__name__}: {e}", flush=True)

    # Convergence sweeps in (Q, maxK) for Maxwellian residuals at representative nmax.
    Q_sweep = _parse_int_list(str(getattr(args, 'tests_Q_sweep', '')))
    maxK_sweep = _parse_int_list(str(getattr(args, 'tests_maxK_sweep', '')))
    if (Q_sweep or maxK_sweep) and (len(nmax_list) > 0):
        nmax_ref2 = int(max(nmax_list))
        print(f"[tests] convergence sweeps at nmax={nmax_ref2} ...", flush=True)

        def _maxw_res(nmax_: int, Q_: int, maxK_: int) -> Tuple[float, float]:
            p_ = nmax_ + 1
            T11s = build_model_tables_np(nmax=nmax_, Q=Q_, maxK=maxK_, ma=sp1.m, mb=sp1.m, vtha=sp1.vth, vthb=sp1.vth, nu_ab=nu_ab)
            Tabs = build_model_tables_np(nmax=nmax_, Q=Q_, maxK=maxK_, ma=spa.m, mb=spb.m, vtha=spa.vth, vthb=spb.vth, nu_ab=nu_ab)
            Tbas = build_model_tables_np(nmax=nmax_, Q=Q_, maxK=maxK_, ma=spb.m, mb=spa.m, vtha=spb.vth, vthb=spa.vth, nu_ab=nu_ba)
            fM1s = np.zeros((p_, p_, p_), dtype=np.float64); fM1s[0, 0, 0] = 1.0 / (sp1.vth**3)
            faMs = np.zeros((p_, p_, p_), dtype=np.float64); faMs[0, 0, 0] = 1.0 / (spa.vth**3)
            fbMs = np.zeros((p_, p_, p_), dtype=np.float64); fbMs[0, 0, 0] = 1.0 / (spb.vth**3)
            r11_ = float(np.linalg.norm(rhs_ab_np(fM1s, fM1s, T11s, use_tt=False, tt_tol=0.0, tt_rmax=1).ravel()) / max(1e-300, np.linalg.norm(fM1s.ravel())))
            rab_ = float(np.linalg.norm(rhs_ab_np(faMs, fbMs, Tabs, use_tt=False, tt_tol=0.0, tt_rmax=1).ravel()))
            rba_ = float(np.linalg.norm(rhs_ab_np(fbMs, faMs, Tbas, use_tt=False, tt_tol=0.0, tt_rmax=1).ravel()))
            r2_ = float((rab_ + rba_) / max(1e-300, (np.linalg.norm(faMs.ravel()) + np.linalg.norm(fbMs.ravel()))))
            return r11_, r2_

        sweep_rows = []
        Q0 = int(args.Q)
        K0 = int(args.maxK)
        for Qv in (Q_sweep or [Q0]):
            r11_, r2_ = _maxw_res(nmax_ref2, int(Qv), K0)
            sweep_rows.append({"nmax": nmax_ref2, "Q": int(Qv), "maxK": K0, "maxw_self": r11_, "maxw_cross": r2_})
        for Kv in (maxK_sweep or [K0]):
            r11_, r2_ = _maxw_res(nmax_ref2, Q0, int(Kv))
            sweep_rows.append({"nmax": nmax_ref2, "Q": Q0, "maxK": int(Kv), "maxw_self": r11_, "maxw_cross": r2_})
        (out_dir / "convergence_Q_maxK.json").write_text(json.dumps({"rows": sweep_rows}, indent=2))

        try:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(6.4, 3.6), dpi=180)
            ax = fig.add_subplot(111)
            ax.set_title(f"Maxwellian residual convergence (nmax={nmax_ref2})")
            ax.set_xlabel("sweep parameter")
            ax.set_ylabel(r"$\\Vert RHS(M)\\Vert/\\Vert M\\Vert$")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            if Q_sweep:
                xs = np.array(Q_sweep, dtype=float)
                _map = {(int(r["Q"]), int(r["maxK"])): (float(r["maxw_self"]), float(r["maxw_cross"])) for r in sweep_rows}
                ys1 = np.array([_map[(int(Qv), int(K0))][0] for Qv in Q_sweep], dtype=float)
                ys2 = np.array([_map[(int(Qv), int(K0))][1] for Qv in Q_sweep], dtype=float)
                ax.plot(xs, ys1, marker="o", label=f"self (maxK={K0})")
                ax.plot(xs, ys2, marker="s", label=f"cross (maxK={K0})")
            if maxK_sweep:
                xs = np.array(maxK_sweep, dtype=float)
                _map = {(int(r["Q"]), int(r["maxK"])): (float(r["maxw_self"]), float(r["maxw_cross"])) for r in sweep_rows}
                ys1 = np.array([_map[(int(Q0), int(Kv))][0] for Kv in maxK_sweep], dtype=float)
                ys2 = np.array([_map[(int(Q0), int(Kv))][1] for Kv in maxK_sweep], dtype=float)
                ax.plot(xs, ys1, marker="^", ls="--", label=f"self (Q={Q0})")
                ax.plot(xs, ys2, marker="v", ls="--", label=f"cross (Q={Q0})")
            ax.legend(frameon=False, loc="best")
            fig.savefig(out_dir / "convergence_maxwellian_residual.png", dpi=220, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass

    # Save CSV + JSON
    csv_path = out_dir / "timings_and_errors.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows, "physics_rows": physics_rows}, indent=2))

    # Write/refresh the small LaTeX snippet used by the companion PDF (kept at repo root for convenience).
    # This lets `pdflatex landau_hermite_jax_companion.tex` always pick up the most recent test sweep.
    try:
        def _as_ms(x: float) -> float:
            return 1e3 * float(x)

        lines = []
        lines.append("% Auto-generated from tests_landau_hermite/latest/summary.json")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Test sweep summary (JAX backend). Reported times are steady-state. Conservation rates are for cross-collision totals $d/dt\\,(n,P,W)$ evaluated on a random near-Maxwellian state.}")
        lines.append("\\label{tab:tests_sweep}")
        lines.append("\\begin{tabular}{r r r r r r r r}")
        lines.append("\\toprule")
        lines.append("$n_{\\max}$ & $N=(n_{\\max}{+}1)^3$ & $t_{\\mathrm{self}}$ [ms] & $t_{\\mathrm{cross}}$ [ms] & $t_{\\mathrm{int}}$ [s] & speedup$_{\\mathrm{self}}$ & speedup$_{\\mathrm{cross}}$ & $\\|RHS(M)\\|/\\|M\\|$ (cross) \\\\")
        lines.append("\\midrule")
        for r in rows:
            # speedup vs NumPy (only available when NumPy was run for this nmax)
            sp_self = "--"
            sp_cross = "--"
            try:
                if np.isfinite(float(r.get("numpy_rhs_self_s", float("nan")))) and np.isfinite(float(r.get("jax_rhs_self_s", float("nan")))):
                    sp_self = f"${float(r['numpy_rhs_self_s'])/max(1e-300,float(r['jax_rhs_self_s'])):.1f}\\times$"
                if np.isfinite(float(r.get("numpy_rhs_cross_s", float('nan')))) and np.isfinite(float(r.get("jax_rhs_cross_s", float('nan')))):
                    sp_cross = f"${float(r['numpy_rhs_cross_s'])/max(1e-300,float(r['jax_rhs_cross_s'])):.1f}\\times$"
            except Exception:
                pass
            lines.append(
                f"{int(r['nmax'])} & {int(r['N'])} & {_as_ms(r['jax_rhs_self_s']):.3f} & {_as_ms(r['jax_rhs_cross_s']):.3f} & {float(r['jax_integrate_s']):.3f} & {sp_self} & {sp_cross} & {float(r['maxwell_relres_cross']):.2e} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        Path("landau_hermite_jax_companion_results.tex").write_text("\n".join(lines) + "\n")
    except Exception:
        pass

    # Plots
    import matplotlib.pyplot as plt

    def _save(fig, name: str):
        fig.savefig(out_dir / f"{name}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    nvals = np.array([r["nmax"] for r in rows], dtype=int)
    Ns = np.array([r["N"] for r in rows], dtype=float)

    fig = plt.figure(figsize=(6.2, 3.4), dpi=180)
    ax = fig.add_subplot(111)
    ax.set_title("RHS backend consistency (JAX vs NumPy)")
    ax.set_xlabel("nmax")
    ax.set_ylabel("relative L2 error")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    y_self = np.array([r["rhs_relerr_self"] for r in rows], dtype=float)
    y_ab = np.array([r["rhs_relerr_ab"] for r in rows], dtype=float)
    y_ba = np.array([r["rhs_relerr_ba"] for r in rows], dtype=float)
    m_self = np.isfinite(y_self)
    m_ab = np.isfinite(y_ab)
    m_ba = np.isfinite(y_ba)
    ax.plot(nvals[m_self], y_self[m_self], marker="o", label="self")
    ax.plot(nvals[m_ab], y_ab[m_ab], marker="s", label="ab")
    ax.plot(nvals[m_ba], y_ba[m_ba], marker="^", label="ba")
    ax.legend(frameon=False, loc="best")
    _save(fig, "rhs_backend_consistency")

    fig = plt.figure(figsize=(6.2, 3.4), dpi=180)
    ax = fig.add_subplot(111)
    ax.set_title("Maxwellian fixed-point residual")
    ax.set_xlabel("nmax")
    ax.set_ylabel(r"$\Vert RHS(M)\Vert/\Vert M\Vert$")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.plot(nvals, [r["maxwell_relres_self"] for r in rows], marker="o", label="self")
    ax.plot(nvals, [r["maxwell_relres_cross"] for r in rows], marker="s", label="2sp cross total")
    ax.legend(frameon=False, loc="best")
    _save(fig, "maxwellian_residual")

    fig = plt.figure(figsize=(6.2, 3.4), dpi=180)
    ax = fig.add_subplot(111)
    ax.set_title("Runtime scaling (per RHS call)")
    ax.set_xlabel(r"$N=(nmax+1)^3$")
    ax.set_ylabel("seconds")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    numpy_self = np.array([r["numpy_rhs_self_s"] for r in rows], dtype=float)
    numpy_cross = np.array([r["numpy_rhs_cross_s"] for r in rows], dtype=float)
    jax_self = np.array([r["jax_rhs_self_s"] for r in rows], dtype=float)
    jax_cross = np.array([r["jax_rhs_cross_s"] for r in rows], dtype=float)
    m_ns = np.isfinite(numpy_self)
    m_nc = np.isfinite(numpy_cross)
    m_js = np.isfinite(jax_self)
    m_jc = np.isfinite(jax_cross)
    ax.plot(Ns[m_ns], numpy_self[m_ns], marker="o", label="NumPy self")
    ax.plot(Ns[m_nc], numpy_cross[m_nc], marker="s", label="NumPy cross")
    ax.plot(Ns[m_js], jax_self[m_js], marker="^", label="JAX self (steady)")
    ax.plot(Ns[m_jc], jax_cross[m_jc], marker="v", label="JAX cross (steady)")
    ax.legend(frameon=False, loc="best")
    _save(fig, "runtime_rhs_scaling")

    # Speedup plot (NumPy vs JAX), for the nmax where NumPy is enabled.
    try:
        fig = plt.figure(figsize=(6.2, 3.4), dpi=180)
        ax = fig.add_subplot(111)
        ax.set_title("Speedup: JAX vs NumPy (steady-state)")
        ax.set_xlabel("nmax")
        ax.set_ylabel("speedup (NumPy time / JAX time)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        sp_self = numpy_self / np.maximum(1e-300, jax_self)
        sp_cross = numpy_cross / np.maximum(1e-300, jax_cross)
        m_sp_self = np.isfinite(sp_self)
        m_sp_cross = np.isfinite(sp_cross)
        ax.plot(nvals[m_sp_self], sp_self[m_sp_self], marker="o", label="self RHS")
        ax.plot(nvals[m_sp_cross], sp_cross[m_sp_cross], marker="s", label="cross RHS")
        ax.legend(frameon=False, loc="best")
        _save(fig, "speedup_numpy_vs_jax")
    except Exception:
        pass

    fig = plt.figure(figsize=(6.2, 3.4), dpi=180)
    ax = fig.add_subplot(111)
    ax.set_title("JAX compile scaling (first call)")
    ax.set_xlabel("nmax")
    ax.set_ylabel("seconds")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.plot(nvals, [r["jax_compile_self_s"] for r in rows], marker="o", label="self")
    ax.plot(nvals, [r["jax_compile_cross_s"] for r in rows], marker="s", label="cross")
    ax.legend(frameon=False, loc="best")
    _save(fig, "jax_compile_scaling")

    fig = plt.figure(figsize=(6.2, 3.4), dpi=180)
    ax = fig.add_subplot(111)
    ax.set_title("Linearization FD self-consistency (NumPy)")
    ax.set_xlabel("nmax")
    ax.set_ylabel("best rel. FD error over eps")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    y_fd = np.array([r["lin_fd_best_relerr"] for r in rows], dtype=float)
    m_fd = np.isfinite(y_fd)
    ax.plot(nvals[m_fd], y_fd[m_fd], marker="o")
    _save(fig, "linearization_fd_best")

    # Linearized operator backend consistency (NumPy explicit linearization vs JAX JVP), where available.
    fig = plt.figure(figsize=(6.2, 3.4), dpi=180)
    ax = fig.add_subplot(111)
    ax.set_title("Linearized operator: NumPy vs JAX consistency")
    ax.set_xlabel("nmax")
    ax.set_ylabel("relative L2 error")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    y_lin_b = np.array([r.get("lin_backend_relerr", float("nan")) for r in rows], dtype=float)
    m_lin_b = np.isfinite(y_lin_b)
    ax.plot(nvals[m_lin_b], y_lin_b[m_lin_b], marker="o")
    _save(fig, "linearized_backend_consistency")

    fig = plt.figure(figsize=(6.2, 3.4), dpi=180)
    ax = fig.add_subplot(111)
    ax.set_title("Cross-collision conservation check (total rates)")
    ax.set_xlabel("nmax")
    ax.set_ylabel("absolute rate")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.plot(nvals, [r["cons_rate_n"] for r in rows], marker="o", label="|dn_tot/dt|")
    ax.plot(nvals, [r["cons_rate_P"] for r in rows], marker="s", label="||dP_tot/dt||")
    ax.plot(nvals, [r["cons_rate_W"] for r in rows], marker="^", label="|dW_tot/dt|")
    ax.legend(frameon=False, loc="best")
    _save(fig, "conservation_rates")

    # Small-run integration timing (if present)
    fig = plt.figure(figsize=(6.2, 3.4), dpi=180)
    ax = fig.add_subplot(111)
    ax.set_title(f"Short integration benchmark (steps={steps}, dt={dt})")
    ax.set_xlabel("nmax")
    ax.set_ylabel("seconds")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.plot(nvals, [r["numpy_integrate_s"] for r in rows], marker="o", label="NumPy (1sp+2sp)")
    ax.plot(nvals, [r["jax_integrate_s"] for r in rows], marker="s", label="JAX (1sp+2sp)")
    ax.legend(frameon=False, loc="best")
    _save(fig, "runtime_integrate_short")

    # Tangent-linear vs nonlinear small-perturbation check (1sp + 2sp) at a representative nmax.
    nmax_ref = 3 if 3 in nmax_list else max(nmax_list)
    try:
        nmax_ref = int(nmax_ref)
    except Exception:
        nmax_ref = int(nmax_list[0])

    if jax is not None and jnp is not None:
        print(f"[tests] small-perturbation linearization check (jax, nmax={nmax_ref}) ...", flush=True)
        p = nmax_ref + 1
        T11 = build_model_tables_np(nmax=nmax_ref, Q=Q, maxK=maxK, ma=sp1.m, mb=sp1.m, vtha=sp1.vth, vthb=sp1.vth, nu_ab=nu_ab)
        Tab = build_model_tables_np(nmax=nmax_ref, Q=Q, maxK=maxK, ma=spa.m, mb=spb.m, vtha=spa.vth, vthb=spb.vth, nu_ab=nu_ab)
        Tba = build_model_tables_np(nmax=nmax_ref, Q=Q, maxK=maxK, ma=spb.m, mb=spa.m, vtha=spb.vth, vthb=spa.vth, nu_ab=nu_ba)

        rhs11_jit = build_jax_functions(T11)
        rhsab_jit = build_jax_functions(Tab)
        rhsba_jit = build_jax_functions(Tba)

        fM1 = np.zeros((p, p, p), dtype=np.float64); fM1[0, 0, 0] = 1.0
        faM = np.zeros((p, p, p), dtype=np.float64); faM[0, 0, 0] = 1.0 / (spa.vth**3)
        fbM = np.zeros((p, p, p), dtype=np.float64); fbM[0, 0, 0] = 1.0 / (spb.vth**3)

        # 1sp anisotropy perturbation direction (energy-neutral)
        h = np.zeros((p, p, p), dtype=np.float64)
        if p > 2:
            h[2, 0, 0] = 2.0
            h[0, 2, 0] = -1.0
            h[0, 0, 2] = -1.0
        eps = 1e-4
        f0_small = fM1 + eps * h

        def rhs_self(x):
            return rhs11_jit(x, x)

        L_apply = make_linearized_rhs_1sp_jax(rhs11_jit, jnp.asarray(fM1))

        integrate_1sp_nl, _ = build_integrators_jax(rhs_self, lambda a, b: (a, b), "ssprk3")
        integrate_1sp_nl = jax.jit(integrate_1sp_nl, static_argnums=(2,))
        integrate_1sp_lin, _ = build_integrators_jax(L_apply, lambda a, b: (a, b), "ssprk3")
        integrate_1sp_lin = jax.jit(integrate_1sp_lin, static_argnums=(2,))

        steps_lin = int(min(steps, 25))
        f_hist_nl = np.array(integrate_1sp_nl(jnp.asarray(f0_small), dt, steps_lin))
        df_hist = np.array(integrate_1sp_lin(jnp.asarray(eps * h), dt, steps_lin))
        f_hist_lin = fM1[None, ...] + df_hist

        rel_t = []
        for k in range(steps_lin + 1):
            num = np.linalg.norm((f_hist_nl[k] - f_hist_lin[k]).ravel())
            den = np.linalg.norm((f_hist_nl[k] - fM1).ravel()) + 1e-300
            rel_t.append(num / den)
        rel_t = np.array(rel_t)

        fig = plt.figure(figsize=(6.2, 3.4), dpi=180)
        ax = fig.add_subplot(111)
        ax.set_title("1sp: nonlinear vs tangent-linear (small perturbation)")
        ax.set_xlabel("step")
        ax.set_ylabel("relative error")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.plot(np.arange(steps_lin + 1), rel_t, marker="o", ms=3)
        _save(fig, "linearization_small_perturb_1sp")

        # 2sp isotropic temperature perturbation
        ha = np.zeros((p, p, p), dtype=np.float64)
        hb = np.zeros((p, p, p), dtype=np.float64)
        if p > 2:
            ha[2, 0, 0] = 1.0; ha[0, 2, 0] = 1.0; ha[0, 0, 2] = 1.0
            hb[2, 0, 0] = -1.0; hb[0, 2, 0] = -1.0; hb[0, 0, 2] = -1.0
        eps2 = 1e-4
        fa0_small = faM + eps2 * ha
        fb0_small = fbM + eps2 * hb

        def rhs_pair(xa, xb):
            return rhsab_jit(xa, xb), rhsba_jit(xb, xa)

        J_apply = make_linearized_rhs_2sp_jax(rhsab_jit, rhsba_jit, jnp.asarray(faM), jnp.asarray(fbM))
        _, integrate_2sp_nl = build_integrators_jax(lambda x: x, rhs_pair, "ssprk3")
        _, integrate_2sp_lin = build_integrators_jax(lambda x: x, J_apply, "ssprk3")
        integrate_2sp_nl = jax.jit(integrate_2sp_nl, static_argnums=(3,))
        integrate_2sp_lin = jax.jit(integrate_2sp_lin, static_argnums=(3,))

        fa_hist_nl, fb_hist_nl = integrate_2sp_nl(jnp.asarray(fa0_small), jnp.asarray(fb0_small), dt, steps_lin)
        dfa_hist, dfb_hist = integrate_2sp_lin(jnp.asarray(eps2 * ha), jnp.asarray(eps2 * hb), dt, steps_lin)
        fa_hist_nl_np = np.array(fa_hist_nl)
        fb_hist_nl_np = np.array(fb_hist_nl)
        fa_hist_lin = faM[None, ...] + np.array(dfa_hist)
        fb_hist_lin = fbM[None, ...] + np.array(dfb_hist)

        dT_err = []
        for k in range(steps_lin + 1):
            Ta_nl = temperature_from_invariants(invariants_from_tensor(fa_hist_nl_np[k], spa))
            Tb_nl = temperature_from_invariants(invariants_from_tensor(fb_hist_nl_np[k], spb))
            Ta_l = temperature_from_invariants(invariants_from_tensor(fa_hist_lin[k], spa))
            Tb_l = temperature_from_invariants(invariants_from_tensor(fb_hist_lin[k], spb))
            num = abs((Ta_nl - Tb_nl) - (Ta_l - Tb_l))
            den = abs(Ta_nl - Tb_nl) + 1e-300
            dT_err.append(num / den)
        dT_err = np.array(dT_err)

        fig = plt.figure(figsize=(6.2, 3.4), dpi=180)
        ax = fig.add_subplot(111)
        ax.set_title("2sp: dT nonlinear vs tangent-linear (small perturbation)")
        ax.set_xlabel("step")
        ax.set_ylabel("relative dT error")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.plot(np.arange(steps_lin + 1), dT_err, marker="o", ms=3)
        _save(fig, "linearization_small_perturb_2sp")

    print(f"[tests] wrote {csv_path}", flush=True)
    print(f"[tests] wrote plots: {out_dir}", flush=True)


# ----------------------------
# Main pipeline
# ----------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Standalone fast SOE→MPO/TT Landau-Hermite (JAX-first): Fig1 panel.")
    ap.add_argument("--backend", choices=["jax", "numpy"], default="jax")
    ap.add_argument("--nmax", type=int, default=4)
    ap.add_argument("--Q", type=int, default=8)
    ap.add_argument("--no_auto_Q", action="store_true", help="Disable heuristic auto-selection of Q for larger nmax (advanced).")
    ap.add_argument("--maxK", type=int, default=256)
    ap.add_argument("--integrator", choices=["rk2", "ssprk3", "rk4"], default="ssprk3")
    ap.add_argument("--linearized", choices=["on", "off"], default="on", help="Include linearized (Maxwellian-background) overlays.")
    ap.add_argument("--linearized_method", choices=["tangent", "matrix"], default="tangent", help="How to compute linearized evolution (default: matrix-free tangent linear).")
    ap.add_argument("--progress_chunks", type=int, default=0, help="If >0, run JAX time stepping in fixed-size chunks and print progress per chunk.")
    ap.add_argument("--quiet", action="store_true", help="Reduce prints.")

    ap.add_argument("--use_tt", action="store_true", help="Use optional TT/MPO contraction in NumPy backend (debug/scaling).")
    ap.add_argument("--tt_tol", type=float, default=1e-12)
    ap.add_argument("--tt_rmax", type=int, default=32)
    ap.add_argument("--dt_1sp", type=float, default=0.04)
    ap.add_argument("--dt_2sp", type=float, default=0.04)
    ap.add_argument("--tmax_1sp", type=float, default=8.0)
    ap.add_argument("--tmax_2sp", type=float, default=8.0)
    ap.add_argument("--steps_1sp", type=int, default=None)
    ap.add_argument("--steps_2sp", type=int, default=None)
    ap.add_argument("--fig1_ic", choices=["poly4", "twostream", "prl_m2"], default="poly4", help="Fig1 IC family for the 1-species case.")
    ap.add_argument("--no_enforce_nonneg_ic", action="store_true", help="Disable IC nonnegativity enforcement on diagnostic grids (not recommended at low nmax).")
    ap.add_argument("--amp1", type=float, default=0.95, help="Fig1 1sp control: for prl_m2 it's the 2nd-moment anisotropy amplitude; for twostream it's the stream separation u in vx/vth; for poly4 it's an anisotropy knob.")
    ap.add_argument("--dT2", type=float, default=0.85, help="Target temperature excess for the hot species in 2sp ICs (positivity-safe).")
    ap.add_argument("--outprefix_fig1", type=str, default="Fig1_panel")
    ap.add_argument("--outprefix", type=str, default=None, help="(compat) Alias for --outprefix_fig1.")
    ap.add_argument("--skip_fig1", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--entropy_nx", type=int, default=22)
    ap.add_argument("--entropy_xlim", type=float, default=5.0)

    ap.add_argument("--run_tests", action="store_true", help="Run internal correctness/performance tests and write plots into tests_landau_hermite/.")
    ap.add_argument("--tests_outdir", type=str, default="tests_landau_hermite")
    ap.add_argument("--tests_keep_old", action="store_true", help="In --run_tests, keep previous run_* folders under --tests_outdir (default: delete old runs).")
    ap.add_argument("--tests_nmax_list", type=str, default="1,2,3,4,5,6")
    ap.add_argument("--tests_reps_rhs", type=int, default=10)
    ap.add_argument("--tests_reps_bench", type=int, default=20)
    ap.add_argument("--tests_reps_integrate", type=int, default=3, help="In --run_tests, repetitions for steady-state integration timing (JAX).")
    ap.add_argument("--tests_steps", type=int, default=20)
    ap.add_argument("--tests_dt", type=float, default=0.05)
    ap.add_argument("--tests_max_seconds_per_nmax", type=float, default=180.0, help="In --run_tests, stop the nmax sweep early if a single nmax takes longer than this many seconds.")
    ap.add_argument("--tests_entropy_nx", type=int, default=18, help="Entropy grid resolution used in --run_tests physics checks (smaller is faster).")
    ap.add_argument("--tests_entropy_xlim", type=float, default=5.0, help="Entropy grid half-width used in --run_tests physics checks.")
    ap.add_argument("--tests_Q_sweep", type=str, default="8,12,16", help="Comma-separated Q values for convergence sweeps in --run_tests.")
    ap.add_argument("--tests_maxK_sweep", type=str, default="128,256,512", help="Comma-separated maxK values for convergence sweeps in --run_tests.")
    ap.add_argument("--tests_max_numpy_nmax", type=int, default=3, help="In --run_tests, only do NumPy-vs-JAX comparisons up to this nmax (NumPy gets slow for larger nmax).")
    args = ap.parse_args()

    backend: BackendName = args.backend  # type: ignore[assignment]
    nmax = int(args.nmax)
    p = nmax + 1
    quiet = bool(args.quiet)

    # For larger nmax, the SOE quadrature needs more nodes to keep the Maxwellian fixed point at roundoff.
    # Empirically, Q≈2*nmax (capped) is a safe default for nmax≳6; users can override by specifying --Q
    # and disabling auto-selection with --no_auto_Q.
    if not bool(getattr(args, "no_auto_Q", False)):
        Q_user = int(args.Q)
        if nmax <= 5:
            Q_rec = 8
        else:
            Q_rec = int(min(24, max(8, 2 * nmax)))
        if Q_user < Q_rec:
            print(f"[cfg] auto_Q: overriding Q={Q_user} -> Q={Q_rec} (recommended for nmax={nmax})", flush=True)
            args.Q = Q_rec

    if args.outprefix is not None:
        args.outprefix_fig1 = str(args.outprefix)

    def vprint(*a, **k):
        if not quiet:
            print(*a, **k, flush=True)

    use_tt = bool(args.use_tt)
    tt_tol = float(args.tt_tol)
    tt_rmax = int(args.tt_rmax)

    if bool(args.run_tests):
        # Default behavior for tests is to keep the output directory tidy (delete old run_* folders).
        # Users can override with --tests_keep_old.
        args.tests_clean = (not bool(getattr(args, "tests_keep_old", False)))
        run_tests(args)
        return

    np.random.default_rng(int(args.seed))  # reserved for future noise ICs

    print("=" * 100, flush=True)
    print(f"[cfg] backend={backend}  nmax={nmax} (p={p})  Q={args.Q} maxK={args.maxK} integrator={args.integrator}", flush=True)
    print(f"[cfg] linearized={args.linearized}  linearized_method={args.linearized_method}", flush=True)
    print(f"[cfg] fig1_out={args.outprefix_fig1}", flush=True)
    print(f"[cfg] 1sp: dt={args.dt_1sp} tmax={args.tmax_1sp} fig1_ic={args.fig1_ic} amp/u={args.amp1}", flush=True)
    print(f"[cfg] 2sp: dt={args.dt_2sp} tmax={args.tmax_2sp} dT_hot={args.dT2}", flush=True)

    # Species (as in the reference PRL panel logic)
    sp1 = Species(m=1.0, vth=1.0)
    Teq = 1.0
    mA, mB = 1.0, 4.0
    vA = math.sqrt(2.0 * Teq / mA)
    vB = math.sqrt(2.0 * Teq / mB)
    spa = Species(m=mA, vth=vA)
    spb = Species(m=mB, vth=vB)

    nu_ab = 1.0
    nu_ba = nu_ab * (spa.m / spb.m) ** 2 * (spa.vth / spb.vth) ** 6

    print(f"[species] 1sp: m={sp1.m} vth={sp1.vth}", flush=True)
    print(f"[species] 2sp: A(m={spa.m},vth={spa.vth:.6g})  B(m={spb.m},vth={spb.vth:.6g})", flush=True)
    print(f"[species] nu_ab={nu_ab:.6g}  nu_ba={nu_ba:.6g}", flush=True)

    t_setup0 = time.perf_counter()
    T11 = build_model_tables_np(nmax=nmax, Q=int(args.Q), maxK=int(args.maxK), ma=sp1.m, mb=sp1.m, vtha=sp1.vth, vthb=sp1.vth, nu_ab=nu_ab)
    Tab = build_model_tables_np(nmax=nmax, Q=int(args.Q), maxK=int(args.maxK), ma=spa.m, mb=spb.m, vtha=spa.vth, vthb=spb.vth, nu_ab=nu_ab)
    Tba = build_model_tables_np(nmax=nmax, Q=int(args.Q), maxK=int(args.maxK), ma=spb.m, mb=spa.m, vtha=spb.vth, vthb=spa.vth, nu_ab=nu_ba)
    t_setup1 = time.perf_counter()
    print(f"[setup] built tables in {t_setup1 - t_setup0:.3f}s", flush=True)

    # Build JAX kernels once (so checks + integration reuse the same compiled code).
    rhs11_jit = None
    rhsab_jit = None
    rhsba_jit = None
    if backend == "jax":
        jax, jnp = _maybe_import_jax("jax")
        assert jax is not None and jnp is not None
        rhs11_jit = build_jax_functions(T11)
        rhsab_jit = build_jax_functions(Tab)
        rhsba_jit = build_jax_functions(Tba)

    enforce_nonneg_ic = (not bool(getattr(args, "no_enforce_nonneg_ic", False)))

    # Fig1 ICs (tensor cube)
    if str(args.fig1_ic) == "twostream":
        ic_tw = build_ic_fig1_1sp_twostream(nmax=nmax, sp=sp1, u=float(args.amp1), enforce_nonneg=enforce_nonneg_ic)
        f0_1sp = ic_tw.f
        print(
            f"[IC/Fig1] 1sp twostream: u={float(args.amp1):.3g} gamma={ic_tw.gamma:.3g}  min(slice,plane)=({ic_tw.min_slice:.3e},{ic_tw.min_plane:.3e})",
            flush=True,
        )
    elif str(args.fig1_ic) == "prl_m2":
        f0_1sp = build_ic_fig1_1sp(nmax=nmax, sp=sp1, amp1=float(args.amp1))
        ms1, mp1 = _min_f_checks_tensor(f0_1sp, nmax=nmax, xlim=6.0, nx=161)
        print(f"[IC/Fig1] 1sp prl_m2: amp={float(args.amp1):.3g}  min(slice,plane)=({ms1:.3e},{mp1:.3e})", flush=True)
    else:
        ic_poly = build_ic_fig1_1sp_poly4(nmax=nmax, sp=sp1, amp1=float(args.amp1))
        f0_1sp = ic_poly.f
        print(
            f"[IC/Fig1] 1sp poly4: amp={float(args.amp1):.3g}  min(slice,plane)=({ic_poly.min_slice:.3e},{ic_poly.min_plane:.3e})",
            flush=True,
        )

    fa0, fb0 = build_ic_fig1_2sp(nmax=nmax, spa=spa, spb=spb, Teq=Teq, dT2=float(args.dT2))
    msa0, mpa0 = _min_f_checks_tensor(fa0, nmax=nmax, xlim=6.0, nx=161)
    msb0, mpb0 = _min_f_checks_tensor(fb0, nmax=nmax, xlim=6.0, nx=161)
    print(f"[IC/Fig1] 2sp A min(slice,plane)=({msa0:.3e},{mpa0:.3e})  B min(slice,plane)=({msb0:.3e},{mpb0:.3e})", flush=True)
    if nmax >= 2:
        Ta0 = temperature_from_invariants(invariants_from_tensor(fa0, spa))
        Tb0 = temperature_from_invariants(invariants_from_tensor(fb0, spb))
        print(f"[IC/Fig1] 2sp temps: Ta={Ta0:.4g}  Tb={Tb0:.4g}  dT={Ta0-Tb0:.4g}", flush=True)

    # Fixed-point checks (Maxwellians)
    fM1 = np.zeros_like(f0_1sp); fM1[0, 0, 0] = f0_1sp[0, 0, 0]
    faM = np.zeros_like(fa0); faM[0, 0, 0] = fa0[0, 0, 0]
    fbM = np.zeros_like(fb0); fbM[0, 0, 0] = fb0[0, 0, 0]
    if backend == "jax":
        jax, jnp = _maybe_import_jax("jax")
        assert jax is not None and jnp is not None and rhs11_jit is not None and rhsab_jit is not None and rhsba_jit is not None
        rhsM1 = np.array(rhs11_jit(jnp.asarray(fM1), jnp.asarray(fM1)))
        rhsMa = np.array(rhsab_jit(jnp.asarray(faM), jnp.asarray(fbM)))
        rhsMb = np.array(rhsba_jit(jnp.asarray(fbM), jnp.asarray(faM)))
    else:
        rhsM1 = rhs_ab_np(fM1, fM1, T11, use_tt=False, tt_tol=tt_tol, tt_rmax=tt_rmax)
        rhsMa = rhs_ab_np(faM, fbM, Tab, use_tt=False, tt_tol=tt_tol, tt_rmax=tt_rmax)
        rhsMb = rhs_ab_np(fbM, faM, Tba, use_tt=False, tt_tol=tt_tol, tt_rmax=tt_rmax)

    relM1 = np.linalg.norm(rhsM1.ravel()) / max(1e-300, np.linalg.norm(fM1.ravel()))
    relM2 = (np.linalg.norm(rhsMa.ravel()) + np.linalg.norm(rhsMb.ravel())) / max(1e-300, np.linalg.norm(faM.ravel()) + np.linalg.norm(fbM.ravel()))
    print(f"[check] ||RHS_11(M)||/||M|| = {relM1:.3e}", flush=True)
    print(f"[check] ||RHS_ab(M)||+||RHS_ba(M)|| / (||Ma||+||Mb||) = {relM2:.3e}", flush=True)
    if nmax <= 4 and (relM1 > 5e-12 or relM2 > 5e-12):
        print("[warn] Maxwellian residual is larger than expected for nmax<=4; verify Q/maxK settings.", flush=True)
    if (relM1 > 1e-10 or relM2 > 1e-10) and int(args.Q) <= 10:
        print("[hint] For larger nmax, increase SOE quadrature nodes: try --Q 16 or --Q 24 (and keep maxK sufficiently large).", flush=True)

    if bool(args.skip_fig1):
        print("[ok] nothing to do: --skip_fig1 is set.", flush=True)
        return

    # Time grids
    if args.steps_1sp is None:
        steps_1 = int(round(float(args.tmax_1sp) / float(args.dt_1sp)))
    else:
        steps_1 = int(args.steps_1sp)
    if args.steps_2sp is None:
        steps_2 = int(round(float(args.tmax_2sp) / float(args.dt_2sp)))
    else:
        steps_2 = int(args.steps_2sp)
    t1 = np.linspace(0.0, steps_1 * float(args.dt_1sp), steps_1 + 1)
    t2 = np.linspace(0.0, steps_2 * float(args.dt_2sp), steps_2 + 1)

    # Entropy grid shared by both panels
    grid = prepare_entropy_grid(nmax=nmax, xlim=float(args.entropy_xlim), nx=int(args.entropy_nx))

    # Backend-specific stepping helpers
    if backend == "numpy":

        def rhs1_np(f):
            return rhs_ab_np(f, f, T11, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax)

        def rhs_pair_np(fa, fb):
            return (
                rhs_ab_np(fa, fb, Tab, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax),
                rhs_ab_np(fb, fa, Tba, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax),
            )

        rhs_self_j = None
        rhs_pair_j = None
        integrate_1sp_jit = None
        integrate_2sp_jit = None
    else:
        if use_tt:
            print("[note] --use_tt is currently only applied in --backend numpy; ignoring for JAX integration.", flush=True)
        jax, jnp = _maybe_import_jax("jax")
        assert jax is not None and jnp is not None and rhs11_jit is not None and rhsab_jit is not None and rhsba_jit is not None

        def rhs_self_j(f):
            return rhs11_jit(f, f)

        def rhs_pair_j(fa, fb):
            return rhsab_jit(fa, fb), rhsba_jit(fb, fa)

        integrate_1sp_jit, integrate_2sp_jit = build_integrators_jax(rhs_self_j, rhs_pair_j, args.integrator)
        integrate_1sp_jit = jax.jit(integrate_1sp_jit, static_argnums=(2,))
        integrate_2sp_jit = jax.jit(integrate_2sp_jit, static_argnums=(3,))

        rhs1_np = None
        rhs_pair_np = None

    # Linearized operator machinery (optional).
    #
    # IMPORTANT: the correct background Maxwellians for linearization depend on the invariants of the
    # particular IC. We therefore *build the equilibrium backgrounds per case* inside `run_case`
    # (instead of reusing the reference equilibrium with only f[0,0,0] nonzero).
    do_linearized = (str(args.linearized).lower() == "on")
    lin_method = str(args.linearized_method)
    expm_multiply = None

    if do_linearized:
        if lin_method == "tangent":
            # Matrix-free tangent-linear evolution (recommended). The per-case background and the
            # resulting linear operator are built inside `run_case`.
            pass
        elif lin_method == "matrix":
            try:
                from scipy.sparse.linalg import expm_multiply as _expm_multiply  # type: ignore

                expm_multiply = _expm_multiply
            except Exception:
                from scipy.linalg import expm  # type: ignore

                def expm_multiply(A, v, start: float, stop: float, num: int, endpoint: bool = True):
                    ts = np.linspace(start, stop, num, endpoint=endpoint)
                    return np.stack([expm(A * t) @ v for t in ts], axis=0)
        else:
            raise ValueError(args.linearized_method)

    def run_case(*, outprefix: str, tag: str, f0_case: np.ndarray, fa0_case: np.ndarray, fb0_case: np.ndarray, do_bench: bool) -> None:
        # Nonlinear integration
        if backend == "numpy":
            vprint(f"[run:{tag}] numpy integration ...")
            t0 = time.perf_counter()
            f_hist1 = integrate_1sp_numpy(rhs1_np, f0_case, float(args.dt_1sp), steps_1, args.integrator)
            fa_hist, fb_hist = integrate_2sp_numpy(rhs_pair_np, fa0_case, fb0_case, float(args.dt_2sp), steps_2, args.integrator)
            t1_run = time.perf_counter()
            vprint(f"[run:{tag}] numpy done in {t1_run - t0:.3f}s")
        else:
            jax, jnp = _maybe_import_jax("jax")
            assert jax is not None and jnp is not None and rhs_self_j is not None and rhs_pair_j is not None and integrate_1sp_jit is not None and integrate_2sp_jit is not None
            vprint(f"[run:{tag}] jax compile+run ...")
            t0 = time.perf_counter()
            if int(args.progress_chunks) > 0:
                f_hist1_j = integrate_1sp_jax_chunked(rhs_self_j, f0_case, float(args.dt_1sp), steps_1, args.integrator, int(args.progress_chunks), label=f"{tag}:1sp", quiet=quiet)
                fa_hist_j, fb_hist_j = integrate_2sp_jax_chunked(rhs_pair_j, fa0_case, fb0_case, float(args.dt_2sp), steps_2, args.integrator, int(args.progress_chunks), label=f"{tag}:2sp", quiet=quiet)
            else:
                f_hist1_j = integrate_1sp_jit(jnp.asarray(f0_case), float(args.dt_1sp), steps_1)
                fa_hist_j, fb_hist_j = integrate_2sp_jit(jnp.asarray(fa0_case), jnp.asarray(fb0_case), float(args.dt_2sp), steps_2)
            jax.block_until_ready(f_hist1_j)
            jax.block_until_ready(fa_hist_j)
            t1_run = time.perf_counter()
            vprint(f"[run:{tag}] jax compile+run: {t1_run - t0:.3f}s")

            if do_bench and (int(args.progress_chunks) <= 0):
                t2_run = time.perf_counter()
                _fh = integrate_1sp_jit(jnp.asarray(f0_case), float(args.dt_1sp), steps_1)
                _fah, _fbh = integrate_2sp_jit(jnp.asarray(fa0_case), jnp.asarray(fb0_case), float(args.dt_2sp), steps_2)
                jax.block_until_ready(_fh)
                jax.block_until_ready(_fah)
                t3_run = time.perf_counter()
                per_step_ms = 1e3 * (t3_run - t2_run) / max(1, (steps_1 + steps_2))
                vprint(f"[run:{tag}] jax steady-state: {t3_run - t2_run:.3f}s  (~{per_step_ms:.3f} ms/step across 1sp+2sp)")

            f_hist1 = np.array(f_hist1_j)
            fa_hist = np.array(fa_hist_j)
            fb_hist = np.array(fb_hist_j)

        # Linearized overlays
        f_hist1_lin = None
        fa_hist_lin = None
        fb_hist_lin = None
        fM1_eq_case = None
        faM_eq_case = None
        fbM_eq_case = None
        if do_linearized:
            vprint(f"[lin:{tag}] computing linearized histories (method={lin_method}) ...")
            t_lin0 = time.perf_counter()

            # Build the *equilibrium* Maxwellians matching the conserved invariants at t=0. This ensures
            # that the linearized evolution relaxes toward the correct Maxwellian and that the
            # nullspace (collision invariants) does not produce an artificial plateau in the quadratic
            # free energy.
            inv10 = invariants_from_tensor(f0_case, sp1)
            inva0 = invariants_from_tensor(fa0_case, spa)
            invb0 = invariants_from_tensor(fb0_case, spb)
            fM1_eq_case = build_maxwellian_tensor_from_invariants(nmax=nmax, sp=sp1, inv=inv10, xlim=10.0, nx=3001)
            faM_eq_case, fbM_eq_case, u_eq, T_eq = build_common_equilibrium_maxwellians_2sp_from_invariants(
                nmax=nmax, spa=spa, spb=spb, inva=inva0, invb=invb0, xlim=10.0, nx=3001
            )
            vprint(
                f"[lin:{tag}] equilibrium backgrounds: 1sp T={_thermal_temperature_from_invariants(inv10, sp1):.4g}  "
                f"2sp T_eq={T_eq:.4g}  u_eq=({u_eq[0]:.3g},{u_eq[1]:.3g},{u_eq[2]:.3g})"
            )

            if lin_method == "tangent":
                if backend == "jax":
                    jax, jnp = _maybe_import_jax("jax")
                    assert jax is not None and jnp is not None and rhs11_jit is not None and rhsab_jit is not None and rhsba_jit is not None
                    assert fM1_eq_case is not None and faM_eq_case is not None and fbM_eq_case is not None
                    L1_apply_j = make_linearized_rhs_1sp_jax(rhs11_jit, jnp.asarray(fM1_eq_case))
                    J_apply_j = make_linearized_rhs_2sp_jax(rhsab_jit, rhsba_jit, jnp.asarray(faM_eq_case), jnp.asarray(fbM_eq_case))
                    df0 = jnp.asarray(f0_case - fM1_eq_case)
                    dfa0 = jnp.asarray(fa0_case - faM_eq_case)
                    dfb0 = jnp.asarray(fb0_case - fbM_eq_case)
                    integrate_1sp_lin, integrate_2sp_lin = build_integrators_jax(lambda x: L1_apply_j(x), lambda xa, xb: J_apply_j(xa, xb), args.integrator)
                    integrate_1sp_lin = jax.jit(integrate_1sp_lin, static_argnums=(2,))
                    integrate_2sp_lin = jax.jit(integrate_2sp_lin, static_argnums=(3,))
                    if int(args.progress_chunks) > 0:
                        df_hist1_j = integrate_1sp_jax_chunked(L1_apply_j, df0, float(args.dt_1sp), steps_1, args.integrator, int(args.progress_chunks), label=f"{tag}:1sp-lin", quiet=quiet)
                        dfa_hist_j, dfb_hist_j = integrate_2sp_jax_chunked(J_apply_j, dfa0, dfb0, float(args.dt_2sp), steps_2, args.integrator, int(args.progress_chunks), label=f"{tag}:2sp-lin", quiet=quiet)
                    else:
                        df_hist1_j = integrate_1sp_lin(df0, float(args.dt_1sp), steps_1)
                        dfa_hist_j, dfb_hist_j = integrate_2sp_lin(dfa0, dfb0, float(args.dt_2sp), steps_2)
                    jax.block_until_ready(df_hist1_j)
                    jax.block_until_ready(dfa_hist_j)
                    f_hist1_lin = np.array(df_hist1_j) + fM1_eq_case[None, ...]
                    fa_hist_lin = np.array(dfa_hist_j) + faM_eq_case[None, ...]
                    fb_hist_lin = np.array(dfb_hist_j) + fbM_eq_case[None, ...]
                else:
                    assert fM1_eq_case is not None and faM_eq_case is not None and fbM_eq_case is not None
                    L1_apply_np = make_linearized_rhs_1sp_numpy(T11, fM1_eq_case, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax)
                    J_apply_np = make_linearized_rhs_2sp_numpy(Tab, Tba, faM_eq_case, fbM_eq_case, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax)
                    df0 = f0_case - fM1_eq_case
                    dfa0 = fa0_case - faM_eq_case
                    dfb0 = fb0_case - fbM_eq_case
                    df_hist1 = integrate_1sp_numpy(L1_apply_np, df0, float(args.dt_1sp), steps_1, args.integrator)
                    dfa_hist, dfb_hist = integrate_2sp_numpy(J_apply_np, dfa0, dfb0, float(args.dt_2sp), steps_2, args.integrator)
                    f_hist1_lin = fM1_eq_case[None, ...] + df_hist1
                    fa_hist_lin = faM_eq_case[None, ...] + dfa_hist
                    fb_hist_lin = fbM_eq_case[None, ...] + dfb_hist
            elif lin_method == "matrix":
                assert expm_multiply is not None
                # Build dense matrices about the per-case equilibrium backgrounds.
                assert fM1_eq_case is not None and faM_eq_case is not None and fbM_eq_case is not None
                if backend == "jax":
                    jax, jnp = _maybe_import_jax("jax")
                    assert jax is not None and jnp is not None and rhs11_jit is not None and rhsab_jit is not None and rhsba_jit is not None
                    vprint(f"[lin:{tag}] building dense matrices via jacfwd (may be slow) ...")
                    fM_flat = jnp.asarray(fM1_eq_case.reshape(-1), dtype=jnp.float64)
                    N = p**3
                    faM_flat = jnp.asarray(faM_eq_case.reshape(-1), dtype=jnp.float64)
                    fbM_flat = jnp.asarray(fbM_eq_case.reshape(-1), dtype=jnp.float64)

                    def rhs_self_flat(x_flat):
                        x = x_flat.reshape(p, p, p)
                        y = rhs11_jit(x, x)
                        return y.reshape(-1)

                    def rhs_pair_flat(y_flat):
                        ya = y_flat[:N].reshape(p, p, p)
                        yb = y_flat[N:].reshape(p, p, p)
                        dya = rhsab_jit(ya, yb)
                        dyb = rhsba_jit(yb, ya)
                        return jnp.concatenate([dya.reshape(-1), dyb.reshape(-1)], axis=0)

                    L1_mat = np.array(jax.jacfwd(rhs_self_flat)(fM_flat))
                    y0_flat = jnp.concatenate([faM_flat, fbM_flat], axis=0)
                    J_mat = np.array(jax.jacfwd(rhs_pair_flat)(y0_flat))
                else:
                    vprint(f"[lin:{tag}] building dense matrices via NumPy assembly (may be slow) ...")
                    L1_mat = linearized_matrix_1sp_fast(T11, fM1_eq_case, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax)
                    J_mat = linearized_matrix_2sp_fast(Tab, Tba, faM_eq_case, fbM_eq_case, use_tt=use_tt, tt_tol=tt_tol, tt_rmax=tt_rmax)

                assert L1_mat is not None and J_mat is not None
                fM_vec = fM1_eq_case.reshape(-1)
                h0 = f0_case.reshape(-1) - fM_vec
                Hlin = expm_multiply(L1_mat, h0, start=float(t1[0]), stop=float(t1[-1]), num=len(t1), endpoint=True)
                f_hist1_lin = (fM_vec[None, :] + Hlin).reshape(len(t1), p, p, p)
                y0 = np.concatenate([faM_eq_case.reshape(-1), fbM_eq_case.reshape(-1)], axis=0)
                dy0 = np.concatenate([fa0_case.reshape(-1) - faM_eq_case.reshape(-1), fb0_case.reshape(-1) - fbM_eq_case.reshape(-1)], axis=0)
                Ylin = expm_multiply(J_mat, dy0, start=float(t2[0]), stop=float(t2[-1]), num=len(t2), endpoint=True)
                Y = y0[None, :] + Ylin
                N = p**3
                fa_hist_lin = Y[:, :N].reshape(len(t2), p, p, p)
                fb_hist_lin = Y[:, N:].reshape(len(t2), p, p, p)
            else:
                raise ValueError(lin_method)

            t_lin1 = time.perf_counter()
            vprint(f"[lin:{tag}] done in {t_lin1 - t_lin0:.3f}s")

        # ----------------------------
        # Diagnostics for plotting + prints (both nonlinear and linearized)
        # ----------------------------

        inv1 = np.vstack([invariants_from_tensor(f_hist1[i], sp1) for i in range(len(t1))])
        inv2a = np.vstack([invariants_from_tensor(fa_hist[i], spa) for i in range(len(t2))])
        inv2b = np.vstack([invariants_from_tensor(fb_hist[i], spb) for i in range(len(t2))])

        inv1_lin = None
        inv2a_lin = None
        inv2b_lin = None
        if (f_hist1_lin is not None) and (fa_hist_lin is not None) and (fb_hist_lin is not None):
            inv1_lin = np.vstack([invariants_from_tensor(f_hist1_lin[i], sp1) for i in range(len(t1))])
            inv2a_lin = np.vstack([invariants_from_tensor(fa_hist_lin[i], spa) for i in range(len(t2))])
            inv2b_lin = np.vstack([invariants_from_tensor(fb_hist_lin[i], spb) for i in range(len(t2))])

        # Invariant drift prints
        def _max_rel_drift(inv: np.ndarray) -> Tuple[float, float, float]:
            n0 = float(inv[0, 0])
            P0 = inv[0, 1:4].astype(np.float64)
            W0 = float(inv[0, 4])
            dn = float(np.max(np.abs(inv[:, 0] - n0) / max(1e-30, abs(n0))))
            dP = float(np.max(np.linalg.norm(inv[:, 1:4] - P0[None, :], axis=1) / max(1e-30, np.linalg.norm(P0) + 1.0)))
            dW = float(np.max(np.abs(inv[:, 4] - W0) / max(1e-30, abs(W0))))
            return dn, dP, dW

        dn1, dP1, dW1 = _max_rel_drift(inv1)
        invT = inv2a + inv2b
        dnT, dPT, dWT = _max_rel_drift(invT)
        print(f"[inv:{tag}] 1sp max rel drift: n={dn1:.3e}  |P|={dP1:.3e}  W={dW1:.3e}", flush=True)
        print(f"[inv:{tag}] 2sp max rel drift: n_tot={dnT:.3e}  |P_tot|={dPT:.3e}  W_tot={dWT:.3e}", flush=True)

        # Entropy time series (1sp and 2sp total)
        print(f"[entropy:{tag}] computing D(t) for 1sp + 2sp ...", flush=True)
        t_ent0 = time.perf_counter()
        D1 = np.array([rel_entropy_KL_local_maxwellian_grid_tensor_precomp(f_hist1[i], sp1, grid) for i in range(len(t1))], dtype=np.float64)
        D2tot = np.array(
            [
                rel_entropy_KL_local_maxwellian_grid_tensor_precomp(fa_hist[i], spa, grid)
                + rel_entropy_KL_local_maxwellian_grid_tensor_precomp(fb_hist[i], spb, grid)
                for i in range(len(t2))
            ],
            dtype=np.float64,
        )
        D10 = float(D1[0]) if (np.isfinite(D1[0]) and D1[0] > 1e-300) else 1.0
        D20 = float(D2tot[0]) if (np.isfinite(D2tot[0]) and D2tot[0] > 1e-300) else 1.0
        D1_ratio = D1 / D10
        D2tot_ratio = D2tot / D20

        D1_ratio_lin = None
        D2tot_ratio_lin = None
        if (f_hist1_lin is not None) and (fa_hist_lin is not None) and (fb_hist_lin is not None):
            # For linearized evolution, use a quadratic free-energy functional about the fixed Maxwellians.
            assert fM1_eq_case is not None and faM_eq_case is not None and fbM_eq_case is not None
            F1l = np.array([linear_free_energy_grid_tensor_precomp(f_hist1_lin[i], fM1_eq_case, sp1, grid) for i in range(len(t1))], dtype=np.float64)
            F2l = np.array(
                [
                    linear_free_energy_grid_tensor_precomp(fa_hist_lin[i], faM_eq_case, spa, grid)
                    + linear_free_energy_grid_tensor_precomp(fb_hist_lin[i], fbM_eq_case, spb, grid)
                    for i in range(len(t2))
                ],
                dtype=np.float64,
            )
            F10l = float(F1l[0]) if (np.isfinite(F1l[0]) and F1l[0] > 1e-300) else 1.0
            F20l = float(F2l[0]) if (np.isfinite(F2l[0]) and F2l[0] > 1e-300) else 1.0
            D1_ratio_lin = F1l / F10l
            D2tot_ratio_lin = F2l / F20l

        t_ent1 = time.perf_counter()
        print(f"[entropy:{tag}] done in {t_ent1 - t_ent0:.3f}s  D1_0={D10:.4g}  D2_0={D20:.4g}", flush=True)

        # Slice deviation time series (used in combined panels)
        xgrid = np.linspace(-4.0, 4.0, 241)
        slice_dev1 = slice_deviation_timeseries(f_hist1, nmax=nmax, xgrid=xgrid)
        slice_dev2a = slice_deviation_timeseries(fa_hist, nmax=nmax, xgrid=xgrid)
        slice_dev2b = slice_deviation_timeseries(fb_hist, nmax=nmax, xgrid=xgrid)
        slice_dev1_lin = None
        slice_dev2a_lin = None
        slice_dev2b_lin = None
        if (f_hist1_lin is not None) and (fa_hist_lin is not None) and (fb_hist_lin is not None):
            slice_dev1_lin = slice_deviation_timeseries(f_hist1_lin, nmax=nmax, xgrid=xgrid)
            slice_dev2a_lin = slice_deviation_timeseries(fa_hist_lin, nmax=nmax, xgrid=xgrid)
            slice_dev2b_lin = slice_deviation_timeseries(fb_hist_lin, nmax=nmax, xgrid=xgrid)

        # Convergence diagnostics: entropy at end and RHS norm at end.
        A_end = float(anisotropy_measure_from_tensor(f_hist1[-1]))
        A0 = float(anisotropy_measure_from_tensor(f_hist1[0]))
        A_ratio_end = A_end / (A0 if abs(A0) > 1e-30 else 1.0)
        Ta_end = temperature_from_invariants(inv2a[-1])
        Tb_end = temperature_from_invariants(inv2b[-1])
        dT_end = (Ta_end - Tb_end)
        dT0 = float(temperature_from_invariants(inv2a[0]) - temperature_from_invariants(inv2b[0]))
        dT_ratio_end = dT_end / (dT0 if abs(dT0) > 1e-30 else 1.0)
        print(f"[diag:{tag}] end: A/A0={A_ratio_end:.3e}  D1/D10={float(D1_ratio[-1]):.3e}  D2/D20={float(D2tot_ratio[-1]):.3e}  dT/dT0={dT_ratio_end:.3e}", flush=True)
        Ta0 = float(temperature_from_invariants(inv2a[0]))
        Tb0 = float(temperature_from_invariants(inv2b[0]))
        print(f"[temps:{tag}] 2sp: Ta {Ta0:.4g}→{Ta_end:.4g}  Tb {Tb0:.4g}→{Tb_end:.4g}", flush=True)
        print(
            f"[shape:{tag}] end slice dev: 1sp={float(slice_dev1[-1]):.3e}  fa={float(slice_dev2a[-1]):.3e}  fb={float(slice_dev2b[-1]):.3e}",
            flush=True,
        )

        # Snapshot positivity diagnostics for the slices shown in the panels.
        xchk = np.linspace(-6.0, 6.0, 401)
        # 1sp: check at t=0 and at the A/A0≈0.3 snapshot time.
        A_ratio_nl = np.array([anisotropy_measure_from_tensor(f_hist1[i]) for i in range(len(t1))], dtype=np.float64)
        A0_here = float(A_ratio_nl[0]) if (np.isfinite(A_ratio_nl[0]) and abs(A_ratio_nl[0]) > 1e-30) else 1.0
        A_ratio_nl = A_ratio_nl / A0_here
        idxA = int(np.argmin(np.abs(A_ratio_nl - 0.3)))
        s0 = reconstruct_slice_vx_tensor(f_hist1[0], nmax=nmax, xgrid=xchk)
        sA = reconstruct_slice_vx_tensor(f_hist1[idxA], nmax=nmax, xgrid=xchk)
        print(f"[pos:{tag}] 1sp min f(vx,0,0): t=0 {float(np.min(s0)):.3e}  @A/A0~0.3 {float(np.min(sA)):.3e}", flush=True)
        if f_hist1_lin is not None:
            A_ratio_l = np.array([anisotropy_measure_from_tensor(f_hist1_lin[i]) for i in range(len(t1))], dtype=np.float64)
            A0_l = float(A_ratio_l[0]) if (np.isfinite(A_ratio_l[0]) and abs(A_ratio_l[0]) > 1e-30) else 1.0
            A_ratio_l = A_ratio_l / A0_l
            idxAl = int(np.argmin(np.abs(A_ratio_l - 0.3)))
            sAl = reconstruct_slice_vx_tensor(f_hist1_lin[idxAl], nmax=nmax, xgrid=xchk)
            print(f"[pos:{tag}] 1sp lin min f(vx,0,0) @A/A0~0.3: {float(np.min(sAl)):.3e}", flush=True)

        # 2sp: check at t=0 and at dT/dT0≈0.3 for each species (on their own normalized vx axes).
        Ta_series = np.array([temperature_from_invariants(inv2a[i]) for i in range(inv2a.shape[0])], dtype=np.float64)
        Tb_series = np.array([temperature_from_invariants(inv2b[i]) for i in range(inv2b.shape[0])], dtype=np.float64)
        dT_series = Ta_series - Tb_series
        dT0_here = float(dT_series[0]) if (np.isfinite(dT_series[0]) and abs(dT_series[0]) > 1e-30) else 1.0
        dT_ratio = dT_series / dT0_here
        idxT = int(np.argmin(np.abs(dT_ratio - 0.3)))
        sa0 = reconstruct_slice_vx_tensor(fa_hist[0], nmax=nmax, xgrid=xchk)
        sb0 = reconstruct_slice_vx_tensor(fb_hist[0], nmax=nmax, xgrid=xchk)
        saT = reconstruct_slice_vx_tensor(fa_hist[idxT], nmax=nmax, xgrid=xchk)
        sbT = reconstruct_slice_vx_tensor(fb_hist[idxT], nmax=nmax, xgrid=xchk)
        print(
            f"[pos:{tag}] 2sp min f(vx,0,0): A t=0 {float(np.min(sa0)):.3e}  @dT~0.3 {float(np.min(saT)):.3e} | "
            f"B t=0 {float(np.min(sb0)):.3e}  @dT~0.3 {float(np.min(sbT)):.3e}",
            flush=True,
        )
        if (fa_hist_lin is not None) and (fb_hist_lin is not None):
            Ta_l = np.array([temperature_from_invariants(inv2a_lin[i]) for i in range(inv2a_lin.shape[0])], dtype=np.float64) if inv2a_lin is not None else Ta_series
            Tb_l = np.array([temperature_from_invariants(inv2b_lin[i]) for i in range(inv2b_lin.shape[0])], dtype=np.float64) if inv2b_lin is not None else Tb_series
            dT_l = Ta_l - Tb_l
            dT0_l = float(dT_l[0]) if (np.isfinite(dT_l[0]) and abs(dT_l[0]) > 1e-30) else 1.0
            idxTl = int(np.argmin(np.abs((dT_l / dT0_l) - 0.3)))
            saTl = reconstruct_slice_vx_tensor(fa_hist_lin[idxTl], nmax=nmax, xgrid=xchk)
            sbTl = reconstruct_slice_vx_tensor(fb_hist_lin[idxTl], nmax=nmax, xgrid=xchk)
            print(
                f"[pos:{tag}] 2sp lin min f(vx,0,0) @dT~0.3: A {float(np.min(saTl)):.3e}  B {float(np.min(sbTl)):.3e}",
                flush=True,
            )

        # RHS norms (to tell 'need longer' vs 'truncated equilibrium').
        rhs1_rel = float("nan")
        rhs2_rel = float("nan")
        if backend == "jax":
            jax, jnp = _maybe_import_jax("jax")
            assert jax is not None and jnp is not None and rhs11_jit is not None and rhsab_jit is not None and rhsba_jit is not None
            r1 = np.array(rhs11_jit(jnp.asarray(f_hist1[-1]), jnp.asarray(f_hist1[-1])))
            ra = np.array(rhsab_jit(jnp.asarray(fa_hist[-1]), jnp.asarray(fb_hist[-1])))
            rb = np.array(rhsba_jit(jnp.asarray(fb_hist[-1]), jnp.asarray(fa_hist[-1])))
            rhs1_rel = float(np.linalg.norm(r1.ravel()) / max(1e-300, np.linalg.norm(f_hist1[-1].ravel())))
            rhs2_rel = float((np.linalg.norm(ra.ravel()) + np.linalg.norm(rb.ravel())) / max(1e-300, (np.linalg.norm(fa_hist[-1].ravel()) + np.linalg.norm(fb_hist[-1].ravel()))))
        else:
            r1 = rhs_ab_np(f_hist1[-1], f_hist1[-1], T11, use_tt=False, tt_tol=tt_tol, tt_rmax=tt_rmax)
            ra = rhs_ab_np(fa_hist[-1], fb_hist[-1], Tab, use_tt=False, tt_tol=tt_tol, tt_rmax=tt_rmax)
            rb = rhs_ab_np(fb_hist[-1], fa_hist[-1], Tba, use_tt=False, tt_tol=tt_tol, tt_rmax=tt_rmax)
            rhs1_rel = float(np.linalg.norm(r1.ravel()) / max(1e-300, np.linalg.norm(f_hist1[-1].ravel())))
            rhs2_rel = float((np.linalg.norm(ra.ravel()) + np.linalg.norm(rb.ravel())) / max(1e-300, (np.linalg.norm(fa_hist[-1].ravel()) + np.linalg.norm(fb_hist[-1].ravel()))))
        print(f"[rhs:{tag}] rel ||rhs_1sp||/||f||={rhs1_rel:.3e}  rel ||rhs_2sp||/||f||={rhs2_rel:.3e}", flush=True)
        if (rhs1_rel > 5e-3) or (rhs2_rel > 5e-3):
            print(f"[hint:{tag}] RHS is not small at t={t1[-1]:.3g}; try increasing --tmax_1sp/--tmax_2sp to run longer.", flush=True)
        elif (float(D1_ratio[-1]) > 1e-2) or (float(D2tot_ratio[-1]) > 1e-2):
            print(f"[hint:{tag}] RHS is small but entropy is not near 0; likely truncation (increase --nmax) and/or increase entropy grid resolution.", flush=True)

        make_fig1_panel(
            outprefix=str(Path(outprefix)),
            sp1=sp1,
            t1=t1,
            f_hist1=f_hist1,
            f_hist1_lin=f_hist1_lin,
            inv1=inv1,
            inv1_lin=inv1_lin,
            D1_ratio=D1_ratio,
            D1_ratio_lin=D1_ratio_lin,
            spa=spa,
            spb=spb,
            t2=t2,
            fa_hist=fa_hist,
            fb_hist=fb_hist,
            fa_hist_lin=fa_hist_lin,
            fb_hist_lin=fb_hist_lin,
            inv2a=inv2a,
            inv2b=inv2b,
            inv2a_lin=inv2a_lin,
            inv2b_lin=inv2b_lin,
            D2tot_ratio=D2tot_ratio,
            D2tot_ratio_lin=D2tot_ratio_lin,
            nmax=nmax,
        )
        print(f"[ok] wrote: {outprefix}.png and {outprefix}.pdf", flush=True)

    run_case(outprefix=str(args.outprefix_fig1), tag="Fig1", f0_case=f0_1sp, fa0_case=fa0, fb0_case=fb0, do_bench=True)


if __name__ == "__main__":
    main()
