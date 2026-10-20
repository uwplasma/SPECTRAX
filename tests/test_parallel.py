"""Tests of the shaped Hermite-DG right-hand side (against the base-commit RHS) and of the flux splittings."""
import inspect
import subprocess
import types
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectrax._initialization import (compute_A_pm_matrices, compute_basis_products, compute_R_pm_matrices,
                                      construct_idx_array)
from spectrax._model import Hermite_DG_rhs, Hermite_DG_system

BASE = "21f506812c9a561d61516adc1afc31a1be26374c"
RHS_NAMES = list(inspect.signature(Hermite_DG_rhs).parameters)[2:-3]

# Small grids: asymmetric 2D with N_DG>1, 1D, 3D, and four species.
A2D = dict(Ns=2, Nn=3, Nm=2, Np=1, N_DG=2, dims=2, Nx=8, Ny=4, Nz=1)
D1 = dict(Ns=2, Nn=4, Nm=1, Np=1, N_DG=3, dims=1, Nx=8, Ny=1, Nz=1)
D1_ONE_CELL = {**D1, "Nx": 4}  # one cell per device for px=4
D3 = dict(Ns=2, Nn=2, Nm=2, Np=2, N_DG=2, dims=3, Nx=4, Ny=2, Nz=4)
S4 = {**A2D, "Ns": 4, "Nn": 2}
NN1 = dict(Ns=2, Nn=1, Nm=2, Np=2, N_DG=2, dims=2, Nx=4, Ny=4, Nz=1)  # no first-order Hermite mode along n


@lru_cache
def basis_tables(dims, N_DG, Nx, Ny, Nz):
    """Basis index array and DG products (jitted and cached: eager construction dominates the test time otherwise)."""
    basis_idx = construct_idx_array(dims, N_DG)
    return basis_idx, jax.jit(compute_basis_products)(4.0 / Nx, 3.0 / Ny, 2.0 / Nz, basis_idx)


A_pm = jax.jit(compute_A_pm_matrices, static_argnums=(0, 1, 2))


def make_args(Ns, Nn, Nm, Np, N_DG, dims, Nx, Ny, Nz, seed=0, edit=None):
    """Deterministic nonconstant state and complete, species-distinct parameters built like initialize_simulation_parameters."""
    rng = np.random.default_rng(seed)
    basis_idx, ip = basis_tables(dims, N_DG, Nx, Ny, Nz)
    Nl = basis_idx.shape[0]
    Lx, Ly, Lz = 4.0, 3.0, 2.0
    alpha, u = jnp.array(rng.uniform(.3, 1.2, (Ns, 3))), jnp.array(rng.normal(0, .4, (Ns, 3)))
    Ap = A_pm(Nn, Nm, Np, alpha, u)
    R_p, R_m = compute_R_pm_matrices(False)
    ladder = lambda n, axis, k: jnp.sqrt(jnp.arange(n) + k).reshape([-1 if i == axis else 1 for i in range(8)])
    args = dict(col=jnp.array(rng.uniform(0, 1, (Np, Nm, Nn))),
                sqrt_n_plus=ladder(Nn, 3, 1), sqrt_n_minus=ladder(Nn, 3, 0), sqrt_m_plus=ladder(Nm, 2, 1),
                sqrt_m_minus=ladder(Nm, 2, 0), sqrt_p_plus=ladder(Np, 1, 1), sqrt_p_minus=ladder(Np, 1, 0),
                basis_idx=basis_idx, inner_mm=ip[0], inner_pm=ip[1], inner_mp=ip[2], inner_pp=ip[3],
                di_inner_product=ip[4], triple_product=ip[5], Ax_p=Ap[0], Ax_m=Ap[1], Ay_p=Ap[2], Ay_m=Ap[3],
                Az_p=Ap[4], Az_m=Ap[5], R_p=R_p, R_m=R_m, Lx=Lx, Ly=Ly, Lz=Lz, nu=.7, D=0., alpha_s=alpha, u_s=u,
                ms=jnp.array([1., 25., 3., 7.][:Ns]), qs=jnp.array([-1., 1., 2., -.5][:Ns]), Omega_ce=.5)
    Ck = jnp.array(rng.normal(size=(Ns, Np * Nm * Nn, Ny, Nx, Nz, Nl)))
    Fk = jnp.array(rng.normal(size=(6, Ny, Nx, Nz, Nl)))
    if edit:  # edit(Ck, args) -> Ck may also modify args (e.g. alpha/u/qs) before the operator tables are rebuilt
        Ck = edit(Ck, args)
        Ap = A_pm(Nn, Nm, Np, args["alpha_s"], args["u_s"])
        args.update(dict(zip(("Ax_p", "Ax_m", "Ay_p", "Ay_m", "Az_p", "Az_m"), Ap)))
    return Ck, Fk, args, (Nn, Nm, Np)


def close(a, b, tol=1e-11):
    a, b = np.asarray(a), np.asarray(b)
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol * np.abs(b).max())


def test_species_blocks_are_distinct():
    Ck, Fk, args, N = make_args(**S4)
    dCk = np.asarray(Hermite_DG_rhs(Ck, Fk, **args, Nn=N[0], Nm=N[1], Np=N[2])[0])
    assert all(np.abs(dCk[i] - dCk[j]).max() > 1e-3 for i in range(4) for j in range(i))


# ------------------------------------------------------------------ single-device parity with the base commit

@lru_cache
def base_model():
    try:
        src = subprocess.run(["git", "show", f"{BASE}:spectrax/_model.py"], cwd=Path(__file__).parents[1], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("base commit not available")
    module = types.ModuleType("_model_base")
    exec(compile(src, "_model_base", "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("case", [A2D, D3, NN1], ids=["2d", "3d", "nn1"])
def test_rhs_and_adapter_match_base_commit(case):
    Ck, Fk, args, (Nn, Nm, Np) = make_args(**case)
    Ns = Ck.shape[0]
    flat = lambda a: [a[k] for k in RHS_NAMES]
    base = np.asarray(base_model().Hermite_DG_system(Ck.reshape(-1, *Ck.shape[2:]), Fk, *flat(args), Nn, Nm, Np, Ns))
    dCk, dFk = Hermite_DG_rhs(Ck, Fk, **args, Nn=Nn, Nm=Nm, Np=Np)
    assert dCk.shape == Ck.shape and dFk.shape == Fk.shape
    close(np.concatenate([np.ravel(dCk), np.ravel(dFk)]), base, 1e-12)
    close(Hermite_DG_system(Ck.reshape(-1, *Ck.shape[2:]), Fk, *flat(args), Nn, Nm, Np, Ns), base, 1e-12)
    flat_params = {**args, "alpha_s": args["alpha_s"].ravel(), "u_s": args["u_s"].ravel()}  # flat (3*Ns,) is accepted too
    close(Hermite_DG_rhs(Ck, Fk, **flat_params, Nn=Nn, Nm=Nm, Np=Np)[0], dCk, 1e-14)


# ------------------------------------------------------------------ A_pm

def eig_split(M):
    """Base-commit construction: eigendecomposition with a general `eig`."""
    Lam, U = jnp.linalg.eig(M)
    D = Lam[..., :, None] * jnp.eye(M.shape[-1])
    part = lambda s: np.real(np.asarray(U @ (D + s * jnp.abs(D)) @ jnp.swapaxes(U, -1, -2) / 2))
    return part(1), part(-1)


@pytest.mark.parametrize("N", [(3, 2, 1), (5, 4, 3)])
def test_A_pm_symmetry_sum_and_eig_reference(N):
    Ns = 3
    rng = np.random.default_rng(1)
    alpha, u = jnp.array(rng.uniform(.3, 1.2, (Ns, 3))), jnp.array(rng.normal(0, .5, (Ns, 3)))
    Ap = compute_A_pm_matrices(*N, alpha, u)
    for k, n in enumerate(N):
        A_p, A_m = (np.asarray(Ap[2 * k + j]).reshape(Ns, n, n) for j in range(2))  # all other axes are size one
        off = np.diag(np.sqrt(np.arange(1, n) / 2), 1) + np.diag(np.sqrt(np.arange(1, n) / 2), -1)
        A = np.asarray(alpha)[:, k, None, None] * off + np.asarray(u)[:, k, None, None] * np.eye(n)
        for M in (A_p, A_m):
            np.testing.assert_allclose(M, np.swapaxes(M, -1, -2), atol=1e-14)
        np.testing.assert_allclose(A_p + A_m, A, atol=1e-13)
        np.testing.assert_allclose(A_p @ A_m, 0, atol=1e-12)  # orthogonal positive/negative parts
        assert np.linalg.eigvalsh(A_p).min() > -1e-12 and np.linalg.eigvalsh(A_m).max() < 1e-12
        ref_p, ref_m = eig_split(jnp.array(A))
        np.testing.assert_allclose(A_p, ref_p, atol=1e-12), np.testing.assert_allclose(A_m, ref_m, atol=1e-12)


def test_R_pm_sum_and_central_flux():
    for upwind in (False, True):
        R_p, R_m = (np.asarray(R).reshape(3, 6, 6) for R in compute_R_pm_matrices(upwind))
        np.testing.assert_allclose(R_p + R_m, np.asarray(compute_R_pm_matrices(False)[0]).reshape(3, 6, 6) * 2, atol=1e-13)
        np.testing.assert_allclose(R_p, np.swapaxes(R_p, -1, -2), atol=1e-14)
