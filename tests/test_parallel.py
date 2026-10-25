"""Multi-device RHS, halo, current, ownership, norm and compiled-collective tests on four logical CPU devices.

Sharded results are compared with the single-device kernel (itself checked against the base-commit RHS); all
comparisons go through numpy because arrays on different device sets cannot be mixed in jnp operations.
"""
import inspect
import re
import subprocess
import types
from functools import lru_cache
from math import prod
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

from spectrax._initialization import (compute_A_pm_matrices, compute_basis_products, compute_R_pm_matrices,
                                      construct_idx_array)
from spectrax._model import Hermite_DG_rhs, Hermite_DG_system, shift_element
from spectrax._runtime import named_sharding
from spectrax._simulation import state_rms_norm

BASE = "21f506812c9a561d61516adc1afc31a1be26374c"
RHS_NAMES = list(inspect.signature(Hermite_DG_rhs).parameters)[2:-4]
SPECIES_NAMES = ("Ax_p", "Ax_m", "Ay_p", "Ay_m", "Az_p", "Az_m", "alpha_s", "u_s", "ms", "qs")

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


@lru_cache
def mesh_of(layout):
    return jax.make_mesh(layout, ("species", "x", "y", "z"), axis_types=(AxisType.Auto,) * 4,
                         devices=jax.devices()[:prod(layout)])


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


def in_shardings(mesh):
    rep, by_species = NamedSharding(mesh, P()), NamedSharding(mesh, P("species"))
    return (named_sharding(mesh, 6, species=True), named_sharding(mesh, 5), {k: by_species if k in SPECIES_NAMES else rep for k in RHS_NAMES})


@lru_cache
def rhs_fn(layout, N):
    """RHS jitted with explicit in/out shardings, as the solver does."""
    mesh = mesh_of(layout)
    ck, fk, _ = shardings = in_shardings(mesh)
    return jax.jit(lambda C, F, a: Hermite_DG_rhs(C, F, **a, Nn=N[0], Nm=N[1], Np=N[2]), in_shardings=shardings, out_shardings=(ck, fk))


def close(a, b, tol=1e-11):
    a, b = np.asarray(a), np.asarray(b)
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol * np.abs(b).max())


def expected_slices(mesh, spec, shape, device):
    """Global index a device must hold for `spec`, from its mesh coordinates alone (independent of JAX's own index map)."""
    coords = dict(zip(mesh.axis_names, np.argwhere(mesh.devices == device)[0]))
    out = []
    for n, name in zip(shape, tuple(spec) + (None,) * (len(shape) - len(spec))):
        size = mesh.shape[name] if name else 1
        out.append((coords[name] * n // size, (coords[name] + 1) * n // size) if name else (0, n))
    return tuple(out)


def assert_owned(mesh, x, spec, ref):
    """Every addressable shard sits on the right device, covers the right global block and holds the right values."""
    assert {s.device for s in x.addressable_shards} == set(mesh.devices.flat)
    for s in x.addressable_shards:
        idx = tuple((sl.start or 0, n if sl.stop is None else sl.stop) for sl, n in zip(s.index, x.shape))
        assert idx == expected_slices(mesh, spec, x.shape, s.device)
        close(s.data, np.asarray(ref)[s.index])


def source_ref(Ck, args, Nn, Nm, Np):
    """Independent numpy species currents (Ns, 3, ...) and the Maxwell source -(1/Omega) [J, 0] they must produce."""
    Ck, alpha, u, q = (np.asarray(v) for v in (Ck, args["alpha_s"], args["u_s"], args["qs"]))
    alpha, u = alpha.reshape(-1, 3), u.reshape(-1, 3)
    zero = np.zeros_like(Ck[:, 0])
    C1 = np.stack([Ck[:, 1] if Nn > 1 else zero, Ck[:, Nn] if Nm > 1 else zero, Ck[:, Nn * Nm] if Np > 1 else zero], axis=1)
    sl = (slice(None), slice(None)) + (None,) * 4
    Js = (q * alpha.prod(1))[:, None, None, None, None, None] * (Ck[:, None, 0] * u[sl] + alpha[sl] * C1 / np.sqrt(2))
    J = Js.sum(0)
    return Js, -np.concatenate([J, np.zeros_like(J)]) / args["Omega_ce"]


def check_layout(layout, case, edit=None, cancel=False):
    """Sharded RHS == single-device RHS: values, shardings, species/spatial ownership and current/field consistency."""
    mesh = mesh_of(layout)
    ps, px, py, pz = layout
    Ck, Fk, args, N = make_args(**case, edit=edit)
    Ns, Nv, Ny, Nx, Nz, Nl = Ck.shape
    ref = rhs_fn((1, 1, 1, 1), N)(Ck, Fk, args)
    ck_s, fk_s, arg_s = in_shardings(mesh)
    Ckd, Fkd, argsd = jax.device_put((Ck, Fk, args), (ck_s, fk_s, arg_s))
    assert_owned(mesh, Ckd, ck_s.spec, Ck)
    assert_owned(mesh, Fkd, fk_s.spec, Fk)
    for k in SPECIES_NAMES:
        assert_owned(mesh, argsd[k], P("species"), args[k])
    dCk, dFk = rhs_fn(layout, N)(Ckd, Fkd, argsd)
    assert dCk.shape == Ck.shape and dFk.shape == Fk.shape
    assert dCk.sharding.is_equivalent_to(ck_s, 6) and dFk.sharding.is_equivalent_to(fk_s, 5)
    assert {s.data.shape for s in dCk.addressable_shards} == {(Ns // ps, Nv, Ny // py, Nx // px, Nz // pz, Nl)}
    assert {s.data.shape for s in dFk.addressable_shards} == {(6, Ny // py, Nx // px, Nz // pz, Nl)}
    assert_owned(mesh, dCk, ck_s.spec, ref[0])  # block-by-block: each shard holds the right species and cells
    assert_owned(mesh, dFk, fk_s.spec, ref[1])
    close(dCk, ref[0]), close(dFk, ref[1])
    # Field/current consistency: dFk depends on Ck only through the species-summed current.
    dFk0 = rhs_fn(layout, N)(jax.device_put(np.zeros(Ck.shape), ck_s), Fkd, argsd)[1]
    Js, source = source_ref(Ck, args, *N)
    np.testing.assert_allclose(np.asarray(dFk) - np.asarray(dFk0), source, rtol=0, atol=1e-11 * np.abs(source).max() + 1e-13 * np.abs(ref[1]).max())
    if cancel:
        assert np.abs(Js).max() > 1e-2  # species currents are individually nonzero ...
        assert np.abs(source).max() < 1e-13 and np.abs(np.asarray(dFk) - np.asarray(dFk0)).max() < 1e-12  # ... but cancel
    return dCk, dFk


def test_species_blocks_are_distinct():
    Ck, Fk, args, N = make_args(**S4)
    dCk = np.asarray(rhs_fn((1, 1, 1, 1), N)(Ck, Fk, args)[0])
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


# ------------------------------------------------------------------ sharded RHS parity (values, shardings, ownership, current)

RHS_LAYOUTS = [
    (A2D, (1, 2, 1, 1)), (A2D, (1, 4, 1, 1)), (A2D, (1, 1, 2, 1)), (A2D, (1, 2, 2, 1)),  # slabs, y split, 2x2 blocks
    (A2D, (2, 1, 1, 1)), (A2D, (2, 2, 1, 1)), (A2D, (2, 1, 2, 1)),  # species only and hybrids
    (D1, (1, 2, 1, 1)), (D1_ONE_CELL, (1, 4, 1, 1)),  # 1D, local extent one
    (D3, (1, 1, 1, 2)), (D3, (2, 1, 1, 2)),  # 3D z split
    (S4, (2, 1, 1, 1)), (S4, (4, 1, 1, 1)), (NN1, (2, 2, 1, 1)),  # two species per shard, one per shard
]


@pytest.mark.parametrize("case,layout", RHS_LAYOUTS, ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else f"Ns{v['Ns']}d{v['dims']}N{v['Nx']}")
def test_sharded_rhs_matches_single_device(case, layout):
    check_layout(layout, case)


# ------------------------------------------------------------------ total current

def only_species(s):
    return lambda Ck, args: Ck.at[jnp.arange(Ck.shape[0]) != s].set(0)


def cancelling_pairs(Ck, args):
    """Species i+Ns/2 gets the data of species i with opposite charge, so the total current vanishes."""
    h = Ck.shape[0] // 2
    for k in ("alpha_s", "u_s", "ms"):
        args[k] = args[k].at[h:].set(args[k][:h])
    args["qs"] = args["qs"].at[h:].set(-args["qs"][:h])
    return Ck.at[h:].set(Ck[:h])


@pytest.mark.parametrize("case,layout,s", [(A2D, (2, 1, 1, 1), 0), (A2D, (2, 2, 1, 1), 1), (S4, (2, 1, 2, 1), 2), (S4, (4, 1, 1, 1), 3), (A2D, (1, 2, 2, 1), 1)])
def test_current_from_one_species_only(case, layout, s):
    check_layout(layout, case, edit=only_species(s))


@pytest.mark.parametrize("case,layout", [(A2D, (2, 1, 1, 1)), (A2D, (2, 2, 1, 1)), (S4, (2, 1, 2, 1)), (S4, (4, 1, 1, 1)), (A2D, (1, 2, 1, 1))])
def test_opposite_species_currents_cancel(case, layout):
    check_layout(layout, case, edit=cancelling_pairs, cancel=True)


# ------------------------------------------------------------------ halo exchange

SHIFTS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1, -1, 0), (-1, 1, 1)]
HALO_CASES = [((1, 4, 1, 1), (1, 4, 1)), ((1, 2, 1, 1), (2, 8, 1)), ((1, 1, 4, 1), (4, 2, 1)), ((1, 1, 2, 1), (4, 2, 1)),
              ((1, 1, 1, 4), (1, 2, 4)), ((1, 1, 1, 2), (2, 2, 2)), ((1, 2, 2, 1), (2, 2, 1)), ((1, 2, 2, 1), (4, 6, 1)),
              ((1, 2, 1, 2), (2, 4, 2)), ((2, 2, 1, 1), (2, 4, 1)), ((2, 1, 2, 1), (4, 2, 1)), ((2, 1, 1, 2), (1, 2, 2))]


@pytest.mark.parametrize("layout,grid", HALO_CASES, ids=lambda v: "x".join(map(str, v)))
def test_shift_element_halos(layout, grid):
    """dx=+1 means cell i receives cell i+1 (periodic over the whole domain, across device seams)."""
    mesh = mesh_of(layout)
    Ny, Nx, Nz = grid
    C = np.arange(2 * 1 * 1 * 2 * Ny * Nx * Nz * 3, dtype=float).reshape(2, 1, 1, 2, Ny, Nx, Nz, 3)  # unique values
    sh = named_sharding(mesh, 8, species=True)
    assert sh.spec == P("species", None, None, None, "y", "x", "z", None)
    outs = jax.jit(lambda c: tuple(shift_element(c, *d) for d in SHIFTS), in_shardings=sh, out_shardings=(sh,) * len(SHIFTS))(jax.device_put(C, sh))
    for (dx, dy, dz), out in zip(SHIFTS, outs):
        expected = C[:, :, :, :, (np.arange(Ny) + dy) % Ny][:, :, :, :, :, (np.arange(Nx) + dx) % Nx][:, :, :, :, :, :, (np.arange(Nz) + dz) % Nz]
        np.testing.assert_array_equal(expected, np.roll(C, (-dy, -dx, -dz), axis=(-4, -3, -2)))
        np.testing.assert_array_equal(np.asarray(out), expected)
        assert out.sharding.is_equivalent_to(sh, 8)
        assert {s.device for s in out.addressable_shards} == set(mesh.devices.flat)


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


# ------------------------------------------------------------------ tree RMS norm

NORM_LAYOUTS = [(1, 1, 1, 1), (1, 2, 1, 1), (1, 4, 1, 1), (1, 2, 2, 1), (1, 1, 1, 2), (2, 1, 1, 1), (2, 2, 1, 1), (2, 1, 2, 1), (2, 1, 1, 2)]
NS, NV, NY, NX, NZ, NL = 2, 6, 4, 8, 2, 3


def norm_scenarios():
    rng = np.random.default_rng(3)
    Ck, Fk = rng.normal(size=(NS, NV, NY, NX, NZ, NL)), rng.normal(size=(6, NY, NX, NZ, NL))
    one_species = np.zeros_like(Ck)
    one_species[1] = Ck[1]
    one_tile = np.zeros_like(Ck)  # nonzero only in the first x tile of a four-way x split
    one_tile[:, :, :, :NX // 4] = Ck[:, :, :, :NX // 4]
    return {"random": (Ck, Fk), "zero": (0 * Ck, 0 * Fk), "field_only": (0 * Ck, Fk), "one_species": (one_species, 0 * Fk),
            "species_only": (Ck, 0 * Fk), "one_spatial_shard": (one_tile, 0 * Fk),
            "complex": (Ck + 1j * rng.normal(size=Ck.shape), Fk + 1j * rng.normal(size=Fk.shape))}


def em_energy(Fk, basis_idx, Omega_ce=.5):
    """Field energy as in spectrax._diagnostics (global sum, normalized by the global cell count)."""
    w = np.prod(1 + 2 * np.asarray(basis_idx), axis=1)
    return 0.5 * Omega_ce ** 2 * np.sum(np.abs(Fk) ** 2 / w) / (Fk.shape[1] * Fk.shape[2] * Fk.shape[3])


@pytest.mark.parametrize("layout", NORM_LAYOUTS, ids=lambda v: "x".join(map(str, v)))
def test_tree_norm_matches_serial_flattened_reference_and_is_layout_invariant(layout):
    """Same global values in every layout: norm == optimistix rms_norm on the serial flattened state, field energy unchanged."""
    mesh = mesh_of(layout)
    ck_s, fk_s = named_sharding(mesh, 6, species=True), named_sharding(mesh, 5)
    rep = NamedSharding(mesh, P())
    basis_idx = construct_idx_array(2, 2)  # Nl = 3
    fn = jax.jit(lambda e: (state_rms_norm(e), 0.5 * .25 * jnp.sum(jnp.abs(e[1]) ** 2 / jnp.prod(1 + 2 * basis_idx, axis=1)) / (NY * NX * NZ)),
                 in_shardings=((ck_s, fk_s),), out_shardings=(rep, rep))
    for name, (Ck, Fk) in norm_scenarios().items():
        ref = float(optx.rms_norm((jnp.array(Ck), jnp.array(Fk))))
        norm, energy = fn(jax.device_put((Ck, Fk), (ck_s, fk_s)))
        assert norm.sharding.is_fully_replicated and energy.sharding.is_fully_replicated, name
        np.testing.assert_allclose(float(norm), ref, rtol=1e-13, atol=0, err_msg=name)
        np.testing.assert_allclose(float(energy), em_energy(Fk, basis_idx), rtol=1e-13, atol=0, err_msg=name)
        np.testing.assert_allclose(float(norm), np.sqrt((np.sum(np.abs(Ck) ** 2) + np.sum(np.abs(Fk) ** 2)) / (Ck.size + Fk.size)), rtol=1e-13, atol=0)


def test_tree_norm_derivative_is_zero_safe_like_optimistix():
    zeros = (jnp.zeros((2, 6, 2, 4, 1, 3)), jnp.zeros((6, 2, 4, 1, 3)))
    for norm in (state_rms_norm, optx.rms_norm):
        assert float(norm(zeros)) == 0
    grad = jax.grad(state_rms_norm)(zeros)
    for g, g_ref in zip(grad, jax.grad(optx.rms_norm)(zeros)):
        assert np.all(np.isfinite(g)) and np.all(np.asarray(g) == 0) and np.all(np.asarray(g_ref) == 0)
    tangent = jax.tree_util.tree_map(lambda z: jnp.ones_like(z), zeros)
    assert float(jax.jvp(state_rms_norm, (zeros,), (tangent,))[1]) == float(jax.jvp(optx.rms_norm, (zeros,), (tangent,))[1]) == 0
    Ck, Fk = (jnp.array(v) for v in norm_scenarios()["random"])
    t = (jnp.array(np.random.default_rng(5).normal(size=Ck.shape)), jnp.array(np.random.default_rng(6).normal(size=Fk.shape)))
    np.testing.assert_allclose(jax.jvp(state_rms_norm, ((Ck, Fk),), (t,)), jax.jvp(optx.rms_norm, ((Ck, Fk),), (t,)), rtol=1e-13)
    # Sharded gradient at zero: finite and zero.
    mesh = mesh_of((2, 2, 1, 1))
    sh = (named_sharding(mesh, 6, species=True), named_sharding(mesh, 5))
    g = jax.jit(jax.grad(state_rms_norm), in_shardings=(sh,), out_shardings=sh)(jax.device_put(zeros, sh))
    assert all(np.all(np.asarray(x) == 0) for x in g)


# ------------------------------------------------------------------ compiled collectives

COLLECTIVES = ("all-reduce", "collective-permute", "all-gather", "all-to-all", "reduce-scatter")
_OP = re.compile(r"=\s*(.*?)\s+(" + "|".join(COLLECTIVES) + r")(?:-start)?\(")
_ITEMSIZE = {"f64": 8, "f32": 4, "c128": 16, "c64": 8, "s32": 4, "s64": 8, "u32": 4, "pred": 1}


def _groups(line, n):
    """Replica groups (lists of logical device ids) from explicit, iota or mesh-axis notation."""
    if m := re.search(r"replica_groups=mesh\[([^\]]*)\]\s*\{([^}]*)\}", line):
        sizes = [int(s) for s in re.findall(r"=(\d+)", m[1])]
        axes = [int(a) for a in re.findall(r"axis_(\d+)", m[2])]
        arr = np.moveaxis(np.arange(n).reshape(sizes), axes, range(len(sizes) - len(axes), len(sizes)))
        return arr.reshape(-1, prod(sizes[a] for a in axes)).tolist()
    if m := re.search(r"replica_groups=\[([\d,]+)\]<=\[([\d,]+)\](?:T\(([\d,]+)\))?", line):
        arr = np.arange(n).reshape([int(s) for s in m[2].split(",")])
        if m[3]:
            arr = arr.transpose([int(s) for s in m[3].split(",")])
        return arr.reshape([int(s) for s in m[1].split(",")]).tolist()
    if m := re.search(r"replica_groups=\{(\{.*?\})\}", line):
        return [[int(v) for v in g.split(",") if v] for g in re.findall(r"\{([\d,]*)\}", m[1])]
    return [list(range(n))]  # replica_groups={}: one group of all devices


def parse_collectives(text, n):
    """List of dicts: op, result shapes, elements, bytes, replica groups (all-reduce/gather...) or source-target pairs."""
    found = []
    for line in text.splitlines():
        if not (m := _OP.search(line)):
            continue
        shapes = [(dt, [int(d) for d in dims.split(",") if d]) for dt, dims in re.findall(r"\b([a-z]+\d*)\[([\d,]*)\]", m[1])]
        item = dict(op=m[2], shapes=[s for _, s in shapes], elements=sum(prod(s) for _, s in shapes),
                    bytes=sum(prod(s) * _ITEMSIZE.get(dt, 8) for dt, s in shapes))
        if m[2] == "collective-permute":
            pm = re.search(r"source_target_pairs=\{((?:\{\d+,\d+\},?)*)\}", line)
            item["pairs"] = [(int(a), int(b)) for a, b in re.findall(r"\{(\d+),(\d+)\}", pm[1])]
        else:
            item["groups"] = _groups(line, n)
        found.append(item)
    return found


def compiled_collectives(fn, mesh, *args):
    return parse_collectives(fn.lower(*args).compile().as_text(), mesh.devices.size)


def coords(mesh, i):
    return tuple(int(c) for c in np.unravel_index(i, mesh.devices.shape))  # logical id = position in mesh.devices.flat


COMPILE_CASE = {**A2D, "Nx": 32, "Ny": 16}


@pytest.mark.parametrize("layout", [(1, 2, 2, 1), (2, 2, 1, 1), (2, 1, 2, 1), (2, 1, 1, 1)], ids=lambda v: "x".join(map(str, v)))
def test_compiled_rhs_communicates_only_faces_and_species_currents(layout):
    mesh = mesh_of(layout)
    ps, px, py, pz = layout
    Ck, Fk, args, N = make_args(**COMPILE_CASE)
    Ns, Nv, Ny, Nx, Nz, Nl = Ck.shape
    shardings = in_shardings(mesh)
    found = compiled_collectives(rhs_fn(layout, N), mesh, *jax.device_put((Ck, Fk, args), shardings))
    ck_tile = (Ns // ps) * Nv * (Ny // py) * (Nx // px) * Nz * Nl
    split_extents = [n for n, p in zip((Ny // py, Nx // px, Nz // pz), (py, px, pz)) if p > 1]
    assert not [c for c in found if c["op"] in ("all-gather", "all-to-all", "reduce-scatter")]
    permutes, reduces = ([c for c in found if c["op"] == op] for op in ("collective-permute", "all-reduce"))
    # Spatial face exchange: one cell along the shifted axis, same species coordinate, a single spatial axis differs.
    for c in permutes:
        assert c["elements"] <= ck_tile // min(split_extents)
        for a, b in c["pairs"]:
            ca, cb = coords(mesh, a), coords(mesh, b)
            assert ca[0] == cb[0] and sum(u != v for u, v in zip(ca[1:], cb[1:])) == 1
    # At most two one-cell faces per split direction and array, far smaller than the local state.
    fk_tile = 6 * (Ny // py) * (Nx // px) * Nz * Nl
    assert sum(c["elements"] for c in permutes) <= (ck_tile + fk_tile) * sum(2 / n for n in split_extents)
    assert sum(c["elements"] for c in permutes) < (ck_tile + fk_tile) / 2
    if ps == 1:
        assert not [c for c in reduces if c["elements"] > 1]  # species sum is local: no non-scalar all-reduce
    else:
        j_tile = 3 * (Ny // py) * (Nx // px) * Nz * Nl
        assert any(c["elements"] == j_tile for c in reduces)  # one stacked 3-component current reduction
        for c in reduces:
            assert c["elements"] == 1 or c["elements"] <= j_tile
            for g in c["groups"]:  # joins species coordinates at FIXED spatial coordinates
                cs = [coords(mesh, i) for i in g]
                assert sorted(x[0] for x in cs) == list(range(ps)) and len({x[1:] for x in cs}) == 1
    if px * py * pz == 1:
        assert not permutes


@pytest.mark.parametrize("layout", [(1, 2, 2, 1), (2, 2, 1, 1)], ids=lambda v: "x".join(map(str, v)))
def test_compiled_tree_norm_has_only_scalar_all_reduces(layout):
    mesh = mesh_of(layout)
    ck_s, fk_s = named_sharding(mesh, 6, species=True), named_sharding(mesh, 5)
    Ck, Fk = (jnp.array(v) for v in norm_scenarios()["random"])
    fn = jax.jit(state_rms_norm, in_shardings=((ck_s, fk_s),), out_shardings=NamedSharding(mesh, P()))
    found = compiled_collectives(fn, mesh, jax.device_put((Ck, Fk), (ck_s, fk_s)))
    assert found and all(c["op"] == "all-reduce" and c["elements"] == len(c["shapes"]) for c in found)  # scalars only
