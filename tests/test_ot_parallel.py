"""Integration tests: tiny Orszag-Tang-like trajectories on 1, 2 and 4 logical CPU devices (spatial, species, hybrid).

``tests/conftest.py`` provides four logical CPU devices. Layouts are written ``(ps, px, py, pz)``. Parallel runs are
compared with the single-device run of the same GLOBAL problem at common saved times; tolerances only have to cover
floating-point reordering (different reduction/collective summation order), not bitwise equality. Compiles dominate the
cost, so each (layout, integrator) pair is run once and cached.
"""
import importlib.util
import io
import os
import subprocess
import sys
import tarfile
from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest
from diffrax import Dopri8, NoProgressMeter, Tsit5
from jax.sharding import NamedSharding, PartitionSpec as P

from spectrax import DG_project, simulation
from spectrax._runtime import make_mesh, parallel_config, select_devices

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "21f506812c9a561d61516adc1afc31a1be26374c"
spec = importlib.util.spec_from_file_location("OT_parallel", ROOT / "Examples" / "OT_parallel.py")
OT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OT)

# Asymmetric Nx != Ny, N_DG > 1 (Nl = 3), dims = 2. Original OT physics (ms, qs, alpha_s, Omega_ce, nu, lengths).
GRID = dict(Nx=8, Ny=4, Nz=1, Nn=2, Nm=2, Np=2, Ns=2, N_DG=2, dims=2, timesteps=3)
OT_INPUT = dict(Lx=50.0, Ly=50.0, Lz=1.0, ms=[1.0, 25.0], qs=[-1.0, 1.0], alpha_s=[.25, .25, .25, .05, .05, .05],
                u_s=[0.0] * 6, Omega_ce=0.5, nu=1.0, D=0.0)
REF = (1, 1, 1, 1)
FIXED_LAYOUTS = [(1, 2, 2, 1), (2, 1, 1, 1), (2, 2, 1, 1)]  # spatial blocks, species-only, hybrid
ADAPTIVE_LAYOUTS = [(2, 2, 1, 1)]


def config(layout, Ns=2, grid=GRID):
    ps, px, py, pz = layout
    return parallel_config(dict(device="cpu", num_devices=ps * px * py * pz, species_partitions=ps, mesh_shape=[px, py, pz]),
                           grid["Nx"], grid["Ny"], grid["Nz"], Ns)


@lru_cache(maxsize=None)
def run(layout, adaptive):
    """Short OT run on ``layout``; fixed steps use Tsit5, adaptive runs use Dopri8 (rtol=atol=1e-8)."""
    cfg = config(layout)
    ip = {**OT_INPUT, "t_max": 0.04 if adaptive else 0.008, "ode_tolerance": 1e-8}
    sp = dict(GRID, parallel=cfg)
    OT.initial_state(ip, sp)
    out = simulation(ip, **GRID, dt=1e-3 if adaptive else 2e-3, solver=Dopri8() if adaptive else Tsit5(),
                     adaptive_time_step=adaptive, parallel=cfg, progress_meter=NoProgressMeter())
    jax.block_until_ready((out["Ck"], out["Fk"]))
    return cfg, out


def host(out, key):
    return np.asarray(out[key])


def assert_close(a, b, rtol=1e-9):
    """Relative agreement at the scale of the whole array (coefficients differ by orders of magnitude)."""
    a, b = np.asarray(a), np.asarray(b)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol * np.max(np.abs(b)) + 1e-300)


def test_reference_run_is_physical(base_run):  # base_run: start the background reference-commit run early
    _, out = run(REF, False)
    assert out["Ck"].shape == (3, 2 * 8, 4, 8, 1, 3) and out["Fk"].shape == (3, 6, 4, 8, 1, 3)
    assert np.all(np.isfinite(host(out, "Ck"))) and not np.allclose(host(out, "Fk")[-1], host(out, "Fk")[0])
    np.testing.assert_allclose(host(out, "time"), [0, 0.004, 0.008])


@pytest.mark.parametrize("layout", FIXED_LAYOUTS)
def test_fixed_step_trajectory_matches_single_device(layout):
    # Same global problem, same fixed steps: only summation order differs, so roundoff-level agreement is expected.
    _, ref = run(REF, False)
    _, out = run(layout, False)
    for key in ("Ck", "Fk", "time"):
        assert_close(out[key], ref[key], 1e-9)
    assert int(out["num_steps"]) == int(ref["num_steps"])


@pytest.mark.parametrize("layout", ADAPTIVE_LAYOUTS)
def test_adaptive_dopri8_trajectory_matches_single_device(layout):
    # Roundoff-level differences in the error norm may flip a step decision, so only the solution is compared, to a
    # small multiple of the integrator tolerance (1e-8); step counts may differ and are not required to be identical.
    _, ref = run(REF, True)
    _, out = run(layout, True)
    for key in ("Ck", "Fk"):
        assert_close(out[key], ref[key], 1e-6)
    assert np.asarray(out["num_steps"]).shape == () and int(out["num_steps"]) >= 1  # one scalar controller
    assert abs(int(out["num_steps"]) - int(ref["num_steps"])) <= 2


@pytest.mark.parametrize("layout", FIXED_LAYOUTS)
def test_diagnostics_agree_and_field_energy_counted_once(layout):
    _, ref = run(REF, False)
    _, out = run(layout, False)
    for key in ("kinetic_energy_species", "kinetic_energy", "EM_energy", "total_energy"):
        assert_close(out[key], ref[key], 1e-9)
    assert out["kinetic_energy_species"].shape == (3, 2)
    # Independent host evaluation from the gathered state (test-only gather). A field buffer replicated over species
    # groups and summed per group would give ps times this value.
    Fk, Ck = host(out, "Fk"), host(out, "Ck")
    w = np.prod(1 + 2 * host(out, "basis_idx"), axis=1)
    np.testing.assert_allclose(host(out, "EM_energy"), 0.5 * 0.5 ** 2 * np.sum(np.abs(Fk) ** 2 / w, axis=(1, 2, 3, 4, 5)) / (8 * 4), rtol=1e-12)
    np.testing.assert_allclose(host(out, "kinetic_energy"), host(out, "kinetic_energy_species").sum(axis=1), rtol=1e-13)
    np.testing.assert_allclose(host(out, "total_energy"), host(out, "kinetic_energy") + host(out, "EM_energy"), rtol=1e-13)
    C = Ck.reshape(3, 2, 8, 4, 8, 1, 3)[..., 0].mean(axis=(3, 4, 5))  # (Nt, Ns, Nv): domain-mean cell-mean coefficients
    a = np.array(OT_INPUT["alpha_s"]).reshape(2, 3)
    want = 0.5 * np.array(OT_INPUT["ms"]) * a.prod(axis=1) * 0.5 * (a ** 2).sum(axis=1) * C[..., 0]  # u_s = 0 and Nn = Nm = Np = 2: only C000 contributes
    np.testing.assert_allclose(host(out, "kinetic_energy_species"), want, rtol=1e-10)


@pytest.mark.parametrize("layout", [REF, *FIXED_LAYOUTS])
def test_state_sharding_and_participating_devices(layout):
    cfg, out = run(layout, False)
    ps, px, py, pz = layout
    mesh = make_mesh(cfg)
    assert mesh.axis_names == ("species", "x", "y", "z") and tuple(mesh.shape.values()) == layout
    devices = set(select_devices(cfg))
    assert len(devices) == ps * px * py * pz
    expected = {  # key: (PartitionSpec, local shard shape)
        "Ck": (P(None, "species", "y", "x", "z", None), (3, 16 // ps, 4 // py, 8 // px, 1 // pz, 3)),
        "Fk": (P(None, None, "y", "x", "z", None), (3, 6, 4 // py, 8 // px, 1 // pz, 3)),
        "Ck_0": (P("species", None, None, None, "y", "x", "z", None), (2 // ps, 2, 2, 2, 4 // py, 8 // px, 1 // pz, 3)),
        "Fk_0": (P(None, "y", "x", "z", None), (6, 4 // py, 8 // px, 1 // pz, 3)),
    }
    for key, (pspec, local) in expected.items():
        a = out[key]
        assert a.sharding.is_equivalent_to(NamedSharding(mesh, pspec), a.ndim), (key, a.sharding)
        assert a.sharding.shard_shape(a.shape) == local
        assert a.sharding.device_set == devices == {s.device for s in a.addressable_shards}  # exactly these devices hold data
    assert out["Ck_0"].shape == (2, 2, 2, 2, 4, 8, 1, 3) and out["Fk_0"].shape == (6, 4, 8, 1, 3)


def test_ot_initial_state_matches_original_formulas_without_all_to_all(monkeypatch):
    # Hybrid layout: the species-sharded OT projection must not communicate (species dependence is broadcast, not stacked),
    # and the state equals the original 2D_Orszag_Tang.py construction (stacked electron/ion velocities) on one device.
    import jax.numpy as jnp
    from jax.scipy.special import factorial
    import spectrax._initialization as init
    hlo, jit = [], init.jit
    monkeypatch.setattr(init, "jit", lambda f, **kw: type("Spy", (), {"__call__": lambda self, *a, g=jit(f, **kw): (hlo.append(g.lower(*a).compile().as_text()), g(*a))[1]})())
    ip, sp = {**OT_INPUT}, dict(GRID, parallel=config((2, 2, 1, 1)))
    OT.initial_state(ip, sp)
    monkeypatch.undo()
    assert hlo and not any("all-to-all" in h for h in hlo)
    dB, Om, U0, k = OT.deltaB, 0.5, OT.deltaB * 0.5 / 5.0, 2 * np.pi / 50.0
    a = np.array(OT_INPUT["alpha_s"]).reshape(2, 3)
    ax, ay, az = (a[:, i, None, None, None, None, None, None, None, None, None] for i in range(3))
    Ue = lambda x, y, z: jnp.array([-U0 * jnp.sin(k * y), U0 * jnp.sin(k * x), -dB * Om * (2 * k * jnp.cos(2 * k * x) + k * jnp.cos(k * y))])
    Ui = lambda x, y, z: jnp.array([-U0 * jnp.sin(k * y), U0 * jnp.sin(k * x), jnp.zeros_like(x)])
    U = lambda x, y, z: jnp.stack([Ue(x, y, z), Ui(x, y, z)])[:, :, None, None, None]
    p, m, n = (np.arange(2).reshape([1] * (1 + i) + [2] + [1] * (8 - i)) for i in range(3))
    C0 = lambda x, y, z: (jnp.sqrt(2 ** (n + m + p) / (factorial(n) * factorial(m) * factorial(p))) / (ax ** (n + 1) * ay ** (m + 1) * az ** (p + 1))
                          * U(x, y, z)[:, 0] ** n * U(x, y, z)[:, 1] ** m * U(x, y, z)[:, 2] ** p)
    ref = DG_project(C0, OT_INPUT, dict(GRID, parallel=config(REF)), species=True)
    assert_close(ip["Ck_0"], ref, 1e-13)


def test_failed_solve_raises_from_host_wrapper():
    ip = {**OT_INPUT, "t_max": 0.008, "ode_tolerance": 1e-8}
    OT.initial_state(ip, dict(GRID, parallel=config(FIXED_LAYOUTS[0])))
    with pytest.raises(RuntimeError):
        simulation(ip, **GRID, dt=2e-3, solver=Tsit5(), adaptive_time_step=False, parallel=config(FIXED_LAYOUTS[0]),
                   max_steps=2, progress_meter=NoProgressMeter())


def test_default_projection_sharded_matches_single_device():
    # Species-sharded, spatially sharded projection of a nonconstant species-dependent function vs one device.
    s = np.array([1.0, -2.0, 0.5, 3.0]).reshape(4, 1, 1, 1, 1, 1, 1)
    f = lambda x, y, z: s * jax.numpy.sin(0.3 * x + 0.5 * y) * (1 + 0.1 * z) + x * y
    ip = dict(Lx=7.0, Ly=5.0, Lz=2.0)
    sp = dict(Nx=8, Ny=4, Nz=1, N_DG=3, dims=2, Ns=4)
    ref = DG_project(f, ip, {**sp, "parallel": config(REF, 4, sp)}, species=True)
    for layout in [(2, 2, 1, 1), (4, 1, 1, 1), (1, 2, 2, 1)]:
        cfg = config(layout, 4, sp)
        out = DG_project(f, ip, {**sp, "parallel": cfg}, species=True)
        assert out.shape == (4, 4, 8, 1, 6)
        assert out.sharding.is_equivalent_to(NamedSharding(make_mesh(cfg), P("species", "y", "x", "z", None)), out.ndim)
        assert_close(out, ref, 1e-13)


def test_supplied_initial_data_skips_default_projection_ns4(monkeypatch):
    # Four species with distinct parameters and full user data; ps=2 puts two species on each shard.
    Ns, grid = 4, dict(GRID, Ns=4)
    rng = np.random.default_rng(0)
    ip = dict(Lx=50.0, Ly=50.0, Lz=1.0, ms=[1.0, 25.0, 2.0, 9.0], qs=[-1.0, 1.0, -2.0, 3.0],
              alpha_s=rng.uniform(.1, .3, 12), u_s=rng.uniform(-.05, .05, 12), Omega_ce=0.5, nu=1.0, D=0.0, t_max=0.004,
              ode_tolerance=1e-8, Ck_0=rng.normal(size=(4, 2, 2, 2, 4, 8, 1, 3)), Fk_0=rng.normal(size=(6, 4, 8, 1, 3)))
    monkeypatch.setattr("spectrax._initialization.DG_project", lambda *a, **k: pytest.fail("default projection was evaluated"))
    outs = [simulation(dict(ip), **grid, dt=2e-3, solver=Tsit5(), adaptive_time_step=False, parallel=config(layout, Ns, grid),
                       progress_meter=NoProgressMeter()) for layout in (REF, (2, 2, 1, 1))]
    assert outs[1]["Ck"].shape == (3, 4 * 8, 4, 8, 1, 3) and outs[1]["Ck_0"].shape == (4, 2, 2, 2, 4, 8, 1, 3)
    assert outs[1]["Ck"].sharding.shard_shape(outs[1]["Ck"].shape) == (3, 16, 4, 4, 1, 3)  # two whole species per shard
    assert outs[1]["Ck_0"].sharding.shard_shape(outs[1]["Ck_0"].shape) == (2, 2, 2, 2, 4, 4, 1, 3)  # NumPy input placed sharded
    for key in ("Ck", "Fk", "kinetic_energy_species"):
        assert_close(outs[1][key], outs[0][key], 1e-9)
    assert not np.allclose(host(outs[1], "kinetic_energy_species")[:, 0], host(outs[1], "kinetic_energy_species")[:, 2])


def test_ns_other_than_two_without_user_data_raises():
    with pytest.raises(ValueError, match="Fk_0"):
        simulation({}, Nx=4, Nn=2, Ns=3, N_DG=1, timesteps=2, progress_meter=NoProgressMeter())
    with pytest.raises(ValueError, match="Inconsistent shapes"):  # right size, wrong layout (Nv, Ns, ...)
        simulation({"Ck_0": np.ones((3, 2, 1, 4, 1, 1))}, Nx=4, Nn=3, N_DG=1, timesteps=2, progress_meter=NoProgressMeter())


ARGS = "Nx=5, Nn=4, Ns=2, N_DG=2, dims=1, timesteps=3, dt=0.01, solver=Tsit5(), adaptive_time_step=False"


@pytest.fixture(scope="module")
def base_run(tmp_path_factory):
    """Start a default-style run of the reference commit's package in a subprocess (its compile overlaps other tests)."""
    tmp = tmp_path_factory.mktemp("base")
    try:
        tar = subprocess.run(["git", "-C", str(ROOT), "archive", BASE_COMMIT, "spectrax"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    tarfile.open(fileobj=io.BytesIO(tar)).extractall(tmp / "base", filter="data")
    (tmp / "base" / "spectrax" / "version.py").write_text("__version__ = '0'\n")  # gitignored, absent from the archive
    script = tmp / "run_base.py"
    script.write_text("import sys, numpy as np\nfrom diffrax import Tsit5\nfrom spectrax import simulation\n"
                      f"out = simulation({{'t_max': 0.5}}, {ARGS})\n"
                      "np.savez(sys.argv[1], Ck=out['Ck'], Fk=out['Fk'], time=out['time'], E=out['total_energy'])\n")
    env = {k: v for k, v in os.environ.items() if k not in ("JAX_NUM_CPU_DEVICES", "XLA_FLAGS", "PYTHONPATH")}
    return subprocess.Popen([sys.executable, str(script), str(tmp / "base.npz")], env={**env, "PYTHONPATH": str(tmp / "base")}, cwd=tmp), tmp / "base.npz"


def test_legacy_single_device_parity_with_base_commit(base_run):
    """Default-style 1D run: public shapes and values equal the reference commit's (the state change, eigh and sharding
    must not alter single-device numerics beyond roundoff)."""
    if base_run is None:
        pytest.skip("base commit is not available in this checkout")
    proc, npz = base_run
    assert proc.wait(timeout=300) == 0
    old = np.load(npz)
    out = eval(f"simulation({{'t_max': 0.5}}, {ARGS}, progress_meter=NoProgressMeter())")
    assert out["Ck"].shape == (3, 2 * 4, 1, 5, 1, 2) and out["Fk"].shape == (3, 6, 1, 5, 1, 2)  # public shapes unchanged
    for key in ("Ck", "Fk", "time"):
        assert out[key].shape == old[key].shape
        assert_close(out[key], old[key], 1e-10)
    assert_close(out["total_energy"], old["E"], 1e-10)
