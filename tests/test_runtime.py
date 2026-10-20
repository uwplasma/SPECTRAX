"""Runtime configuration: in-process validation tests and fresh-subprocess bootstrap/device-selection tests."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from spectrax._runtime import ParallelConfig, make_mesh, named_sharding, parallel_config, read_input

ROOT = Path(__file__).resolve().parents[1]
TOML = """[input_parameters]
ode_tolerance = 1e-7
[solver_parameters]
Nx = 8
Ny = 8
Nz = 1
Ns = 2
{parallel}"""
PRELUDE = """
import json, sys
def out(**kw): print("RESULT" + json.dumps(kw))
def mesh_info(cfg):
    from spectrax import make_mesh
    m = make_mesh(cfg)
    return dict(ndev=int(m.devices.size), shape=list(m.devices.shape), names=list(m.axis_names),
                types=[str(t) for t in m.axis_types], platforms=sorted({d.platform for d in m.devices.flat}))
"""

def run(tmp_path, body, parallel="", env=None):
    """Run ``body`` in a fresh python process (path of a run TOML in ``P``); returns the dict passed to ``out``."""
    (tmp_path / "run.toml").write_text(TOML.format(parallel=parallel))
    child_env = {k: v for k, v in os.environ.items() if k not in ("JAX_NUM_CPU_DEVICES", "XLA_FLAGS")}
    child_env.update(PYTHONPATH=str(ROOT), **(env or {}))
    code = PRELUDE + f"P = {str(tmp_path / 'run.toml')!r}\n" + textwrap.dedent(body)
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=child_env, capture_output=True, text=True, timeout=110)
    lines = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
    assert r.returncode == 0 and lines, f"child failed ({r.returncode}):\n{r.stdout}\n{r.stderr[-3000:]}"
    return json.loads(lines[-1][6:])

LOAD = """
    from spectrax import load_parameters
    cfg = load_parameters(P)[1]["parallel"]
    out(cfg=[cfg.device, cfg.num_devices, cfg.species_partitions, list(cfg.mesh_shape)], **mesh_info(cfg))
"""

# ---------------------------------------------------------------- fresh-process tests through load_parameters
@pytest.mark.parametrize("device", ["cpu", "auto"])
@pytest.mark.parametrize("n, shape", [(1, [1, 1, 1, 1]), (2, [1, 2, 1, 1]), (4, [1, 4, 1, 1])])
def test_toml_logical_cpu_devices(tmp_path, device, n, shape):
    res = run(tmp_path, LOAD, f'[parallel]\ndevice = "{device}"\nnum_devices = {n}\n')
    assert res["ndev"] == n and res["shape"] == shape and res["platforms"] == ["cpu"]
    assert res["names"] == ["species", "x", "y", "z"] and set(res["types"]) == {"AxisType.Auto"}

@pytest.mark.parametrize("parallel, shape", [
    ('device = "cpu"\nnum_devices = 4\nmesh_shape = [2, 2, 1]', [1, 2, 2, 1]),
    ('device = "cpu"\nnum_devices = 4\nspecies_partitions = 2\nmesh_shape = [2, 1, 1]', [2, 2, 1, 1]),
    ('device = "auto"\nnum_devices = 2\nspecies_partitions = 2', [2, 1, 1, 1])])
def test_toml_species_and_spatial_layouts(tmp_path, parallel, shape):
    assert run(tmp_path, LOAD, "[parallel]\n" + parallel)["shape"] == shape

def test_no_parallel_section_is_single_auto_device(tmp_path):
    res = run(tmp_path, LOAD)
    assert res["cfg"] == ["auto", 1, 1, [1, 1, 1]] and res["shape"] == [1, 1, 1, 1] and res["ndev"] == 1

def test_lazy_imports_do_not_initialize_backend(tmp_path):
    res = run(tmp_path, """
        import spectrax
        no_jax = "jax" not in sys.modules
        from spectrax import load_parameters, simulation, DG_project
        cfg = load_parameters(P)[1]["parallel"]
        out(no_jax=no_jax, **mesh_info(cfg))
    """, '[parallel]\ndevice = "cpu"\nnum_devices = 4\n')
    assert res["no_jax"] and res["ndev"] == 4

def test_standalone_dg_project_bootstraps_and_submodule_imports_enable_x64(tmp_path):
    res = run(tmp_path, """
        from spectrax._model import Hermite_DG_rhs  # direct submodule import still enables x64
        import jax.numpy as jnp
        from spectrax import DG_project
        c = DG_project(lambda x, y, z: jnp.sin(x), {"Lx": 1.0, "Ly": 1.0, "Lz": 1.0},
                       {"Nx": 8, "Ny": 1, "Nz": 1, "N_DG": 2, "dims": 1, "parallel": {"device": "cpu", "num_devices": 4}})
        out(dtype=str(c.dtype), ndev=len(c.sharding.device_set))
    """)
    assert res == {"dtype": "float64", "ndev": 4}

def test_late_incompatible_backend_gives_fresh_process_error(tmp_path):
    res = run(tmp_path, """
        import jax; jax.devices()                    # backend initialized with 1 CPU device
        from spectrax import load_parameters
        cfg = load_parameters(P)[1]["parallel"]
        try: mesh_info(cfg)
        except RuntimeError as e: out(msg=str(e))
    """, '[parallel]\ndevice = "cpu"\nnum_devices = 4\n')
    assert "fresh" in res["msg"] and "num_devices=4" in res["msg"] and "1 local cpu" in res["msg"]

def test_gpu_unavailable_no_fallback(tmp_path):
    res = run(tmp_path, """
        from spectrax import load_parameters
        cfg = load_parameters(P)[1]["parallel"]
        try: mesh_info(cfg)
        except RuntimeError as e: out(msg=str(e))
    """, '[parallel]\ndevice = "gpu"\nnum_devices = 1\n')
    assert "'gpu'" in res["msg"] and "no fallback" in res["msg"]

# ---------------------------------------------------------------- fresh-process tests using only _runtime
def test_read_input_bootstraps_four_cpu_devices(tmp_path):
    res = run(tmp_path, """
        from spectrax._runtime import read_input
        inp, sol = read_input(P)
        out(solver=sol.get("solver"), tol=inp["ode_tolerance"], **mesh_info(sol["parallel"]))
    """, '[parallel]\ndevice = "cpu"\nnum_devices = 4\nmesh_shape = [2, 2, 1]\n')
    assert res["ndev"] == 4 and res["shape"] == [1, 2, 2, 1] and res["solver"] is None and res["tol"] == 1e-7

def test_smaller_subset_of_available_devices(tmp_path):
    res = run(tmp_path, """
        import jax; jax.config.update("jax_num_cpu_devices", 4); jax.devices()
        from spectrax import bootstrap
        out(**mesh_info(bootstrap(dict(device="cpu", num_devices=2), Nx=8)))
    """)
    assert res["ndev"] == 2 and res["shape"] == [1, 2, 1, 1]

def test_existing_cpu_settings_are_respected(tmp_path):
    res = run(tmp_path, """
        import jax, os
        from spectrax import bootstrap
        bootstrap(dict(device="cpu", num_devices=2), Nx=8)       # a larger user setting is not decreased
        out(n=jax.config.jax_num_cpu_devices, ndev=len(jax.devices()), xla=os.environ.get("XLA_FLAGS"))
    """, env={"JAX_NUM_CPU_DEVICES": "3"})
    assert res["n"] == 3 and res["ndev"] == 3 and res["xla"] is None
    res = run(tmp_path, """
        import jax
        from spectrax import bootstrap
        bootstrap(dict(device="cpu", num_devices=4), Nx=8)       # XLA_FLAGS forcing a count is left alone
        out(n=jax.config.jax_num_cpu_devices, ndev=len(jax.devices()))
    """, env={"XLA_FLAGS": "--xla_force_host_platform_device_count=2"})
    assert res["n"] == -1 and res["ndev"] == 2

# ---------------------------------------------------------------- in-process validation
def cfg_of(parallel, **grid):
    return parallel_config(parallel, **{"Nx": 8, "Ny": 8, "Nz": 1, "Ns": 2, **grid})

def test_defaults_and_passthrough():
    assert parallel_config() == ParallelConfig("auto", 1, 1, (1, 1, 1)) and parallel_config({}).shape == (1, 1, 1, 1)
    c = ParallelConfig("cpu", 4, 2, (2, 1, 1))
    assert parallel_config(c, Nx=8) == c and c.shape == (2, 2, 1, 1) and hash(c) == hash(ParallelConfig("cpu", 4, 2, (2, 1, 1)))
    with pytest.raises(ValueError, match="must divide"):
        parallel_config(c, Nx=7)
    assert cfg_of({"num_devices": __import__("numpy").int64(2)}).num_devices == 2  # NumPy integers are counts too

@pytest.mark.parametrize("parallel, exc, match", [
    ({"devices": "cpu"}, ValueError, "Unknown.*devices"),
    ("cpu", TypeError, "mapping"),
    (4, TypeError, "mapping"),
    ({"device": "tpu"}, ValueError, "device must be"),
    ({"device": 1}, ValueError, "device must be"),
    ({"num_devices": True}, TypeError, "num_devices must be an integer"),
    ({"species_partitions": True}, TypeError, "species_partitions must be an integer"),
    ({"mesh_shape": [True, 1, 1]}, TypeError, "mesh_shape entries"),
    ({"num_devices": 2.0}, TypeError, "num_devices"),
    ({"num_devices": "2"}, TypeError, "num_devices"),
    ({"mesh_shape": [1, 1.5, 1]}, TypeError, "mesh_shape entries"),
    ({"num_devices": 0}, ValueError, "positive"),
    ({"num_devices": -2}, ValueError, "positive"),
    ({"species_partitions": 0}, ValueError, "positive"),
    ({"mesh_shape": [0, 1, 1]}, ValueError, "positive"),
    ({"num_devices": 3, "species_partitions": 2}, ValueError, "not divisible by species_partitions=2"),
    ({"num_devices": 4, "species_partitions": 4}, ValueError, "must divide Ns=2"),               # ps > Ns
    ({"mesh_shape": [1, 1]}, ValueError, "3 entries"),
    ({"mesh_shape": [1, 1, 1, 1]}, ValueError, "3 entries"),
    ({"mesh_shape": "111"}, ValueError, "3 entries"),
    ({"num_devices": 4, "mesh_shape": [2, 1, 1]}, ValueError, "must equal"),
    ({"num_devices": 2, "species_partitions": 2, "mesh_shape": [2, 1, 1]}, ValueError, "must equal"),
    ({"num_devices": 3, "mesh_shape": [3, 1, 1]}, ValueError, "px=3 must divide Nx=8"),
    ({"num_devices": 3, "mesh_shape": [1, 3, 1]}, ValueError, "py=3 must divide Ny=8"),
    ({"num_devices": 2, "mesh_shape": [1, 1, 2]}, ValueError, "pz=2 must divide Nz=1.*singleton"),
    ({"num_devices": 3}, ValueError, "No valid mesh_shape"),
])
def test_invalid_parallel(parallel, exc, match):
    with pytest.raises(exc, match=match):
        cfg_of(parallel)

def test_species_partitions_must_divide_ns_and_pz_divides_nz():
    with pytest.raises(ValueError, match="must divide Ns=3"):
        cfg_of({"num_devices": 2, "species_partitions": 2}, Ns=3)
    with pytest.raises(ValueError, match="pz=2 must divide Nz=3"):
        cfg_of({"num_devices": 2, "mesh_shape": [1, 1, 2]}, Nz=3)
    assert cfg_of({"num_devices": 3, "mesh_shape": [1, 1, 3]}, Nz=3).shape == (1, 1, 1, 3)

@pytest.mark.parametrize("n, grid, expected", [
    (1, dict(Nx=8, Ny=8), (1, 1, 1)),
    (4, dict(Nx=8, Ny=8), (4, 1, 1)),             # cost tie with (2,2,1): fewer split directions wins
    (4, dict(Nx=16, Ny=4), (4, 1, 1)),            # 4/16 < 4/4
    (4, dict(Nx=4, Ny=16), (1, 4, 1)),
    (16, dict(Nx=16, Ny=16), (4, 4, 1)),          # 4/16+4/16 < 16/16
    (8, dict(Nx=8, Ny=8, Nz=8), (4, 2, 1)),             # (4,2,1) and (2,2,2) tie on cost; fewer split directions
    (4, dict(Nx=4, Ny=4, Nz=4), (4, 1, 1)),       # cost tie: fewer split directions, then x
    (4, dict(Nx=2, Ny=2, Nz=2), (2, 2, 1)),       # all costs equal and two split directions: x before y before z
    (6, dict(Nx=8, Ny=9), (2, 3, 1)),             # only valid triple
    (4, dict(Nx=1, Ny=8), (1, 4, 1)),             # singleton x forced to 1
    (4, dict(Nx=1, Ny=1, Nz=4), (1, 1, 4)),
    (4, dict(Nx=2, Ny=2, Nz=1), (2, 2, 1)),
])
def test_auto_mesh_selection(n, grid, expected):
    assert cfg_of({"num_devices": n}, **grid).mesh_shape == expected

def test_auto_mesh_never_enables_species_sharding():
    assert cfg_of({"num_devices": 2}).shape == (1, 2, 1, 1)
    assert cfg_of({"num_devices": 4, "species_partitions": 2}).shape == (2, 2, 1, 1)
    assert cfg_of({"num_devices": 4, "species_partitions": 4}, Ns=4).shape == (4, 1, 1, 1)
    assert cfg_of({"num_devices": 4}, Ns=4).species_partitions == 1

@pytest.mark.parametrize("parallel, shape", [   # documented valid examples for Ns=2
    ({"device": "cpu", "num_devices": 4, "species_partitions": 1, "mesh_shape": [2, 2, 1]}, (1, 2, 2, 1)),
    ({"device": "cpu", "num_devices": 4, "species_partitions": 2, "mesh_shape": [2, 1, 1]}, (2, 2, 1, 1)),
    ({"device": "cpu", "num_devices": 2, "species_partitions": 2, "mesh_shape": [1, 1, 1]}, (2, 1, 1, 1)),
    ({"device": "cpu", "num_devices": 1, "species_partitions": 1, "mesh_shape": [1, 1, 1]}, (1, 1, 1, 1)),
    ({"device": "cpu", "num_devices": 1}, (1, 1, 1, 1)),
])
def test_documented_examples(parallel, shape):
    assert cfg_of(parallel, Nx=32, Ny=32).shape == shape

def test_ns4_two_species_per_shard():
    assert cfg_of({"num_devices": 4, "species_partitions": 2, "mesh_shape": [2, 1, 1]}, Ns=4).shape == (2, 2, 1, 1)

def test_read_input_validates_parallel_with_grid(tmp_path):
    f = tmp_path / "r.toml"
    f.write_text(TOML.format(parallel='[parallel]\nnum_devices = 3\nmesh_shape = [3, 1, 1]\n'))
    with pytest.raises(ValueError, match="px=3 must divide Nx=8"):
        read_input(f)
    f.write_text(TOML.format(parallel='[parallel]\nnum_devices = 4\nmesh_shape = [1, 4, 1]\n'))
    inp, sol = read_input(f)     # 4 CPU devices are available in-process (tests/conftest.py)
    assert sol["parallel"] == ParallelConfig("auto", 4, 1, (1, 4, 1)) and sol["Nx"] == 8 and inp["ode_tolerance"] == 1e-7

# ---------------------------------------------------------------- mesh and shardings (4 CPU devices from conftest)
@pytest.fixture(scope="module")
def mesh():
    return make_mesh(ParallelConfig("cpu", 4, 2, (2, 1, 1)))

def test_make_mesh_cached_axes_and_devices(mesh):
    import jax
    assert mesh is make_mesh(ParallelConfig("cpu", 4, 2, (2, 1, 1)))
    assert mesh.axis_names == ("species", "x", "y", "z") and mesh.devices.shape == (2, 2, 1, 1)
    assert mesh.axis_types == (jax.sharding.AxisType.Auto,) * 4 and len(set(mesh.devices.flat)) == 4
    assert make_mesh(ParallelConfig("cpu", 2, 1, (2, 1, 1))).devices.size == 2
    with pytest.raises(RuntimeError, match="only 4 local cpu"):
        make_mesh(ParallelConfig("cpu", 8, 1, (8, 1, 1)))

@pytest.mark.parametrize("kwargs, spec", [
    (dict(ndim=6, species=True), ("species", None, "y", "x", "z", None)),                  # Ck
    (dict(ndim=5), (None, "y", "x", "z", None)),                                           # Fk
    (dict(ndim=8, species=True), ("species", None, None, None, "y", "x", "z", None)),      # rank-8 Ck
    (dict(ndim=7, species=True, lead=1), (None, "species", None, "y", "x", "z", None)),    # saved Ck
    (dict(ndim=6, lead=1), (None, None, "y", "x", "z", None)),                             # saved Fk
    (dict(ndim=9, tail=3), (None, None, None, "y", "x", "z", None, None, None)),           # projection integrand
    (dict(ndim=8, species=True, tail=3), ("species", None, "y", "x", "z", None, None, None)),
    (dict(ndim=6, tail=3), ("y", "x", "z", None, None, None)),                             # no leading slot
    (dict(ndim=4), ("y", "x", "z", None)),                                                 # (Ny,Nx,Nz,Nl)
    (dict(ndim=5, lead=1), (None, "y", "x", "z", None)),                                   # lead, no slot
])
def test_named_sharding_specs(mesh, kwargs, spec):
    assert tuple(named_sharding(mesh, **kwargs).spec) == spec

def test_named_sharding_errors(mesh):
    with pytest.raises(ValueError, match="species=True needs a species axis"):
        named_sharding(mesh, 4, species=True)
    with pytest.raises(ValueError, match="species=True needs a species axis"):
        named_sharding(mesh, 6, species=True, tail=3)
    with pytest.raises(ValueError, match="too small"):
        named_sharding(mesh, 3)
    with pytest.raises(ValueError, match="too small"):
        named_sharding(mesh, 5, tail=3)

def test_named_sharding_places_array(mesh):
    import jax, numpy as np
    x = jax.device_put(np.zeros((2, 3, 4, 8, 1, 2)), named_sharding(mesh, 6, species=True))
    assert x.sharding.shard_shape(x.shape) == (1, 3, 4, 4, 1, 2)
