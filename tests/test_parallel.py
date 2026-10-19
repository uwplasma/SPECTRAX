import subprocess
import sys
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding
from spectrax import make_mesh, check_mesh, setup_devices, load_parameters, simulation, initialize_simulation_parameters
from spectrax._parallel import state_specs, args_specs
from spectrax._simulation import _RHS_PARAMETER_NAMES, ode_system

# The 8 virtual CPU devices created in conftest.py (asking again reuses them).
DEVICES = setup_devices("cpu", 8)


def test_make_mesh_axes_and_devices():
    mesh = make_mesh(DEVICES, species=2, y=1, x=4)
    assert dict(mesh.shape) == {"s": 2, "z": 1, "y": 1, "x": 4}
    assert mesh.axis_types == (AxisType.Auto,) * 4
    assert mesh.devices.size == 8

def test_make_mesh_requires_split_to_match_devices():
    with pytest.raises(ValueError, match="must match"):
        make_mesh(DEVICES, x=16)
    with pytest.raises(ValueError, match="must match"):
        make_mesh(DEVICES, x=4)

def test_check_mesh_divisibility():
    mesh = make_mesh(DEVICES, species=2, x=2, z=2)
    check_mesh(mesh, Nx=32, Ny=1, Nz=4, Ns=4)
    with pytest.raises(ValueError, match="Nx"):
        check_mesh(mesh, Nx=33, Ny=1, Nz=4, Ns=4)
    with pytest.raises(ValueError, match="Nz"):
        check_mesh(mesh, Nx=32, Ny=1, Nz=1, Ns=4)
    with pytest.raises(ValueError, match="Ns"):
        check_mesh(mesh, Nx=32, Ny=1, Nz=4, Ns=3)
    with pytest.raises(ValueError, match="Ny"):
        check_mesh(make_mesh(DEVICES[:2], y=2), Nx=32, Ny=1, Nz=1, Ns=4)

def test_check_mesh_rejects_foreign_axis_names():
    mesh = jax.make_mesh((2, 4), ("a", "b"), devices=DEVICES)
    with pytest.raises(ValueError, match="axes"):
        check_mesh(mesh, Nx=32, Ny=1, Nz=1, Ns=2)

def test_setup_devices_works_after_importing_spectrax():
    # Runs in a fresh process, because this test session has already started JAX.
    code = ("import spectrax, jax; devices = spectrax.setup_devices('cpu', 3); "
            "assert len(devices) == 3 and len(jax.devices('cpu')) == 3; "
            "assert jax.numpy.zeros(1).devices() == {devices[0]}")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

def test_setup_devices_after_jax_has_run():
    assert setup_devices("cpu", 8) == DEVICES  # The CPU devices already exist, so they are reused.
    assert setup_devices("cpu", 2) == DEVICES[:2]
    with pytest.raises(RuntimeError, match="already run a computation"):
        setup_devices("cpu", 9)

def test_setup_devices_rejects_unknown_device_type():
    with pytest.raises(ValueError, match="device_type"):
        setup_devices("tpu", 1)

def test_setup_devices_rejects_more_gpus_than_present():
    try:
        n_gpus = len(jax.devices("gpu"))
    except RuntimeError:
        n_gpus = 0
    with pytest.raises((RuntimeError, ValueError), match="GPU"):
        setup_devices("gpu", n_gpus + 1)

def test_partition_specs_fit_simulation_arrays():
    Nx, Ny, Nn, Ns = 8, 4, 6, 2
    parameters = initialize_simulation_parameters({}, Nx=Nx, Ny=Ny, Nn=Nn, Ns=Ns, N_DG=2, dims=2)
    Nl = parameters["basis_idx"].shape[0]
    state = {"Ck": jnp.zeros((Ns, 1, 1, Nn, Ny, Nx, 1, Nl)), "Fk": jnp.zeros((6, Ny, Nx, 1, Nl))}
    args = {"col": parameters["collision_matrix"], **{name: parameters[name] for name in _RHS_PARAMETER_NAMES}}
    mesh = make_mesh(DEVICES, species=2, y=2, x=2)
    check_mesh(mesh, Nx, Ny, 1, Ns)

    place = lambda tree, specs: {k: jax.device_put(v, NamedSharding(mesh, specs[k])) for k, v in tree.items()}
    state = place(state, state_specs())
    args = place(args, args_specs(args))

    assert state["Ck"].addressable_shards[0].data.shape == (1, 1, 1, Nn, Ny // 2, Nx // 2, 1, Nl)
    assert state["Fk"].addressable_shards[0].data.shape == (6, Ny // 2, Nx // 2, 1, Nl)
    assert args["alpha_s"].addressable_shards[0].data.shape == (3,)
    assert args["Ax_p"].addressable_shards[0].data.shape == (1, Nn, Nn)
    assert args["triple_product"].sharding.is_fully_replicated

def write_toml(path, parallel_table=""):
    path.write_text(
        "[input_parameters]\nt_max = 1.0\node_tolerance = 1e-8\n\n"
        "[solver_parameters]\nNx = 8\nNn = 4\nsolver = \"Dopri8\"\n\n" + parallel_table
    )
    return path

def test_load_parameters_without_parallel_table(tmp_path):
    _, solver_parameters = load_parameters(write_toml(tmp_path / "serial.toml"))
    assert "mesh" not in solver_parameters

def test_load_parameters_builds_mesh(tmp_path):
    toml = write_toml(tmp_path / "parallel.toml", '[parallel]\ndevice = "cpu"\nnum_devices = 8\nspecies = 2\nx = 4\n')
    _, solver_parameters = load_parameters(toml)
    assert dict(solver_parameters["mesh"].shape) == {"s": 2, "z": 1, "y": 1, "x": 4}
    assert list(solver_parameters["mesh"].devices.flat) == DEVICES

def test_load_parameters_single_device_needs_no_mesh(tmp_path):
    _, solver_parameters = load_parameters(write_toml(tmp_path / "one.toml", '[parallel]\ndevice = "cpu"\nnum_devices = 1\n'))
    assert "mesh" not in solver_parameters

def test_load_parameters_rejects_unknown_parallel_entries(tmp_path):
    with pytest.raises(ValueError, match="Unknown entries"):
        load_parameters(write_toml(tmp_path / "typo.toml", '[parallel]\ndevice = "cpu"\nnum_devices = 8\nX = 8\n'))

def test_load_parameters_requires_device_and_count(tmp_path):
    with pytest.raises(ValueError, match="num_devices"):
        load_parameters(write_toml(tmp_path / "incomplete.toml", '[parallel]\ndevice = "cpu"\nx = 8\n'))

def test_load_parameters_rejects_split_that_does_not_use_all_devices(tmp_path):
    with pytest.raises(ValueError, match="must match"):
        load_parameters(write_toml(tmp_path / "mismatch.toml", '[parallel]\ndevice = "cpu"\nnum_devices = 8\nx = 4\n'))

def test_simulation_with_mesh_matches_serial():
    # A fixed time step makes both runs take identical steps, so they agree to round-off.
    settings = dict(Nx=8, Nn=8, timesteps=3, dt=0.01, adaptive_time_step=False)
    serial = simulation({"t_max": 1.0}, **settings)
    with_mesh = simulation({"t_max": 1.0}, **settings, mesh=make_mesh(DEVICES, species=2, x=4))
    assert jnp.allclose(with_mesh["Ck"], serial["Ck"], rtol=1e-12, atol=1e-14)
    assert jnp.allclose(with_mesh["Fk"], serial["Fk"], rtol=1e-12, atol=1e-14)
    assert jnp.allclose(with_mesh["total_energy"], serial["total_energy"], rtol=1e-12)
    # The saved solution stays distributed: each device holds one species and a quarter of the elements.
    assert len(with_mesh["Ck"].sharding.device_set) == 8
    assert with_mesh["Ck"].addressable_shards[0].data.shape == (3, 8, 1, 2, 1, 2)

def test_simulation_rejects_grid_that_does_not_split():
    with pytest.raises(ValueError, match="Nx"):
        simulation({"t_max": 1.0}, Nx=9, Nn=8, timesteps=3, mesh=make_mesh(DEVICES[:4], x=4))

@pytest.mark.parametrize("grid, mesh_shape", [
    (dict(Nx=8, Ny=1, Nn=32, Nm=1, Np=1, N_DG=2, dims=1), dict(species=2, x=4)),  # matmul path of the flux
    (dict(Nx=4, Ny=4, Nn=3, Nm=3, Np=3, N_DG=2, dims=2), dict(species=2, x=2, y=2)),
])
def test_rhs_only_exchanges_halos_and_species_sums(grid, mesh_shape):
    Ns, Nz = 2, 1
    parameters = initialize_simulation_parameters({}, Nx=grid["Nx"], Ny=grid["Ny"], Nz=Nz, Nn=grid["Nn"], Nm=grid["Nm"],
                                                  Np=grid["Np"], Ns=Ns, N_DG=grid["N_DG"], dims=grid["dims"])
    Nl = parameters["basis_idx"].shape[0]
    mesh = make_mesh(DEVICES, **mesh_shape)
    state = {"Ck": jnp.zeros((Ns, grid["Np"], grid["Nm"], grid["Nn"], grid["Ny"], grid["Nx"], Nz, Nl)),
             "Fk": jnp.zeros((6, grid["Ny"], grid["Nx"], Nz, Nl))}
    args = {"col": parameters["collision_matrix"], **{name: parameters[name] for name in _RHS_PARAMETER_NAMES}}
    place = lambda tree, specs: {k: jax.device_put(v, NamedSharding(mesh, specs[k])) for k, v in tree.items()}
    static = dict(Nx=grid["Nx"], Ny=grid["Ny"], Nz=Nz, Nn=grid["Nn"], Nm=grid["Nm"], Np=grid["Np"], Ns=Ns, dims=grid["dims"])
    compiled = ode_system.lower(0.0, place(state, state_specs()), place(args, args_specs(args)), **static, mesh=mesh).compile().as_text()
    assert "collective-permute" in compiled
    assert "all-gather" not in compiled and "all-to-all" not in compiled
