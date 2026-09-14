import os
import subprocess
import sys

import pytest
import jax.numpy as jnp
from diffrax import Euler, RESULTS
from spectrax import make_phase_space_mesh, make_species_mesh, simulation
from spectrax._simulation import _mapped_solve


def test_mapped_solve_reuses_equivalent_configurations():
    _mapped_solve.cache_clear()
    first = _mapped_solve(5, 1, 1, 1, 1, Euler(), False, make_species_mesh())
    second = _mapped_solve(5, 1, 1, 1, 1, Euler(), False, make_species_mesh())
    assert first is second
    assert _mapped_solve.cache_info().hits == 1

def test_simulation_runs():
    """Test if the simulation runs without errors with default parameters."""
    result = simulation(throw=False, species_mesh=make_species_mesh())
    assert isinstance(result, dict), "Simulation did not return a dictionary."
    assert "Ck" in result, "Missing Ck in output."
    assert "Fk" in result, "Missing Fk in output."
    assert result["Ck"].shape[1:] == (result["Ns"] * result["Nn"] * result["Nm"] * result["Np"],
                                      result["Ny"], result["Nx"] // 2 + 1, result["Nz"])
    assert result["Fk"].shape[1:] == (6, result["Ny"], result["Nx"] // 2 + 1, result["Nz"])
    assert result["solver_result"] == RESULTS.successful

def test_electric_field_update():
    """Test that the electric field updates and does not remain zero."""
    result = simulation()
    assert not jnp.all(result["Fk"] == 0), "Electric field did not update."

def test_two_device_simulation_matches_serial():
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    code = """
import numpy as np
from spectrax import make_species_mesh, simulation

rng = np.random.default_rng(4)
shape = (4, 1, 3, 1)
parameters = {
    "t_max": 0.001,
    "Ck_0": (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * 1e-3,
    "Fk_0": (rng.standard_normal((6, 1, 3, 1))
              + 1j * rng.standard_normal((6, 1, 3, 1))) * 1e-3,
    "qs": np.array([-1.0, 2.0]), "Omega_cs": np.array([0.5, 0.02]),
    "alpha_s": np.array([0.25, 0.30, 0.35, 0.05, 0.06, 0.07]),
}
kwargs = dict(Nx=5, Ny=1, Nz=1, Nn=2, Nm=1, Np=1, Ns=2,
              timesteps=2, dt=0.001, throw=False)
serial = simulation(parameters, **kwargs)
parallel = simulation(parameters, species_mesh=make_species_mesh(), **kwargs)
np.testing.assert_allclose(parallel["Ck"], serial["Ck"], rtol=0, atol=1e-12)
np.testing.assert_allclose(parallel["Fk"], serial["Fk"], rtol=0, atol=1e-12)
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)

def test_phase_space_simulation_matches_serial():
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    code = """
import numpy as np
from spectrax import make_phase_space_mesh, simulation

rng = np.random.default_rng(4)
shape = (8, 1, 3, 1)
initial = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * 1e-3
parameters = {"t_max": 0.01, "ode_tolerance": 1e-8, "Ck_0": initial,
              "Fk_0": np.zeros((6, 1, 3, 1), dtype=complex)}
adaptive = dict(Nx=5, Ny=1, Nz=1, Nn=4, Nm=1, Np=1, Ns=2,
                timesteps=2, dt=0.001, throw=False)
serial = simulation(parameters, **adaptive)
parallel = simulation(parameters, species_mesh=make_phase_space_mesh(2),
                      **adaptive)
np.testing.assert_allclose(parallel["Ck"], serial["Ck"], rtol=0, atol=1e-12)
np.testing.assert_allclose(parallel["Fk"], serial["Fk"], rtol=0, atol=1e-12)
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)

if __name__ == "__main__":
    pytest.main()
