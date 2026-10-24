import pytest
import jax.numpy as jnp
from spectrax import simulation

@pytest.mark.parametrize("Ns", [1, 2, 3, 4])
def test_perturbation_removes_every_background_density_and_keeps_mean_current(Ns):
    """dCk differs from Ck only in each species' (n=0, k=0) density slot; the k=0 first Hermite
    moment (the physical mean current) and every k != 0 coefficient are kept unchanged."""
    Nn, Nx = 2, 4
    Ck_0 = jnp.zeros((Ns * Nn, 1, Nx // 2 + 1, 1), dtype=jnp.complex128)
    for s in range(Ns):
        Ck_0 = Ck_0.at[s * Nn, 0, 0, 0].set(1.0 + s)             # background density
        Ck_0 = Ck_0.at[s * Nn + 1, 0, 0, 0].set(0.1 * (s + 1))   # uniform drift: mean current
        Ck_0 = Ck_0.at[s * Nn, 0, 1, 0].set(1e-3j * (s + 1))     # density perturbation at kx=1
    parameters = {
        "Ck_0": Ck_0,
        "Fk_0": jnp.zeros((6, 1, Nx // 2 + 1, 1), dtype=jnp.complex128),
        "qs": jnp.array([(-1.0) ** (s + 1) for s in range(Ns)]),
        "alpha_s": jnp.ones(3 * Ns),
        "u_s": jnp.zeros(3 * Ns),
        "Omega_cs": jnp.ones(Ns),
        "t_max": 0.01,
    }
    result = simulation(parameters, Nx=Nx, Nn=Nn, Ns=Ns, timesteps=2, dt=0.001, adaptive_time_step=False)
    Ck, dCk = result["Ck"], result["dCk"]
    density = jnp.arange(Ns) * Nn
    assert jnp.all(dCk[:, density, 0, 0, 0] == 0)
    assert jnp.all(Ck[:, density, 0, 0, 0] != 0)
    keep = jnp.ones(Ck.shape[1:], dtype=bool).at[density, 0, 0, 0].set(False)
    assert jnp.array_equal(jnp.where(keep, dCk, 0), jnp.where(keep, Ck, 0))
    assert jnp.all(dCk[:, density + 1, 0, 0, 0] != 0)  # mean current survives in the perturbation
    assert result["Fk"][-1, 0, 0, 0, 0] != 0  # the net mean current drives the uniform Ex

def test_simulation_runs():
    """Test if the simulation runs without errors with default parameters."""
    result = simulation()
    assert isinstance(result, dict), "Simulation did not return a dictionary."
    assert "Ck" in result, "Missing Ck in output."
    assert "Fk" in result, "Missing Fk in output."
    assert result["solver_stats"]["num_steps"] == (
        result["solver_stats"]["num_accepted_steps"]
        + result["solver_stats"]["num_rejected_steps"]
    )

def test_electric_field_update():
    """Test that the electric field updates and does not remain zero."""
    result = simulation()
    assert not jnp.all(result["Fk"] == 0), "Electric field did not update."

if __name__ == "__main__":
    pytest.main()
