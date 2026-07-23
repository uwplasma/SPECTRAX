import pytest
import jax.numpy as jnp
from spectrax import simulation

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

def test_multispecies_background_removed_from_perturbation():
    """Remove every species' homogeneous density from dCk."""
    Ns = 3
    Ck_0 = jnp.zeros((Ns, 1, 2, 1), dtype=jnp.complex128)
    Ck_0 = Ck_0.at[:, 0, 0, 0].set(jnp.arange(1, Ns + 1))
    parameters = {
        "Ck_0": Ck_0,
        "Fk_0": jnp.zeros((6, 1, 2, 1), dtype=jnp.complex128),
        "qs": jnp.array([-1.0, 1.0, 1.0]),
        "alpha_s": jnp.ones(3 * Ns),
        "u_s": jnp.zeros(3 * Ns),
        "Omega_cs": jnp.ones(Ns),
        "t_max": 0.001,
    }

    result = simulation(parameters, Nx=3, Nn=1, Ns=Ns, timesteps=2,
                        dt=0.001, adaptive_time_step=False)
    assert jnp.all(result["dCk"][:, :, 0, 0, 0] == 0)

if __name__ == "__main__":
    pytest.main()
