import pytest
import jax.numpy as jnp
from jax.scipy.special import factorial
from spectrax import compute_C_nmp, simulation

def test_maxwellian_initialization_is_stable_at_high_order():
    velocity = jnp.linspace(-0.1, 0.1, 6).reshape(2, 3, 1, 1, 1)
    alpha = jnp.array([0.25, 0.3, 0.35, 0.05, 0.06, 0.07])
    shift = jnp.zeros(6)
    high_order = compute_C_nmp(velocity, alpha, shift, 256, 4, 4, 2)
    assert jnp.all(jnp.isfinite(high_order))

    Nn, Nm, Np = 5, 3, 2
    n = jnp.arange(Nn)[None, None, None, :, None, None, None]
    m = jnp.arange(Nm)[None, None, :, None, None, None, None]
    p = jnp.arange(Np)[None, :, None, None, None, None, None]
    scale = alpha.reshape(2, 3)
    drift = shift.reshape(2, 3)
    reference = (
        jnp.sqrt(2.0 ** (n + m + p) / (factorial(n) * factorial(m) * factorial(p)))
        * ((velocity[:, 0, None, None, None] - drift[:, 0, None, None, None, None, None, None])
           / scale[:, 0, None, None, None, None, None, None]) ** n
        * ((velocity[:, 1, None, None, None] - drift[:, 1, None, None, None, None, None, None])
           / scale[:, 1, None, None, None, None, None, None]) ** m
        * ((velocity[:, 2, None, None, None] - drift[:, 2, None, None, None, None, None, None])
           / scale[:, 2, None, None, None, None, None, None]) ** p
        / jnp.prod(scale, axis=1)[:, None, None, None, None, None, None]
    )
    expected = jnp.fft.rfftn(reference, axes=(-1, -3, -2), norm="forward")
    assert jnp.allclose(compute_C_nmp(velocity, alpha, shift, Nn, Nm, Np, 2),
                        expected, rtol=1e-13, atol=1e-13)

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
