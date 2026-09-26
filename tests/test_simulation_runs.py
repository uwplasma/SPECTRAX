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

def test_solver_state_keeps_species_and_hermite_axes(monkeypatch):
    """The integrator sees (Ck, Fk) with their species, Hermite and grid axes, not one flat vector."""
    import spectrax._simulation as sim
    captured = {}
    def stop(term, **kwargs):
        captured.update(kwargs, rhs=term.vector_field(0.0, kwargs["y0"], kwargs["args"]))
        raise RuntimeError("captured")
    monkeypatch.setattr(sim, "diffeqsolve", stop)
    with pytest.raises(RuntimeError, match="captured"):
        simulation(Nx=7, Nn=4, timesteps=3)
    shapes = [(2, 1, 1, 4, 1, 4, 1), (6, 1, 4, 1)]
    assert [y.shape for y in captured["y0"]] == shapes
    assert [y.shape for y in captured["rhs"]] == shapes

def test_electric_field_update():
    """Test that the electric field updates and does not remain zero."""
    result = simulation()
    assert not jnp.all(result["Fk"] == 0), "Electric field did not update."

if __name__ == "__main__":
    pytest.main()
