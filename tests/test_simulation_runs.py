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


def _three_species_unequal_orders():
    import numpy as np
    Ns, Nn, Nm, Np, Ny, Nx, Nz = 3, 3, 2, 4, 2, 5, 3
    H = Nn * Nm * Np
    rng = np.random.default_rng(3)
    shape = (Ns * H, Ny, Nx // 2 + 1, Nz)
    Ck_0 = jnp.asarray(1e-2 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)))
    Ck_0 = Ck_0.at[jnp.arange(Ns) * H, 0, 0, 0].set(1.0)
    params = {"Ck_0": Ck_0, "Fk_0": jnp.zeros((6,) + shape[1:], dtype=jnp.complex128),
              "qs": jnp.array([-1.0, 1.0, 2.0]), "alpha_s": jnp.linspace(0.4, 0.9, 3 * Ns),
              "u_s": jnp.linspace(-0.2, 0.2, 3 * Ns), "Omega_cs": jnp.array([1.0, 0.5, 0.25]), "t_max": 0.2}
    grid = dict(Nx=Nx, Ny=Ny, Nz=Nz, Nn=Nn, Nm=Nm, Np=Np, Ns=Ns)
    return params, grid


def test_structured_state_maps_flat_hermite_index_to_species_p_m_n(monkeypatch):
    """With unequal (Nn, Nm, Np) and 3 species, the solver state axis order is (s, p, m, n): flat
    output index s*Nn*Nm*Np + p*Nn*Nm + m*Nn + n lands at state[s, p, m, n]."""
    import spectrax._simulation as sim
    params, g = _three_species_unequal_orders()
    captured = {}
    def stop(term, **kwargs):
        captured["y0"] = kwargs["y0"]
        raise RuntimeError("captured")
    monkeypatch.setattr(sim, "diffeqsolve", stop)
    import jax
    with jax.disable_jit(), pytest.raises(RuntimeError, match="captured"):
        simulation(params, **g, timesteps=2)
    Ck = captured["y0"][0]
    assert Ck.shape == (g["Ns"], g["Np"], g["Nm"], g["Nn"], g["Ny"], g["Nx"] // 2 + 1, g["Nz"])
    Nn, Nm, Np = g["Nn"], g["Nm"], g["Np"]
    for s, p, m, n in [(0, 0, 0, 1), (1, 3, 1, 2), (2, 2, 0, 0)]:
        flat = s * Nn * Nm * Np + p * Nn * Nm + m * Nn + n
        assert jnp.array_equal(Ck[s, p, m, n], params["Ck_0"][flat])


def test_continuation_from_output_matches_single_run():
    """Restarting from a saved (Ck, Fk) snapshot reproduces the uninterrupted fixed-step trajectory."""
    from diffrax import Tsit5
    params, g = _three_species_unequal_orders()
    kw = dict(dt=0.01, solver=Tsit5(), adaptive_time_step=False)
    full = simulation({**params, "t_max": 0.2}, **g, timesteps=3, **kw)        # t = 0, 0.1, 0.2
    first = simulation({**params, "t_max": 0.1}, **g, timesteps=2, **kw)
    second = simulation({**params, "t_max": 0.1, "Ck_0": first["Ck"][-1], "Fk_0": first["Fk"][-1]},
                        **g, timesteps=2, **kw)
    assert jnp.allclose(second["Ck"][-1], full["Ck"][-1], rtol=0, atol=1e-13)
    assert jnp.allclose(second["Fk"][-1], full["Fk"][-1], rtol=0, atol=1e-13)
