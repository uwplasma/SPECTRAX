"""Physics and observation regressions for the PSP chirp example (Examples/psp_chirp_model.py)."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffrax import ForwardMode, NoProgressMeter, RecursiveCheckpointAdjoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Examples"))
from psp_chirp_model import (antenna_voltage, gauss_residual, initial_state, kinetic_root, moments, probe, pulse,
                             spectrogram, thermal_alpha)
from spectrax import simulation

M, Nx, H = 1836.0, 64, 24
SIGMA_C = thermal_alpha(10.0, M) / np.sqrt(2)
BASE = dict(Lx=400 * SIGMA_C / np.sqrt(1 / M), n0=1.0, nb0=0.05, xs=0.0, ell0=None, Ub=4 * SIGMA_C, sigmab=SIGMA_C,
            age=0.0, alpha_e=np.sqrt(20 * M) * SIGMA_C, alpha_c=np.sqrt(2) * SIGMA_C, M=M, t_max=20.0, nu=1.0, tolerance=1e-10)


def make_pulse(**overrides):
    p = dict(BASE, ell0=BASE["Lx"] / 16, xs=0.3 * BASE["Lx"], age=BASE["Lx"] / (40 * SIGMA_C))
    return {**p, **overrides}


def run(p, timesteps=3, **kwargs):
    kwargs.setdefault("progress_meter", NoProgressMeter())
    return simulation(initial_state(p, Nx, H), Nx=Nx, Ny=1, Nz=1, Nn=H, Nm=1, Np=1, Ns=3, timesteps=timesteps, **kwargs)


def test_pulse_moments_match_quadrature_and_free_streaming():
    """Eqs. (10)-(14): direct phase-space quadrature of the free-streamed Gaussian gives the closed-form moments."""
    x, v = np.linspace(-15, 30, 901), np.linspace(-6, 10, 1601)
    xs, ell0, Ub, sigmab, nb0, age = 0.5, 1.2, 2.0, 0.6, 0.3, 3.0
    X, V = np.meshgrid(x, v, indexing="ij")
    f = nb0 / (np.sqrt(2 * np.pi) * sigmab) * np.exp(-(X - V * age - xs) ** 2 / (2 * ell0**2) - (V - Ub) ** 2 / (2 * sigmab**2))
    n = np.trapezoid(f, v, axis=1)
    U = np.trapezoid(f * V, v, axis=1) / n
    var = np.trapezoid(f * V**2, v, axis=1) / n - U**2
    n_b, U_a, sigma_a, n_c, U_e = pulse(x, 1.0, nb0, xs, ell0, Ub, sigmab, age)
    core = n > 1e-6 * n.max()
    np.testing.assert_allclose(n, n_b, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(U[core], U_a[core], rtol=1e-8)
    np.testing.assert_allclose(var[core], float(sigma_a) ** 2, rtol=1e-8)
    np.testing.assert_allclose(np.trapezoid(n_b, x), nb0 * np.sqrt(2 * np.pi) * ell0, rtol=1e-8)   # particles conserved
    np.testing.assert_allclose(n_c + n_b, 1.0) and np.testing.assert_allclose(n_b * U_a, U_e)     # neutral, current-free


def test_initial_state_moments_neutrality_gauss_and_proton_acceleration():
    p = make_pulse()
    params = initial_state(p, Nx, H)
    n, U, P = moments(params["Ck_0"], params["alpha_s"], params["u_s"], H, Nx)
    x = jnp.arange(Nx) * p["Lx"] / Nx
    n_b, U_a, sigma_a, n_c, U_e = pulse(x, p["n0"], p["nb0"], p["xs"], p["ell0"], p["Ub"], p["sigmab"], p["age"])
    resolved = n_b > 1e-3 * p["nb0"]
    np.testing.assert_allclose(n, jnp.stack([jnp.ones(Nx), n_c, n_b]), rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(U[2][resolved], U_a[resolved], rtol=1e-8)
    np.testing.assert_allclose(P[2][resolved] / n[2][resolved], sigma_a**2, rtol=1e-7)
    charge = jnp.sum(params["qs"][:, None] * n, axis=0)
    current = jnp.sum(params["qs"][:, None] * n * U, axis=0)
    np.testing.assert_allclose(charge, 0.0, atol=1e-12) and np.testing.assert_allclose(current, 0.0, atol=1e-14)
    np.testing.assert_allclose((params["qs"] * params["Omega_cs"])[1:], 1 / M)       # same q/m, charge counted once
    assert float(jnp.max(jnp.abs(params["Fk_0"]))) < 1e-15                    # neutral: no initial field


def test_seeded_state_satisfies_gauss_and_evolution_preserves_it():
    seed = jnp.zeros(Nx // 2 + 1, complex).at[3].set(1e-4).at[5].set(-2e-4j)
    out = run(make_pulse(seed=seed, t_max=200.0), timesteps=4)
    for i in range(4):
        assert float(gauss_residual(out["Ck"][i], out["Fk"][i], out["qs"], out["alpha_s"], H, out["Lx"])) < 1e-8
    assert float(jnp.max(jnp.abs(out["Fk"][:, 1:]))) == 0.0                         # no transverse fields appear
    assert jnp.all(jnp.isfinite(out["Fk"][:, 0, 0, 0, 0]))   # mean field obeys dE0/dt = -J0: recorded, never subtracted


@pytest.mark.parametrize("state", ["uniform", "split"])
def test_uniform_state_stays_at_rest_and_splitting_protons_is_bookkeeping(state):
    """A uniform, drift-free plasma has no dynamics; splitting one proton Maxwellian into two components changes nothing."""
    p = dict(BASE, Ub=0.0, sigmab=SIGMA_C, nb0=0.0 if state == "uniform" else 0.3)
    out = run(p)
    np.testing.assert_allclose(out["Ck"][-1], out["Ck"][0], atol=1e-12 * float(jnp.max(jnp.abs(out["Ck"][0]))))
    assert float(jnp.max(jnp.abs(out["Fk"]))) < 1e-14


def test_probe_and_antenna_response_parity_and_doppler_sign():
    Lx, N = 10.0, 16
    x = jnp.linspace(0, Lx, 37)
    for j in (0, 3, N // 2):                                             # mean, paired and Nyquist modes
        Ek = jnp.zeros(N // 2 + 1, complex).at[j].set(0.7 - 0.2j)
        grid_field = jnp.fft.irfft(Ek, n=N, norm="forward")
        np.testing.assert_allclose(probe(Ek, jnp.arange(N) * Lx / N, Lx, N), grid_field, atol=1e-13)
    k, ell, j = 2 * np.pi * 3 / Lx, 0.9, 3
    Ek = jnp.zeros(N // 2 + 1, complex).at[j].set(0.5)                   # E = cos(k x)
    V = antenna_voltage(Ek, x + ell / 2, x - ell / 2, Lx, N)
    np.testing.assert_allclose(V, -jnp.sin(k * ell / 2) / (k / 2) * jnp.cos(k * x), atol=1e-12)   # -int E dx: |sin(kl/2)/(kl/2)| E l
    np.testing.assert_allclose(antenna_voltage(jnp.zeros(N // 2 + 1).at[0].set(2.0), 1.0, 0.5, Lx, N), -1.0)   # uniform field
    omega, V_sc, t = 0.3, -1.7, jnp.linspace(0, 20, 4000)               # spacecraft at x = V_sc t sees omega - k V_sc
    trace = probe(Ek[None] * jnp.exp(-1j * omega * t)[:, None], V_sc * t, Lx, N)
    spectrum = jnp.abs(jnp.fft.rfft(trace))
    f = jnp.fft.rfftfreq(t.size, float(t[1] - t[0])) * 2 * np.pi
    assert abs(float(f[jnp.argmax(spectrum)]) - (omega - k * V_sc)) < 2 * np.pi / 20


def test_spectrogram_is_differentiable_and_dispersion_root_is_exact():
    s = jnp.sin(0.3 * jnp.arange(256.0))
    assert jnp.all(jnp.isfinite(jax.grad(lambda a: jnp.sum(jnp.log(spectrogram(a * s, 64, 16) + 1e-12)))(1.0)))
    k = 2 * np.pi / 60
    root = kinetic_root(0.36 + 0.016j, k, np.array([1, 0.05, 1.05]), np.sqrt([1, 1, 18360]), np.array([0, 5, 0.25 / 1.05]),
                        np.array([1, 1, -1836]))
    np.testing.assert_allclose([root.real, root.imag], [0.366742494372020, 0.016259569553166], rtol=1e-10)


def test_seed_gradient_reverse_forward_and_finite_differences_agree():
    """Directional derivative of a probe observable with respect to the source age, three ways."""
    def observable(age, adjoint):
        p = make_pulse(age=age, t_max=100.0, seed=jnp.zeros(Nx // 2 + 1, complex).at[4].set(1e-3))
        out = run(p, adjoint=adjoint)
        return jnp.sum(probe(out["Fk"][-1, 0, 0, :, 0], jnp.linspace(0, p["Lx"], 7), p["Lx"], Nx) ** 2)
    a = make_pulse()["age"]
    reverse = jax.grad(observable)(a, RecursiveCheckpointAdjoint(checkpoints=8))
    forward = jax.jvp(lambda s: observable(s, ForwardMode()), (a,), (1.0,))[1]
    plain = RecursiveCheckpointAdjoint()
    fd = (observable(a * (1 + 1e-5), plain) - observable(a * (1 - 1e-5), plain)) / (2e-5 * a)
    np.testing.assert_allclose(reverse, forward, rtol=1e-8)
    np.testing.assert_allclose(reverse, fd, rtol=1e-4)                     # the plan's initial directional target
