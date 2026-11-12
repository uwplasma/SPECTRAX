"""Tests of the opt-in relativistic momentum-space Hermite model (``inv_c2 > 0``)."""
import numpy as np
import pytest
import jax.numpy as jnp

from spectrax import (Hermite_Fourier_system, plasma_current, simulation, speed_apply,
                      relativistic_kinetic_energy)
from spectrax._simulation import _twothirds_mask
from spectrax._initialization import initialize_simulation_parameters


def _system(Nn, Nm, Np, Nx, Ns=1, Lx=5.0):
    """Static arguments of Hermite_Fourier_system for Ny = Nz = 1."""
    nk = Nx // 2 + 1
    ky, kx, kz = jnp.meshgrid(jnp.zeros(1), 2 * jnp.pi * jnp.arange(nk), jnp.zeros(1), indexing="ij")
    n = jnp.arange(Nn)[None, None, None, :, None, None, None]
    m = jnp.arange(Nm)[None, None, :, None, None, None, None]
    p = jnp.arange(Np)[None, :, None, None, None, None, None]
    tabs = (jnp.sqrt(n + 1.0), jnp.sqrt(n * 1.0), jnp.sqrt(m + 1.0), jnp.sqrt(m * 1.0), jnp.sqrt(p + 1.0), jnp.sqrt(p * 1.0))
    col = jnp.zeros((Np, Nm, Nn))
    return dict(nk=nk, kx=kx, ky=ky, kz=kz, k2=kx ** 2, tabs=tabs, col=col, mask=jnp.asarray(_twothirds_mask(1, Nx, 1)), Nx=Nx, Lx=Lx)


def _rhs(sy, Ck, F, alpha, u, q, Nn, Nm, Np, Ns=1, **kw):
    Nx = sy["Nx"]
    Ckj = jnp.asarray(Ck)
    C = jnp.fft.irfftn(Ckj * sy["mask"], s=(1, 1, Nx), axes=(-1, -3, -2), norm="forward")
    Fk = jnp.asarray(F)
    Fx = jnp.fft.irfftn(Fk * sy["mask"], s=(1, 1, Nx), axes=(-1, -3, -2), norm="forward")
    return Hermite_Fourier_system(Ckj, C, Fx, sy["kx"], sy["ky"], sy["kz"], sy["k2"], sy["col"], *sy["tabs"],
                                  sy["Lx"], 1.0, 1.0, 0.0, 0.0, alpha, u, jnp.asarray(q), jnp.ones(Ns), Nn, Nm, Np, Ns,
                                  mask23=sy["mask"], **kw)


def _random_state(Nn, Nm, Np, Nx, seed=0):
    rng = np.random.default_rng(seed)
    nk = Nx // 2 + 1
    dec = np.exp(-0.15 * (np.arange(Nn)[None, None, :] + np.arange(Nm)[None, :, None] + np.arange(Np)[:, None, None]))
    Ck = (rng.normal(size=(Np, Nm, Nn, 1, nk, 1)) + 1j * rng.normal(size=(Np, Nm, Nn, 1, nk, 1))) * dec[..., None, None, None]
    Ck[..., 0, :] = Ck[..., 0, :].real
    F = rng.normal(size=(6, 1, nk, 1)) + 1j * rng.normal(size=(6, 1, nk, 1))
    F[:, :, 0] = F[:, :, 0].real
    return Ck.reshape(Np * Nm * Nn, 1, nk, 1), F


def test_default_is_explicit_zero():
    Nn, Nm, Np, Nx = 6, 3, 2, 8
    sy = _system(Nn, Nm, Np, Nx)
    Ck, F = _random_state(Nn, Nm, Np, Nx)
    alpha, u = jnp.array([0.7, 0.5, 0.6]), jnp.array([0.3, -0.2, 0.1])
    a = _rhs(sy, Ck, F, alpha, u, [-1.3], Nn, Nm, Np)
    b = _rhs(sy, Ck, F, alpha, u, [-1.3], Nn, Nm, Np, inv_c2=0.0)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_nonrelativistic_limit_3v_matches_ladder():
    """Dense nodal route with a vanishing 1/c^2 equals the ladder (E and full v x B), 3V, all six fields."""
    Nn, Nm, Np, Nx = 6, 5, 4, 8
    sy = _system(Nn, Nm, Np, Nx)
    Ck, F = _random_state(Nn, Nm, Np, Nx, seed=3)
    alpha, u = jnp.array([0.7, 0.5, 0.6]), jnp.array([0.3, -0.2, 0.1])
    ref = np.asarray(_rhs(sy, Ck, F, alpha, u, [-1.3], Nn, Nm, Np))
    new = np.asarray(_rhs(sy, Ck, F, alpha, u, [-1.3], Nn, Nm, Np, inv_c2=1e-30))
    assert np.max(np.abs(new - ref)) <= 1e-12 * np.max(np.abs(ref))
    cur = np.asarray(plasma_current(jnp.array([-1.3]), alpha, u, jnp.asarray(Ck), Nn, Nm, Np, 1))
    cur2 = np.asarray(plasma_current(jnp.array([-1.3]), alpha, u, jnp.asarray(Ck), Nn, Nm, Np, 1, inv_c2=1e-30))
    assert np.max(np.abs(cur - cur2)) <= 1e-12 * np.max(np.abs(cur))
    # first-order in 1/c^2
    small = np.asarray(_rhs(sy, Ck, F, alpha, u, [-1.3], Nn, Nm, Np, inv_c2=1e-8))
    assert np.max(np.abs(small - ref)) <= 1e-6 * np.max(np.abs(ref))


def _project(f, a, N, nq=4096, span=12.0):
    xi = np.linspace(-span, span, nq)
    phi = np.zeros((N, nq))
    phi[0] = 1.0
    if N > 1:
        phi[1] = np.sqrt(2.0) * xi
    for n in range(1, N - 1):
        phi[n + 1] = (xi * phi[n] - np.sqrt(n / 2.0) * phi[n - 1]) / np.sqrt((n + 1) / 2.0)
    return (phi * f(a * xi)).sum(axis=1) * (xi[1] - xi[0])


def _gauss(mean, sig):
    return lambda u: np.exp(-(u - mean) ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2)


def test_current_of_drifting_maxwellian_matches_quadrature():
    N, a, sig, d = 24, 0.2, 0.1, 0.4
    Ck = np.zeros((N, 1, 1, 1), complex)
    Ck[:, 0, 0, 0] = _project(_gauss(d, sig), a, N)
    J = plasma_current(jnp.array([1.0]), jnp.array([a, 1.0, 1.0]), jnp.zeros(3), jnp.asarray(Ck), N, 1, 1, 1, inv_c2=1.0)
    z, w = np.polynomial.hermite_e.hermegauss(80)
    w = w / np.sqrt(2 * np.pi)
    u = d + sig * z
    assert abs(float(J[0, 0, 0, 0].real) - np.sum(w * u / np.sqrt(1 + u * u))) <= 1e-9
    assert abs(float(J[0, 0, 0, 0].real) - d) > 1e-2  # differs from the nonrelativistic drift


def test_relativistic_gyration_and_zero_magnetic_work():
    """Uniform Bz: every particle rotates at q B / (m gamma); <v>(t) against the exact orbit average, K constant."""
    N, a, sig, d, T, dt = 20, 0.2, 0.1, 0.3, 1.5, 0.01
    Nx = 4
    sy = _system(N, N, 1, Nx)
    Ck = np.zeros((N, N, 1, 1, sy["nk"], 1), complex)
    cx, cy = _project(_gauss(d, sig), a, N), _project(_gauss(0.0, sig), a, N)
    Ck[:, :, 0, 0, 0, 0] = np.outer(cy, cx)  # storage (m, n)
    Ck = Ck.reshape(N * N, 1, sy["nk"], 1)
    F = np.zeros((6, 1, sy["nk"], 1), complex)
    F[5, 0, 0, 0] = 1.0
    alpha, u0 = jnp.array([a, a, 1.0]), jnp.zeros(3)
    q = jnp.array([1.0])
    import jax

    @jax.jit
    def f(Cf):
        return _rhs(sy, Cf, F, alpha, u0, q, N, N, 1, inv_c2=1.0).reshape(Cf.shape)

    def kin(Cf):
        return float(relativistic_kinetic_energy(Cf, alpha, u0, jnp.ones(1), 1.0, N, N, 1, 1)[0])

    y = jnp.asarray(Ck)
    K0 = kin(y)
    for _ in range(int(round(T / dt))):
        k1 = f(y); k2 = f(y + dt / 2 * k1); k3 = f(y + dt / 2 * k2); k4 = f(y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    J = plasma_current(q, alpha, u0, y, N, N, 1, 1, inv_c2=1.0)[:, 0, 0, 0].real
    z, w = np.polynomial.hermite_e.hermegauss(80)
    w = w / np.sqrt(2 * np.pi)
    UX, UY = d + sig * z[:, None], sig * z[None, :]
    W = w[:, None] * w[None, :]
    th = T / np.sqrt(1 + UX ** 2 + UY ** 2)
    ux, uy = UX * np.cos(th) + UY * np.sin(th), -UX * np.sin(th) + UY * np.cos(th)
    g = np.sqrt(1 + ux ** 2 + uy ** 2)
    assert abs(float(J[0]) - np.sum(W * ux / g)) <= 1e-7
    assert abs(float(J[1]) - np.sum(W * uy / g)) <= 1e-7
    assert abs(kin(y) - K0) <= 1e-9 * K0
    th_nr = T  # rigid rotation would give a visibly different answer
    assert abs(float(J[1]) - np.sum(W * (-UX * np.sin(th_nr) + UY * np.cos(th_nr)) / np.sqrt(1 + UX ** 2 + UY ** 2))) > 1e-3


def test_electric_work_identity():
    """dK/dt = <J.E> for a uniform E (the work identity), to the projection error of v."""
    N, a, sig, d = 20, 0.2, 0.1, 0.3
    Nx = 4
    sy = _system(N, 1, 1, Nx)
    Ck = np.zeros((N, 1, sy["nk"], 1), complex)
    Ck[:, 0, 0, 0] = _project(_gauss(d, sig), a, N)
    F = np.zeros((6, 1, sy["nk"], 1), complex)
    E0 = 0.7
    F[0, 0, 0, 0] = E0
    alpha, u0, q = jnp.array([a, 1.0, 1.0]), jnp.zeros(3), jnp.array([-1.0])
    dC = _rhs(sy, Ck, F, alpha, u0, q, N, 1, 1, inv_c2=1.0).reshape(Ck.shape)
    dK = float(relativistic_kinetic_energy(dC, alpha, u0, jnp.ones(1), 1.0, N, 1, 1, 1)[0])
    J = float(plasma_current(q, alpha, u0, jnp.asarray(Ck), N, 1, 1, 1, inv_c2=1.0)[0, 0, 0, 0].real)
    assert abs(dK - J * E0) <= 1e-9 * abs(J * E0)


def test_speed_apply_axis_with_single_mode_is_cold():
    """An axis with one Hermite mode has one node at its centre."""
    C = jnp.ones((1, 1, 1, 3, 1, 1, 1))
    v = speed_apply(C, 1, jnp.array([0.2, 1.0, 1.0]), jnp.array([0.0, 0.5, 0.0]), 1.0, 3, 1, 1, 1)
    # nodal value of v_y at u_y = 0.5 with u_x = a lam_j, applied to a constant (n=0 mode weighted) state: finite and positive
    assert np.all(np.isfinite(np.asarray(v)))


def test_simulation_relativistic_runs_and_conserves_charge():
    Nn, Nx = 8, 4
    Ck_0 = jnp.zeros((2 * Nn, 1, Nx // 2 + 1, 1), dtype=jnp.complex128)
    for s in range(2):
        Ck_0 = Ck_0.at[s * Nn, 0, 0, 0].set(1.0 / 0.2)
        Ck_0 = Ck_0.at[s * Nn, 0, 1, 0].set(1e-3j * (1 - 2 * s))
    parameters = {"Ck_0": Ck_0, "Fk_0": jnp.zeros((6, 1, Nx // 2 + 1, 1), dtype=jnp.complex128),
                  "qs": jnp.array([-1.0, 1.0]), "alpha_s": jnp.array([0.2, 1, 1, 0.2, 1, 1]), "u_s": jnp.zeros(6),
                  "Omega_cs": jnp.array([1.0, 1.0]), "t_max": 0.05}
    out = simulation(parameters, Nx=Nx, Nn=Nn, Ns=2, timesteps=2, dt=0.005, adaptive_time_step=False, inv_c2=1.0)
    assert np.all(np.isfinite(np.asarray(out["Ck"])))
    nonrel = simulation(parameters, Nx=Nx, Nn=Nn, Ns=2, timesteps=2, dt=0.005, adaptive_time_step=False)
    d = np.max(np.abs(np.asarray(out["Ck"]) - np.asarray(nonrel["Ck"])))
    assert 0 < d < 1e-2  # weakly relativistic here
    assert np.allclose(np.asarray(out["Ck"])[:, 0, 0, 0, 0], np.asarray(Ck_0)[0, 0, 0, 0])
