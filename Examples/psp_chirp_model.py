"""Units, impulsive-proton source, moments and synthetic antenna for the Parker Solar Probe chirp example.

The model is the longitudinal electrostatic subspace of the unchanged SPECTRAX Vlasov-Maxwell solver: 1D1V,
``Ny=Nz=Nm=Np=1`` and three populations (electron, core proton, beam proton). Core and beam are bookkeeping parts
of one proton distribution, with identical charge-to-mass ratio.

Units are fixed by a reference electron density ``n_star``, independent of the fitted density:
``t_star = 1/omega_pe_star``, ``x_star = c/omega_pe_star``, ``v_star = c``, ``E_star = m_e c omega_pe_star / e``.
Then ``qs = (-1, 1, 1)`` and ``Omega_cs = (1, 1/M, 1/M)``: the force coefficient ``qs * Omega_cs`` is the
charge-to-mass ratio, so charge is not counted twice. The Hermite width is ``alpha = sqrt(2) sigma / c`` with
``sigma = sqrt(T/m)``. Transverse widths are bookkeeping factors set to one, so only parallel moments are physical
and densities are ``n = alpha_x C0``. ``nu`` is numerical high-Hermite damping, not a Coulomb collision rate.
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import wofz

from spectrax import compute_C_nmp

jax.config.update("jax_enable_x64", True)

C_LIGHT, E_CHARGE, EPS0, M_E = 299792458.0, 1.602176634e-19, 8.8541878128e-12, 9.1093837015e-31
M_P_OVER_M_E = 1836.15267343


def units(n_star_cm3):
    """Reference scales in SI for an electron density in cm^-3: omega_pe, t_star, x_star and E_star."""
    omega_pe = np.sqrt(n_star_cm3 * 1e6 * E_CHARGE**2 / (EPS0 * M_E))
    return dict(omega_pe=omega_pe, t_star=1 / omega_pe, x_star=C_LIGHT / omega_pe, E_star=M_E * C_LIGHT * omega_pe / E_CHARGE)


def thermal_alpha(T_eV, mass_ratio):
    """Hermite width alpha = sqrt(2 T / m) / c for a temperature in eV and a mass in electron masses."""
    return np.sqrt(2 * T_eV * E_CHARGE / (mass_ratio * M_E)) / C_LIGHT


def pulse(x, n0, nb0, xs, ell0, Ub, sigmab, age):
    """Free-streaming Gaussian proton pulse at ballistic age ``age`` (all in code units).

    Returns the beam density and conditional mean velocity, the constant local width, and the core density and
    electron drift that make the state charge- and current-neutral: ``n_c = n0 - n_b``, ``n_e U_e = n_b U_a``.
    The spatial width ``sqrt(D)`` grows with age while the local velocity width ``sigma_a`` shrinks: the apparent
    cooling is velocity sorting, not a loss of beam energy.
    """
    D = ell0**2 + sigmab**2 * age**2
    sigma_a = sigmab * ell0 / jnp.sqrt(D)
    n_b = nb0 * ell0 / jnp.sqrt(D) * jnp.exp(-(x - xs - Ub * age) ** 2 / (2 * D))
    U_a = (Ub * ell0**2 + sigmab**2 * age * (x - xs)) / D
    return n_b, U_a, sigma_a, n0 - n_b, n_b * U_a / n0


def density_weighted_maxwellian(U, alpha, basis_u, density, H):
    """Hermite-Fourier coefficients of Maxwellians with drift ``U(x)`` and density ``density(x)``, shape (Ns*H, 1, Nx//2+1, 1).

    ``compute_C_nmp`` returns unit-density coefficients; they are transformed back once, multiplied by the density
    in real space (never pointwise in Fourier space, which would be a convolution), and transformed forward. Where the
    density is negligible the drift is relaxed to the basis drift first, so an unbounded linear drift in the empty
    tails cannot create overflowing ``(U - u)^n`` terms (0 * inf); the relaxation is exact where density > 1e-12 max.
    """
    Ns, Nx = U.shape
    taper = density / (density + 1e-12 * jnp.max(density, axis=1, keepdims=True) + 1e-300)
    U = basis_u[:, None] + (U - basis_u[:, None]) * taper
    grid = jnp.zeros((Ns, 3, 1, Nx, 1)).at[:, 0, 0, :, 0].set(U)
    alpha_s = jnp.stack([alpha, jnp.ones(Ns), jnp.ones(Ns)], 1).ravel()
    u_s = jnp.stack([basis_u, jnp.zeros(Ns), jnp.zeros(Ns)], 1).ravel()
    ck = compute_C_nmp(grid, alpha_s, u_s, H, 1, 1, Ns)
    cx = jnp.fft.irfftn(ck, s=(1, 1, Nx), axes=(-1, -3, -2), norm="forward") * density[:, None, None, None, None, :, None]
    return jnp.fft.rfftn(cx, axes=(-1, -3, -2), norm="forward").reshape(Ns * H, 1, Nx // 2 + 1, 1), alpha_s, u_s


def initial_state(p, Nx, H):
    """SPECTRAX ``input_parameters`` for a pulse ``p`` (dict of code-unit scalars, see ``pulse``) plus a Gauss-consistent seed.

    Required keys: ``Lx, n0, nb0, Ub, sigmab, alpha_e, alpha_c, M, t_max`` and, for a pulse, ``xs, ell0, age``;
    without ``ell0`` the beam is homogeneous (the linear benchmark). Optional: ``seed`` (a fixed complex array of
    core-proton density perturbation per rFFT index, zero mean), ``nu``, ``tolerance`` and ``basis_ub`` (beam basis
    drift, default ``Ub``). The seed sets ``E_k = rho_k / (i k)`` for k != 0 and E_0 = 0.
    """
    x = jnp.arange(Nx) * p["Lx"] / Nx
    if p.get("ell0") is None:
        n_b, U_a, sigma_a = jnp.full(Nx, p["nb0"]), jnp.full(Nx, p["Ub"]), p["sigmab"]
        n_c, U_e = p["n0"] - n_b, n_b * U_a / p["n0"]
    else:
        n_b, U_a, sigma_a, n_c, U_e = pulse(x, p["n0"], p["nb0"], p["xs"], p["ell0"], p["Ub"], p["sigmab"], p["age"])
    seed = jnp.zeros(Nx // 2 + 1) if p.get("seed") is None else jnp.asarray(p["seed"])
    n_c = n_c + p["n0"] * jnp.fft.irfft(seed, n=Nx, norm="forward")
    density = jnp.stack([jnp.full(Nx, p["n0"]), n_c, n_b])
    alpha = jnp.stack([p["alpha_e"], p["alpha_c"], jnp.sqrt(2.0) * sigma_a])
    basis_u = jnp.stack([0.0, 0.0, p.get("basis_ub", p["Ub"])])
    Ck_0, alpha_s, u_s = density_weighted_maxwellian(jnp.stack([U_e, jnp.zeros(Nx), U_a]), alpha, basis_u, density, H)
    k = 2 * jnp.pi * jnp.arange(Nx // 2 + 1) / p["Lx"]
    rho_k = jnp.fft.rfft(-density[0] + density[1] + density[2], norm="forward")
    E_k = jnp.where(k > 0, rho_k / (1j * jnp.where(k > 0, k, 1.0)), 0.0)
    Fk_0 = jnp.zeros((6, 1, Nx // 2 + 1, 1), complex).at[0, 0, :, 0].set(E_k)
    return dict(Lx=p["Lx"], Ly=1.0, Lz=1.0, qs=jnp.array([-1.0, 1.0, 1.0]), ms=jnp.array([1.0, p["M"], p["M"]]),
                Omega_cs=jnp.array([1.0, 1 / p["M"], 1 / p["M"]]), alpha_s=alpha_s, u_s=u_s, nu=p.get("nu", 0.0), D=0.0,
                t_max=p["t_max"], ode_tolerance=p.get("tolerance", 1e-10), Ck_0=Ck_0, Fk_0=Fk_0)


def moments(Ck, alpha_s, u_s, H, Nx):
    """Real-space density, mean velocity and parallel pressure per population (Appendix A of the plan), shape (Ns, Nx).

    ``Ck`` is one saved state, shape (Ns*H, 1, Nx//2+1, 1). Pressure is ``n <(v - U)^2>`` in code units (times mass
    for physical pressure); a negative value marks an unresolved state and is reported, never clipped.
    """
    Ns = Ck.shape[0] // H
    C = jnp.fft.irfft(Ck.reshape(Ns, H, Nx // 2 + 1), n=Nx, norm="forward")
    a, u = alpha_s.reshape(Ns, 3)[:, 0, None], u_s.reshape(Ns, 3)[:, 0, None]
    c0, c1, c2 = C[:, 0], C[:, 1] if H > 1 else 0 * C[:, 0], C[:, 2] if H > 2 else 0 * C[:, 0]
    n = a * c0
    flux = a * (u * c0 + a / jnp.sqrt(2.0) * c1)
    second = a * ((u**2 + a**2 / 2) * c0 + jnp.sqrt(2.0) * u * a * c1 + a**2 / jnp.sqrt(2.0) * c2)
    U = flux / n
    return n, U, second - n * U**2


def gauss_residual(Ck, Fk, qs, alpha_s, H, Lx):
    """Max |i k E_k - rho_k| over k != 0 relative to max |rho_k|; the mean-field E_0 is reported separately."""
    Ns, Nk = Ck.shape[0] // H, Fk.shape[-2]
    rho = jnp.sum(qs[:, None] * alpha_s.reshape(Ns, 3)[:, 0, None] * Ck.reshape(Ns, H, Nk)[:, 0], axis=0)
    k = 2 * jnp.pi * jnp.arange(Nk) / Lx
    return jnp.max(jnp.abs(1j * k[1:] * Fk[0, 0, 1:, 0] - rho[1:])) / jnp.max(jnp.abs(rho[1:]))


def probe(Ek, x, Lx, Nx):
    """E(x) from the rFFT coefficients ``Ek`` (last axis Nx//2+1) at continuous positions ``x`` (no grid lookup).

    Paired modes contribute ``2 Re[E_k e^{ikx}]``; an even grid's Nyquist singleton contributes ``Re[E_N/2 e^{ikx}]``.
    """
    j = jnp.arange(Ek.shape[-1])
    weight = jnp.where((j == 0) | ((Nx % 2 == 0) & (j == Nx // 2)), 1.0, 2.0)
    return jnp.real(jnp.sum(weight * Ek * jnp.exp(2j * jnp.pi * j * jnp.asarray(x)[..., None] / Lx), axis=-1))


def antenna_voltage(Ek, x1, x2, Lx, Nx):
    """Ideal voltage phi(x1) - phi(x2) with phi_k = i E_k / k, plus the uniform-field part -E_0 (x1 - x2).

    For a single mode this has the finite-baseline response |sin(k l/2) / (k l/2)| relative to E l.
    """
    j = jnp.arange(1, Ek.shape[-1])
    phi = 1j * Ek[..., 1:] / (2 * jnp.pi * j / Lx)
    weight = jnp.where((Nx % 2 == 0) & (j == Nx // 2), 1.0, 2.0)
    phase = lambda x: jnp.exp(2j * jnp.pi * j * jnp.asarray(x)[..., None] / Lx)
    return jnp.real(jnp.sum(weight * phi * (phase(x1) - phase(x2)), axis=-1)) - jnp.real(Ek[..., 0]) * (x1 - x2)


def spectrogram(signal, window, hop):
    """Fixed-window short-time power ``|FFT(w s)|^2`` in (frame, frequency) order; smooth in ``signal`` (no argmax)."""
    frames = (signal.shape[-1] - window) // hop + 1
    idx = jnp.arange(window)[None, :] + hop * jnp.arange(frames)[:, None]
    taper = 0.5 - 0.5 * jnp.cos(2 * jnp.pi * jnp.arange(window) / window)
    return jnp.abs(jnp.fft.rfft(signal[..., idx] * taper, axis=-1)) ** 2


def dielectric(omega, k, n, sigma, U, q_over_m):
    """Full kinetic Maxwellian dielectric 1 + sum chi_s (Faddeeva form), k > 0; plasma frequency^2 = n q (q/m)."""
    z = (omega - k * U) / (np.sqrt(2) * k * sigma)
    Z = 1j * np.sqrt(np.pi) * wofz(z)
    return 1 + np.sum(n * q_over_m * np.sign(q_over_m) / (k * sigma) ** 2 * (1 + z * Z))


def kinetic_root(omega0, k, n, sigma, U, q_over_m, iterations=60):
    """Newton root of ``dielectric`` from ``omega0``; continue roots in k or parameters from the previous root."""
    omega, h = complex(omega0), 1e-7
    for _ in range(iterations):
        f = dielectric(omega, k, n, sigma, U, q_over_m)
        step = f * 2 * h * abs(omega) / (dielectric(omega * (1 + h), k, n, sigma, U, q_over_m)
                                          - dielectric(omega * (1 - h), k, n, sigma, U, q_over_m))
        omega -= step
        if abs(step) < 1e-15 * abs(omega):
            break
    return omega
