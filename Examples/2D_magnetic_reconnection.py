"""Periodic double-Harris-sheet magnetic reconnection.

The short defaults are a smoke test. For a clear nonlinear island movie, run

    python Examples/2D_magnetic_reconnection.py --nx 65 --ny 17 \
        --t-max 500 --chunk-time 50 --snapshots 51 --perturbation 0.1

Use ``--perturbation 1e-4`` instead for a linear tearing-growth diagnostic.
The parameters follow Camporeale et al. (2006), section IV.C; the periodic
double-sheet geometry and half-sheet magnetic diagnostic follow Koshkarov et
al. (2021).
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from diffrax import Dopri8
from matplotlib.animation import FuncAnimation, PillowWriter

from spectrax import simulation


def initial_conditions(nx, ny, hermite, perturbation, nu):
    a = jnp.sqrt(2.0)
    lx, ly = 32.0 * a, 4.0 * jnp.pi * a
    x = jnp.linspace(0.0, lx, nx, endpoint=False)
    y = jnp.linspace(0.0, ly, ny, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, y, jnp.zeros(1), indexing="xy")
    sheets = (
        jnp.cosh((X - lx / 4.0) / a) ** -2,
        jnp.cosh((X - 3.0 * lx / 4.0) / a) ** -2,
    )
    density = jnp.stack((sheets[0], sheets[1], sheets[0], sheets[1]))
    alpha = a * jnp.array(([1.0 / 6.0] * 6) + ([1.0 / 24.0] * 6))
    coefficients = jnp.zeros(
        (4, hermite, hermite, hermite, ny, nx, 1), dtype=jnp.float64
    )
    coefficients = coefficients.at[:, 0, 0, 0].set(
        density / jnp.prod(alpha.reshape(4, 3), axis=1)[:, None, None, None]
    )

    kx, ky = 2.0 * jnp.pi / lx, 2.0 * jnp.pi / ly
    magnetic = jnp.array(
        [
            -perturbation * ky * jnp.sin(ky * Y) * jnp.sin(kx * X),
            jnp.tanh((X - lx / 4.0) / a)
            - jnp.tanh((X - 3.0 * lx / 4.0) / a)
            - 1.0
            - perturbation * kx * jnp.cos(ky * Y) * jnp.cos(kx * X),
            5.0 * jnp.ones_like(X),
        ]
    )
    transform = lambda value: jnp.fft.rfftn(
        value, axes=(-1, -3, -2), norm="forward"
    )
    return {
        "Lx": lx,
        "Ly": ly,
        "Lz": 1.0,
        "mi_me": 16.0,
        "qs": jnp.array((-1.0, -1.0, 1.0, 1.0)),
        "alpha_s": alpha,
        "u_s": jnp.array(
            (0.0, 0.0, -1.0 / 6.0, 0.0, 0.0, 1.0 / 6.0,
             0.0, 0.0, 1.0 / 6.0, 0.0, 0.0, -1.0 / 6.0)
        ) / a,
        "Omega_cs": jnp.array((1.0 / 3.0, 1.0 / 3.0, 1.0 / 48.0, 1.0 / 48.0)),
        "nu": nu,
        "D": 0.0,
        "ode_tolerance": 1e-7,
        "Ck_0": transform(coefficients),
        "Fk_0": transform(jnp.concatenate((jnp.zeros_like(magnetic), magnetic))),
    }


def fields_and_flux(fk, nx, lx, ly):
    ny = fk.shape[-3]
    fields = np.fft.irfftn(
        fk, s=(1, ny, nx), axes=(-1, -3, -2), norm="forward"
    ).real
    kx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    KX, KY = np.meshgrid(kx, ky)
    bx, by = fk[:, 3, :, :, 0], fk[:, 4, :, :, 0]
    k2 = KX**2 + KY**2
    az_k = np.zeros_like(bx)
    np.divide(-1j * KY * bx + 1j * KX * by, k2, out=az_k, where=k2 != 0.0)
    jz_k = 1j * KX * by - 1j * KY * bx
    az = np.fft.irfftn(az_k, s=(ny, nx), axes=(-2, -1), norm="forward").real
    jz = np.fft.irfftn(jz_k, s=(ny, nx), axes=(-2, -1), norm="forward").real
    return fields, az, jz


def save_diagnostics(output_dir, t, fk, nx, lx, ly):
    fields, az, jz = fields_and_flux(fk, nx, lx, ly)
    half = (nx + 1) // 2
    magnetic = fields[:, 3:, :, :half, 0]
    phase = np.exp(2j * np.pi * np.arange(magnetic.shape[2]) / magnetic.shape[2])
    amplitude = np.linalg.norm(
        np.sum(magnetic * phase[None, None, :, None], axis=(-2, -1)), axis=1
    ) / np.sqrt(magnetic.shape[2] * magnetic.shape[3])
    peaks = np.flatnonzero(
        (amplitude[1:-1] > amplitude[:-2]) & (amplitude[1:-1] > amplitude[2:])
    ) + 1
    peaks = peaks[t[peaks] >= t[-1] / 4.0]
    fit = None
    if len(peaks) >= 3:
        gamma, intercept = np.polyfit(t[peaks], np.log(amplitude[peaks]), 1)
        fit = np.exp(intercept + gamma * t)
    else:
        gamma = np.nan

    sheet = np.argmin(abs(np.linspace(0.0, lx, nx, endpoint=False) - lx / 4.0))
    flux = np.ptp(az[:, :, sheet], axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True, constrained_layout=True)
    axes[0].semilogy(t, amplitude, color="0.65", label="half-sheet magnetic mode")
    if fit is not None:
        axes[0].semilogy(t[peaks], amplitude[peaks], "o", ms=3)
        axes[0].semilogy(t, fit, "k--", label=rf"envelope $\gamma={gamma:.5f}$")
    axes[0].set_ylabel("magnetic-mode amplitude")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, flux)
    axes[1].set(xlabel=r"$\omega_{pe}t$", ylabel=r"reconnected flux $\Delta A_z$")
    fig.savefig(output_dir / "reconnection-growth.png", dpi=180)
    plt.close(fig)

    x = np.linspace(0.0, lx, nx, endpoint=False)[:half]
    y = np.linspace(-ly / 2.0, ly / 2.0, fk.shape[-3], endpoint=False)
    az = np.roll(az, fk.shape[-3] // 2, axis=1)[:, :, :half]
    jz = np.roll(jz, fk.shape[-3] // 2, axis=1)[:, :, :half]
    window = (x >= lx / 8.0) & (x <= 3.0 * lx / 8.0)
    x, az, jz = x[window], az[:, :, window], jz[:, :, window]
    jz -= jz.mean(axis=1, keepdims=True)
    limit = max(np.quantile(abs(jz[-1]), 0.99), 1e-12)
    sheet = np.argmin(abs(x - lx / 4.0))
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    image = ax.pcolormesh(x, y, jz[0], shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    contours = [None]
    title = ax.set_title("")
    ax.set(xlabel="x", ylabel="y")
    fig.colorbar(image, ax=ax, label=r"$J_z-\langle J_z\rangle_y$")

    def update(frame):
        image.set_array(jz[frame].ravel())
        if contours[0] is not None:
            contours[0].remove()
        center = az[frame, :, sheet].mean()
        span = max(1.25 * np.ptp(az[frame, :, sheet]), 1e-12)
        contours[0] = ax.contour(
            x, y, az[frame], center + np.linspace(-span, span, 19),
            colors="k", linewidths=0.55,
        )
        title.set_text(rf"$\omega_{{pe}}t={t[frame]:.1f}$")
        return image, title, contours[0]

    frames = range(0, len(t), max(1, len(t) // 100))
    animation = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)
    animation.save(output_dir / "reconnection-topology.gif", PillowWriter(fps=12), dpi=110)
    plt.close(fig)
    return amplitude, gamma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=33)
    parser.add_argument("--ny", type=int, default=9)
    parser.add_argument("--hermite", type=int, default=4)
    parser.add_argument("--nu", type=float, default=1.0)
    parser.add_argument("--perturbation", type=float, default=1e-4)
    parser.add_argument("--t-max", type=float, default=10.0)
    parser.add_argument("--chunk-time", type=float, default=10.0)
    parser.add_argument("--snapshots", type=int, default=21)
    parser.add_argument("--output-dir", type=Path, default=Path("reconnection_output"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    values = initial_conditions(args.nx, args.ny, args.hermite, args.perturbation, args.nu)
    times, fields, offset = [], [], 0.0
    start = time.perf_counter()
    while offset < args.t_max:
        duration = min(args.chunk_time, args.t_max - offset)
        values["t_max"] = duration
        output = simulation(
            values, Nx=args.nx, Ny=args.ny, Nz=1,
            Nn=args.hermite, Nm=args.hermite, Np=args.hermite, Ns=4,
            timesteps=args.snapshots, dt=1e-3, solver=Dopri8(),
        )
        jax.block_until_ready(output["Ck"])
        first = 0 if not times else 1
        times.append(np.asarray(output["time"])[first:] + offset)
        fields.append(np.asarray(output["Fk"])[first:])
        values["Ck_0"] = jnp.asarray(output["Ck"][-1])
        values["Fk_0"] = jnp.asarray(output["Fk"][-1])
        del output
        offset += duration
        print(f"t={offset:g}")

    t, fk = np.concatenate(times), np.concatenate(fields)
    amplitude, gamma = save_diagnostics(
        args.output_dir, t, fk, args.nx, float(values["Lx"]), float(values["Ly"])
    )
    coefficients = np.asarray(values["Ck_0"]).reshape(
        4, args.hermite, args.hermite, args.hermite, args.ny, args.nx // 2 + 1, 1
    )
    power = abs(coefficients) ** 2
    tails = {
        name: float(np.take(power, -1, axis=axis).sum() / power.sum())
        for name, axis in (("p", 1), ("m", 2), ("n", 3))
    }
    np.savez(args.output_dir / "reconnection.npz", time=t, amplitude=amplitude, Fk=fk)
    linear = args.perturbation <= 1e-3
    print(json.dumps({
        "runtime_s": time.perf_counter() - start,
        "regime": "linear diagnostic" if linear else "finite-seed nonlinear",
        "gamma": None if not np.isfinite(gamma) else float(gamma),
        "expected_gamma": 0.256 / 48.0 if linear else None,
        "hermite_tail_fraction": tails,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
