"""Doubly periodic GEM Harris-sheet reconnection.

The defaults are a smoke test. A resolved nonlinear run is

    python Examples/2D_GEM_reconnection.py --nx 129 --ny 129 --hermite 6 \
        --t-max 625 --chunk-time 25 --snapshots 4

Lengths and times are reported in ion units, although SPECTRAX evolves in
``d_e`` and ``omega_pe^-1``. Two sheets replace GEM's conducting y boundaries.
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from diffrax import Dopri8
from matplotlib.animation import FuncAnimation, PillowWriter

from spectrax import simulation


DI, OMEGA_CI = 5.0, 0.02


def initial_conditions(nx, ny, hermite, nu):
    """Return the periodic GEM equilibrium in SPECTRAX electron units."""
    lx = ly = 25.6 * DI
    width, psi = 0.5 * DI, 0.1 * DI
    x = jnp.linspace(0.0, lx, nx, endpoint=False)
    y = jnp.linspace(0.0, ly, ny, endpoint=False)
    X, Y, _ = jnp.meshgrid(x, y, jnp.zeros(1), indexing="xy")
    y1, y2 = ly / 4.0, 3.0 * ly / 4.0
    sheets = jnp.stack((jnp.cosh((Y - y1) / width) ** -2,
                        jnp.cosh((Y - y2) / width) ** -2))
    density = jnp.concatenate((sheets, sheets, 0.2 * jnp.ones((2,) + Y.shape)))

    alpha_e, alpha_i = jnp.sqrt(25.0 / 6.0) * 0.1, jnp.sqrt(5.0 / 6.0) * 0.1
    alpha = jnp.repeat(jnp.array((alpha_e, alpha_e, alpha_i, alpha_i,
                                  alpha_e, alpha_i)), 3)
    coefficients = jnp.zeros((6, hermite, hermite, hermite, ny, nx, 1))
    coefficients = coefficients.at[:, 0, 0, 0].set(
        density / jnp.prod(alpha.reshape(6, 3), axis=1)[:, None, None, None]
    )

    kx, ky = 2.0 * jnp.pi / lx, 2.0 * jnp.pi / ly
    phase = ky * (Y - y1)
    bx = (jnp.tanh((Y - y1) / width) - jnp.tanh((Y - y2) / width) - 1.0
          - psi * ky * jnp.cos(kx * X) * jnp.sin(phase))
    by = psi * kx * jnp.sin(kx * X) * jnp.cos(phase)
    magnetic = jnp.stack((bx, by, jnp.zeros_like(bx)))
    def transform(a):
        return jnp.fft.rfftn(a, axes=(-1, -3, -2), norm="forward")

    return {
        "Lx": lx, "Ly": ly, "Lz": 1.0, "mi_me": 25.0,
        "ms": jnp.array((1.0, 1.0, 25.0, 25.0, 1.0, 25.0)),
        "qs": jnp.array((-1.0, -1.0, 1.0, 1.0, -1.0, 1.0)),
        "alpha_s": alpha,
        "u_s": jnp.array((0, 0, 1/30, 0, 0, -1/30,
                            0, 0, -1/6, 0, 0, 1/6,
                            0, 0, 0, 0, 0, 0)),
        "Omega_cs": jnp.array((0.5, 0.5, 0.02, 0.02, 0.5, 0.02)),
        "nu": nu, "D": 0.0, "ode_tolerance": 1e-7,
        "Ck_0": transform(coefficients),
        "Fk_0": transform(jnp.concatenate((jnp.zeros_like(magnetic), magnetic))),
    }


def fields_and_flux(fk, nx, lx, ly):
    ny = fk.shape[-3]
    magnetic = np.fft.irfftn(fk[:, 3:5], s=(1, ny, nx),
                              axes=(-1, -3, -2), norm="forward").real
    kx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    KX, KY = np.meshgrid(kx, ky)
    bx, by = fk[:, 3, :, :, 0], fk[:, 4, :, :, 0]
    k2 = KX**2 + KY**2
    az_k = np.zeros_like(bx)
    np.divide(-1j * KY * bx + 1j * KX * by, k2, out=az_k, where=k2 != 0.0)
    az = np.fft.irfftn(az_k, s=(ny, nx), axes=(-2, -1), norm="forward").real
    return np.linalg.norm(magnetic, axis=1)[..., 0], az


def save_plots(output_dir, t, fk, nx, lx, ly):
    magnitude, az = fields_and_flux(fk, nx, lx, ly)
    tau, x = OMEGA_CI * t, np.linspace(0.0, lx / DI, nx, endpoint=False)
    y = np.linspace(0.0, ly / DI, fk.shape[-3], endpoint=False)
    sheet = np.argmin(abs(y - ly / (4.0 * DI)))
    flux = np.ptp(az[:, sheet], axis=1) / DI

    fig, ax = plt.subplots(figsize=(5.8, 3.4), constrained_layout=True)
    ax.plot(tau, flux)
    ax.set(xlabel=r"$\Omega_{ci}t$", ylabel=r"reconnected flux $\Delta A_z/(B_0d_i)$")
    fig.savefig(output_dir / "gem-fourier-flux.png", dpi=180)
    plt.close(fig)

    window = np.arange(len(y)) < len(y) // 2
    y, az, magnitude = y[window], az[:, window], magnitude[:, window]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    image = ax.pcolormesh(x, y, magnitude[0], shading="auto", cmap="viridis",
                          vmin=magnitude.min(), vmax=magnitude.max())
    contours, title = [None], ax.set_title("")
    ax.set(xlabel=r"$x/d_i$", ylabel=r"$y/d_i$")
    fig.colorbar(image, ax=ax, label=r"$|B|/B_0$")

    def update(frame):
        image.set_array(magnitude[frame].ravel())
        if contours[0] is not None:
            contours[0].remove()
        levels = np.linspace(az[frame].min(), az[frame].max(), 24)
        contours[0] = ax.contour(x, y, az[frame], levels, colors="k", linewidths=0.5)
        title.set_text(rf"$\Omega_{{ci}}t={tau[frame]:.2f}$")
        return image, title, contours[0]

    frames = range(0, len(t), max(1, (len(t) + 99) // 100))
    animation = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)
    animation.save(output_dir / "gem-fourier.gif", PillowWriter(fps=12), dpi=100)
    update(len(t) - 1)
    fig.savefig(output_dir / "gem-fourier-final.png", dpi=180)
    plt.close(fig)
    return flux


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=33)
    parser.add_argument("--ny", type=int, default=33)
    parser.add_argument("--hermite", type=int, default=4)
    parser.add_argument("--nu", type=float, default=1.0)
    parser.add_argument("--t-max", type=float, default=10.0)
    parser.add_argument("--chunk-time", type=float, default=10.0)
    parser.add_argument("--snapshots", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("gem_fourier_output"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    values = initial_conditions(args.nx, args.ny, args.hermite, args.nu)

    times, fields, offset, start = [], [], 0.0, time.perf_counter()
    while offset < args.t_max:
        duration = min(args.chunk_time, args.t_max - offset)
        values["t_max"] = duration
        output = simulation(values, Nx=args.nx, Ny=args.ny, Nz=1,
                            Nn=args.hermite, Nm=args.hermite, Np=args.hermite,
                            Ns=6, timesteps=args.snapshots, dt=1e-3, solver=Dopri8())
        jax.block_until_ready(output["Ck"])
        first = bool(times)
        times.append(np.asarray(output["time"])[first:] + offset)
        fields.append(np.asarray(output["Fk"])[first:])
        values["Ck_0"], values["Fk_0"] = output["Ck"][-1], output["Fk"][-1]
        offset += duration
        print(f"t={offset:g}")

    t, fk = np.concatenate(times), np.concatenate(fields)
    flux = save_plots(args.output_dir, t, fk, args.nx, float(values["Lx"]),
                      float(values["Ly"]))
    np.savez(args.output_dir / "gem-fourier.npz", time=t, flux=flux, Fk=fk)
    power = abs(np.asarray(values["Ck_0"]).reshape(6, args.hermite, args.hermite,
                                                    args.hermite, -1)) ** 2
    total = power.sum(axis=(1, 2, 3, 4))
    tails = {name: float(np.max(np.take(power, -1, axis=axis).sum(axis=(1, 2, 3)) / total))
             for name, axis in (("p", 1), ("m", 2), ("n", 3))}
    print({"runtime_s": time.perf_counter() - start,
           "max_reconnected_flux": float(flux.max()), "hermite_tails": tails})


if __name__ == "__main__":
    main()
