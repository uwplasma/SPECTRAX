"""Automatic diagnostic figure for any SPECTRAX run.

:func:`plot` takes the dictionary returned by :func:`spectrax.simulation` and draws, for any number of
species and any dimension, the energy budget (electric, magnetic and kinetic energy of each species), the
relative error in total energy, the amplitude of the strongest field modes, the Hermite spectrum of each
species over time, the phase space of each species at the last saved time, and, in 2D or 3D, the field in
the plane.
"""

import matplotlib.pyplot as plt
import numpy as np
from ._inverse_transform import inverse_HF_transform

__all__ = ["plot"]


def _positive(a):
    return np.where(a > 0, a, np.nan)            # zeros would stretch a log axis to 1e-300


def _species_names(output, Ns):
    return list(output.get("species_names", [f"species {s + 1}" for s in range(Ns)]))


def phase_space(output, species=0, time_index=-1, n_velocity=201):
    """Distribution function f(position, velocity) of one species at one saved time.

    Position is the longest spatial direction and velocity the matching velocity component (the first one
    with more than one Hermite mode). Other coordinates are taken at zero. Returns (position, velocity, f).
    """
    Nn, Nm, Np, Nx, Ny, Nz = (int(output[k]) for k in ("Nn", "Nm", "Np", "Nx", "Ny", "Nz"))
    H, alpha, u = Nn * Nm * Np, np.asarray(output["alpha_s"]).reshape(-1, 3), np.asarray(output["u_s"]).reshape(-1, 3)
    axis = int(np.argmax([Nx, Ny, Nz]))
    vaxis = axis if (Nn, Nm, Np)[axis] > 1 else int(np.argmax([Nn, Nm, Np]))
    v = u[species, vaxis] + alpha[species, vaxis] * np.linspace(-4, 4, n_velocity)
    xi = [np.zeros((1, n_velocity, 1)) for _ in range(3)]
    xi[vaxis] = ((v - u[species, vaxis]) / alpha[species, vaxis]).reshape(1, n_velocity, 1)
    Ck = output["Ck"][time_index:][:1, species * H:(species + 1) * H]
    f = np.asarray(inverse_HF_transform(Ck, Nn, Nm, Np, Nx, Ny, Nz, *xi))[0, ..., 0, :, 0]   # (Ny, Nx, Nz, Nv)
    f = np.moveaxis(f, (1, 0, 2)[axis], 0)[:, 0, 0]
    L = float(output[("Lx", "Ly", "Lz")[axis]])
    return np.arange(f.shape[0]) * L / f.shape[0], v, f


def plot(output, save=None, show=True):
    """Draw the standard diagnostic figure, save it to ``save`` if given, and show it if ``show``."""
    t = np.asarray(output["time"])
    Ns = int(np.asarray(output["alpha_s"]).size // 3)
    names = _species_names(output, Ns)
    Nn, Nm, Np, Nx, Ny, Nz = (int(output[k]) for k in ("Nn", "Nm", "Np", "Nx", "Ny", "Nz"))
    H = Nn * Nm * Np
    multi_d = sum(N > 1 for N in (Nx, Ny, Nz)) > 1
    n_rows = 2 + (Ns + multi_d + 2) // 3
    fig = plt.figure(figsize=(13, 3.6 * n_rows), constrained_layout=True)
    grid = fig.add_gridspec(n_rows, 3)

    ax = fig.add_subplot(grid[0, 0])
    kinetic = np.asarray(output["kinetic_energy_species"])
    for s in range(Ns):
        ax.semilogy(t, _positive(np.abs(kinetic[:, s] - kinetic[0, s])), label=f"kinetic change, {names[s]}")
    for key, style in (("electric_energy", "-"), ("magnetic_energy", "--")):
        if key in output and np.any(np.asarray(output[key]) > 0):
            ax.semilogy(t, _positive(np.asarray(output[key])), "k" + style, label=key.split("_")[0])
    ax.set(xlabel=r"$t\,\omega_{pe}$", ylabel="energy", title="Field energy and kinetic energy exchange")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(grid[0, 1])
    W = np.asarray(output["total_energy"])
    ax.semilogy(t[1:], np.abs(W[1:] / W[0] - 1) + 1e-17, "k")
    ax.set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$|W(t)/W(0) - 1|$", title="Energy conservation")

    ax = fig.add_subplot(grid[0, 2])
    Ek = np.abs(np.asarray(output["Fk"])[:, :3]) ** 2
    mode_energy = Ek.sum(axis=1)
    mode_energy[:, 0, 0, 0] = 0                               # the uniform field is not a wave
    flat = mode_energy.reshape(len(t), -1)
    for index in np.argsort(flat.max(axis=0))[::-1][:4]:
        if flat[:, index].max() > 0:
            iy, ix, iz = np.unravel_index(index, mode_energy.shape[1:])
            k = [2 * np.pi * i / float(output[L]) for i, L in ((ix, "Lx"), (int(np.fft.fftfreq(Ny, 1 / Ny)[iy]), "Ly"),
                                                                (int(np.fft.fftfreq(Nz, 1 / Nz)[iz]), "Lz"))]
            ax.semilogy(t, np.sqrt(flat[:, index]), label="k = (" + ", ".join(f"{q:.3g}" for q in k) + r") $\omega_{pe}/c$")
    ax.set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$|E_k|$", title="Strongest electric-field modes")
    ax.legend(fontsize=7)

    order = np.add.outer(np.add.outer(np.arange(Np), np.arange(Nm)), np.arange(Nn)).ravel()
    weights = np.full(Nx // 2 + 1, 2.0)
    weights[0] = 1.0
    if Nx % 2 == 0:
        weights[-1] = 1.0
    power = (np.abs(np.asarray(output["dCk"])) ** 2 * weights[None, None, None, :, None]).sum(axis=(-3, -2, -1))
    for s in range(min(Ns, 3)):
        spectrum = np.array([power[:, s * H:(s + 1) * H][:, order == n].sum(axis=1) for n in range(order.max() + 1)]).T
        ax = fig.add_subplot(grid[1, s])
        image = ax.pcolormesh(np.arange(order.max() + 1), t, np.log10(spectrum + 1e-300), shading="auto",
                              vmin=np.log10(spectrum.max() + 1e-300) - 14, vmax=np.log10(spectrum.max() + 1e-300))
        fig.colorbar(image, ax=ax, label=r"$\log_{10}\sum_k |\delta C_n|^2$")
        ax.set(xlabel="Hermite order", ylabel=r"$t\,\omega_{pe}$", title=f"Hermite spectrum, {names[s]}")

    panels = [(s, None) for s in range(Ns)] + ([(None, "field")] if multi_d else [])
    for i, (s, kind) in enumerate(panels):
        ax = fig.add_subplot(grid[2 + i // 3, i % 3])
        if kind == "field":
            F = np.fft.irfftn(np.asarray(output["Fk"])[-1], s=(Nz, Ny, Nx), axes=(-1, -3, -2), norm="forward")
            component = int(np.argmax(np.abs(F - F.mean(axis=(1, 2, 3), keepdims=True)).max(axis=(1, 2, 3))))
            image = ax.imshow(F[component, :, :, 0], origin="lower", cmap="RdBu_r",
                              extent=(0, float(output["Lx"]), 0, float(output["Ly"])), aspect="auto")
            fig.colorbar(image, ax=ax)
            ax.set(xlabel="x", ylabel="y", title=f"{['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'][component]} at the last time")
            continue
        x, v, f = phase_space(output, s)
        df = f - f.mean(axis=0)
        limit = np.abs(df).max() or 1.0
        image = ax.pcolormesh(x, v, df.T, shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        fig.colorbar(image, ax=ax, label=r"$f - \langle f \rangle_x$")
        ax.set(xlabel=r"position ($c/\omega_{pe}$)", ylabel="velocity ($c$)", title=f"Phase-space perturbation, {names[s]}, last time")
    if save:
        fig.savefig(save, dpi=200)
    if show:
        plt.show()
    return fig
