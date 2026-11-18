"""Periodic double-Harris-sheet magnetic reconnection.

Parameters follow Camporeale et al. (2006), section IV.C; the periodic double-sheet geometry and half-sheet magnetic
diagnostic follow Koshkarov et al. (2021). The defaults are a short smoke test with a linear seed. For a nonlinear
island movie set NX, NY = 65, 17, T_MAX = 500, CHUNK_TIME = 50, SNAPSHOTS = 51 and PERTURBATION = 0.1.

Growth observable: amplitude of the first y Fourier mode (ky = 2 pi / Ly) of (Bx, By) over the left half of the box,
which contains the sheet at Lx/4. gamma is a least-squares fit of its logarithm over FIT_WINDOW (default: second half
of the run); the rms log residual and the growth rate of each half of the window are printed, and a linear eigenmode
needs them to agree. The linear reference is 0.256 Omega_ci = 5.33e-3 omega_pe.
"""
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from diffrax import Dopri8
from matplotlib.animation import FuncAnimation, PillowWriter

from spectrax import plasma_current, simulation

NX, NY, HERMITE, NU = 33, 9, 4, 1.0
PERTURBATION = 1e-4      # 1e-4: linear tearing diagnostic; 0.1: nonlinear islands
T_MAX, CHUNK_TIME, SNAPSHOTS = 10.0, 10.0, 21
FIT_WINDOW = None        # (t_start, t_end); None: second half of the run
OUTPUT_DIR = Path("reconnection_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Geometry, Harris sheets at Lx/4 and 3Lx/4, four populations (electrons and ions on each sheet).
a = np.sqrt(2.0)
lx, ly = 32.0 * a, 4.0 * np.pi * a
x, y = np.linspace(0.0, lx, NX, endpoint=False), np.linspace(0.0, ly, NY, endpoint=False)
X, Y, _ = jnp.meshgrid(x, y, jnp.zeros(1), indexing="xy")
sheets = [jnp.cosh((X - c * lx) / a) ** -2 for c in (0.25, 0.75)]
alpha = a * jnp.array([1.0 / 6.0] * 6 + [1.0 / 24.0] * 6)
density = jnp.stack(sheets + sheets)
Ck = jnp.zeros((4, HERMITE, HERMITE, HERMITE, NY, NX, 1)).at[:, 0, 0, 0].set(
    density / jnp.prod(alpha.reshape(4, 3), axis=1)[:, None, None, None])
kx, ky = 2.0 * jnp.pi / lx, 2.0 * jnp.pi / ly
B = jnp.array([
    -PERTURBATION * ky * jnp.sin(ky * Y) * jnp.sin(kx * X),
    jnp.tanh((X - lx / 4.0) / a) - jnp.tanh((X - 3.0 * lx / 4.0) / a) - 1.0
    - PERTURBATION * kx * jnp.cos(ky * Y) * jnp.cos(kx * X),
    5.0 * jnp.ones_like(X)])
transform = lambda v: jnp.fft.rfftn(v, axes=(-1, -3, -2), norm="forward")
omega_ce = 1.0 / 3.0
values = {
    "Lx": lx, "Ly": ly, "Lz": 1.0, "mi_me": 16.0, "qs": jnp.array((-1.0, -1.0, 1.0, 1.0)), "alpha_s": alpha,
    "u_s": jnp.array((0, 0, -1 / 6, 0, 0, 1 / 6, 0, 0, 1 / 6, 0, 0, -1 / 6)) / a,
    "Omega_cs": jnp.array((omega_ce, omega_ce, 1.0 / 48.0, 1.0 / 48.0)), "nu": NU, "D": 0.0, "ode_tolerance": 1e-7,
    "Ck_0": transform(Ck), "Fk_0": transform(jnp.concatenate((jnp.zeros_like(B), B)))}

# Run in chunks so only the saved snapshots stay in memory; each chunk restarts from the previous final state.
times, fields, currents, offset, start = [], [], [], 0.0, time.perf_counter()
while offset < T_MAX:
    duration = min(CHUNK_TIME, T_MAX - offset)
    values["t_max"] = duration
    output = simulation(values, Nx=NX, Ny=NY, Nz=1, Nn=HERMITE, Nm=HERMITE, Np=HERMITE, Ns=4,
                        timesteps=SNAPSHOTS, dt=1e-3, solver=Dopri8())
    jax.block_until_ready(output["Ck"])
    first = 0 if not times else 1
    # Kinetic J_z / Omega_ce from the Hermite moments (equals (curl B)_z only when dE_z/dt vanishes).
    jz = jax.vmap(lambda ck: plasma_current(output["qs"], output["alpha_s"], output["u_s"], ck,
                                            HERMITE, HERMITE, HERMITE, 4))(output["Ck"])[:, 2, :, :, 0]
    times.append(np.asarray(output["time"])[first:] + offset)
    fields.append(np.asarray(output["Fk"])[first:])
    currents.append(np.asarray(jz)[first:] / omega_ce)
    values["Ck_0"], values["Fk_0"] = jnp.asarray(output["Ck"][-1]), jnp.asarray(output["Fk"][-1])
    del output
    offset += duration
    print(f"t={offset:g}")
t, fk, jz_k = np.concatenate(times), np.concatenate(fields), np.concatenate(currents)

# Real-space fields and the flux function A_z (B = curl A_z z) from the Fourier coefficients.
fields = np.fft.irfftn(fk, s=(1, NY, NX), axes=(-1, -3, -2), norm="forward").real
KX, KY = np.meshgrid(2.0 * np.pi * np.fft.rfftfreq(NX, d=lx / NX), 2.0 * np.pi * np.fft.fftfreq(NY, d=ly / NY))
k2 = KX**2 + KY**2
az_k = np.zeros_like(fk[:, 3, :, :, 0])
np.divide(-1j * KY * fk[:, 3, :, :, 0] + 1j * KX * fk[:, 4, :, :, 0], k2, out=az_k, where=k2 != 0.0)
az = np.fft.irfftn(az_k, s=(NY, NX), axes=(-2, -1), norm="forward").real
jz = np.fft.irfftn(jz_k, s=(NY, NX), axes=(-2, -1), norm="forward").real

# Growth: first y mode of (Bx, By) over the left half of the box, least-squares fit of its logarithm.
half = (NX + 1) // 2
magnetic = fields[:, 3:, :, :half, 0]
phase = np.exp(2j * np.pi * np.arange(NY) / NY)
amplitude = np.linalg.norm(np.sum(magnetic * phase[None, None, :, None], axis=(-2, -1)), axis=1) / np.sqrt(NY * half)
window = FIT_WINDOW or (T_MAX / 2.0, T_MAX)
inside = (t >= window[0]) & (t <= window[1]) & (amplitude > 0)
ts, logs = t[inside], np.log(amplitude[inside])
gamma, intercept = np.polyfit(ts, logs, 1)
rms = np.sqrt(np.mean((logs - intercept - gamma * ts) ** 2))
h = len(ts) // 2
halves = [np.polyfit(ts[s], logs[s], 1)[0] for s in (slice(None, h + 1), slice(h, None))]

sheet = np.argmin(abs(x - lx / 4.0))
fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True, constrained_layout=True)
axes[0].semilogy(t, amplitude, color="0.65", label="half-sheet magnetic mode")
axes[0].axvspan(*window, color="0.9", zorder=0)
axes[0].set(ylabel="magnetic-mode amplitude", title=rf"window fit $\gamma={gamma:.5f}$")
axes[0].legend(fontsize=8)
axes[1].plot(t, np.ptp(az[:, :, sheet], axis=1))
axes[1].set(xlabel=r"$\omega_{pe}t$", ylabel=r"reconnected flux $\Delta A_z$")
fig.savefig(OUTPUT_DIR / "reconnection-growth.png", dpi=180)
plt.close(fig)

# Island topology movie: kinetic J_z (colour) and A_z contours around the sheet at Lx/4.
yc = np.linspace(-ly / 2.0, ly / 2.0, NY, endpoint=False)
region = (x[:half] >= lx / 8.0) & (x[:half] <= 3.0 * lx / 8.0)
xc = x[:half][region]
az_c = np.roll(az, NY // 2, axis=1)[:, :, :half][:, :, region]
jz_c = np.roll(jz, NY // 2, axis=1)[:, :, :half][:, :, region]
jz_c -= jz_c.mean(axis=1, keepdims=True)
limit = max(np.quantile(abs(jz_c[-1]), 0.99), 1e-12)
sheet = np.argmin(abs(xc - lx / 4.0))
fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
image = ax.pcolormesh(xc, yc, jz_c[0], shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
title = ax.set_title("")
ax.set(xlabel="x", ylabel="y")
fig.colorbar(image, ax=ax, label=r"kinetic $(J_z-\langle J_z\rangle_y)/\Omega_{ce}$")
contours = []


def update(frame):
    image.set_array(jz_c[frame].ravel())
    for c in contours:
        c.remove()
    span = max(1.25 * np.ptp(az_c[frame, :, sheet]), 1e-12)
    levels = az_c[frame, :, sheet].mean() + np.linspace(-span, span, 19)
    contours[:] = [ax.contour(xc, yc, az_c[frame], levels, colors="k", linewidths=0.55)]
    title.set_text(rf"$\omega_{{pe}}t={t[frame]:.1f}$")
    return image, title


FuncAnimation(fig, update, frames=range(0, len(t), max(1, len(t) // 100)), interval=80).save(
    OUTPUT_DIR / "reconnection-topology.gif", PillowWriter(fps=12), dpi=110)
plt.close(fig)

np.savez(OUTPUT_DIR / "reconnection.npz", time=t, amplitude=amplitude, Fk=fk)
print(f"runtime {time.perf_counter() - start:.1f} s, {'linear diagnostic' if PERTURBATION <= 1e-3 else 'finite-seed nonlinear'}")
print(f"gamma = {gamma:.5g} (window {window}, rms log residual {rms:.2g}, halves {halves[0]:.4g} / {halves[1]:.4g})")
if PERTURBATION <= 1e-3:
    print(f"expected gamma = {0.256 / 48.0:.5g}")
