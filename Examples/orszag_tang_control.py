"""Orszag–Tang control problem shared by the gradient examples.

The controls are the amplitudes and phases of the M lowest Fourier modes of the in-plane magnetic stream
function, normalised to a fixed in-plane magnetic energy; electrons carry the consistent out-of-plane current,
as in ``2D_Orszag_Tang.py``. With ``fixed_amplitudes`` only the phases vary, so every mode energy stays fixed.
Objectives are real scalars of the ``simulation`` output; add any other one to ``OBJECTIVES``.
"""

import os
import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import Dopri8, NoProgressMeter, RecursiveCheckpointAdjoint

from spectrax import simulation, compute_C_nmp, plasma_current

jax.config.update("jax_enable_x64", True)
Lx = Ly = 50.0
Omega_ce, mi_me, deltaB = 0.5, 25.0, 0.2          # as in input_2D_orszag_tang.toml
alpha_s, u_s = jnp.array([0.25] * 3 + [0.05] * 3), jnp.zeros(6)
COLORS = {"initial": "#2a78d6", "optimized": "#eb6834", "third": "#1baf7a", "muted": "#7a7975"}
STYLE = {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
         "grid.linewidth": 0.6, "pdf.fonttype": 42, "savefig.dpi": 300, "lines.linewidth": 1.6}


def setup(theta, grid, hermite, t_max, nu=1.0, fixed_amplitudes=False, tolerance=1e-7, mass_ratio=mi_me, guide_field=1.0,
          amplitude=deltaB):
    """Initial condition from controls ``theta = (log amplitudes, phases)``, or phases only, at fixed in-plane magnetic energy."""
    M = theta.size if fixed_amplitudes else theta.size // 2
    pairs = sorted(((m, n) for m in range(6) for n in range(-5, 6) if m > 0 or n > 0), key=lambda mn: (mn[0] ** 2 + mn[1] ** 2, mn))
    k = jnp.array(pairs[:M], dtype=float) * 2 * jnp.pi / Lx        # the M lowest wavevectors, one of each +-k pair
    a = jnp.ones(M) if fixed_amplitudes else jnp.exp(theta[:M])
    a = a * amplitude / jnp.sqrt(0.5 * jnp.sum(a**2 * jnp.sum(k**2, axis=1)))   # <|B_perp|^2> = amplitude^2
    x = jnp.arange(grid) * Lx / grid
    X, Y = jnp.meshgrid(x, x, indexing="xy")
    phase = k[:, 0, None, None] * X + k[:, 1, None, None] * Y + theta[-M:, None, None]
    Bx = -jnp.sum(a[:, None, None] * k[:, 1, None, None] * jnp.sin(phase), axis=0)
    By = jnp.sum(a[:, None, None] * k[:, 0, None, None] * jnp.sin(phase), axis=0)
    Jz = jnp.sum(a[:, None, None] * jnp.sum(k**2, axis=1)[:, None, None] * jnp.cos(phase), axis=0)
    U0, k0 = amplitude * Omega_ce / jnp.sqrt(mass_ratio), 2 * jnp.pi / Lx
    flow = jnp.stack([-U0 * jnp.sin(k0 * Y), U0 * jnp.sin(k0 * X), jnp.zeros_like(X)])
    Us = jnp.stack([flow.at[2].set(-Omega_ce * Jz), flow])[..., None]      # electrons carry the current
    F = jnp.concatenate([jnp.zeros((3, grid, grid)), jnp.stack([Bx, By, guide_field * jnp.ones_like(X)])])[..., None]
    Ck_0 = compute_C_nmp(Us, alpha_s, u_s, hermite, hermite, hermite, 2).reshape(2 * hermite**3, grid, grid // 2 + 1, 1)
    return dict(Lx=Lx, Ly=Ly, Lz=1.0, mi_me=mass_ratio, qs=jnp.array([-1.0, 1.0]), Omega_cs=jnp.array([Omega_ce, Omega_ce / mass_ratio]),
                alpha_s=alpha_s, u_s=u_s, nu=nu, D=0.0, t_max=t_max, ode_tolerance=tolerance,
                Ck_0=Ck_0, Fk_0=jnp.fft.rfftn(F, axes=(-1, -3, -2), norm="forward"))


def solve(parameters, grid, hermite, checkpoints=32, timesteps=2, **kwargs):
    """``simulation`` with a fixed checkpoint budget, so reverse-mode memory does not grow with the number of steps."""
    kwargs = dict(dt=0.01, solver=Dopri8(), max_steps=100_000, adjoint=RecursiveCheckpointAdjoint(checkpoints=checkpoints)) | kwargs
    return simulation(parameters, Nx=grid, Ny=grid, Nz=1, Nn=hermite, Nm=hermite, Np=hermite, Ns=2, timesteps=timesteps,
                      progress_meter=NoProgressMeter(), **kwargs)


def inplane_magnetic_energy(output, i=-1):
    weights = jnp.where(jnp.arange(output["Fk"].shape[-2]) == 0, 1.0, 2.0)    # real-FFT Parseval weights
    return 0.5 * Omega_ce**2 * jnp.sum(jnp.abs(output["Fk"][i, 3:5]) ** 2 * weights[None, None, :, None])


def current_density(output, i=-1):
    """Out-of-plane current density Jz(x, y) at snapshot ``i`` from the Hermite moments."""
    Ck = output["Ck"][i]
    hermite = round((Ck.shape[0] // 2) ** (1 / 3))
    Jk = plasma_current(output["qs"], output["alpha_s"], output["u_s"], Ck, hermite, hermite, hermite, 2)
    return jnp.fft.irfftn(Jk[2], s=(1, *Ck.shape[1:2], 2 * (Ck.shape[2] - 1)), axes=(-1, -3, -2), norm="forward")[:, :, 0]


def peak(J, p=8):
    """Smooth maximum of |J| (the p-norm of the spatial mean)."""
    return jnp.mean(J**p) ** (1 / p)


OBJECTIVES = {
    "conversion": lambda output: inplane_magnetic_energy(output) / inplane_magnetic_energy(output, 0),
    "peak_current": lambda output: -peak(current_density(output)),
    "mean_conversion": lambda output: jnp.mean(jnp.stack([inplane_magnetic_energy(output, i) for i in range(1, output["Fk"].shape[0])]))
    / inplane_magnetic_energy(output, 0),
}


def initial_controls(M, seed, fixed_amplitudes=False):
    phases = jnp.asarray(np.random.default_rng(seed).uniform(-np.pi, np.pi, M))
    return phases if fixed_amplitudes else jnp.concatenate([jnp.zeros(M), phases])


def provenance():
    """Commit, JAX version and device, recorded in every report."""
    def run(*command):
        return subprocess.run(command, capture_output=True, text=True).stdout.strip() if shutil.which(command[0]) else ""
    device = jax.devices()[0].platform.upper()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    name = run("nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", visible) if device == "GPU" else run("sysctl", "-n", "machdep.cpu.brand_string")
    return dict(commit=run("git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD") or None,
                device=f"{device}, {name}" if name else device, jax=jax.__version__)
