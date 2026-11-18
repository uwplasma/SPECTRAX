"""Gradient-based inverse design of a 2D Orszag-Tang-like vortex.

The controls are the phases of the M lowest Fourier modes of the in-plane magnetic stream function (fixed mode
energies); electrons carry the consistent out-of-plane current, as in ``2D_Orszag_Tang.py``. The objective is the
in-plane magnetic energy at t_max over its initial value. ``jax.value_and_grad`` differentiates through ``simulation``
with a checkpointed Diffrax adjoint, the gradient is checked against centred differences, and L-BFGS-B minimises it.

  python 2D_Orszag_Tang_optimization.py --modes 4 --grid 16 --hermite 3 --t-max 50 --iterations 5
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import Dopri8, NoProgressMeter, RecursiveCheckpointAdjoint
from scipy.optimize import minimize

from spectrax import simulation, compute_C_nmp

jax.config.update("jax_enable_x64", True)
parser = argparse.ArgumentParser()
for name, default in (("modes", 4), ("grid", 16), ("hermite", 3), ("t-max", 50.0), ("iterations", 5), ("seed", 0)):
    parser.add_argument(f"--{name}", type=type(default), default=default)
args = parser.parse_args()

L, Omega_ce, mi_me, deltaB = 50.0, 0.5, 25.0, 0.2          # as in input_2D_orszag_tang.toml
alpha_s, u_s, N, H = jnp.array([0.25] * 3 + [0.05] * 3), jnp.zeros(6), args.grid, args.hermite
pairs = sorted(((m, n) for m in range(6) for n in range(-5, 6) if m > 0 or n > 0), key=lambda mn: (mn[0] ** 2 + mn[1] ** 2, mn))
k = jnp.array(pairs[:args.modes], dtype=float) * 2 * jnp.pi / L
a = deltaB / jnp.sqrt(0.5 * jnp.sum(k**2))                  # equal mode amplitudes with <|B_perp|^2> = deltaB^2
X, Y = jnp.meshgrid(jnp.arange(N) * L / N, jnp.arange(N) * L / N, indexing="xy")


def inplane_magnetic_energy(Fk):
    """Grid mean of Bx^2 + By^2: evaluated in real space, so no real-FFT parity weights are needed."""
    B = jnp.fft.irfftn(Fk[3:5], s=(1, N, N), axes=(-1, -3, -2), norm="forward")
    return jnp.mean(jnp.sum(B ** 2, axis=0))


def conversion(phases):
    phase = k[:, 0, None, None] * X + k[:, 1, None, None] * Y + phases[:, None, None]
    Bx, By = -a * jnp.sum(k[:, 1, None, None] * jnp.sin(phase), 0), a * jnp.sum(k[:, 0, None, None] * jnp.sin(phase), 0)
    Jz = a * jnp.sum(jnp.sum(k**2, 1)[:, None, None] * jnp.cos(phase), 0)
    U0, k0 = deltaB * Omega_ce / jnp.sqrt(mi_me), 2 * jnp.pi / L
    flow = jnp.stack([-U0 * jnp.sin(k0 * Y), U0 * jnp.sin(k0 * X), jnp.zeros_like(X)])
    Us = jnp.stack([flow.at[2].set(-Omega_ce * Jz), flow])[..., None]
    F = jnp.concatenate([jnp.zeros((3, N, N)), jnp.stack([Bx, By, jnp.ones_like(X)])])[..., None]
    parameters = dict(Lx=L, Ly=L, Lz=1.0, mi_me=mi_me, qs=jnp.array([-1.0, 1.0]), Omega_cs=jnp.array([Omega_ce, Omega_ce / mi_me]),
                      alpha_s=alpha_s, u_s=u_s, nu=1.0, D=0.0, t_max=args.t_max, ode_tolerance=1e-8,
                      Ck_0=compute_C_nmp(Us, alpha_s, u_s, H, H, H, 2).reshape(2 * H**3, N, N // 2 + 1, 1),
                      Fk_0=jnp.fft.rfftn(F, axes=(-1, -3, -2), norm="forward"))
    output = simulation(parameters, Nx=N, Ny=N, Nz=1, Nn=H, Nm=H, Np=H, Ns=2, timesteps=2, solver=Dopri8(), max_steps=100_000,
                        adjoint=RecursiveCheckpointAdjoint(checkpoints=32), progress_meter=NoProgressMeter())
    return inplane_magnetic_energy(output["Fk"][-1]) / inplane_magnetic_energy(output["Fk"][0])


value_and_grad = jax.jit(jax.value_and_grad(conversion))
theta0 = jnp.asarray(np.random.default_rng(args.seed).uniform(-np.pi, np.pi, args.modes))
value, gradient = value_and_grad(theta0)
eps = 1e-5
fd = np.array([(conversion(theta0 + eps * e) - conversion(theta0 - eps * e)) / (2 * eps) for e in jnp.eye(args.modes)])
print(f"objective {value:.8g}\nreverse-mode gradient {np.asarray(gradient)}\ncentred differences   {fd}\n"
      f"max relative difference {np.max(np.abs(gradient - fd)) / np.max(np.abs(fd)):.1e}")
result = minimize(lambda th: tuple(np.asarray(v, dtype=float) for v in value_and_grad(jnp.asarray(th))), np.asarray(theta0),
                  jac=True, method="L-BFGS-B", options=dict(maxiter=args.iterations))
print(f"L-BFGS-B: {float(value):.6g} -> {result.fun:.6g} in {result.nit} iterations ({result.nfev} gradient evaluations)")
