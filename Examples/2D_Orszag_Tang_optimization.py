"""Example: gradient-based optimization of the 2D Orszag–Tang vortex.

The in-plane flow and field are written as U = z x grad(phi), B = grad(psi) x z,
with phi and psi expanded in the lowest Fourier modes. The amplitudes are rescaled
so the initial flow plus magnetic energy is always E0. We maximize the fraction of
E0 converted from electromagnetic to plasma energy by t_max. The Orszag–Tang
state is the starting point. One reverse-mode gradient costs a few forward solves
for any number of controls, against 2 * n_controls solves for central differences.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from time import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy.fft import rfftn, irfftn
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from spectrax import simulation, load_parameters, compute_C_nmp

toml_file = os.path.join(os.path.dirname(__file__), 'input_2D_orszag_tang.toml')
input_parameters, solver_parameters = load_parameters(toml_file)
N, H, t_max = 25, 4, 100.0  # Reduced resolution so every forward solve takes seconds.
solver_parameters.update(Nx=N, Ny=N, Nn=H, Nm=H, Np=H, timesteps=2)
input_parameters["t_max"] = t_max

Lx, Ly, mi_me = input_parameters["Lx"], input_parameters["Ly"], input_parameters["mi_me"]
Omega_ce = input_parameters["Omega_cs"][0]
deltaB = 0.2
E0 = deltaB**2 * Omega_ce**2  # Orszag–Tang flow plus in-plane magnetic energy.
x = jnp.linspace(0, Lx, N, endpoint=False)
y = jnp.linspace(0, Ly, N, endpoint=False)
X, Y = [a[..., None] for a in jnp.meshgrid(x, y, indexing='xy')]  # (Ny, Nx, 1)

# Modes (m, n) for phi and psi, each with a cos and a sin amplitude.
modes = jnp.array([(1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2), (2, 1), (1, 2)])
kx, ky = 2 * jnp.pi * modes[:, 0] / Lx, 2 * jnp.pi * modes[:, 1] / Ly
phase = kx[:, None, None, None] * X + ky[:, None, None, None] * Y
n_modes = len(modes)

def potential(c):
    """Return f, df/dx, df/dy, laplacian(f) for amplitudes c = (cos, sin)."""
    a, b = c[:n_modes, None, None, None], c[n_modes:, None, None, None]
    f, g = a * jnp.cos(phase) + b * jnp.sin(phase), -a * jnp.sin(phase) + b * jnp.cos(phase)
    k = lambda w: w[:, None, None, None]
    return (f.sum(0), (k(kx) * g).sum(0), (k(ky) * g).sum(0), -(k(kx**2 + ky**2) * f).sum(0))

# Orszag–Tang: phi = -U0 (cos(ky y)/ky + cos(kx x)/kx), psi = deltaB (cos(ky y)/ky + cos(2 kx x)/(2 kx)).
U0 = deltaB * Omega_ce / jnp.sqrt(mi_me)
theta0 = jnp.zeros(4 * n_modes)
theta0 = theta0.at[jnp.array([0, 1])].set(-U0 * Lx / (2 * jnp.pi))
theta0 = theta0.at[2 * n_modes + jnp.array([1, 4])].set(deltaB * Lx / (2 * jnp.pi) * jnp.array([1.0, 0.5]))

def initial_state(theta):
    """Build (Ck_0, Fk_0, W_K, W_B) for controls theta = (phi amplitudes, psi amplitudes)."""
    _, phx, phy, _ = potential(theta[:2 * n_modes])
    _, psx, psy, lap = potential(theta[2 * n_modes:])
    W_K = 0.5 * (1 + mi_me) * jnp.mean(phx**2 + phy**2)
    W_B = 0.5 * Omega_ce**2 * jnp.mean(psx**2 + psy**2)
    s = jnp.sqrt(E0 / (W_K + W_B))
    Ux, Uy, Bx, By, Jz = -s * phy, s * phx, s * psy, -s * psx, -s * lap
    Ue = jnp.stack([Ux, Uy, -Omega_ce * Jz])  # Electrons carry Jz.
    Ui = jnp.stack([Ux, Uy, jnp.zeros_like(Ux)])
    p = {**input_parameters}
    p["Ck_0"] = compute_C_nmp(jnp.stack([Ue, Ui]), input_parameters["alpha_s"], input_parameters["u_s"], H, H, H, 2)
    F = jnp.stack([0 * Bx, 0 * Bx, 0 * Bx, Bx, By, jnp.ones_like(Bx)])
    p["Fk_0"] = rfftn(F, axes=(-1, -3, -2), norm="forward")
    return p, s**2 * W_K, s**2 * W_B

def conversion(theta):
    """Fraction of E0 moved from fields to particles by t_max."""
    EM = simulation(initial_state(theta)[0], **solver_parameters)["EM_energy"]
    return (EM[0] - EM[-1]) / E0

J = jax.jit(conversion)
dJ = jax.jit(jax.value_and_grad(conversion))

print(f'{4 * n_modes} controls, grid {N}x{N}, Hermite {H}^3, t_max = {t_max}')
J(theta0).block_until_ready(); dJ(theta0)[1].block_until_ready()  # Compile.
start = time(); J0 = J(theta0).block_until_ready(); t_forward = time() - start
start = time(); _, g0 = dJ(theta0); g0.block_until_ready(); t_gradient = time() - start
print(f'J = {J0:.6f}, forward {t_forward:.2f} s, gradient {t_gradient:.2f} s, '
      f'central differences {2 * len(theta0) * t_forward:.2f} s')

# Central differences along the gradient direction and per control.
d = g0 / jnp.linalg.norm(g0)
steps = np.logspace(-9, -1, 17)
fd_dir = np.array([(J(theta0 + h * d) - J(theta0 - h * d)) / (2 * h) for h in steps])
dir_error = np.abs(fd_dir - jnp.dot(g0, d)) / jnp.abs(jnp.dot(g0, d))
h = steps[np.argmin(dir_error)]
eye = jnp.eye(len(theta0))
g_fd = np.array([(J(theta0 + h * e) - J(theta0 - h * e)) / (2 * h) for e in eye])
print(f'Directional derivative relative error {dir_error.min():.1e} at h = {h:.0e}')

history = []
def objective(theta):
    value, grad = dJ(jnp.asarray(theta))
    W_B = initial_state(jnp.asarray(theta))[2]
    history.append((float(value), float(W_B / E0)))
    print(f'  eval {len(history):3d}  J = {value:.6f}  W_B/E0 = {W_B / E0:.3f}')
    return -float(value), -np.asarray(grad)

start = time()
result = minimize(objective, np.asarray(theta0), jac=True, method='L-BFGS-B', options=dict(maxiter=30))
print(f'Optimization: {len(history)} gradients in {time() - start:.1f} s, J {J0:.4f} -> {-result.fun:.4f}')
theta_opt = jnp.asarray(result.x)

# Full histories and final current sheets for the initial and optimized states.
plot_parameters = {**solver_parameters, "timesteps": 101}
runs = [simulation(initial_state(t)[0], **plot_parameters) for t in (theta0, theta_opt)]

def Jz(output, i):
    """Real-space Jz = dBy/dx - dBx/dy at save index i."""
    Fk = output["Fk"][i, 3:5]
    Fk = Fk * 1j * jnp.stack([output["ky_grid"] / Ly, output["kx_grid"] / Lx])
    return irfftn(Fk[1] - Fk[0], s=(1, N, N), axes=(-1, -3, -2), norm="forward")[:, :, 0]

names = ('Orszag–Tang', 'optimized')
values, W_B = map(np.array, zip(*history))
fig = plt.figure(figsize=(15, 9), constrained_layout=True)
grid = fig.add_gridspec(2, 4)

ax = fig.add_subplot(grid[0, 0])
ax.plot(np.arange(1, len(values) + 1), values, 'o-', ms=3)
ax.axhline(J0, color='k', ls=':', lw=1)
ax.set(xlabel='gradient evaluation', ylabel=r'$J=\Delta W_{EM}/E_0$', title='(a) objective')

ax = fig.add_subplot(grid[0, 1])
sc = ax.scatter(W_B, values, c=np.arange(len(values)), cmap='viridis', s=18, zorder=3)
ax.plot(W_B, values, color='0.6', lw=0.8)
ax.plot(W_B[0], values[0], 'k*', ms=12, label='Orszag–Tang')
ax.plot(W_B[-1], values[-1], 'r*', ms=12, label='optimum')
ax.set(xlabel=r'$W_B(0)/E_0$  (1 - flow fraction)', ylabel=r'$J$', title='(b) trajectory')
ax.legend(); fig.colorbar(sc, ax=ax, label='evaluation')

ax = fig.add_subplot(grid[0, 2])
ax.loglog(steps, dir_error, 'o-')
ax.set(xlabel='finite-difference step $h$', ylabel='relative error', title='(c) AD vs central FD, gradient direction')

ax = fig.add_subplot(grid[0, 3])
lim = np.abs(g_fd).max() * 1.1
ax.plot([-lim, lim], [-lim, lim], 'k:', lw=1)
ax.plot(g_fd, g0, 'o', ms=4)
ax.set(xlabel='central FD', ylabel='reverse-mode AD', title=f'(d) {len(theta0)} components: AD {t_gradient:.0f} s, FD {2 * len(theta0) * t_forward:.0f} s')

ax = fig.add_subplot(grid[1, 0])
for output, name, c in zip(runs, names, ('k', 'r')):
    ax.plot(output["time"], (output["EM_energy"][0] - output["EM_energy"]) / E0, c, label=name)
ax.set(xlabel=r'$\omega_{pe} t$', ylabel=r'$\Delta W_{EM}/E_0$', title='(e) field-to-plasma conversion'); ax.legend()

panels = [(runs[0], -1, 'Orszag–Tang'), (runs[1], 0, 'optimized'), (runs[1], -1, 'optimized')]
for col, (output, i, name) in enumerate(panels):
    ax = fig.add_subplot(grid[1, 1 + col])
    im = ax.imshow(Jz(output, i), origin='lower', cmap='RdBu_r', extent=(0, Lx, 0, Ly))
    ax.set(title=f'({"fgh"[col]}) $J_z$, {name}, $t={float(output["time"][i]):.0f}$', xlabel='$x/d_e$', ylabel='$y/d_e$')
    fig.colorbar(im, ax=ax)
fig.savefig('OT_optimization.png', dpi=200)
plt.show()
