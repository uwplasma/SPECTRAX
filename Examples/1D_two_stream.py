"""Example: 1D two-stream instability (Discontinuous Galerkin).

Loads parameters from ``input_1D_two_stream_DG_paper.toml``, builds the standard two-stream
initial conditions in Legendre-DG/Hermite space, runs the solver, and plots results.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from time import time
import jax.numpy as jnp
from jax import block_until_ready, config
config.update("jax_enable_x64", True)
from spectrax import simulation, load_parameters, DG_project, inverse_HDG_transform
from scipy.special import eval_hermite
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt

# Read from input.toml
toml_file = os.path.join(os.path.dirname(__file__), 'input_1D_two_stream.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

alpha_s = input_parameters["alpha_s"]
u_s = input_parameters["u_s"]
nx = input_parameters["nx"]
Lx = input_parameters["Lx"]
Omega_ce = input_parameters["Omega_ce"]
dn1 = input_parameters["dn1"]
dn2 = input_parameters["dn2"]
Nx = solver_parameters["Nx"]
Ny = solver_parameters["Ny"]
Nz = solver_parameters["Nz"]
Nn = solver_parameters["Nn"]
N_DG = solver_parameters["N_DG"]
dims = solver_parameters["dims"]

alpha = jnp.array(alpha_s).reshape(2, 3)
a0 = alpha[:, 0]
a1 = alpha[:, 1]
a2 = alpha[:, 2]
u = jnp.array(u_s).reshape(2, 3)
u0 = u[:, 0]
u1 = u[:, 1]
u2 = u[:, 2]

# Initialize distribution function as a two-stream instability
Ex = lambda x: -(dn1 + dn2) * Lx / (2 * jnp.pi * nx * Omega_ce) * jnp.sin(2 * nx * jnp.pi * x / Lx)
F0 = lambda x, y, z: jnp.array([Ex(x), jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)])
input_parameters["Fk_0"] = DG_project(F0, input_parameters, solver_parameters)

C10 = lambda x: (1 / (alpha_s[0] ** 3)) * (1 + dn1 * jnp.cos(2 * nx * jnp.pi * x / Lx))
C20 = lambda x: (1 / (alpha_s[3] ** 3)) * (1 + dn2 * jnp.cos(2 * nx * jnp.pi * x / Lx))
C0 = lambda x, y, z: jnp.concatenate([jnp.array([C10(x)]), jnp.broadcast_to(jnp.zeros_like(x), (Nn-1,) + jnp.zeros_like(x).shape), jnp.array([C20(x)]), jnp.broadcast_to(jnp.zeros_like(x), (Nn-1,) + jnp.zeros_like(x).shape)])
input_parameters["Ck_0"] = DG_project(C0, input_parameters, solver_parameters)

# Simulate
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

# Plot results

Ck = output["Ck"]
Fk = output["Fk"]
t = output["time"]

Nt = len(t)

fig, ax = plt.subplots()

vx = jnp.linspace(-5 * alpha_s[0], 5 * alpha_s[0], 2001)
Vx, Vy, Vz = jnp.meshgrid(vx, jnp.array([0.]), jnp.array([0.]), indexing='xy')
f1 = inverse_HDG_transform(Ck[:, 0:Nn, ...], Nn, 1, 1, 
                                (Vx - u_s[0]) / alpha_s[0], 
                                (Vy - u_s[1]) / alpha_s[1], 
                                (Vz - u_s[2]) / alpha_s[2])
f2 = inverse_HDG_transform(Ck[:, Nn:2*Nn, ...], Nn, 1, 1, 
                                (Vx - u_s[3]) / alpha_s[3], 
                                (Vy - u_s[4]) / alpha_s[4], 
                                (Vz - u_s[5]) / alpha_s[5])
f = f1 + f2
electron_phase_plot = ax.imshow(jnp.transpose(f[0, 0, :, 0, 0, :, 0]), extent=(0, Lx, vx[0], vx[-1]),
                                        cmap='jet', origin='lower', interpolation='sinc')
plt.colorbar(electron_phase_plot, ax=ax, label="$f$")
ax.set(xlabel="x/d_e", ylabel="v/c", title="Electron Distribution")
ax.set_aspect('auto', adjustable='box')
electron_phase_text = ax.text(
0.5, 0.9, "", transform=ax.transAxes, ha="center", va="top",
fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

def update(frame):
        electron_phase_plot.set_array(jnp.transpose(f[frame, 0, :, 0, 0, :, 0]))
        electron_phase_plot.set_clim(vmin=f[frame].min(), vmax=f[frame].max())
        electron_phase_text.set_text(f"Time: {t[frame]:.1f} * ωₚ")

        return [electron_phase_plot, electron_phase_text]

anim = FuncAnimation(fig, update, frames=Nt, blit=False, interval=1, repeat_delay=100)
# anim.save(os.path.join(os.path.dirname(__file__), 'Figures/two_stream_f_DG.gif'), writer=PillowWriter(fps=25))

U_EM = jnp.sum(0.5 * Omega_ce**2 * jnp.abs(Fk[:, :, 0, :, 0, :])**2 / (1 + 2 * jnp.arange(0, N_DG)), axis=(-2, -1)) / Nx # shape (Nt, 6)

C_ave = (jnp.sum(Ck[:, :, 0, :, 0, 0], axis=-1) / Nx).reshape(Nt, 2, 1, 1, Nn)

C000 = C_ave[:, :, 0, 0, 0]

C100 = C_ave[:, :, 0, 0, 1]
C010 = 0 # C_ave[:, :, 0, 1, 0] Out of range
C001 = 0 # C_ave[:, :, 1, 0, 0]

C200 = C_ave[:, :, 0, 0, 2]
C020 = 0 # C_ave[:, :, 0, 2, 0] Out of range
C002 = 0 # C_ave[:, :, 2, 0, 0]

term0 = (0.5 * (a0**2 + a1**2 + a2**2) + (u0**2 + u1**2 + u2**2)) * C000 # (Ns,)
term1 = jnp.sqrt(2.0) * (a0 * u0 * C100 + a1 * u1 * C010 + a2 * u2 * C001)  # (Nt, Ns)
term2 = (1.0 / jnp.sqrt(2.0)) * (a0**2 * C200 + a1**2 * C020 + a2**2 * C002)  # (Nt, Ns)

kinetic_energy_species = 0.5 * a0 * a1 * a2 * (term0 + term1 + term2)  # (Nt, Ns)
kinetic_energy = jnp.sum(kinetic_energy_species, axis=1)  # (Nt,)

E_tot = kinetic_energy + jnp.sum(U_EM, axis=1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogy(t, jnp.sum(U_EM[:, :3], axis=1))
ax.set_xlabel(r'$\omega_{\rm{pe}}t$', fontsize=20)
ax.set_ylabel('Electric energy', fontsize=20)
ax.set_xlim([0, jnp.max(t)])
ax.tick_params(axis='both', labelsize=14)
ax.text(0.03, 0.97, '(a)', transform=ax.transAxes,
        ha='left', va='top', color='black',
        fontsize=ax.xaxis.label.get_fontsize(), zorder=10)

# Show energy conservation without using a separate figure.
relative_energy_error = jnp.abs(E_tot - E_tot[0]) / jnp.abs(E_tot[0])
inset_ax = ax.inset_axes([0.52, 0.12, 0.45, 0.46])
inset_ax.semilogy(t, relative_energy_error, color='tab:red', linewidth=1.5, label='Relative total\nenergy error')
inset_ax.set_xlabel(r'$\omega_{\rm{pe}}t$', fontsize=14, labelpad=1)
inset_ax.set_ylabel(r'$|U-U_0|/U_0$', fontsize=14, labelpad=1)
inset_ax.set_xlim([0, jnp.max(t)])
inset_ax.tick_params(axis='both', which='both', labelsize=12)
inset_ax.grid(True, which='both', alpha=0.25, linewidth=0.5)
inset_ax.legend(fontsize=12, loc='upper right', framealpha=0.5)

fig.tight_layout()

plt.show()

# plt.savefig(os.path.join(os.path.dirname(__file__), 'two_stream_energy.png'), dpi=300)

# print('Saving results...')
# jnp.savez(os.path.join(os.path.dirname(__file__), 'output_two-stream.npz'), **output)
