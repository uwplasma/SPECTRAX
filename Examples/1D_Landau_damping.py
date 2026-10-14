"""Example: 1D Landau damping.

Runs a 1D Landau damping setup defined in ``input_1D_landau_damping.toml`` and
plots standard diagnostics.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from time import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import block_until_ready
from spectrax import simulation, load_parameters, construct_idx_array, legT, inverse_HDG_transform
from scipy.special import eval_hermite
from jax.scipy.special import factorial
from matplotlib.animation import FuncAnimation, PillowWriter

# Read from input.toml
toml_file = os.path.join(os.path.dirname(__file__), 'input_1D_landau_damping.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

alpha_s = input_parameters["alpha_s"]
u_s = input_parameters["u_s"]
ms = input_parameters["ms"]
nx = input_parameters["nx"]
ny = input_parameters["ny"]
nz = input_parameters["nz"]
Lx = input_parameters["Lx"]
Ly = input_parameters["Ly"]
Lz = input_parameters["Lz"]
Omega_ce = input_parameters["Omega_ce"]
dn = input_parameters["dn1"]
Nx = solver_parameters["Nx"]
Ny = solver_parameters["Ny"]
Nz = solver_parameters["Nz"]
Nn = solver_parameters["Nn"]
Nm = solver_parameters["Nm"]
Np = solver_parameters["Np"]
Nt = solver_parameters["timesteps"]
N_DG = solver_parameters["N_DG"]
dims = solver_parameters["dims"]

basis_idx = construct_idx_array(dims, N_DG)
F0 = lambda x, y, z: jnp.concatenate([jnp.array([dn * Lx / (2 * jnp.pi * nx * Omega_ce) * jnp.cos(2*jnp.pi*nx*x/Lx)]), jnp.broadcast_to(jnp.zeros_like(x), (5,) + jnp.zeros_like(x).shape)])
input_parameters["Fk_0"] = legT(F0, basis_idx, N_DG, Lx, Nx)

Ci0_0 = lambda x: 1 / (alpha_s[3] ** 3) * jnp.ones_like(x)
Ce0_0 = lambda x: 1 / (alpha_s[0] ** 3) * (1 + dn * jnp.sin(2*jnp.pi*nx*x/Lx))
C0 = lambda x, y, z: jnp.concatenate([jnp.array([Ce0_0(x)]), jnp.broadcast_to(jnp.zeros_like(x), (Nn-1,) + jnp.zeros_like(x).shape), jnp.array([Ci0_0(x)]), jnp.broadcast_to(jnp.zeros_like(x), (Nn-1,) + jnp.zeros_like(x).shape)])
input_parameters["Ck_0"] = legT(C0, basis_idx, N_DG, Lx, Nx)

# Simulate
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

Fk = output['Fk'][:, :, 0, :, 0, :] # (Nt, 6, Nx, Nl)
Ck = output['Ck'][:, :, 0, :, 0, :] # (Nt, Ns*Nn, Nx, Nl)
C = Ck[:, :, :, 0] # (Nt, Ns*Nn, Nx)
t = output["time"] # (Nt,)

# Velocity space reconstruction
v_max = 3 * alpha_s[0]
Nv = 100
v = jnp.linspace(-v_max, v_max, Nv)
Vx, Vy, Vz = jnp.meshgrid(v, jnp.array([0.]), jnp.array([0.]), indexing='xy')
f_e = inverse_HDG_transform(output['Ck'][:, :Nn*Nm*Np], Nn, Nm, Np,
                            (Vx - u_s[0]) / alpha_s[0],
                            (Vy - u_s[1]) / alpha_s[1],
                            (Vz - u_s[2]) / alpha_s[2])[:, 0, :, 0, 0, :, 0] # (Nt, Nx, Nv)

print(f_e.shape)

# Animate electron f(x, vx) (DG element averages, plotted at element centers)
x = Lx / Nx * (jnp.arange(Nx) + 0.5)
# Perturbation relative to the spatial average, since dn is small
df_e = f_e - jnp.mean(f_e, axis=1, keepdims=True) # (Nt, Nx, Nv)
frames = range(0, len(t), 5) # Every 5th output time

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
im_f = ax[0].pcolormesh(x, v, f_e[0].T, cmap='jet', shading='nearest', vmin=jnp.min(f_e), vmax=jnp.max(f_e))
plt.colorbar(im_f, ax=ax[0], label=r"$f_e$")
ax[0].set(xlabel=r"$x$", ylabel=r"$v_x$", title=rf"$f_e(x, v_x)$ at $t = {t[0]:.2f}$")
im_df = ax[1].pcolormesh(x, v, df_e[0].T, cmap='RdBu_r', shading='nearest')
plt.colorbar(im_df, ax=ax[1], label=r"$\delta f_e$")
ax[1].set(xlabel=r"$x$", ylabel=r"$v_x$", title=rf"$\delta f_e(x, v_x)$ at $t = {t[0]:.2f}$")
fig.tight_layout()

def update(frame):
    im_f.set_array(f_e[frame].T)
    im_df.set_array(df_e[frame].T)
    # delta f damps away, so rescale the color limits every frame to keep the structure visible
    df_max = jnp.max(jnp.abs(df_e[frame]))
    im_df.set_clim(-df_max, df_max)
    ax[0].set_title(rf"$f_e(x, v_x)$ at $t = {t[frame]:.2f}$")
    ax[1].set_title(rf"$\delta f_e(x, v_x)$ at $t = {t[frame]:.2f}$")
    return im_f, im_df

phase_space_anim = FuncAnimation(fig, update, frames=frames, interval=50)
phase_space_anim.save(os.path.join(os.path.dirname(__file__), 'Phase_space_Landau.gif'), writer=PillowWriter(fps=20))

# Compute energies
Ue_0 = alpha_s[0]**2 / 4 # Initial electron kinetic energy
U_EM = 0.5 * Omega_ce**2 * jnp.sum(jnp.abs(Fk) ** 2 / (1 + 2 * jnp.arange(0, N_DG)), axis=(-2, -1)) / Nx
ve2 = alpha_s[0] * alpha_s[1] * alpha_s[2] * (alpha_s[0]**2 * C[:,2] / jnp.sqrt(2) + jnp.sqrt(2) * u_s[0] * alpha_s[0] * C[:,1] + (alpha_s[0]**2 / 2 + u_s[0]**2) * C[:,0])
vi2 = alpha_s[3] * alpha_s[4] * alpha_s[5] * (alpha_s[3]**2 * C[:,Nn*Nm*Np+2] / jnp.sqrt(2) + jnp.sqrt(2) * u_s[3] * alpha_s[3] * C[:,Nn*Nm*Np+1] + (alpha_s[3]**2 / 2 + u_s[3]**2) * C[:,Nn*Nm*Np])
U_Ke = 0.5 * ms[0] * jnp.sum(ve2, axis=(-1)) / Nx
U_Ki = 0.5 * ms[1] * jnp.sum(vi2, axis=(-1)) / Nx
U_tot = jnp.sum(U_EM, axis=(-1)) + U_Ke + U_Ki

fig, ax = plt.subplots()
ax.plot(t, U_EM[:, 0], label=r"$E_x$ energy")
ax.set_xlabel(r"$\omega_p t$")
ax.set_ylabel(r"$\langle E^2 \rangle/2$")
ax.set_title("Electric field energy growth")

fig, ax = plt.subplots()
ax.semilogy(t, U_EM[:, 0], label=r"$E_x$ energy")
ax.semilogy(t, U_Ke, label=r"Electron kinetic energy")
ax.semilogy(t, U_Ki, label=r"Ion kinetic energy")
ax.semilogy(t, U_tot, label=r"Total energy")
ax.semilogy(t, jnp.abs(U_tot / U_tot[0] - 1), label=r"Energy error")
ax.legend()
ax.set_xlabel(r"$\omega_p t$")
ax.set_ylabel(r"Energies")
plt.savefig(os.path.join(os.path.dirname(__file__), 'Energies_Landau.png'))

# Plot perturbation energies
fig, ax = plt.subplots()
ax.plot(t, U_Ke - Ue_0, label="Electron perturbation energy")
ax.plot(t, U_Ki - U_Ki[0], label="Ion perturbation energy")
ax.plot(t, jnp.sum(U_EM, axis=(-1)), label="Field energy")
ax.set(xlabel=r"$t\omega_{pe}$", ylabel=r"$U$")
ax.legend()

fig, ax = plt.subplots(1, 2)

C_Fourier = jnp.fft.fftshift(jnp.fft.fft(C, axis=-1), axes=-1)
mode = (Nx - 1) // 2 + nx

im = ax[0].imshow(jnp.abs(C_Fourier[:, :Nn, mode] / jnp.max(jnp.abs(C_Fourier[:, :Nn, mode]))), aspect='auto', cmap='jet', interpolation='none', origin='lower', extent=(0, Nn, 0, jnp.max(t)), norm="linear", vmin=1e-6, vmax=1.0)
cbar = plt.colorbar(im, ax=ax[0])
title = ax[0].set_title(r"Electron $C_n(t)$")
ax[0].set_xlabel("n")
ax[0].set_ylabel("t")

im = ax[1].imshow(jnp.abs(C_Fourier[:, Nn:, mode] / jnp.max(jnp.abs(C_Fourier[:, Nn:, mode]))), aspect='auto', cmap='jet', interpolation='none', origin='lower', extent=(0, Nn, 0, jnp.max(t)), norm="linear", vmin=1e-6, vmax=1.0)
cbar = plt.colorbar(im, ax=ax[1])
title = ax[1].set_title(r"Ion $C_n(t)$")
ax[1].set_xlabel("n")
ax[1].set_ylabel("t")

fig.tight_layout()

plt.show()

# print('Saving results...')
# jnp.savez('output_landau-damping.npz', **output)
