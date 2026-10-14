import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
number_of_processors_to_use = 1 # Parallelization, this should divide total resolution
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
from jax import block_until_ready
from spectrax import simulation, load_parameters, construct_idx_array, legT
import matplotlib.pyplot as plt

# Read from input.toml
# input_parameters, solver_parameters = load_parameters('input_1D_landau_damping.toml')
toml_file = os.path.join(os.path.dirname(__file__), 'input_1D_Weibel.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

alpha_s = input_parameters["alpha_s"]
u_s = input_parameters["u_s"]
ms = input_parameters["ms"]
dE = input_parameters["dE"]
Nx = solver_parameters["Nx"]
Ny = solver_parameters["Ny"]
Nz = solver_parameters["Nz"]
Nn = solver_parameters["Nn"]
Nm = solver_parameters["Nm"]
Np = solver_parameters["Np"]
nx = input_parameters["nx"]
Lx = input_parameters["Lx"]
Ly = input_parameters["Ly"]
Lz = input_parameters["Lz"]
Omega_ce = input_parameters["Omega_ce"]
N_DG = solver_parameters["N_DG"]
dims = solver_parameters["dims"]

basis_idx = construct_idx_array(dims, N_DG)

Ez = lambda x: dE * jnp.cos(2 * nx * jnp.pi * x / Lx)
F0 = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), Ez(x), jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)]) 
input_parameters["Fk_0"] = legT(F0, basis_idx, N_DG, Lx, Nx)

C10_0 = lambda x: 1 / (alpha_s[0] * alpha_s[1] * alpha_s[2]) * jnp.ones_like(x)
C20_0 = lambda x: 1 / (alpha_s[3] * alpha_s[4] * alpha_s[5]) * jnp.ones_like(x)
C0 = lambda x, y, z: jnp.concatenate([jnp.array([C10_0(x)]), jnp.broadcast_to(jnp.zeros_like(x), (Nn*Nm*Np-1,) + jnp.zeros_like(x).shape), jnp.array([C20_0(x)]), jnp.broadcast_to(jnp.zeros_like(x), (Nn*Nm*Np-1,) + jnp.zeros_like(x).shape)])
input_parameters["Ck_0"] = legT(C0, basis_idx, N_DG, Lx, Nx)

# Simulate
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

F = output["Fk"][:, :, 0, :, 0, :] # [time, component, x, l]
C = output["Ck"][:, :, 0, :, 0, :] # [time, Hermite mode, x, l]
t = output["time"]

Fk = jnp.fft.fftshift(jnp.fft.fft(F[..., 0], axis=-1, norm='forward'), axes=-1)

# Compute energies
U_EM = 0.5 * Omega_ce**2 * jnp.sum(jnp.abs(F) ** 2 / (1 + 2 * jnp.arange(0, N_DG)), axis=(-2, -1)) / Nx  # Average field energy density
# Squared velocities
ve2 = alpha_s[0] * alpha_s[1] * alpha_s[2] * (alpha_s[0]**2 * C[:,2] / jnp.sqrt(2) + jnp.sqrt(2) * u_s[0] * alpha_s[0] * C[:,1] + (alpha_s[0]**2 / 2 + u_s[0]**2) * C[:,0]
                                                + alpha_s[2]**2 * C[:,2*Nn*Nm] / jnp.sqrt(2) + jnp.sqrt(2) * u_s[2] * alpha_s[2] * C[:,Nn*Nm] + (alpha_s[2]**2 / 2 + u_s[2]**2) * C[:,0])
vi2 = alpha_s[3] * alpha_s[4] * alpha_s[5] * (alpha_s[3]**2 * C[:,Nn*Nm*Np+2] / jnp.sqrt(2) + jnp.sqrt(2) * u_s[3] * alpha_s[3] * C[:,Nn*Nm*Np+1] + (alpha_s[3]**2 / 2 + u_s[3]**2) * C[:,Nn*Nm*Np]
                                              + alpha_s[5]**2 * C[:,Nn*Nm*Np+2*Nn*Nm] / jnp.sqrt(2) + jnp.sqrt(2) * u_s[5] * alpha_s[5] * C[:,Nn*Nm*Np+Nn*Nm] + (alpha_s[5]**2 / 2 + u_s[5]**2) * C[:,Nn*Nm*Np])

ve2x = alpha_s[0] * alpha_s[1] * alpha_s[2] * (alpha_s[0]**2 * C[:,2] / jnp.sqrt(2) + jnp.sqrt(2) * u_s[0] * alpha_s[0] * C[:,1] + (alpha_s[0]**2 / 2 + u_s[0]**2) * C[:,0])
ve2z = alpha_s[0] * alpha_s[1] * alpha_s[2] * (alpha_s[2]**2 * C[:,2*Nn*Nm] / jnp.sqrt(2) + jnp.sqrt(2) * u_s[2] * alpha_s[2] * C[:,Nn*Nm] + (alpha_s[2]**2 / 2 + u_s[2]**2) * C[:,0])

# Kinetic energy densities
U_Ke = 0.5 * ms[0] * jnp.sum(ve2[..., 0], axis=-1) / Nx
U_Ki = 0.5 * ms[1] * jnp.sum(vi2[..., 0], axis=-1) / Nx

U_Kex = 0.5 * ms[0] * jnp.sum(ve2x[..., 0], axis=-1) / Nx
U_Kez = 0.5 * ms[0] * jnp.sum(ve2z[..., 0], axis=-1) / Nx

U_tot = jnp.sum(U_EM, axis=(-1)) + U_Ke + U_Ki

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
plt.subplots_adjust(hspace=0.2, wspace=0.2)
fig.suptitle(rf'$N_x = {Nx}$, $N_y = {Ny}$, $N_z = {Nz}$, $N_n = {Nn}$, $N_m = {Nm}$, $N_p = {Np}$', fontsize=14)
field_names = ["$E_x$", "$E_y$", "$E_z$", "$B_x$", "$B_y$", "$B_z$"]
for i, ax in enumerate(axes.flat):
    im = ax.imshow(F[:, i, :, 0], aspect='auto', cmap='jet',
                   interpolation='sinc', origin='lower', extent=(0, Lx, 0, t[-1]))
    ax.set(title=field_names[i], xlabel=r"$x/d_e$", ylabel=r"$t\omega_{pe}$")#, ylim=[1e-5, None])
    fig.colorbar(im, ax=ax)
fig.tight_layout()

fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex=True)
labels = [rf'$\langle {f}^2 \rangle / 2$' for f in ['E_x', 'E_y', 'E_z', 'B_x', 'B_y', 'B_z']]  # matches U_EM with Omega_ce = 1
for i, ax in enumerate(axes.flat):
    ax.semilogy(t, U_EM[:, i], linewidth=1.5)
    ax.set_title(labels[i], fontsize=16)
    ax.grid(True, which='both', alpha=0.3)
axes[0, 0].set_xlim([t[0], t[-1]])
fig.supxlabel(r'$\omega_{\rm{pe}}t$', fontsize=14)
fig.supylabel(r'Average field energy density', fontsize=14)
fig.tight_layout()

# Energy plots
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle(rf'$N_x = {Nx}$, $N_n = {Nn}$, $N_m = {Nm}$, $N_p = {Np}$, $N_DG = {N_DG}$', fontsize=14)

axes[0].plot(t, U_EM[:, 0], label=r"$|E_x|^2$")
axes[0].plot(t, U_EM[:, 2], label=r"$|E_z|^2$")
axes[0].plot(t, U_EM[:, 4], label=r"$|B_y|^2$")
axes[0].plot(t, U_Ke, label=r"Electron kinetic energy")
axes[0].plot(t, U_Ki, label=r"Ion kinetic energy")
axes[0].plot(t, U_tot, label=r"Total Energy")
axes[0].plot(t, jnp.abs(U_tot / U_tot[0] - 1), label=r"Relative Energy Error")
axes[0].set(xlabel=r"$t\omega_{pe}$", ylabel=r"Energies", yscale="log")#, ylim=[1e-5, None])
axes[0].legend()

# Plot perturbation energies
axes[1].plot(t, U_Kex - U_Kex[0], label="Electron perturbation energy (x-direction)")
axes[1].plot(t, U_Kez - U_Kez[0], label="Electron perturbation energy (z-direction)")
axes[1].plot(t, U_Ke - U_Ke[0], label="Electron perturbation energy")
axes[1].plot(t, U_Ki - U_Ki[0], label="Ion perturbation energy")
axes[1].plot(t, jnp.sum(U_EM, axis=(-1)), label="Field energy")
axes[1].plot(t, jnp.sum(U_EM[:, :3], axis=(-1)), label="E Field energy")
axes[1].plot(t, jnp.sum(U_EM[:, 3:], axis=(-1)), label="B Field energy")
axes[1].plot(t, jnp.sum(U_EM, axis=(-1)) + U_Ke - U_Ke[0] + U_Ki - U_Ki[0], label="total perturbation energy")
axes[1].set(xlabel=r"$t\omega_{pe}$", ylabel=r"$U$")
axes[1].legend()

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(t, U_Kez / U_Kex)
ax.set(xlabel=r"$t\omega_{pe}$", ylabel=r"$T_z / T_x$", title=r"Electron temperature anisotropy $T_z / T_x$")

plt.show()

# print('Saving results...')
# jnp.savez('output_Weibel.npz', **output)