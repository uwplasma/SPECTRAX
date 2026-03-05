import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
number_of_processors_to_use = 1 # Parallelization, this should divide total resolution
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
from jax import block_until_ready
from spectrax import simulation, load_parameters, plot, construct_idx_array, legT
from scipy.special import eval_legendre, roots_legendre, eval_hermite
from matplotlib.animation import FuncAnimation, PillowWriter
from jax.numpy.fft import ifftn, ifftshift
import matplotlib.pyplot as plt

# Read from input.toml
# input_parameters, solver_parameters = load_parameters('input_1D_landau_damping.toml')
toml_file = os.path.join(os.path.dirname(__file__), 'input_1D_Weibel.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

alpha_s = input_parameters["alpha_s"]
u_s = input_parameters["u_s"]
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
# output = simulation(input_parameters, **solver_parameters)
print(f"Runtime: {time() - start_time} seconds")

# Plot results
# plot(output)

F = output["Fk"][:, :, 0, :, 0, :] # [time, component, x, l]
C = output["Ck"][:, :, 0, :, 0, :] # [time, Hermite mode, x, l]
t = output["time"]
Lx = output["Lx"]
Ny = output["Ny"]
Nz = output["Nz"]
nu = output["nu"]
uz = output["u_s"][2]

# p = jnp.polyfit(t[800:], jnp.log(jnp.sum(jnp.abs(Fk[800:,4,...]) ** 2, axis=(-1,-2,-3))), 1)
# print(p[0])
# p2 = jnp.polyfit(t[800:], jnp.log(jnp.sum(jnp.abs(Fk[800:,0,...]) ** 2, axis=(-1,-2,-3))), 1)
# print(p2[0])
# p3 = jnp.polyfit(t[1200:], jnp.log(jnp.sum(jnp.abs(Fk[1200:,2,...]) ** 2, axis=(-1,-2,-3))), 1)
# print(p3[0])

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
plt.subplots_adjust(hspace=0.2, wspace=0.2)
fig.suptitle(rf'$N_x = {Nx}$, $N_y = {Ny}$, $N_z = {Nz}$, $N_n = {Nn}$, $N_m = {Nm}$, $N_p = {Np}$, $\nu = {nu}$, $|u_z| = {uz}$', fontsize=14)


im00 = axes[0, 0].imshow(F[:, 0, :, 0], aspect='auto', cmap='jet', 
        interpolation='sinc', origin='lower', extent=(0, Lx, 0, t[-1]))
axes[0, 0].set(title="$E_x$", xlabel=r"$x/d_e$", ylabel=r"$t\omega_{pe}$")#, ylim=[1e-5, None])
fig.colorbar(im00)

im01 = axes[0, 1].imshow(F[:, 1, :, 0], aspect='auto', cmap='jet', 
        interpolation='sinc', origin='lower', extent=(0, Lx, 0, t[-1]))
axes[0, 1].set(title="$E_y$", xlabel=r"$x/d_e$", ylabel=r"$t\omega_{pe}$")#, ylim=[1e-5, None])
fig.colorbar(im01)

im02 = axes[0, 2].imshow(F[:, 2, :, 0], aspect='auto', cmap='jet', 
        interpolation='sinc', origin='lower', extent=(0, Lx, 0, t[-1]))
axes[0, 2].set(title="$E_z$", xlabel=r"$x/d_e$", ylabel=r"$t\omega_{pe}$")#, ylim=[1e-5, None])
fig.colorbar(im02)

im10 = axes[1, 0].imshow(F[:, 3, :, 0], aspect='auto', cmap='jet', 
        interpolation='sinc', origin='lower', extent=(0, Lx, 0, t[-1]))
axes[1, 0].set(title="$B_x$", xlabel=r"$x/d_e$", ylabel=r"$t\omega_{pe}$")#, ylim=[1e-5, None])
fig.colorbar(im10)

im11 = axes[1, 1].imshow(F[:, 4, :, 0], aspect='auto', cmap='jet', 
        interpolation='sinc', origin='lower', extent=(0, Lx, 0, t[-1]))
axes[1, 1].set(title="$B_y$", xlabel=r"$x/d_e$", ylabel=r"$t\omega_{pe}$")#, ylim=[1e-5, None])
fig.colorbar(im11)

im12 = axes[1, 2].imshow(F[:, 5, :, 0], aspect='auto', cmap='jet', 
        interpolation='sinc', origin='lower', extent=(0, Lx, 0, t[-1]))
axes[1, 2].set(title="$B_z$", xlabel=r"$x/d_e$", ylabel=r"$t\omega_{pe}$")#, ylim=[1e-5, None])
fig.colorbar(im12)
fig.tight_layout()

# fig, axes = plt.subplots(1, 2, figsize=(15, 9))
# fig.suptitle(rf'$N_x = {Nx}$, $N_n = {Nn}$, $N_m = {Nm}$, $N_p = {Np}$, $\nu = {nu}$, $|u_z| = {uz}$', fontsize=14)

# Energy plots
# axes[0].plot(t, output["EM_energy"], label="EM energy")
# axes[0].plot(t, output["kinetic_energy"], label="Kinetic energy")
# axes[0].plot(t, output["kinetic_energy_species1"], label="Kinetic energy species 1")
# axes[0].plot(t, output["kinetic_energy_species2"], label="Kinetic energy species 2")
# axes[0].plot(t, output["total_energy"], label="Total energy")
# axes[0].set(title="Energy", xlabel=r"$t\omega_{pe}$", ylabel="Energy", yscale="log")#, ylim=[1e-5, None])
# axes[0].legend()

# axes[1].plot(t[1:], jnp.abs(output["total_energy"][1:]-output["total_energy"][0])/(output["total_energy"][0]+1e-9), label="Relative energy error")
# axes[1].set(xlabel=r"$t\omega_{pe}$", ylabel="Relative Energy Error", yscale="log")#, ylim=[1e-5, None])

# Compute number
ne = alpha_s[0] * alpha_s[1] * alpha_s[2] * jnp.sum(C[:, 0] / (1 + 2 * jnp.arange(0, N_DG)), axis=(-2, -1)) / Nx
ni = alpha_s[3] * alpha_s[4] * alpha_s[5] * jnp.sum(C[:, Nn*Nm*Np] / (1 + 2 * jnp.arange(0, N_DG)), axis=(-2, -1)) / Nx

# Compute energies
U_EM = 0.5 * Omega_ce**2 * jnp.sum(jnp.abs(F) ** 2 / (1 + 2 * jnp.arange(0, N_DG)), axis=(-2, -1)) / Nx
ve2 = alpha_s[0] * alpha_s[1] * alpha_s[2] * (alpha_s[0]**2 * C[:,2] / jnp.sqrt(2) + jnp.sqrt(2) * u_s[0] * alpha_s[0] * C[:,1] + (alpha_s[0]**2 / 2 + u_s[0]**2) * C[:,0]
                                                + alpha_s[2]**2 * C[:,2*Nn*Nm] / jnp.sqrt(2) + jnp.sqrt(2) * u_s[2] * alpha_s[2] * C[:,Nn*Nm] + (alpha_s[2]**2 / 2 + u_s[2]**2) * C[:,0])
vi2 = alpha_s[3] * alpha_s[4] * alpha_s[5] * (alpha_s[3]**2 * C[:,Nn*Nm*Np+2] / jnp.sqrt(2) + jnp.sqrt(2) * u_s[3] * alpha_s[3] * C[:,Nn*Nm*Np+1] + (alpha_s[3]**2 / 2 + u_s[3]**2) * C[:,Nn*Nm*Np]
                                              + alpha_s[5]**2 * C[:,Nn*Nm*Np+2*Nn*Nm] / jnp.sqrt(2) + jnp.sqrt(2) * u_s[5] * alpha_s[5] * C[:,Nn*Nm*Np+Nn*Nm] + (alpha_s[5]**2 / 2 + u_s[5]**2) * C[:,Nn*Nm*Np])
U_Ke = 0.5 * 1 * jnp.sum(ve2 / (1 + 2 * jnp.arange(0, N_DG)), axis=(-2, -1)) / Nx
U_Ki = 0.5 * 10000.0 * jnp.sum(vi2 / (1 + 2 * jnp.arange(0, N_DG)), axis=(-2, -1)) / Nx

ve2x = alpha_s[0] * alpha_s[1] * alpha_s[2] * (alpha_s[0]**2 * C[:,2] / jnp.sqrt(2) + jnp.sqrt(2) * u_s[0] * alpha_s[0] * C[:,1] + (alpha_s[0]**2 / 2 + u_s[0]**2) * C[:,0])
ve2z = alpha_s[0] * alpha_s[1] * alpha_s[2] * (alpha_s[2]**2 * C[:,2*Nn*Nm] / jnp.sqrt(2) + jnp.sqrt(2) * u_s[2] * alpha_s[2] * C[:,Nn*Nm] + (alpha_s[2]**2 / 2 + u_s[2]**2) * C[:,0])

U_Kex = 0.5 * 1 * jnp.sum(ve2x / (1 + 2 * jnp.arange(0, N_DG)), axis=(-2, -1)) / Nx
U_Kez = 0.5 * 1 * jnp.sum(ve2z / (1 + 2 * jnp.arange(0, N_DG)), axis=(-2, -1)) / Nx

U_tot = jnp.sum(U_EM, axis=(-1)) + U_Ke + U_Ki

Jx = - alpha_s[0] * alpha_s[1] * alpha_s[2] * (u_s[0] * C[:, 0, :, 0] + alpha_s[0] * C[:, 1, :, 0] / jnp.sqrt(2))
Jz = - alpha_s[0] * alpha_s[1] * alpha_s[2] * (u_s[2] * C[:, 0, :, 0] + alpha_s[2] * C[:, Nn*Nm, :, 0] / jnp.sqrt(2))

Jx_ave = jnp.sum(Jx, axis=(-1)) / Nx
Jz_ave = jnp.sum(Jz, axis=(-1)) / Nx

# Energy plots
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle(rf'$N_x = {Nx}$, $N_n = {Nn}$, $N_m = {Nm}$, $N_p = {Np}$, $N_DG = {N_DG}$, $\nu = {nu}$, $|u_z| = {uz}$', fontsize=14)

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

# Number plots
fig2, ax = plt.subplots()
ax.plot(t, ne, label="Electron number")
ax.plot(t, ni, label="Ion number")
ax.plot(t, jnp.abs(ne / ne[0] - 1), label="Electron number error")
ax.plot(t, jnp.abs(ni / ni[0] - 1), label="Ion number error")
ax.set(xlabel=r"$t\omega_{pe}$", ylabel=r"$nd_e^3$", yscale="log")#, ylim=[1e-5, None])
ax.legend()

fig, ax = plt.subplots(1, 3, figsize=(14, 7))
ax[0].plot(t, Jx_ave, label="Jx")
ax[0].plot(t, Jz_ave, label="Jz")
ax[0].set(xlabel=r"$t\omega_{pe}$", ylabel=r"current $J$")
ax[0].legend()

Jxim = ax[1].imshow(Jx, aspect='auto', cmap='jet', 
        interpolation='sinc', origin='lower', extent=(0, Lx, 0, t[-1]))
ax[1].set(title="$J_x$", xlabel=r"$x/d_e$", ylabel=r"$t\omega_{pe}$")#, ylim=[1e-5, None])
fig.colorbar(Jxim)

Jzim = ax[2].imshow(Jz, aspect='auto', cmap='jet', 
        interpolation='sinc', origin='lower', extent=(0, Lx, 0, t[-1]))
ax[2].set(title="$J_z$", xlabel=r"$x/d_e$", ylabel=r"$t\omega_{pe}$")#, ylim=[1e-5, None])
fig.colorbar(Jzim)

fig, ax = plt.subplots(1, 3, figsize=(14, 6))
ax[0].plot(t, U_Kez / U_Kex)
ax[0].set(xlabel=r"$t\omega_{pe}$", ylabel=r"$T_z / T_x$")

C_ave = jnp.sum(C / (1 + 2 * jnp.arange(0, N_DG)), axis=(-2, -1)) / Nx
Nt = len(t)
C_ave = C_ave.reshape(Nt, 2, Np, Nn)
from jax.scipy.special import factorial
n_vals = jnp.arange(Nn)
p_vals = jnp.arange(Np)
v_max_x_e = 6 * alpha_s[0]
v_max_x_i = 3 * alpha_s[3]
v_max_z_e = 6 * alpha_s[2]
v_max_z_i = 3 * alpha_s[5]
Nv = 150
v_x_e = jnp.linspace(-v_max_x_e, v_max_x_e, Nv)
v_z_e = jnp.linspace(-v_max_z_e, v_max_z_e, Nv)
v_x_i = jnp.linspace(-v_max_x_i, v_max_x_i, Nv)
v_z_i = jnp.linspace(-v_max_z_i, v_max_z_i, Nv)
xi_x_e = (v_x_e - u_s[0]) / alpha_s[0]
xi_z_e = (v_z_e - u_s[2]) / alpha_s[2]
xi_x_i = (v_x_i - u_s[3]) / alpha_s[3]
xi_z_i = (v_z_i - u_s[5]) / alpha_s[5]

her_vals_x_e = jnp.array(eval_hermite(n_vals[:, None], xi_x_e[None, :])) * jnp.exp(-xi_x_e**2) / jnp.sqrt(jnp.pi * (2 ** n_vals) * factorial(n_vals))[:, None]
her_vals_z_e = jnp.array(eval_hermite(p_vals[:, None], xi_z_e[None, :])) * jnp.exp(-xi_z_e**2) / jnp.sqrt(jnp.pi * (2 ** p_vals) * factorial(p_vals))[:, None]
her_vals_x_i = jnp.array(eval_hermite(n_vals[:, None], xi_x_i[None, :])) * jnp.exp(-xi_x_i**2) / jnp.sqrt(jnp.pi * (2 ** n_vals) * factorial(n_vals))[:, None]
her_vals_z_i = jnp.array(eval_hermite(p_vals[:, None], xi_z_i[None, :])) * jnp.exp(-xi_z_i**2) / jnp.sqrt(jnp.pi * (2 ** p_vals) * factorial(p_vals))[:, None]
f_e = jnp.tensordot(C_ave[:, 0, :, :], her_vals_x_e[None, :, None, :] * her_vals_z_e[:, None, :, None], axes=((-2, -1), (0, 1)))
f_i = jnp.tensordot(C_ave[:, 0, :, :], her_vals_x_i[None, :, None, :] * her_vals_z_i[:, None, :, None], axes=((-2, -1), (0, 1)))

im_e = ax[1].imshow(jnp.transpose(f_e[0]), aspect='auto', cmap='jet', interpolation='none', origin='lower', extent=(-v_max_x_e, v_max_x_e, -v_max_z_e, v_max_z_e))
cbar = plt.colorbar(im_e, ax=ax[1])
title_e = ax[1].set_title(r"Electron $t = 0$")
ax[1].set_xlabel("$v_x/c$")  # Set x-axis label
ax[1].set_ylabel("$v_z/c$")  # Set y-axis label

im_i = ax[2].imshow(jnp.transpose(f_i[0]), aspect='auto', cmap='jet', interpolation='none', origin='lower', extent=(-v_max_x_i, v_max_x_i, -v_max_z_i, v_max_z_i))
cbar = plt.colorbar(im_i, ax=ax[2])
title_i = ax[2].set_title(r"Ion $t = 0$")
ax[2].set_xlabel("$v_x/c$")  # Set x-axis label
ax[2].set_ylabel("$v_z/c$")  # Set y-axis label

def update(frame):
    im_e.set_array(jnp.transpose(f_e[frame]))
    im_i.set_array(jnp.transpose(f_i[frame]))
    title_e.set_text(f"Electrons t={t[frame]}")
    title_i.set_text(f"Ions t={t[frame]}")
    return [im_e, title_e, im_i, title_i]

# Create the animation
anim = FuncAnimation(
    fig, update, frames=f_e.shape[0], interval=50, blit=True  # Adjust interval as needed
)
fig.tight_layout()

anim.save("f_Weibel.gif", writer=PillowWriter(fps=10))
# Work = jnp.sum(Jx * F[:, 0, :, 0] + Jz * F[:, 2, :, 0], axis=(-1)) / Nx
# fig, ax = plt.subplots()
# ax.plot(t, U_Ke - U_Ke[0], label="perturbation energy")
# ax.plot(t[1:], (U_Ke[1:] - U_Ke[:-1]) / (jnp.max(t) / jnp.shape(t)[0]), label="perturbation energy derivative")
# ax.plot(t, Work, label="Work")
# ax.grid()
# ax.legend()

plt.show()

# print('Saving results...')
# jnp.savez('output_landau-damping.npz', **output)