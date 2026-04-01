import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
number_of_processors_to_use = 1 # Parallelization, this should divide total resolution
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
from jax import block_until_ready, config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import factorial
from spectrax import simulation, load_parameters, construct_idx_array, legT
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

toml_file = os.path.join(os.path.dirname(__file__), 'input_2D_orszag_tang.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

print('Setting up simulation parameters...')
start_time = time()
deltaB = 0.2 # In-plane magnetic field amplitude. 
U0 = deltaB * input_parameters["Omega_ce"] / jnp.sqrt(input_parameters["ms"][1] / input_parameters["ms"][0])

kx = 2 * jnp.pi / input_parameters["Lx"]
ky = 2 * jnp.pi / input_parameters["Ly"]

Lx = input_parameters["Lx"]
Ly = input_parameters["Ly"]
Nx = solver_parameters["Nx"]
Ny = solver_parameters["Ny"]
Nz = solver_parameters["Nz"]
Nn = solver_parameters["Nn"]
Nm = solver_parameters["Nm"]
Np = solver_parameters["Np"]
Ns = solver_parameters["Ns"]
N_DG = solver_parameters["N_DG"]
dims = solver_parameters["dims"]
alpha_s = input_parameters["alpha_s"]

alpha = jnp.array(alpha_s).reshape(Ns, 3)
alpha_x = alpha[:, 0, None, None, None, None, None, None, None, None, None]
alpha_y = alpha[:, 1, None, None, None, None, None, None, None, None, None]
alpha_z = alpha[:, 2, None, None, None, None, None, None, None, None, None]

basis_idx = construct_idx_array(dims, N_DG)
# Electron and ion fluid velocities.
Ue = lambda x, y, z: jnp.array([-U0 * jnp.sin(ky * y), U0 * jnp.sin(kx * x), -deltaB * input_parameters["Omega_ce"] * (2 * kx * jnp.cos(2 * kx * x) + ky * jnp.cos(ky * y))])
Ui = lambda x, y, z: jnp.array([-U0 * jnp.sin(ky * y), U0 * jnp.sin(kx * x), jnp.zeros_like(x)])

# Magnetic and electric fields.
B = lambda x, y, z: jnp.array([-deltaB * jnp.sin(ky * y), deltaB * jnp.sin(2 * kx * x), jnp.ones_like(x)])
E = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)])

p = jnp.arange(Np)[None, :, None, None, None, None, None, None, None, None]
m = jnp.arange(Nm)[None, None, :, None, None, None, None, None, None, None]
n = jnp.arange(Nn)[None, None, None, :, None, None, None, None, None, None]

Us = lambda x, y, z: jnp.stack([Ue(x, y, z), Ui(x, y, z)], axis=0)[:, :, None, None, None]
U_x = lambda x, y, z: Us(x, y, z)[:, 0]
U_y = lambda x, y, z: Us(x, y, z)[:, 1]
U_z = lambda x, y, z: Us(x, y, z)[:, 2]

C0 = lambda x, y, z: (jnp.sqrt(2 ** (n + m + p) / (factorial(n) * factorial(m) * factorial(p))) 
     * (1 / (alpha_x ** (n + 1) * alpha_y ** (m + 1) * alpha_z ** (p + 1)))
     * U_x(x, y, z) ** n * U_y(x, y, z) ** m * U_z(x, y, z) ** p)

input_parameters["Ck_0"] = legT(C0, basis_idx, N_DG, Lx, Nx, Ly, Ny) # shape (Ns, Np, Nm, Nn, Ny, Nx, Nz, N_DG)

F0 = lambda x, y, z: jnp.concatenate([E(x, y, z), B(x, y, z)])
input_parameters["Fk_0"] = legT(F0, basis_idx, N_DG, Lx, Nx, Ly, Ny) # shape (6, Ny, Nx, Nz, N_DG)

C = input_parameters["Ck_0"][:, :, :, :, :, :, 0, 0] # (Ns, Np, Nm, Nn, Ny, Nx)
F = input_parameters["Fk_0"][:, :, :, 0, 0] # (6, Ny, Nx)

ne = alpha_s[0] * alpha_s[1] * alpha_s[2] * C[0, 0, 0, 0, :, :] # (Ny, Nx)
ni = alpha_s[3] * alpha_s[4] * alpha_s[5] * C[1, 0, 0, 0, :, :]

Jz = (1 / jnp.sqrt(2)) * (alpha_s[3] * alpha_s[4] * (alpha_s[5]**2) * C[1, 1, 0, 0, :, :] - 
                          alpha_s[0] * alpha_s[1] * (alpha_s[2]**2) * C[0, 1, 0, 0, :, :]) # (Ny, Nx)

fig, ax = plt.subplots(1, 3, figsize=(16, 6))
im0 = ax[0].imshow(ne, aspect='auto', cmap='jet', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))#, vmin=-10, vmax=10)
fig.colorbar(im0, label=r'$n_e$').ax.yaxis.label.set_size(16)

im1 = ax[1].imshow(ni, aspect='auto', cmap='jet', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))#, vmin=-10, vmax=10)
fig.colorbar(im1, label=r'$n_i$').ax.yaxis.label.set_size(16)

im2 = ax[2].imshow(Jz, aspect='auto', cmap='jet', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))#, vmin=-10, vmax=10)
fig.colorbar(im2, label=r'$J_z$').ax.yaxis.label.set_size(16)

fig, ax = plt.subplots(2, 3, figsize=(16, 6))
im0 = ax[0, 0].imshow(F[0], aspect='auto', cmap='jet', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))
fig.colorbar(im0, label=r'$Ex$').ax.yaxis.label.set_size(14)

im1 = ax[0, 1].imshow(F[1], aspect='auto', cmap='jet', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))
fig.colorbar(im1, label=r'$Ey$').ax.yaxis.label.set_size(14)

im2 = ax[0, 2].imshow(F[2], aspect='auto', cmap='jet', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))
fig.colorbar(im2, label=r'$Ez$').ax.yaxis.label.set_size(14)

im3 = ax[1, 0].imshow(F[3], aspect='auto', cmap='jet', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))
fig.colorbar(im3, label=r'$Bx$').ax.yaxis.label.set_size(14)

im4 = ax[1, 1].imshow(F[4], aspect='auto', cmap='jet', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))
fig.colorbar(im4, label=r'$By$').ax.yaxis.label.set_size(14)

im5 = ax[1, 2].imshow(F[5], aspect='auto', cmap='jet', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))
fig.colorbar(im5, label=r'$Bz$').ax.yaxis.label.set_size(14)

print('Starting simulation...')
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

# Results.
alpha_s = input_parameters["alpha_s"]
Lx = input_parameters["Lx"]
Ly = input_parameters["Ly"]
Nx = solver_parameters["Nx"]
Ny = solver_parameters["Ny"]
Nz = solver_parameters["Nz"]
Nn = solver_parameters["Nn"]
Nm = solver_parameters["Nm"]
Np = solver_parameters["Np"]
t = output["time"]
nu = output["nu"]
uz = output["u_s"][2]
Ck = output["Ck"][:, :, :, :, 0, :] # (Nt, Ns*Np*Nm*Nn, Ny, Nx, Nl)
C = output["Ck"][:, :, :, :, 0, 0] # (Nt, Ns*Np*Nm*Nn, Ny, Nx)
F = output["Fk"][:, :, :, :, 0, :] # (Nt, 6, Ny, Nx, Nl)

# Densities
ne = alpha_s[0] * alpha_s[1] * alpha_s[2] * C[:, 0, :, :]
ni = alpha_s[3] * alpha_s[4] * alpha_s[5] * C[:, Nn * Nm * Np, :, :]

# Currents and velocities
Jz = (1 / jnp.sqrt(2)) * (alpha_s[3] * alpha_s[4] * (alpha_s[5]**2) * C[:, Nn * Nm * Np + Nn * Nm, :, :] - alpha_s[0] * alpha_s[1] * (alpha_s[2]**2) * C[:, Nn * Nm, :, :])
Uex = (1 / jnp.sqrt(2)) * alpha_s[0]**2 * alpha_s[1] * alpha_s[2] * C[:, 1, :, :] / ne
Uey = (1 / jnp.sqrt(2)) * alpha_s[0] * alpha_s[1]**2 * alpha_s[2] * C[:, Nn, :, :] / ne
Uix = (1 / jnp.sqrt(2)) * alpha_s[3]**2 * alpha_s[4] * alpha_s[5] * C[:, Nn * Nm * Np + 1, :, :] / ni
Uiy = (1 / jnp.sqrt(2)) * alpha_s[3] * alpha_s[4]**2 * alpha_s[5] * C[:, Nn * Nm * Np + Nn, :, :] / ni

Ue_perp = jnp.stack([Uex, Uey], axis=1)
Ui_perp = jnp.stack([Uix, Uiy], axis=1)

# EM Energies. Size (Nt, 6)
U_EM = 0.5 * input_parameters["Omega_ce"]**2 * jnp.sum(jnp.abs(F) ** 2 / ((1 + 2 * basis_idx[:, 0]) * (1 + 2 * basis_idx[:, 1])), axis=(-3, -2, -1)) / (Nx * Ny)
U_E, U_B = jnp.sum(U_EM[:, :3], axis=-1), jnp.sum(U_EM[:, 3:], axis=-1) # Size Nt
U_EM_tot = U_E + U_B

# Single fluid MHD variables and alignment
rho = (ne + ni * input_parameters["ms"][1] / input_parameters["ms"][0])[:, None, :, :]
U_perp = (Ue_perp * ne[:, None, :, :] + Ui_perp * ni[:, None, :, :] * input_parameters["ms"][1] / input_parameters["ms"][0]) / rho
VA_perp = input_parameters["Omega_ce"] * F[:, 3:5, :, :, 0] / jnp.sqrt(rho)

cross_helicity = jnp.sum(rho * VA_perp * U_perp, axis=(-3, -2, -1)) / (Nx * Ny)
total_fluctuation_energy = jnp.sum(rho * (U_perp**2 + VA_perp**2), axis=(-3, -2, -1)) / (Nx * Ny)
alignment_corr = cross_helicity / total_fluctuation_energy

fig, ax = plt.subplots()
ax.plot(t, alignment_corr)
ax.set_xlabel(r'$\omega_p t$')
ax.set_ylabel(r'$\kappa$')
ax.set_title(r'$\kappa = \frac{\langle \rho u_{\perp} \cdot v_{A\perp}\rangle}{\langle \rho(u_{\perp}^2 + v_{A\perp}^2)\rangle}$')
ax.title.set_fontsize(16)
ax.grid()
fig.savefig('OT_tests_alignment.png')

plt.show()

# Plot densities
plt.figure(figsize=(8, 6))
plt.imshow(ne[-1], aspect='auto', cmap='plasma', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))#, vmin=-10, vmax=10)
plt.colorbar(label=r'$n_e$').ax.yaxis.label.set_size(16)

plt.figure(figsize=(8, 6))
plt.imshow(ni[-1], aspect='auto', cmap='plasma', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))#, vmin=-10, vmax=10)
plt.colorbar(label=r'$n_i$').ax.yaxis.label.set_size(16)

# Plot energies
fig, ax = plt.subplots(figsize=(12, 8))
ax.semilogy(t, U_EM[:, 0], label=r'$E_x$ Energy')
ax.semilogy(t, U_EM[:, 1], label=r'$E_y$ Energy')
ax.semilogy(t, U_EM[:, 2], label=r'$E_z$ Energy')
ax.semilogy(t, U_EM[:, 3], label=r'$B_x$ Energy')
ax.semilogy(t, U_EM[:, 4], label=r'$B_y$ Energy')
ax.semilogy(t, U_EM[:, 5], label=r'$B_z$ Energy')

ax.semilogy(t, U_E, label=r'Total electric energy')
ax.semilogy(t, U_B, label=r'Total magnetic energy')
ax.semilogy(t, U_EM_tot, label=r'Total electromagnetic energy')

ax.set_xlabel(r'$\omega_p t$')
ax.set_ylabel(r'Energy')
ax.grid()
ax.legend()
ax.set_title('Orszag-Tang vortex: electromagentic field energies')
fig.tight_layout()
fig.savefig('OT_tests_energies.png')

plt.show()

def plot_field_evolution(field, interval=50, save_file='', save_fps=10):
    fig, ax = plt.subplots()
    im = ax.imshow(field[0], aspect='auto', cmap='plasma', interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))
    cbar = plt.colorbar(im, ax=ax)
    title = ax.set_title(r"$t = 0$")
    ax.set_xlabel("x/d_e")  # Set x-axis label
    ax.set_ylabel("y/d_e")  # Set y-axis label

    def update(frame):
        im.set_array(field[frame])
        im.set_clim(vmin=field[frame].min(), vmax=field[frame].max())
        title.set_text(f"$t = {t[frame]}$")
        cbar.update_normal(im)
        return [im, cbar.ax, title]

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=field.shape[0], interval=interval, blit=False  # Adjust interval as needed
    )

    if save_file != '':
        anim.save(save_file, writer=PillowWriter(fps=save_fps))

    # plt.show()

plot_field_evolution(Uex, 2000, 'OT_tests_Uex.gif')

plot_field_evolution(Uey, 2000, 'OT_tests_Uey.gif')

plot_field_evolution(ne, 2000, 'OT_tests_ne.gif')

plot_field_evolution(Jz, 2000, 'OT_tests_Jz.gif')

plt.show()