import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
number_of_processors_to_use = 1 # Parallelization, this should divide total resolution
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f'0.95' # Fraction of memory to allocate on each device
from time import time
from jax import block_until_ready
import jax.numpy as jnp
from jax.scipy.special import factorial
from spectrax import simulation, load_parameters, plot, compute_C_nmp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from jax.numpy.fft import fftn, ifftn, fftshift, ifftshift

# Read from input.toml
toml_file = os.path.join(os.path.dirname(__file__), 'input_2D_Tearing_camporeale06_Fourier.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

print('Setting up simulation parameters...')
start_time = time()
a = 1 # Current sheet thickness
ka = 0.5 # Normalized perturbation wavenumber
Lx = 32 * a # input_parameters["Lx"]
Ly = 2 * jnp.pi * a / ka
Lz = input_parameters["Lz"]
ky = 2 * jnp.pi / Ly
kx = 2 * jnp.pi / Lx
Bz0 = 5.0 # Out of plane magnetic field
Br = 1.0 # Reconnecting field
dB = 0.01 * Br

Nx = solver_parameters["Nx"]
Ny = solver_parameters["Ny"]
Nz = solver_parameters["Nz"]
Ns = solver_parameters["Ns"]
Nn = solver_parameters["Nn"]
Nm = solver_parameters["Nm"]
Np = solver_parameters["Np"]
alpha_s = jnp.array(input_parameters["alpha_s"]) * a
input_parameters["alpha_s"] = alpha_s
u_s = jnp.array(input_parameters["u_s"]) * Br / a
input_parameters["u_s"] = u_s
Omega_cs = jnp.array(input_parameters["Omega_cs"])
qs = jnp.array(input_parameters["qs"])
mi_me = input_parameters["mi_me"]
ms = jnp.array([1.0, 1.0, mi_me, mi_me])

alpha = jnp.array(alpha_s).reshape(Ns, 3)
u = jnp.array(u_s).reshape(Ns, 3)
alpha_x = alpha[:, 0, None, None, None, None, None, None]
alpha_y = alpha[:, 1, None, None, None, None, None, None]
alpha_z = alpha[:, 2, None, None, None, None, None, None]

p = jnp.arange(Np)[None, :, None, None, None, None, None]
m = jnp.arange(Nm)[None, None, :, None, None, None, None]
n = jnp.arange(Nn)[None, None, None, :, None, None, None]

# Electron and ion fluid velocities.
Ue1 = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)])
Ue2 = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)])
Ui1 = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)])
Ui2 = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)])

# Electron and ion densities.
ne1 = lambda x, y, z: 1 / (jnp.cosh((x - Lx / 4) / a)**2)
ni1 = lambda x, y, z: 1 / (jnp.cosh((x - Lx / 4) / a)**2)
ne2 = lambda x, y, z: 1 / (jnp.cosh((x - 3 * Lx / 4) / a)**2)
ni2 = lambda x, y, z: 1 / (jnp.cosh((x - 3 * Lx / 4) / a)**2)

ns = lambda x, y, z: jnp.stack([ne1(x, x, z), ne2(x, x, z), ni1(x, x, z), ni2(x, x, z)], axis=0)[:, None, None, None]
Us = lambda x, y, z: jnp.stack([Ue1(x, y, z), Ue2(x, y, z), Ui1(x, y, z), Ui2(x, y, z)], axis=0)[:, :, None, None, None]
U_x = lambda x, y, z: Us(x, y, z)[:, 0]
U_y = lambda x, y, z: Us(x, y, z)[:, 1]
U_z = lambda x, y, z: Us(x, y, z)[:, 2]

C0 = lambda x, y, z: ns(x, y, z) * (jnp.sqrt(2 ** (n + m + p) / (factorial(n) * factorial(m) * factorial(p))) 
     * (1 / (alpha_x ** (n + 1) * alpha_y ** (m + 1) * alpha_z ** (p + 1)))
     * U_x(x, y, z) ** n * U_y(x, y, z) ** m * U_z(x, y, z) ** p)

x = jnp.linspace(0, Lx, Nx, endpoint=False)
y = jnp.linspace(0, Ly, Ny, endpoint=False)
z = jnp.linspace(0, Lz, Nz, endpoint=False)
X, Y, Z = jnp.meshgrid(x, y, z, indexing='xy')

input_parameters["Ck_0"] = fftshift(fftn(C0(X, Y, Z), axes=(-3, -2, -1), norm="forward"), axes=(-3, -2, -1))

# Magnetic and electric fields.
B = lambda x, y, z: jnp.array([- dB * ky * jnp.sin(ky * y) * jnp.sin(kx * x), Br * (jnp.tanh((x - Lx / 4) / a) - jnp.tanh((x - 3 * Lx / 4) / a) - jnp.ones_like(x)) - dB * kx * jnp.cos(ky * y) * jnp.cos(kx * x), Bz0 * jnp.ones_like(x)])
E = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)])

F0 = lambda x, y, z: jnp.concatenate([E(x, y, z), B(x, y, z)])
input_parameters["Fk_0"] =  fftshift(fftn(F0(X, Y, Z), axes=(-3, -2, -1), norm="forward"), axes=(-3, -2, -1))

########################################################################################

# Plot initial fields
C = ifftn(ifftshift(input_parameters["Ck_0"], axes=(-3, -2, -1)), axes=(-3, -2, -1), norm="forward").real
F = ifftn(ifftshift(input_parameters["Fk_0"], axes=(-3, -2, -1)), axes=(-3, -2, -1), norm="forward").real

ne = alpha_s[0] * alpha_s[1] * alpha_s[2] * C[0, 0, 0, 0, :, :, 0] + alpha_s[3] * alpha_s[4] * alpha_s[5] * C[1, 0, 0, 0, :, :, 0] # (Ny, Nx)
ni = alpha_s[6] * alpha_s[7] * alpha_s[8] * C[2, 0, 0, 0, :, :, 0] + alpha_s[9] * alpha_s[10] * alpha_s[11] * C[3, 0, 0, 0, :, :, 0]

Jz = (alpha_s[6] * alpha_s[7] * alpha_s[8] * (alpha_s[8] * C[2, 1, 0, 0, :, :, 0] / jnp.sqrt(2) + u_s[8] * C[2, 0, 0, 0, :, :, 0]) 
        + alpha_s[9] * alpha_s[10] * alpha_s[11] * (alpha_s[11] * C[3, 1, 0, 0, :, :, 0] / jnp.sqrt(2) + u_s[11] * C[3, 0, 0, 0, :, :, 0])
        - alpha_s[0] * alpha_s[1] * alpha_s[2] * (alpha_s[2] * C[0, 1, 0, 0, :, :, 0] / jnp.sqrt(2) + u_s[2] * C[0, 0, 0, 0, :, :, 0]) 
        - alpha_s[3] * alpha_s[4] * alpha_s[5] * (alpha_s[5] * C[1, 1, 0, 0, :, :, 0] / jnp.sqrt(2) + u_s[5] * C[1, 0, 0, 0, :, :, 0])) # (Ny, Nx)

fig, ax = plt.subplots(1, 3, figsize=(16, 6))
# n_e
im0 = ax[0].imshow(ne, aspect='auto', cmap='jet', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))#, vmin=-10, vmax=10)
fig.colorbar(im0, label=r'$n_e$').ax.yaxis.label.set_size(16)

# n_i
im1 = ax[1].imshow(ni, aspect='auto', cmap='jet', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))#, vmin=-10, vmax=10)
fig.colorbar(im1, label=r'$n_i$').ax.yaxis.label.set_size(16)

# By
im2 = ax[2].imshow(F[4, :, :, 0], aspect='auto', cmap='jet', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))#, vmin=-10, vmax=10)
fig.colorbar(im2, label=r'$J_z$').ax.yaxis.label.set_size(16)

fig.savefig('compo_F_initial.png')

################################################################################

print('Starting simulation...')
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

################################################################################

# Results.
t = output["time"]
Nt = len(t)
Ck = output["Ck"].reshape(Nt, Ns, Np, Nm, Nn, Ny, Nx, Nz)[:, :, :, :, :, :, :, 0]
Fk = output["Fk"].reshape(Nt, 6, Ny, Nx, Nz)[:, :, :, :, 0]
C = ifftn(ifftshift(Ck, axes=(-2, -1)), axes=(-2, -1), norm="forward").real
F = ifftn(ifftshift(Fk, axes=(-2, -1)), axes=(-2, -1), norm="forward").real

# Extract relevant low-order coefficients
Ck000 = Ck[:, :, 0, 0, 0, :, :]
Ck200 = Ck[:, :, 2, 0, 0, :, :]
Ck020 = Ck[:, :, 0, 2, 0, :, :]
Ck002 = Ck[:, :, 0, 0, 2, :, :]
Ck100 = Ck[:, :, 1, 0, 0, :, :]
Ck010 = Ck[:, :, 0, 1, 0, :, :]
Ck001 = Ck[:, :, 0, 0, 1, :, :]

C000 = C[:, :, 0, 0, 0, :, :]
C200 = C[:, :, 2, 0, 0, :, :]
C020 = C[:, :, 0, 2, 0, :, :]
C002 = C[:, :, 0, 0, 2, :, :]
C100 = C[:, :, 1, 0, 0, :, :]
C010 = C[:, :, 0, 1, 0, :, :]
C001 = C[:, :, 0, 0, 1, :, :]

# Species energy. Shape (Nt, Ns, Ny, Nx)
U_s = 0.5 * ms[:, None, None] * alpha[:, 0, None, None] * alpha[:, 1, None, None] * alpha[:, 2, None, None] * (0.5 * Ck000 * (alpha[:, 0, None, None]**2 + alpha[:, 1, None, None]**2 + alpha[:, 2, None, None]**2)
                                                                                                                 + (Ck002 * alpha[:, 0, None, None]**2 + Ck020 * alpha[:, 1, None, None]**2 + Ck200 * alpha[:, 2, None, None]**2) / jnp.sqrt(2)
                                                                                                                 + jnp.sqrt(2) * (Ck001 * alpha[:, 0, None, None] * u[:, 0, None, None] + Ck010 * alpha[:, 1, None, None] * u[:, 1, None, None] + Ck100 * alpha[:, 2, None, None] * u[:, 2, None, None])
                                                                                                                 + u[:, 0, None, None]**2 + u[:, 1, None, None]**2 + u[:, 2, None, None]**2) # Kinetic energy for each species.
U_k = jnp.sum(U_s[:, :, Ny//2, Nx//2], axis=(-1)) # Total kinetic energy. Shape (Nt,)
U_em = 0.5 * Omega_cs[0]**2 * jnp.sum(Fk**2, axis=(-2, -1)) # Electromagnetic field energies. Shape (Nt, 6)

U = U_k + jnp.sum(U_em, axis=-1) # Total energies. Shape (Nt,)

ne = alpha_s[0] * alpha_s[1] * alpha_s[2] * C000[:, 0] + alpha_s[3] * alpha_s[4] * alpha_s[5] * C000[:, 1]
ni = alpha_s[6] * alpha_s[7] * alpha_s[8] * C000[:, 2] + alpha_s[9] * alpha_s[10] * alpha_s[11] * C000[:, 3]

Bx = F[:, 3]
By = F[:, 4]
Bz = F[:, 5]
dBz = Bz - Bz0

# Plots 2D field evolution in time. Limited to left half of the domain since the second current sheet is a mirror of the first.
def plot_field_evolution(field, interval=50, save_file='', save_fps=10, include_contours=False):
    fig, ax = plt.subplots()
    im = ax.imshow(field[0, :, :Nx//2 + 1], aspect='auto', cmap='bwr', interpolation='none', origin='lower', extent=(0, Lx * (Nx//2 + 1) / Nx, 0, Ly))
    cbar = plt.colorbar(im, ax=ax)
    title = ax.set_title(r"$t = 0$")
    ax.set_xlabel("x/d_e")  # Set x-axis label
    ax.set_ylabel("y/d_e")  # Set y-axis label

    if include_contours:
        contours = ax.contour(field[0, :, :Nx//2 + 1], levels=10, colors='white', linewidths=1, extent=(0, Lx * (Nx//2 + 1) / Nx, 0, Ly))

    def update(frame):
        im.set_array(field[frame, :, :Nx//2 + 1])
        im.set_clim(vmin=field[frame].min(), vmax=field[frame].max())
        if include_contours:
            nonlocal contours
            contours.remove()
            contours = ax.contour(field[frame, :, :Nx//2 + 1], levels=10, colors='white', linewidths=1, extent=(0, Lx * (Nx//2 + 1) / Nx, 0, Ly))
        title.set_text(f"$t = {t[frame]}$")
        cbar.update_normal(im)
        return [im, cbar.ax, title]

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=field.shape[0], interval=interval, blit=False  # Adjust interval as needed
    )

    if save_file != '':
        anim.save(save_file, writer=PillowWriter(fps=save_fps))

Jz_e = -(alpha_s[0] * alpha_s[1] * alpha_s[2] * (alpha_s[2] * C100[:, 0] / jnp.sqrt(2) + u_s[2] * C000[:, 0]) 
            + alpha_s[3] * alpha_s[4] * alpha_s[5] * (alpha_s[5] * C100[:, 1] / jnp.sqrt(2) + u_s[5] * C000[:, 1]))
Jz_i = (alpha_s[6] * alpha_s[7] * alpha_s[8] * (alpha_s[8] * C100[:, 2] / jnp.sqrt(2) + u_s[8] * C000[:, 2])
            + alpha_s[9] * alpha_s[10] * alpha_s[11] * (alpha_s[11] * C100[:, 3] / jnp.sqrt(2) + u_s[11] * C000[:, 3]))

Jz = Jz_e + Jz_i

# Zoomed-in plot of the magnetic field lines
fig, ax = plt.subplots()
x, y = jnp.linspace(0, Lx, Nx), jnp.linspace(0, Ly, Ny)
l_ind, u_ind = jnp.int32(3 * Nx / 16), jnp.int32(5 * Nx / 16)
im = ax.streamplot(np.array(x[l_ind:u_ind]), np.array(y), np.array(Bx[0, :, l_ind:u_ind]), np.array(By[0, :, l_ind:u_ind]), density=1, broken_streamlines=False, color='b')
title = ax.set_title(r"$t=0$")
ax.set_xlabel("x/d_e")  # Set x-axis label
ax.set_ylabel("y/d_e")  # Set y-axis label
ax.set_xlim(x[l_ind], x[u_ind])
ax.set_ylim(0, Ly)

def update(frame):
    ax.collections[0].remove()
    for patch in ax.patches: patch.remove()

    im = ax.streamplot(np.array(x[l_ind:u_ind]), np.array(y), np.array(Bx[frame, :, l_ind:u_ind]), np.array(By[frame, :, l_ind:u_ind]), density=1, broken_streamlines=False, color='b')
    # ax.set_xlabel("x/d_e")  # Set x-axis label
    # ax.set_ylabel("y/d_e")  # Set y-axis label
    # ax.set_xlim(0, Lx)
    # ax.set_ylim(0, Ly)

    title.set_text(f"t = {t[frame]:.1f}")
    return [im.lines, title]

anim = FuncAnimation(
    fig, update, frames=By.shape[0], interval=200, blit=True  # Adjust interval as needed
)

anim.save("compo_F_field_lines.gif", writer=PillowWriter(fps=4))

# Zoomed in plot of out-of-plane current density Jz
fig, ax = plt.subplots()
im = ax.imshow(Jz[0, :, l_ind:u_ind], aspect='auto', cmap='bwr', interpolation='none', origin='lower', extent=(x[l_ind], x[u_ind], 0, Ly))
cbar = plt.colorbar(im, ax=ax)
title = ax.set_title(r"$t = 0$")
ax.set_xlabel("x/d_e")  # Set x-axis label
ax.set_ylabel("y/d_e")  # Set y-axis label
contours = ax.contour(Jz[0, :, l_ind:u_ind], levels=16, colors='white', linewidths=1, extent=(x[l_ind], x[u_ind], 0, Ly))

def update(frame):
    im.set_array(Jz[frame, :, l_ind:u_ind])
    im.set_clim(vmin=Jz[frame].min(), vmax=Jz[frame].max())
    global contours
    contours.remove()
    contours = ax.contour(Jz[frame, :, l_ind:u_ind], levels=16, colors='black', linewidths=1, extent=(x[l_ind], x[u_ind], 0, Ly))
    title.set_text(f"$t = {t[frame]}$")
    cbar.update_normal(im)
    return [im, cbar.ax, title]

# Create the animation
anim = FuncAnimation(
    fig, update, frames=Jz.shape[0], interval=200, blit=False  # Adjust interval as needed
)

anim.save("compo_F_Jz_zoom.gif", writer=PillowWriter(fps=2))

plot_field_evolution(ne, 500, 'compo_F_ne.gif', save_fps=2)
plot_field_evolution(By, 500, 'compo_F_By.gif', save_fps=2)
plot_field_evolution(Bx, 500, 'compo_F_Bx.gif', save_fps=2)
plot_field_evolution(dBz, 500, 'compo_F_dBz.gif', save_fps=2)
plot_field_evolution(Jz, 500, 'compo_F_Jz.gif', save_fps=2, include_contours=True)
# plot_field_evolution(Uey, 2000, 'Reconnection_Uey.gif')

Bx_modes = jnp.abs(Fk[:, 3, :, Nx//2:])
plot_field_evolution(Bx_modes, 500, 'compo_F_Bx_modes.gif', save_fps=1)

gam, b = jnp.polyfit(t[Nt//2:], jnp.log(jnp.abs(Fk[Nt//2:, 3, Ny//2 + 1, Nx//2 + 1])), deg=1)
fit = jnp.exp(b) * jnp.exp(gam * t)

# Plot evolution of dominant Bx mode and its exponential fit
fig, ax = plt.subplots()
ax.semilogy(t, jnp.abs(Fk[:, 3, Ny//2 + 1, Nx//2 + 1]), label='Bx')
ax.semilogy(t[Nt//2:], fit[Nt//2:], 'k--')
i_text = int(0.75 * (Nt - 1))
ax.text(
    float(t[i_text]+50),
    float(fit[i_text]),
    rf"$\gamma = {float(gam):.4g}$",
    fontsize=12,
    ha="left",
    va="bottom"
)

ax.set_xlabel(r'$\omega_{\text{pe}}t$', fontsize=16)
ax.set_ylabel(r'$|\hat{B}_x(0, 1)|$', fontsize=16)
plt.savefig('compo_F_Bx_mode.png')

jnp.savez('output_2D_Tearing_Camporeale06_Fourier.npz', **output)