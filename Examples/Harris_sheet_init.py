import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
number_of_processors_to_use = 1 # Parallelization, this should divide total resolution
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
from jax import block_until_ready
import jax.numpy as jnp
from jax.scipy.special import factorial
from spectrax import simulation, load_parameters, plot, initialize_xv
import matplotlib.pyplot as plt
from jax.numpy.fft import ifftn, ifftshift, fftn, fftshift
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

# Read from input.toml
toml_file = os.path.join(os.path.dirname(__file__), 'input_Harris_sheet_init.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

print('Setting up simulation parameters...')
start_time = time()
Lx = input_parameters["Lx"]
#Ly = input_parameters["Ly"]
Ly = 64 * jnp.pi / jnp.sqrt(3)
Bz = 1.0 # Out of plane magnetic field
Br = 1.0 * Bz # Reconnecting field
ky = 2 * jnp.pi / Ly
kx = 2 * jnp.pi / Lx

Nx = solver_parameters["Nx"]
Ny = solver_parameters["Ny"]
Nz = solver_parameters["Nz"]
Ns = solver_parameters["Ns"]
Nn = solver_parameters["Nn"]
Nm = solver_parameters["Nm"]
Np = solver_parameters["Np"]
alpha_s = input_parameters["alpha_s"]

alpha = jnp.array(alpha_s).reshape(Ns, 3)
alpha_x = alpha[:, 0, None, None, None, None, None, None]
alpha_y = alpha[:, 1, None, None, None, None, None, None]
alpha_z = alpha[:, 2, None, None, None, None, None, None]

a = 16 / jnp.sqrt(3) # Current sheet thickness
dB = 0.05 * Br

# Magnetic and electric fields.
B = lambda x, y, z: jnp.array([- dB * ky * jnp.sin(ky * y) * jnp.sin(kx * x), Br * (jnp.tanh((x - Lx / 4) / a) - jnp.tanh((x - 3 * Lx / 4) / a) - jnp.ones_like(x)) - dB * kx * jnp.cos(ky * y) * jnp.cos(kx * x), Bz * jnp.ones_like(x)])
E = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)])

# Electron and ion fluid velocities.
Ue = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), -input_parameters["Omega_cs"][0] / (a) * jnp.ones_like(x)])
Ui = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), input_parameters["Omega_cs"][0] / (a) * jnp.ones_like(x)])
# Otherwise use U * (2 * jnp.heaviside(x, Lx / 2) - 1)

ne = lambda x, y, z: 1 / (jnp.cosh((x - Lx / 4) / a)**2)
ni = lambda x, y, z: 1 / (jnp.cosh((x - 3 * Lx / 4) / a)**2)

p = jnp.arange(Np)[None, :, None, None, None, None, None]
m = jnp.arange(Nm)[None, None, :, None, None, None, None]
n = jnp.arange(Nn)[None, None, None, :, None, None, None]

x = jnp.linspace(0, input_parameters["Lx"], Nx)
y = jnp.linspace(0, input_parameters["Ly"], Ny)
z = jnp.linspace(0, input_parameters["Lz"], Nz)
X, Y, Z = jnp.meshgrid(x, y, z, indexing='xy')

Us_grid = jnp.stack([Ue(X, Y, Z), Ui(X, Y, Z)], axis=0)  # shape (Ns, 3, Ny, Nx, Nz)
U_x = Us_grid[:, 0, None, None, None, :, :, :] # shape (Ns, 1, 1, 1, Ny, Nx, Nz)
U_y = Us_grid[:, 1, None, None, None, :, :, :] # shape (Ns, 1, 1, 1, Ny, Nx, Nz)
U_z = Us_grid[:, 2, None, None, None, :, :, :] # shape (Ns, 1, 1, 1, Ny, Nx, Nz)

n_ei = jnp.stack([ne(X, Y, Z), ni(X, Y, Z)], axis=0)[:, None, None, None, :, :, :]

C = n_ei * (jnp.sqrt(2 ** (n + m + p) / (factorial(n) * factorial(m) * factorial(p))) 
     * (1 / (alpha_x ** (n + 1) * alpha_y ** (m + 1) * alpha_z ** (p + 1)))
     * U_x ** n * U_y ** m * U_z ** p)


input_parameters["Ck_0"] = fftshift(fftn(C, axes=(-3, -2, -1), norm="forward"), axes=(-3, -2, -1))

B_grid = B(X, Y, Z)  # shape (3, Ny, Nx, Nz)
E_grid = E(X, Y, Z)  # shape (3, Ny, Nx, Nz)
F_grid = jnp.concatenate((E_grid, B_grid), axis=0)
input_parameters["Fk_0"] = fftshift(fftn(F_grid, axes=(-3, -2, -1), norm="forward"), axes=(-3, -2, -1))



print('Starting simulation...')
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

time_vals = output["time"]

# Results.
Ck = output["Ck"]
Fk = output["Fk"]
Bk = Fk[:, 3:6, :, :, 0]

C = ifftn(ifftshift(Ck, axes=(-3, -2, -1)), axes=(-3, -2, -1), norm="forward").real
EB = ifftn(ifftshift(Fk, axes=(-3, -2, -1)), axes=(-3, -2, -1), norm = "forward").real

Bx = EB[:, 3, :, :, 0]
By = EB[:, 4, :, :, 0]
Bz = EB[:, 5, :, :, 0]

B_x_mode = Bk[:, 0, (Ny // 2) + 1, (Nx // 2) + 1]
B_y_mode = Bk[:, 1, (Ny // 2) + 1, (Nx // 2) + 1]
B_mode = jnp.sqrt(jnp.abs(B_x_mode)**2 + 0*jnp.abs(B_y_mode)**2)

B_hat_x = jnp.sum(Bx[:, :, :Nx // 2 + 1] * jnp.exp(1j * ky * y).reshape(1, Ny, 1), axis=(-2, -1))
B_hat_y = jnp.sum(By[:, :, :Nx // 2 + 1] * jnp.exp(1j * ky * y).reshape(1, Ny, 1), axis=(-2, -1))
B_hat_z = jnp.sum(Bz[:, :, :Nx // 2 + 1] * jnp.exp(1j * ky * y).reshape(1, Ny, 1), axis=(-2, -1))

B_hat = jnp.sqrt(jnp.abs(B_hat_x)**2 + jnp.abs(B_hat_y)**2 + 0 * jnp.abs(B_hat_z)) / (Nx * Ny)


ne = alpha_s[0] * alpha_s[1] * alpha_s[2] * C[:, 0, :, :, 0] # n = m = p = 0
ni = alpha_s[3] * alpha_s[4] * alpha_s[5] * C[:, Nn * Nm * Np, :, :, 0] # n = m = p = 0 for second species

Jz = - (1 / jnp.sqrt(2)) * (alpha_s[3] * alpha_s[4] * (alpha_s[5]**2) * C[:, Nn * Nm * Np + Nn * Nm, :, :, 0] + alpha_s[0] * alpha_s[1] * (alpha_s[2]**2) * C[:, Nn * Nm, :, :, 0]) # n = m = 0, p = 1 for both species
J_z_plot = Jz[0, Ny // 2, :]

# plt.figure(figsize=(8, 6))
# plt.plot(x, J_z_plot, '-')
# plt.plot(x, Br * input_parameters["Omega_cs"][0] / a * (1/jnp.cosh((x - Lx / 4) / a)**2 - 1/jnp.cosh((x - 3 * Lx / 4) / a)**2), '--')
# plt.xlabel(r'$\omega_{\text{pe}}t$', fontsize=16)
# plt.ylabel(r'$|\hat{B}(1, 1)|$', fontsize=16)
# plt.savefig('B_mode.pdf')
# plt.show()


U_e_y = alpha_s[1] / jnp.sqrt(2) * C[:, Nn, :, :, 0] / C[:, 0, :, :, 0]
U_i_y = alpha_s[4] / jnp.sqrt(2) * C[:, Nn * Nm * Np + Nn, :, :, 0] / C[:, Nn * Nm * Np, :, :, 0]

plt.figure(figsize=(8, 6))
plt.imshow(By[0], aspect='auto', cmap='jet', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))#, vmin=-10, vmax=10)
plt.colorbar(label=r'$B_y$').ax.yaxis.label.set_size(16)
plt.xlabel('x/d_e', fontsize=16)
plt.ylabel('y/d_e', fontsize=16)
# plt.show()

plt.figure(figsize=(8, 6))
plt.streamplot(np.array(x), np.array(y), np.array(Bx[0]), np.array(By[0]), density=1, broken_streamlines=False)
plt.xlabel('x/d_e', fontsize=16)
plt.ylabel('y/d_e', fontsize=16)
# plt.show()

fig, ax = plt.subplots()
im = ax.streamplot(np.array(x), np.array(y), np.array(Bx[0]), np.array(By[0]), density=1, broken_streamlines=False, color='b')
title = ax.set_title(r"$t=0$")
ax.set_xlabel("x/d_e")  # Set x-axis label
ax.set_ylabel("y/d_e")  # Set y-axis label
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)

def update(frame):
    ax.collections[0].remove()
    for patch in ax.patches: patch.remove()

    im = ax.streamplot(np.array(x), np.array(y), np.array(Bx[frame]), np.array(By[frame]), density=1, broken_streamlines=False, color='b')
    # ax.set_xlabel("x/d_e")  # Set x-axis label
    # ax.set_ylabel("y/d_e")  # Set y-axis label
    # ax.set_xlim(0, Lx)
    # ax.set_ylim(0, Ly)

    title.set_text(f"t = {time_vals[frame]:.1f}")
    return [im.lines, title]

anim = FuncAnimation(
    fig, update, frames=By.shape[0], interval=200, blit=True  # Adjust interval as needed
)

anim.save("field_lines.gif", writer=PillowWriter(fps=5))



# Animation for By
fig, ax = plt.subplots()
im = ax.imshow(By[0], extent=(0, Lx, 0, Ly), cmap='jet', vmin=-Br, vmax=Br, interpolation='nearest', origin='lower')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("$B_y$") 
title = ax.set_title(r"$t=0$")
ax.set_xlabel("x/d_e")  # Set x-axis label
ax.set_ylabel("y/d_e")  # Set y-axis label

def update(frame):
    im.set_array(By[frame])
    
    # im.set_clim(vmin=By[frame
    # ].min(), vmax=By[frame].max())
    # cbar.update_normal(im)
    
    title.set_text(f"t = {time_vals[frame]:.1f}")
    return [im, title]

anim = FuncAnimation(
    fig, update, frames=By.shape[0], interval=200, blit=True  # Adjust interval as needed
)

anim.save("By.gif", writer=PillowWriter(fps=5))
# plt.show()

fig, ax = plt.subplots()
im = ax.imshow(Bx[0], extent=(0, Lx, 0, Ly), cmap='jet', vmin=-0.005, vmax=0.005, interpolation='nearest', origin='lower')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("$B_x$") 
title = ax.set_title(r"$t=0$")
ax.set_xlabel("x/d_e")  # Set x-axis label
ax.set_ylabel("y/d_e")  # Set y-axis label

def update(frame):
    im.set_array(Bx[frame])
    
    # im.set_clim(vmin=Bx[frame
    # ].min(), vmax=Bx[frame].max())
    # cbar.update_normal(im)
    
    title.set_text(f"t = {time_vals[frame]:.1f}")
    return [im, title]

anim = FuncAnimation(
    fig, update, frames=Bx.shape[0], interval=200, blit=True  # Adjust interval as needed
)

anim.save("Bx.gif", writer=PillowWriter(fps=5))
# plt.show()

fig, ax = plt.subplots()
im = ax.imshow(U_e_y[0], extent=(0, Lx, 0, Ly), cmap='jet', vmin=-0.1, vmax=0.1, interpolation='nearest', origin='lower')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("$U_ey$") 
title = ax.set_title(r"$t=0$")
ax.set_xlabel("x/d_e")  # Set x-axis label
ax.set_ylabel("y/d_e")  # Set y-axis label

def update(frame):
    im.set_array(U_e_y[frame])
    
    # im.set_clim(vmin=U_e_y[frame
    # ].min(), vmax=U_e_y[frame].max())
    # cbar.update_normal(im)
    
    title.set_text(f"t = {time_vals[frame]:.1f}")
    return [im, title]

anim = FuncAnimation(
    fig, update, frames=U_e_y.shape[0], interval=200, blit=True  # Adjust interval as needed
)

anim.save("U_e_y.gif", writer=PillowWriter(fps=5))
# plt.show()

# Animation for Jz
fig, ax = plt.subplots()
im = ax.imshow(Jz[0], extent=(0, Lx, 0, Ly), cmap='jet', vmin=-0.2, vmax=0.2, interpolation='nearest', origin='lower')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("$J_z$") 
title = ax.set_title(r"$t=0$")
ax.set_xlabel("x/d_e")  # Set x-axis label
ax.set_ylabel("y/d_e")  # Set y-axis label

def update(frame):
    im.set_array(Jz[frame])
    
    # im.set_clim(vmin=Jz[frame].min(), vmax=Jz[frame].max())
    # cbar.update_normal(im)
    
    title.set_text(f"t = {time_vals[frame]:.1f}")
    return [im, title]

anim = FuncAnimation(
    fig, update, frames=Jz.shape[0], interval=200, blit=True  # Adjust interval as needed
)

anim.save("Jz.gif", writer=PillowWriter(fps=2))
# plt.show()

plt.figure(figsize=(8, 6))
plt.plot(time_vals, B_hat, '-')
plt.xlabel(r'$\omega_{\text{pe}}t$', fontsize=16)
plt.ylabel(r'$|\hat{B}(1, 1)|$', fontsize=16)
plt.savefig('B_mode.pdf')
#plt.show()