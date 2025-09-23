import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
number_of_processors_to_use = 1 # Parallelization, this should divide total resolution
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
from jax import block_until_ready
import jax.numpy as jnp
from spectrax import simulation, load_parameters, plot, initialize_xv
import matplotlib.pyplot as plt
from jax.numpy.fft import ifftn, ifftshift
from matplotlib.animation import FuncAnimation, PillowWriter

# Read from input.toml
# input_parameters, solver_parameters = load_parameters('input_2D_orszag_tang.toml')
toml_file = os.path.join(os.path.dirname(__file__), 'input_2D_orszag_tang.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

print('Setting up simulation parameters...')
start_time = time()
deltaB = 0.2 # In-plane magnetic field amplitude. 
U0 = deltaB * input_parameters["Omega_cs"][0] / jnp.sqrt(input_parameters["mi_me"])
kx = 2 * jnp.pi / input_parameters["Lx"]
ky = 2 * jnp.pi / input_parameters["Ly"]
Nv = 125 
nvxyz = 40 # Number of velocity points in each dimension for the velocity grid.
max_min_v_factor = 5 # Factor for the maximum and minimum velocity in each dimension.

# Electron and ion fluid velocities.
Ue = lambda x, y, z: U0 * jnp.array([-jnp.sin(ky * y), jnp.sin(kx * x), -deltaB * input_parameters["Omega_cs"][0] * (2 * kx * jnp.cos(2 * kx * x) + ky * jnp.cos(ky * y))])
Ui = lambda x, y, z: U0 * jnp.array([-jnp.sin(ky * y), jnp.sin(kx * x), jnp.zeros_like(x)])

# Magnetic and electric fields.
B = lambda x, y, z: jnp.array([-deltaB * jnp.sin(ky * y), deltaB * jnp.sin(2 * kx * x), jnp.ones_like(x)])
E = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)])

# Electron and ion distribution functions.
fe = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * input_parameters["vte"] ** 3) * 
                                jnp.exp(-((vx - Ue(x, y, z)[0])**2 + (vy - Ue(x, y, z)[1])**2 + (vz - Ue(x, y, z)[2])**2) / (2 * input_parameters["vte"] ** 2))))
fi = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * input_parameters["vti"] ** 3) * 
                                jnp.exp(-((vx - Ui(x, y, z)[0])**2 + (vy - Ui(x, y, z)[1])**2 + (vz - Ui(x, y, z)[2])**2) / (2 * input_parameters["vti"] ** 2))))

input_parameters["Ck_0"], input_parameters["Fk_0"] =  block_until_ready(
    initialize_xv(B, E, fe, fi, Nv, nvxyz, max_min_v_factor, input_parameters, **solver_parameters))
print(f"took {time() - start_time} seconds to initialize the simulation parameters.")

print('Starting simulation...')
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

# print('Plotting results...')
# plot(output)


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
Ck = output["Ck"]

C = ifftn(ifftshift(Ck, axes=(-3, -2, -1)), axes=(-3, -2, -1)).real

ne = alpha_s[0] * alpha_s[1] * alpha_s[2] * C[:, 0, :, :, 0]
ni = alpha_s[3] * alpha_s[4] * alpha_s[5] * C[:, Nn * Nm * Np, :, :, 0]

Jz = (1 / jnp.sqrt(2)) * (alpha_s[3] * alpha_s[4] * (alpha_s[5]**2) * C[:, Nn * Nm * Np + Nn * Nm, :, :, 0] - alpha_s[0] * alpha_s[1] * (alpha_s[2]**2) * C[:, Nn * Nm, :, :, 0])



fig, axes = plt.subplots(1, 2, figsize=(15, 9))
fig.suptitle(rf'$N_x = {Nx}$, $N_n = {Nn}$, $N_m = {Nm}$, $N_p = {Np}$, $\nu = {nu}$, $|u_z| = {uz}$', fontsize=14)

# Energy plots
axes[0].plot(t, output["EM_energy"], label="EM energy")
axes[0].plot(t, output["kinetic_energy"], label="Kinetic energy")
axes[0].plot(t, output["kinetic_energy_species1"], label="Kinetic energy species 1")
axes[0].plot(t, output["kinetic_energy_species2"], label="Kinetic energy species 2")
axes[0].plot(t, output["total_energy"], label="Total energy")
axes[0].set(title="Energy", xlabel=r"$t\omega_{pe}$", ylabel="Energy", yscale="log")#, ylim=[1e-5, None])
axes[0].legend()

axes[1].plot(t[1:], jnp.abs(output["total_energy"][1:]-output["total_energy"][0])/(output["total_energy"][0]+1e-9), label="Relative energy error")
axes[1].set(xlabel=r"$t\omega_{pe}$", ylabel="Relative Energy Error", yscale="log")#, ylim=[1e-5, None])

plt.figure(figsize=(8, 6))
plt.imshow(ne[50], aspect='auto', cmap='jet', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))#, vmin=-10, vmax=10)
plt.colorbar(label=r'$n_e$').ax.yaxis.label.set_size(16)

# plt.imshow(C2[:10000, :Nn * Nm * Np], aspect='auto', cmap='viridis', 
# interpolation='none', origin='lower', extent=(0, Nn, 0, 1000))
# plt.colorbar(label=r'$\langle |C_{e,n}|^2\rangle (t)$').ax.yaxis.label.set_size(16)

# plt.plot(jnp.arange(Nn) + 0.5, 3.6*jnp.sqrt(jnp.arange(Nn)), label='$3.60\sqrt{n}$', linestyle='-', color='black', linewidth=3.0)
plt.xlabel('x/d_e', fontsize=16)
plt.ylabel('y/d_e', fontsize=16)
# plt.title(rf'$\nu ={nu}, L_x/d_e = {Lx}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=14)
# plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(ni[50], aspect='auto', cmap='jet', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))#, vmin=-10, vmax=10)
plt.colorbar(label=r'$n_i$').ax.yaxis.label.set_size(16)

# plt.imshow(C2[:10000, :Nn * Nm * Np], aspect='auto', cmap='viridis', 
# interpolation='none', origin='lower', extent=(0, Nn, 0, 1000))
# plt.colorbar(label=r'$\langle |C_{e,n}|^2\rangle (t)$').ax.yaxis.label.set_size(16)

# plt.plot(jnp.arange(Nn) + 0.5, 3.6*jnp.sqrt(jnp.arange(Nn)), label='$3.60\sqrt{n}$', linestyle='-', color='black', linewidth=3.0)
plt.xlabel('x/d_e', fontsize=16)
plt.ylabel('y/d_e', fontsize=16)
# plt.title(rf'$\nu ={nu}, L_x/d_e = {Lx}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=14)
# plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(Jz[50], aspect='auto', cmap='jet', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))#, vmin=-10, vmax=10)
plt.colorbar(label=r'$J_z$').ax.yaxis.label.set_size(16)

# plt.imshow(C2[:10000, :Nn * Nm * Np], aspect='auto', cmap='viridis', 
# interpolation='none', origin='lower', extent=(0, Nn, 0, 1000))
# plt.colorbar(label=r'$\langle |C_{e,n}|^2\rangle (t)$').ax.yaxis.label.set_size(16)

# plt.plot(jnp.arange(Nn) + 0.5, 3.6*jnp.sqrt(jnp.arange(Nn)), label='$3.60\sqrt{n}$', linestyle='-', color='black', linewidth=3.0)
plt.xlabel('x/d_e', fontsize=16)
plt.ylabel('y/d_e', fontsize=16)
# plt.title(rf'$\nu ={nu}, L_x/d_e = {Lx}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=14)
# plt.legend()
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(jnp.transpose(Jz[0]), extent=(0, Lx, 0, Ly), cmap='jet', interpolation='nearest', origin='lower')

# Add a color bar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("$J_z$") 

# Set up the plot aesthetics
title = ax.set_title("Frame 0")
ax.set_xlabel("x/d_e")  # Set x-axis label
ax.set_ylabel("y/d_e")  # Set y-axis label
# ax.tick_params(axis='both', which='both', direction='in')  # Show tick marks
# ax.axis('off')  
# ax.set_aspect(0.5)

# Update function for the animation
def update(frame):
    im.set_array(jnp.transpose(Jz[frame]))
    
    im.set_clim(vmin=Jz[frame].min(), vmax=Jz[frame].max())
    cbar.update_normal(im)
    
    title.set_text(f"Frame {frame}")
    return [im, title]

# Create the animation
anim = FuncAnimation(
    fig, update, frames=Jz.shape[0], interval=50, blit=True  # Adjust interval as needed
)

# Save the animation as a GIF
anim.save("/Users/csvega/Desktop/Madison/Code/Simulations/Orszag_Tang/S8/Jz.gif", writer=PillowWriter(fps=5))

# Display the animation
plt.show()

print('Saving results...')
jnp.savez('/Users/csvega/Desktop/Madison/Code/Simulations/Orszag_Tang/S8/output_orszag.npz', **output)

# # print("Loading results...")
# # output = jnp.load('output_orszag.npz')