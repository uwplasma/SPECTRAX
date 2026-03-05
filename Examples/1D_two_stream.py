import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
number_of_processors_to_use = 1 # Parallelization, this should divide total resolution
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax
import jax.numpy as jnp
from jax import block_until_ready, config
config.update("jax_enable_x64", True)
from spectrax import simulation, load_parameters, plot, construct_idx_array
from scipy.special import eval_legendre, roots_legendre, eval_hermite
from matplotlib.animation import FuncAnimation, PillowWriter

# Read from input.toml
# input_parameters, solver_parameters = load_parameters('input_1D_two_stream.toml')
toml_file = os.path.join(os.path.dirname(__file__), 'input_1D_two_stream.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

alpha_s = input_parameters["alpha_s"]
nx = input_parameters["nx"]
Lx = input_parameters["Lx"]
Ly = input_parameters["Ly"]
Lz = input_parameters["Lz"]
Omega_ce = input_parameters["Omega_ce"]
dn1 = input_parameters["dn1"]
dn2 = input_parameters["dn2"]
Nx = solver_parameters["Nx"]
Ny = solver_parameters["Ny"]
Nz = solver_parameters["Nz"]
Nn = solver_parameters["Nn"]
N_DG = solver_parameters["N_DG"]
dims = solver_parameters["dims"]

basis_idx = construct_idx_array(dims, N_DG)
def legT(f, N_DG=3, Nq=10):
        dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz
        x_elements = dx / 2 + jnp.linspace(0, Lx, Nx, endpoint=False)
        y_elements = dy / 2 + jnp.linspace(0, Ly, Ny, endpoint=False)
        z_elements = dz / 2 + jnp.linspace(0, Lz, Nz, endpoint=False)
        quad_points, w_quad = jnp.array(roots_legendre(Nq))
        leg_vals = jnp.array(eval_legendre(jnp.arange(N_DG)[:, None], quad_points)) # Create array of shape (N_DG, Nq) evaluating the n'th Legendre polynomial at each quadrature point
        leg_vals3d = leg_vals[basis_idx[:, 1], :][:, :, None, None] * leg_vals[basis_idx[:, 0], :][:, None, :, None] * leg_vals[basis_idx[:, 2], :][:, None, None, :] # create values of evaluated Legendre polynomials on 3d local grid. Shape (Nl, Nq, Nq, Nq) where Nq is the number of quadrature points.
        leg_vals3d = jnp.transpose(leg_vals3d, (1, 2, 3, 0))
        w_array = w_quad[:, None, None] * w_quad[None, :, None] * w_quad[None, None, :]
        
        Fk_0 = jnp.zeros((*f(0, 0, 0).shape, Ny, Nx, Nz, basis_idx.shape[0])) # Initialize array
        for i, x in enumerate(x_elements): # Loop over all elements
              for j, y in enumerate(y_elements):
                    for k, z in enumerate(z_elements):
                          X, Y, Z = jnp.meshgrid(x+dx*quad_points/2, y+dy*quad_points/2, z+dz*quad_points/2, indexing='xy') # Create local mesh
                          integral = jnp.sum(f(X, Y, Z)[..., None] * leg_vals3d * w_array[:, :, :, None], axis=(-4, -3, -2))
                          Fk_0 = Fk_0.at[..., j, i, k, :].set((2 * basis_idx[:, 0] + 1) * (2 * basis_idx[:, 1] + 1) * (2 * basis_idx[:, 2] + 1) * integral / 8)

        return Fk_0

# Initialize distribution function as a two-stream instability
values  = (dn1 + dn2) * Lx / (4 * jnp.pi * nx * Omega_ce) # Initial perturbation is then 2 * values * cos(2\pi*x/L)
F0 = lambda x, y, z: jnp.concatenate([jnp.array([2 * values * jnp.cos(2*jnp.pi*x/Lx)]), jnp.broadcast_to(jnp.zeros_like(x), (5,) + jnp.zeros_like(x).shape)])
input_parameters["Fk_0"] = legT(F0, N_DG)

C10 = lambda x: 1 / (alpha_s[0] ** 3) - dn1 * (1 / (alpha_s[0] ** 3)) * jnp.sin(2*jnp.pi*x/Lx)
C20 = lambda x: 1 / (alpha_s[3] ** 3) - dn2 * (1 / (alpha_s[3] ** 3)) * jnp.sin(2*jnp.pi*x/Lx)
C0 = lambda x, y, z: jnp.concatenate([jnp.array([C10(x)]), jnp.broadcast_to(jnp.zeros_like(x), (Nn-1,) + jnp.zeros_like(x).shape), jnp.array([C20(x)]), jnp.broadcast_to(jnp.zeros_like(x), (Nn-1,) + jnp.zeros_like(x).shape)])
input_parameters["Ck_0"] = legT(C0, N_DG)
# Simulate
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

# Plot results
#plot(output)
Ex = output['Fk'][:, 0, 0, :, 0, 0]
Ck = output['Ck'][:, :, 0, :, 0, 0]

Jx1 = -1 * 2**(-3/2) * (Ck[:, 0, :] * 1 + 2**(-1/2) * Ck[:, 1, :] / jnp.sqrt(2))
Jx2 = -1 * 2**(-3/2) * (Ck[:, Nn, :] * -1 + 2**(-1/2) * Ck[:, Nn+1, :] / jnp.sqrt(2))
Jx = Jx1 + Jx2

t = output["time"]

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, figsize=(10, 4))
ax[0].plot(jnp.linspace(0, Lx, Nx), Ex[-1])
ax[0].plot(jnp.linspace(0, Lx, Nx), Ex[0]) 
ax[0].set_title("Frame 0")
ax[0].set_xlabel("x/d_e")  # Set x-axis label
ax[0].set_ylabel("Ex")

ax[1].plot(jnp.linspace(0, Lx, Nx), Jx[-1])
ax[1].plot(jnp.linspace(0, Lx, Nx), Jx[0]) 
ax[1].set_title("Frame 0")
ax[1].set_xlabel("x/d_e")  # Set x-axis label
ax[1].set_ylabel("Jx")

UE = jnp.sum(Ex ** 2, axis=-1) * (1/ Nx) / 2
fig, ax = plt.subplots()
ax.semilogy(t, UE)
ax.set_xlabel(r"$\omega_p t$")
ax.set_ylabel(r"$E^2/2$")
ax.set_title("Electric field energy growth: FEM")
plt.savefig('Ex_growth_2NDG.png')
# plt.show()
# a, b = jnp.polyfit(t[200:401], jnp.log(UE[200:401]), 1)
# print(f"Growth rate: {a}")
alpha_s = input_parameters["alpha_s"]

from jax.scipy.special import factorial
n_vals = jnp.arange(Nn)
v_max = 0.4
Nv = 100
v = jnp.linspace(-v_max, v_max, Nv)
xi1 = (v - 0.1) / alpha_s[0]
xi2 = (v + 0.1) / alpha_s[3]
her_vals1 = jnp.array(eval_hermite(n_vals[:, None], xi1[None, :])) * jnp.exp(-xi1**2) / jnp.sqrt(jnp.pi * (2 ** n_vals) * factorial(n_vals))[:, None]
her_vals2 = jnp.array(eval_hermite(n_vals[:, None], xi2[None, :])) * jnp.exp(-xi2**2) / jnp.sqrt(jnp.pi * (2 ** n_vals) * factorial(n_vals))[:, None]
f1 = jnp.tensordot(Ck[:, :Nn, :], her_vals1, axes=(1, 0))
f2 = jnp.tensordot(Ck[:, Nn:, :], her_vals2, axes=(1, 0))
f = f1 + f2

fig, ax = plt.subplots()
im = ax.imshow(jnp.transpose(f[0]), aspect='auto', cmap='jet', interpolation='none', origin='lower', extent=(0, Lx, -v_max, v_max))
cbar = plt.colorbar(im, ax=ax)
title = ax.set_title(r"$t = 0$")
ax.set_xlabel("x/d_e")  # Set x-axis label
ax.set_ylabel("v/c")  # Set y-axis label

def update(frame):
    im.set_array(jnp.transpose(f[frame]))
    title.set_text(f"Frame {frame}")
    return [im, title]

# Create the animation
anim = FuncAnimation(
    fig, update, frames=f.shape[0], interval=50, blit=True  # Adjust interval as needed
)

anim.save("f_test_2NDG.gif", writer=PillowWriter(fps=10))

# im.set_array(f_vals)
# im.set_clim(vmin=f_vals.min(), vmax=f_vals.max())
# el_line.set_ydata(np.asarray(E_vals))
# ax[1].relim()
# ax[1].autoscale_view()
# cbar.update_normal(im)
# title.set_text(f'$t= {t}$')
# plt.draw()
# plt.pause(0.01)


# print('Saving results...')
# jnp.savez('output_two-stream.npz', **output)

# print("Loading results...")
# output = jnp.load('output_orszag.npz')