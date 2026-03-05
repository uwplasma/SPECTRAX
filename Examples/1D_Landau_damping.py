import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
number_of_processors_to_use = 1 # Parallelization, this should divide total resolution
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
from jax import block_until_ready
from spectrax import simulation, load_parameters, plot, construct_idx_array
from scipy.special import eval_legendre, roots_legendre, eval_hermite
from matplotlib.animation import FuncAnimation, PillowWriter
# import pandas as pd

# Read from input.toml
# input_parameters, solver_parameters = load_parameters('input_1D_landau_damping.toml')
toml_file = os.path.join(os.path.dirname(__file__), 'input_1D_landau_damping.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

alpha_s = input_parameters["alpha_s"]
u_s = input_parameters["u_s"]
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

F0 = lambda x, y, z: jnp.concatenate([jnp.array([2 * dn * Lx / (4 * jnp.pi * Omega_ce) * jnp.cos(2*jnp.pi*x/Lx)]), jnp.broadcast_to(jnp.zeros_like(x), (5,) + jnp.zeros_like(x).shape)])
input_parameters["Fk_0"] = legT(F0, N_DG)

Ci0_0 = lambda x: 1 / (alpha_s[3] ** 3) * jnp.ones_like(x)
Ce0_0 = lambda x: 1 / (alpha_s[0] ** 3) * (1 + dn * jnp.sin(2*jnp.pi*x/Lx))
C0 = lambda x, y, z: jnp.concatenate([jnp.array([Ce0_0(x)]), jnp.broadcast_to(jnp.zeros_like(x), (Nn-1,) + jnp.zeros_like(x).shape), jnp.array([Ci0_0(x)]), jnp.broadcast_to(jnp.zeros_like(x), (Nn-1,) + jnp.zeros_like(x).shape)])
input_parameters["Ck_0"] = legT(C0, N_DG)

Ue_0 = alpha_s[0]**2 / 4

# Simulate
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

Fk = output['Fk'][:, :, 0, :, 0, :]
Ck = output['Ck'][:, :, 0, :, 0, :]
C = Ck[:, :, :, 0]
t = output["time"]

import matplotlib.pyplot as plt
from jax.scipy.special import factorial
n_vals = jnp.arange(Nn)
v_max = 3 * alpha_s[0]
Nv = 100
v = jnp.linspace(-v_max, v_max, Nv)
xi1 = (v - u_s[0]) / alpha_s[0]
xi2 = (v + u_s[3]) / alpha_s[3]
her_vals1 = jnp.array(eval_hermite(n_vals[:, None], xi1[None, :])) * jnp.exp(-xi1**2) / jnp.sqrt(jnp.pi * (2 ** n_vals) * factorial(n_vals))[:, None]
her_vals2 = jnp.array(eval_hermite(n_vals[:, None], xi2[None, :])) * jnp.exp(-xi2**2) / jnp.sqrt(jnp.pi * (2 ** n_vals) * factorial(n_vals))[:, None]
f = jnp.tensordot(C[:, :Nn, :], her_vals1, axes=(1, 0))

# Compute energies
U_EM = 0.5 * Omega_ce**2 * jnp.sum(jnp.abs(Fk) ** 2 / (1 + 2 * jnp.arange(0, N_DG)), axis=(-2, -1)) / Nx
ve2 = alpha_s[0] * alpha_s[1] * alpha_s[2] * (alpha_s[0]**2 * Ck[:,2] / jnp.sqrt(2) + jnp.sqrt(2) * u_s[0] * alpha_s[0] * Ck[:,1] + (alpha_s[0]**2 / 2 + u_s[0]**2) * Ck[:,0])
vi2 = alpha_s[3] * alpha_s[4] * alpha_s[5] * (alpha_s[3]**2 * Ck[:,Nn*Nm*Np+2] / jnp.sqrt(2) + jnp.sqrt(2) * u_s[3] * alpha_s[3] * Ck[:,Nn*Nm*Np+1] + (alpha_s[3]**2 / 2 + u_s[3]**2) * Ck[:,Nn*Nm*Np])
U_Ke = 0.5 * 1 * jnp.sum(ve2 / (1 + 2 * jnp.arange(0, N_DG)), axis=(-2, -1)) / Nx
U_Ki = 0.5 * 1836 * jnp.sum(vi2 / (1 + 2 * jnp.arange(0, N_DG)), axis=(-2, -1)) / Nx
U_tot = jnp.sum(U_EM, axis=(-1)) + U_Ke + U_Ki

fig, ax = plt.subplots()
ax.semilogy(t, U_EM[:, 0], label=r"$E_x$ energy")
ax.set_xlabel(r"$\omega_p t$")
ax.set_ylabel(r"$E^2/2$")
ax.set_title("Electric field energy growth: FEM")
plt.savefig('Ex_growth_Landau.png')

fig, ax = plt.subplots()
ax.semilogy(t, U_EM[:, 0], label=r"$E_x$ energy")
ax.semilogy(t, U_Ke, label=r"Electron kinetic energy")
ax.semilogy(t, U_Ki, label=r"Ion kinetic energy")
ax.semilogy(t, U_tot, label=r"Total energy")
ax.semilogy(t, jnp.abs(U_tot / U_tot[0] - 1), label=r"Energy error")
ax.legend()
ax.set_xlabel(r"$\omega_p t$")
ax.set_ylabel(r"Energies")
ax.set_title("Energies")

Jx = - alpha_s[0] * alpha_s[1] * alpha_s[2] * (u_s[0] * C[:, 0, :] + alpha_s[0] * C[:, 1, :] / jnp.sqrt(2))
Work = jnp.sum(Jx * Fk[:, 0, :, 0] , axis=(-1)) / Nx

# Plot perturbation energies
fig, ax = plt.subplots()
ax.plot(t, U_Ke - Ue_0, label="Electron perturbation energy")
ax.plot(t, U_Ki - U_Ki[0], label="Ion perturbation energy")
ax.plot(t, jnp.sum(U_EM, axis=(-1)), label="Field energy")
ax.plot(t, Work, label="Work")
ax.set(xlabel=r"$t\omega_{pe}$", ylabel=r"$U$")
ax.legend()

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

# anim.save("f_Landau.gif", writer=PillowWriter(fps=10))

plt.show()

# Plot results
#plot(output)

# print('Saving results...')
# jnp.savez('output_landau-damping.npz', **output)
