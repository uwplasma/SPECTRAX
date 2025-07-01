import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
number_of_processors_to_use = 5 # Parallelization, this should divide total resolution
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'
from time import time
import jax.numpy as jnp
from jax import block_until_ready
from spectrax import simulation, load_parameters, plot
from jax.numpy.fft import ifftn, ifftshift
import matplotlib.pyplot as plt

# Read from input.toml
# input_parameters, solver_parameters = load_parameters('input_1D_landau_damping.toml')
toml_file = os.path.join(os.path.dirname(__file__), 'input_1D_Weibel.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

alpha_s = input_parameters["alpha_s"]
dE = input_parameters["dE"]
Nx = solver_parameters["Nx"]
Nn = solver_parameters["Nn"]
Nm = solver_parameters["Nm"]
Np = solver_parameters["Np"]
nx = input_parameters["nx"]
Lx = input_parameters["Lx"]
Omega_cs = input_parameters["Omega_cs"]

# Fourier components of magnetic and electric fields.
Fk_0 = jnp.zeros((6, 1, Nx, 1), dtype=jnp.complex128)
Fk_0 = Fk_0.at[2, 0, int((Nx-1)/2-nx), 0].set(dE / 2)
Fk_0 = Fk_0.at[2, 0, int((Nx-1)/2+nx), 0].set(dE / 2)
input_parameters["Fk_0"] = Fk_0

# Hermite-Fourier components of electron and ion distribution functions.
C10_0 = 1 / (alpha_s[0] * alpha_s[1] * alpha_s[2]) + 0 * 1j
C20_0 = 1 / (alpha_s[3] * alpha_s[4] * alpha_s[5]) + 0 * 1j
Ck_0 = jnp.zeros((2 * Nn * Nm * Np, 1, Nx, 1), dtype=jnp.complex128)
Ck_0 = Ck_0.at[0, 0, int((Nx-1)/2), 0].set(C10_0)
Ck_0 = Ck_0.at[Nn * Nm * Np, 0, int((Nx-1)/2), 0].set(C20_0)
input_parameters["Ck_0"] = Ck_0

# Simulate
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

# Plot results
# plot(output)

Fk = output["Fk"]
t = output["time"]
Lx = output["Lx"]
Ny = output["Ny"]
Nz = output["Nz"]
nu = output["nu"]
uz = output["u_s"][2]

F = ifftn(ifftshift(Fk, axes=(-3, -2, -1)), axes=(-3, -2, -1)).real


fig, axes = plt.subplots(2, 3, figsize=(15, 9))
plt.subplots_adjust(hspace=0.2, wspace=0.2)
fig.suptitle(rf'$N_x = {Nx}$, $N_y = {Ny}$, $N_z = {Nz}$, $N_n = {Nn}$, $N_m = {Nm}$, $N_p = {Np}$, $\nu = {nu}$, $|u_z| = {uz}$', fontsize=14)


im00 = axes[0, 0].imshow(F[:, 0, 0, :, 0], aspect='auto', cmap='jet', 
        interpolation='sinc', origin='lower', extent=(0, Lx, 0, t[-1]))
axes[0, 0].set(title="$E_x$", xlabel=r"$x/d_e$", ylabel=r"$t\omega_{pe}$")#, ylim=[1e-5, None])
fig.colorbar(im00)

im01 = axes[0, 1].imshow(F[:, 1, 0, :, 0], aspect='auto', cmap='jet', 
        interpolation='sinc', origin='lower', extent=(0, Lx, 0, t[-1]))
axes[0, 1].set(title="$E_y$", xlabel=r"$x/d_e$", ylabel=r"$t\omega_{pe}$")#, ylim=[1e-5, None])
fig.colorbar(im01)

im02 = axes[0, 2].imshow(F[:, 2, 0, :, 0], aspect='auto', cmap='jet', 
        interpolation='sinc', origin='lower', extent=(0, Lx, 0, t[-1]))
axes[0, 2].set(title="$E_z$", xlabel=r"$x/d_e$", ylabel=r"$t\omega_{pe}$")#, ylim=[1e-5, None])
fig.colorbar(im02)

im10 = axes[1, 0].imshow(F[:, 3, 0, :, 0], aspect='auto', cmap='jet', 
        interpolation='sinc', origin='lower', extent=(0, Lx, 0, t[-1]))
axes[1, 0].set(title="$B_x$", xlabel=r"$x/d_e$", ylabel=r"$t\omega_{pe}$")#, ylim=[1e-5, None])
fig.colorbar(im10)

im11 = axes[1, 1].imshow(F[:, 4, 0, :, 0], aspect='auto', cmap='jet', 
        interpolation='sinc', origin='lower', extent=(0, Lx, 0, t[-1]))
axes[1, 1].set(title="$B_y$", xlabel=r"$x/d_e$", ylabel=r"$t\omega_{pe}$")#, ylim=[1e-5, None])
fig.colorbar(im11)

im12 = axes[1, 2].imshow(F[:, 5, 0, :, 0], aspect='auto', cmap='jet', 
        interpolation='sinc', origin='lower', extent=(0, Lx, 0, t[-1]))
axes[1, 2].set(title="$B_z$", xlabel=r"$x/d_e$", ylabel=r"$t\omega_{pe}$")#, ylim=[1e-5, None])
fig.colorbar(im12)


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

fig, axes = plt.subplots(figsize=(15, 9))
fig.suptitle(rf'$N_x = {Nx}$, $N_n = {Nn}$, $N_m = {Nm}$, $N_p = {Np}$, $\nu = {nu}$, $|u_z| = {uz}$', fontsize=14)

# Energy plots
axes.plot(t, jnp.sum(jnp.abs(Fk[:,0,...]) ** 2, axis=(-1,-2,-3)), label=r"$|E_x|^2$")
axes.plot(t, jnp.sum(jnp.abs(Fk[:,2,...]) ** 2, axis=(-1,-2,-3)), label=r"$|E_z|^2$")
axes.plot(t, jnp.sum(jnp.abs(Fk[:,4,...]) ** 2, axis=(-1,-2,-3)), label=r"$|B_y|^2$")
axes.set(xlabel=r"$t\omega_{pe}$", ylabel=r"|Fields|$^2$", yscale="log")#, ylim=[1e-5, None])
axes.legend()


plt.show()

# print('Saving results...')
# jnp.savez('output_landau-damping.npz', **output)