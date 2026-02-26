"""Example: 1D free streaming.

Compares the simulated density perturbation against the analytic free-streaming
solution for a sinusoidal perturbation in 1D.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from time import time
import jax
import jax.numpy as jnp
from jax import block_until_ready, config
config.update("jax_enable_x64", True)
from spectrax import simulation, load_parameters, plot
from jax.numpy.fft import ifftn, ifftshift, fftn, fftshift
import matplotlib.pyplot as plt


# Read from input.toml
toml_file = os.path.join(os.path.dirname(__file__), 'input_1D_free_streaming.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

alpha_s = input_parameters["alpha_s"]
nx = input_parameters["nx"]
Lx = input_parameters["Lx"]
Omega_cs = input_parameters["Omega_cs"]
dn = input_parameters["dn"]
Nx = solver_parameters["Nx"]
Nn = solver_parameters["Nn"]

# Initialize a small sinusoidal density perturbation in Hermite–Fourier space.
Fk_0    = jnp.zeros((6, 1, Nx, 1), dtype=jnp.complex128)
input_parameters["Fk_0"] = Fk_0

C10     = jnp.array([
        0 + 1j * (1 / (2 * alpha_s[0] ** 3)) * dn,
        1 / (alpha_s[0] ** 3) + 0 * 1j,
        0 - 1j * (1 / (2 * alpha_s[0] ** 3)) * dn
])

indices = jnp.array([int((Nx-1)/2-nx), int((Nx-1)/2), int((Nx-1)/2+nx)])
Ck_0    = jnp.zeros((2 * Nn, 1, Nx, 1), dtype=jnp.complex128)
Ck_0    = Ck_0.at[0,  0, indices, 0].set(C10)
Ck_0    = Ck_0.at[Nn, 0, indices, 0].set(C10)
input_parameters["Ck_0"] = Ck_0

# Simulate
start_time = time()
output = block_until_ready(simulation(input_parameters, **solver_parameters))
print(f"Runtime: {time() - start_time} seconds")

Ck = output["Ck"]
dCk = output["dCk"]
t = output["time"]
nu = input_parameters["nu"]



x = jnp.linspace(0, Lx, 201)
kx = 2 * jnp.pi / Lx
n = -alpha_s[0] ** 3 * 2 * Ck[: , 0, 0, 2, 0, None].imag * jnp.sin(kx * x[None, :])
n_exact = dn * jnp.sin(kx * x[None, :]) * jnp.exp(- (kx * alpha_s[0] * t[:, None]) ** 2 / 4)


fig, axes = plt.subplots(figsize=(15, 9))
# fig.suptitle(rf'$N_x = {Nx}$, $N_n = {Nn}$, $\nu = {nu}$', fontsize=18)

# Energy plots
axes.plot(x, n[0], label=r"$\delta n_{SPEC}$, $t\omega_{pe}=0$")
axes.plot(x, n[200], label=r"$\delta n_{SPEC}$, $t\omega_{pe}=2$")
axes.plot(x, n[500], label=r"$\delta n_{SPEC}$, $t\omega_{pe}=5$")
axes.plot(x, n_exact[0], linestyle=':', color='black', label=r"$\delta n_{exact}$, $t\omega_{pe}=0$")
axes.plot(x, n_exact[200], linestyle='--', color='black', label=r"$\delta n_{exact}$, $t\omega_{pe}=2$")
axes.plot(x, n_exact[500], linestyle='-.', color='black', label=r"$\delta n_{exact}$, $t\omega_{pe}=5$")
axes.set_xlabel(r"$x/d_e$", fontsize=18)
axes.set_ylabel(r"$\delta n/n_0$", fontsize=18)
axes.legend(fontsize=18).set_draggable(True)
axes.set_title(rf'$N_x = {Nx}$, $N_n = {Nn}$, $\nu = {nu}$', fontsize=18)
# axins = axes.inset_axes([0.1, 0.1, 0.4, 0.3])
# axins.plot(x, n[500], label=r"$\delta n_{SPEC}$")
# axins.plot(x, n_exact[500], linestyle='-.', color='black', label=r"$\delta n_{exact}$")
# # axins.set_xlim(0.0, 1.0)
# # axins.set_ylim(-0.5, 0.5)
# axins.set_xlabel(r'$x/d_e$', fontsize=12)
# axins.set_ylabel(r'$\delta n/n_0$', fontsize=12)
# axins.legend(fontsize=10).set_draggable(True)
# axins.set_title(r'$t\omega_{pe}=5$', fontsize=14)
plt.show()


fig, axes = plt.subplots(figsize=(15, 9))
axes.plot(x, n[0], label=r"$\delta n_{SPEC}$, $t\omega_{pe}=0$")
axes.plot(x, n[243], linestyle='--', label=r"$\delta n_{SPEC}$, $t\omega_{pe}=24.3$")
axes.plot(x, n[245], linestyle=':', label=r"$\delta n_{SPEC}$, $t\omega_{pe}=24.5$")
axes.set_xlabel(r"$x/d_e$", fontsize=18)
axes.set_ylabel(r"$\delta n/n_0$", fontsize=18)
axes.legend(fontsize=18).set_draggable(True)
axes.set_title(rf'$N_x = {Nx}$, $N_n = {Nn}$, $\nu = {nu}$, ' + r'$T\omega_{pe}=24.5$', fontsize=18)
plt.show()


Nm = solver_parameters["Nm"]
Np = solver_parameters["Np"]
t_max = input_parameters["t_max"]
dCk2 = jnp.mean(jnp.abs(dCk)**2, axis=(-3,-2,-1))



plt.figure(figsize=(8, 6))
plt.imshow(jnp.log10(dCk2[:, :Nn * Nm * Np]), aspect='auto', cmap='viridis', 
interpolation='none', origin='lower', extent=(0, Nn, 0, t_max))
plt.colorbar(label=r'$\log_{10}[\langle |C_n^k|^2\rangle_k (t)]$').ax.yaxis.label.set_size(18)
# plt.colorbar(label=r'$\langle |C_n^k|^2\rangle_k (t)$').ax.yaxis.label.set_size(18)

# plt.plot(jnp.arange(Nn) + 0.5, 3.6*jnp.sqrt(jnp.arange(Nn)), label='$3.60\sqrt{n}$', linestyle='-', color='black', linewidth=3.0)
plt.xlabel(r'$n$', fontsize=18)
plt.ylabel(r'$t\omega_{pe}$', fontsize=18)
plt.title(rf'$N_x = {Nx}$, $N_n = {Nn}$, $\nu = {nu}$', fontsize=18)
# plt.legend()
plt.show()
