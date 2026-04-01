"""Example: 1D Landau damping damping-rate curve.

This script runs a parameter scan over the perturbation wavenumber for the
standard 1D Landau damping setup and compares the *measured* damping rate from
SPECTRAX against a tabulated analytical/theoretical curve.

Configuration is read from ``input_1D_landau_damping_dr_curve.toml`` in the
same directory. For each target wavenumber ``kx`` we adjust the domain length
``Lx`` so that the excited Fourier mode corresponds to that ``kx`` value, run a
simulation, and then estimate the damping rate by fitting an exponential decay
to the peaks of the electromagnetic energy:

    EM_energy(t) ~ exp(2 * gamma * t)  =>  gamma = 0.5 * d/dt log(EM_energy)

Notes
-----
- The peak-based fit is done on the first 100 time samples to stay in the
  linear regime (adjust if needed).
- The theoretical curve is loaded from ``damping_rate_1836.csv``.
"""

import sys
import os
# Allow running this example directly from the repo without installing the package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax.numpy as jnp
from jax import block_until_ready
from spectrax import simulation, load_parameters
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt


# Read run configuration from the example TOML file.
toml_file = os.path.join(os.path.dirname(__file__), 'input_1D_landau_damping_dr_curve.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

# Unpack frequently used parameters for readability.
alpha_s = input_parameters["alpha_s"]
u_s = input_parameters["u_s"]
nx = input_parameters["nx"]
ny = input_parameters["ny"]
nz = input_parameters["nz"]
Ly = input_parameters["Ly"]
Lz = input_parameters["Lz"]
Omega_cs = input_parameters["Omega_cs"]
dn = input_parameters["dn1"]
Nx = solver_parameters["Nx"]
Ny = solver_parameters["Ny"]
Nz = solver_parameters["Nz"]
Nn = solver_parameters["Nn"]
Nm = solver_parameters["Nm"]
Np = solver_parameters["Np"]
Nt = solver_parameters["timesteps"]
mi_me = input_parameters["mi_me"]
qs = input_parameters["qs"]
nu = input_parameters["nu"]
dt = solver_parameters["dt"]
t_max = input_parameters["t_max"]

# Wavevector bookkeeping: (nx, ny, nz) selects which Fourier mode is excited.
n = nx + ny + nz

# Map the wavevector orientation to which E-field component gets initialized.
# For the typical 1D case (ny=nz=0), this selects the x-component.
E_field_component = int(jnp.sign(ny) + 2 * jnp.sign(nz))



# ---------------------------------------------------------------------------
# Initial condition: Hermite-Fourier components of electron and ion
# distribution functions (background + small sinusoidal perturbation).
# ---------------------------------------------------------------------------
Ce0_mk, Ce0_0, Ce0_k = 0 + 1j * (1 / (2 * alpha_s[0] ** 3)) * dn, 1 / (alpha_s[0] ** 3) + 0 * 1j, 0 - 1j * (1 / (2 * alpha_s[0] ** 3)) * dn
Ci0_0 = 1 / (alpha_s[3] ** 3) + 0 * 1j
Ck_0 = jnp.zeros((2 * Nn * Nm * Np, Ny, Nx, Nz), dtype=jnp.complex128)
Ck_0 = Ck_0.at[0, int((Ny-1)/2-ny), int((Nx-1)/2-nx), int((Nz-1)/2-nz)].set(Ce0_mk)
Ck_0 = Ck_0.at[0, int((Ny-1)/2), int((Nx-1)/2), int((Nz-1)/2)].set(Ce0_0)
Ck_0 = Ck_0.at[0, int((Ny-1)/2+ny), int((Nx-1)/2+nx), int((Nz-1)/2+nz)].set(Ce0_k)
Ck_0 = Ck_0.at[Nn * Nm * Np, int((Ny-1)/2), int((Nx-1)/2), int((Nz-1)/2)].set(Ci0_0)
input_parameters["Ck_0"] = Ck_0

# Target dimensionless wavenumbers to scan.
kx = jnp.arange(0.1, 1.02, 0.02)

# Store the measured damping rate gamma(kx).
damping_rate = jnp.zeros_like(kx)
for j in jnp.arange(len(kx)):
    # Set the box length in x so that the chosen Fourier mode corresponds to
    # this target wavenumber.
    Lx = jnp.sqrt(2) * jnp.pi * alpha_s[0] / kx[j]
    input_parameters["Lx"] = Lx

    # Initialize Fourier components of the electromagnetic fields.
    Fk_0 = jnp.zeros((6, Ny, Nx, Nz), dtype=jnp.complex128)
    Fk_0 = Fk_0.at[E_field_component, int((Ny-1)/2-ny), int((Nx-1)/2-nx), int((Nz-1)/2-nz)].set(dn * Lx / (4 * jnp.pi * n * Omega_cs[0]))
    Fk_0 = Fk_0.at[E_field_component, int((Ny-1)/2+ny), int((Nx-1)/2+nx), int((Nz-1)/2+nz)].set(dn * Lx / (4 * jnp.pi * n * Omega_cs[0]))
    input_parameters["Fk_0"] = Fk_0

    # Run the simulation (block until the JAX computation is complete).
    output = block_until_ready(simulation(input_parameters, **solver_parameters))

    t = output["time"]
    EM_energy = output["EM_energy"]

    # Estimate gamma by fitting log(EM_energy) at its oscillation peaks:
    # log(EM_energy) ~ (2 * gamma) * t + const.
    peaks, _ = find_peaks(EM_energy[:100])
    p = jnp.polyfit(t[peaks], jnp.log(EM_energy[peaks]), 1)
    damping_rate = damping_rate.at[j].set(p[0] / 2)

# Load the reference/theoretical curve (tabulated).
damping_rate_theo_file = os.path.join(os.path.dirname(__file__), 'damping_rate_1836.csv')
data_damping_rate = pd.read_csv(damping_rate_theo_file, header=None)
damping_rate_theo = jnp.array(data_damping_rate.values)


# Plot measured damping-rate curve against the reference.
plt.figure(figsize=(8, 6))
plt.plot(kx[:40], damping_rate[:40], label=r'$\gamma_{SPECTRAX}$', linestyle='None', marker='o', color='red', linewidth=3.0)
plt.plot(damping_rate_theo[:, 0], damping_rate_theo[:, 1], label=r'$\gamma_{theo}$', linestyle='-', color='blue', linewidth=3.0)
plt.xlim(0.0, 1.0)
plt.ylim(-1.4, 0.1)
plt.ylabel(r'$\gamma/\omega_{pe}$', fontsize=16)
plt.xlabel(r'$kv_{e,th}/\omega_{pe}$', fontsize=16)
plt.title(rf'$\nu = {nu}, u_e = {u_s[0]}, \alpha_e = {alpha_s[0]:.3},'
                +rf'N_x = {Nx}, N_n = {Nn}, \delta n = {dn}$', fontsize=18)
plt.legend(fontsize=20).set_draggable(True)
plt.show()
