"""Example: 1D two-stream instability growth-rate curve.

This script runs a scan over the perturbation wavenumber for a 1D two-stream
instability setup and compares the *measured* growth rate from SPECTRAX against
a tabulated analytical/theoretical curve.

Configuration is read from ``input_1D_two_stream_gr_curve.toml`` in the same
directory. For each target wavenumber ``kx`` we adjust the domain length ``Lx``
so the excited Fourier mode corresponds to that ``kx`` value, run a simulation,
and estimate the growth rate by fitting an exponential to the electromagnetic
energy time series in the linear growth regime:

    EM_energy(t) ~ exp(2 * gamma * t)  =>  gamma = 0.5 * d/dt log(EM_energy)

Notes
-----
- The fit is performed on a late-time window (``t[300:]`` here) to avoid
  transient behavior; adjust this slice if the growth window changes.
- The theoretical curve is loaded from ``growth_rate.csv``.
"""

import sys
import os
# Allow running this example directly from the repo without installing the package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax.numpy as jnp
from jax import block_until_ready
from spectrax import simulation, load_parameters
import pandas as pd
import matplotlib.pyplot as plt

# Read run configuration from the example TOML file.
toml_file = os.path.join(os.path.dirname(__file__), 'input_1D_two_stream_gr_curve.toml')
input_parameters, solver_parameters = load_parameters(toml_file)

# Unpack frequently used parameters for readability.
alpha_s = input_parameters["alpha_s"]
u_s = input_parameters["u_s"]
mi_me = input_parameters["mi_me"]
nx = input_parameters["nx"]
Omega_cs = input_parameters["Omega_cs"]
dn1 = input_parameters["dn1"]
dn2 = input_parameters["dn2"]
Nx = solver_parameters["Nx"]
Ny = solver_parameters["Ny"]
Nz = solver_parameters["Nz"]
Nn = solver_parameters["Nn"]
Nm = solver_parameters["Nm"]
Np = solver_parameters["Np"]
nu = input_parameters["nu"]

# ---------------------------------------------------------------------------
# Initial condition: two counter-streaming populations (Hermite-Fourier form)
# with a small density perturbation at the selected Fourier mode index ±nx.
# ---------------------------------------------------------------------------

C10     = jnp.array([
        1 / (alpha_s[0] ** 3) + 0 * 1j,
        0 - 1j * (1 / (2 * alpha_s[0] ** 3)) * dn1
])
C20     = jnp.array([
        1 / (alpha_s[3] ** 3) + 0 * 1j,
        0 - 1j * (1 / (2 * alpha_s[3] ** 3)) * dn2
])

# Indices corresponding to (0, k) for the chosen Fourier mode in x.
indices = jnp.array([0, nx])

# Ck_0 stores the Hermite-Fourier coefficients for each species.
# Here Ns=2 and Nm=Nz=1 so the shape is (2*Nn, 1, Nx//2+1, 1).
Ck_0    = jnp.zeros((2 * Nn, 1, Nx//2+1, 1), dtype=jnp.complex128)
Ck_0    = Ck_0.at[0,  0, indices, 0].set(C10)
Ck_0    = Ck_0.at[Nn, 0, indices, 0].set(C20)
input_parameters["Ck_0"] = Ck_0

# Target dimensionless wavenumbers to scan.
kx = jnp.arange(0.1, 1.02, 0.02)

# Store the measured growth rate gamma(kx).
growth_rate = jnp.zeros_like(kx)
for j in jnp.arange(len(kx)):
    # Set the box length in x so that the chosen Fourier mode corresponds to
    # this target wavenumber.
    Lx = jnp.sqrt(2) * jnp.pi * alpha_s[0] / kx[j]
    input_parameters["Lx"] = Lx

    # Initialize Fourier components of the electromagnetic fields.
    value  = (dn1 + dn2) * Lx / (4 * jnp.pi * nx * Omega_cs[0])
    Fk_0    = jnp.zeros((6, 1, Nx//2+1, 1), dtype=jnp.complex128).at[0, 0, nx, 0].set(value)
    input_parameters["Fk_0"] = Fk_0

    # Run the simulation (block until the JAX computation is complete).
    output = block_until_ready(simulation(input_parameters, **solver_parameters))

    t = output["time"]
    EM_energy = output["EM_energy"]

    # Estimate gamma by fitting log(EM_energy) in an exponential growth window:
    # log(EM_energy) ~ (2 * gamma) * t + const.
    p = jnp.polyfit(t[300:], jnp.log(EM_energy[300:]), 1)
    growth_rate = growth_rate.at[j].set(p[0] / 2)


# Load the reference/theoretical curve (tabulated).
growth_rate_theo_file = os.path.join(os.path.dirname(__file__), 'growth_rate.csv')
data_growth_rate = pd.read_csv(growth_rate_theo_file, header=None)
growth_rate_theo = jnp.array(data_growth_rate.values)

# Plot measured growth-rate curve against the reference.
plt.figure(figsize=(8, 6))
plt.plot(kx[:40], growth_rate[:40], label=r'$\gamma_{SPECTRAX}$', linestyle='None', marker='o', color='red', linewidth=3.0)
plt.plot(growth_rate_theo[:, 0], growth_rate_theo[:, 1], label=r'$\gamma_{theo}$', linestyle='-', color='blue', linewidth=3.0)
plt.xlim(0.0, 1.0)
plt.ylim(-0.5, 0.5)
plt.ylabel(r'$\gamma/\omega_{pe}$', fontsize=16)
plt.xlabel(r'$kv_{e,th}/\omega_{pe}$', fontsize=16)
plt.title(rf'$\nu = {nu}, u_e = \pm{u_s[0]}, \alpha_e = {alpha_s[0]:.3},'
                +rf'N_x = {Nx}, N_n = {Nn}, \delta n = {dn1}$', fontsize=18)
plt.legend(fontsize=20).set_draggable(True)
plt.show()
