"""Landau damping rate against wavenumber, compared with the kinetic dispersion relation (damping_rate_1836.csv).

Each run is input_1D_landau_damping.toml with the box length set so that its first mode has k vth/omega_pe = kx.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from spectrax import load_parameters, simulation

here = Path(__file__).parent
vth = load_parameters(here / "input_1D_landau_damping.toml")[0]["alpha_s"][0]
kx = np.arange(0.1, 1.02, 0.02)
rate = []
for k in kx:
    input_parameters, solver_parameters = load_parameters(here / "input_1D_landau_damping.toml", Lx=np.sqrt(2) * np.pi * vth / k)
    output = simulation(input_parameters, **solver_parameters)
    t, W = np.asarray(output["time"])[:100], np.asarray(output["EM_energy"])[:100]
    peaks = [i for i in range(1, len(W) - 1) if W[i - 1] < W[i] > W[i + 1]]
    rate.append(np.polyfit(t[peaks], np.log(W[peaks]), 1)[0] / 2)      # field energy decays at twice the amplitude rate

theory = np.loadtxt(here / "damping_rate_1836.csv", delimiter=",")
plt.plot(kx, rate, "o", color="tab:red", label="SPECTRAX")
plt.plot(theory[:, 0], theory[:, 1], color="tab:blue", label="kinetic theory")
plt.xlim(0, 1), plt.ylim(-1.4, 0.1)
plt.xlabel(r"$k v_{th}/\omega_{pe}$"), plt.ylabel(r"$\gamma/\omega_{pe}$"), plt.legend()
plt.savefig(here / "1D_Landau_damping_dr_curve.png", dpi=200)
plt.show()
