"""Two-stream growth rate against wavenumber, compared with the dispersion relation (growth_rate.csv).

Each run is input_1D_two_stream.toml with 3 Fourier modes, a 1e-8 seed, and the box length set so that its
first mode has k vth/omega_pe = kx. The growth rate is fitted to the field energy after the transient.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from spectrax import load_parameters, simulation

here = Path(__file__).parent
vth = load_parameters(here / "input_1D_two_stream.toml")[0]["alpha_s"][0]
kx = np.arange(0.1, 1.02, 0.02)
rate = []
for k in kx:
    input_parameters, solver_parameters = load_parameters(here / "input_1D_two_stream.toml", Lx=np.sqrt(2) * np.pi * vth / k,
                                                          Nx=3, Nn=100, t_max=80.0, timesteps=801,
                                                          species={name: {"perturbation_amplitude": 1e-8} for name in ("beam_right", "beam_left")})
    output = simulation(input_parameters, **solver_parameters)
    t, W = np.asarray(output["time"]), np.asarray(output["EM_energy"])
    rate.append(np.polyfit(t[300:], np.log(W[300:]), 1)[0] / 2)

theory = np.loadtxt(here / "growth_rate.csv", delimiter=",")
plt.plot(kx, rate, "o", color="tab:red", label="SPECTRAX")
plt.plot(theory[:, 0], theory[:, 1], color="tab:blue", label="dispersion relation")
plt.xlim(0, 1), plt.ylim(-0.5, 0.5)
plt.xlabel(r"$k v_{th}/\omega_{pe}$"), plt.ylabel(r"$\gamma/\omega_{pe}$"), plt.legend()
plt.savefig(here / "1D_two_stream_gr_curve.png", dpi=200)
plt.show()
