import sys

sys.path.append(r'/Users/csvega/Desktop/Madison/Code/Simulations')
sys.path.append(r'/Users/csvega/Desktop/Madison/Code/Vlasov-Maxwell_Spectral_Solver/Vlasov-MaxwellSpectralSolver')

import json
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from JAX_VM_solver import VM_simulation
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time

   
# Simulation parameters.
Nx, Ny, Nz = 3, 1, 1
Lx, Ly, Lz = 1.0, 1.0, 1.0
Nn, Nm, Np, Ns = 80, 1, 1, 2 
mi_me, Ti_Te = 1836.0, 1.0
Omega_cs = jnp.array([1.0, 1.0 / mi_me])
qs = jnp.array([-1, 1])
alpha_e = jnp.array([0.1, 0.1, 0.1])
alpha_s = jnp.concatenate([alpha_e, alpha_e * jnp.sqrt(Ti_Te / mi_me)])
u_s = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
nu = 1.0
t_steps, t_max, dt = 401, 40, 0.01

# Initial conditions

kx = 2 * jnp.pi / Lx # Wavenumber.
dn = 0.0001 # Density fluctuation.

# Fourier components of magnetic and electric fields.
Fk_0 = jnp.zeros((6, 1, 3, 1), dtype=jnp.complex128)
Fk_0 = Fk_0.at[0, 0, 0, 0].set(dn / (2 * kx * Omega_cs[0]))
Fk_0 = Fk_0.at[0, 0, 2, 0].set(dn / (2 * kx * Omega_cs[0]))
    
# Hermite-Fourier components of electron and ion distribution functions.
Ce0_mk, Ce0_0, Ce0_k = 0 + 1j * (1 / (2 * alpha_s[0] ** 3)) * dn, 1 / (alpha_s[0] ** 3) + 0 * 1j, 0 - 1j * (1 / (2 * alpha_s[0] ** 3)) * dn
Ci0_0 = 1 / (alpha_s[3] ** 3) + 0 * 1j
Ck_0 = jnp.zeros((2 * Nn, 1, 3, 1), dtype=jnp.complex128)
Ck_0 = Ck_0.at[0, 0, 0, 0].set(Ce0_mk)
Ck_0 = Ck_0.at[0, 0, 1, 0].set(Ce0_0)
Ck_0 = Ck_0.at[0, 0, 2, 0].set(Ce0_k)
Ck_0 = Ck_0.at[Nn, 0, 1, 0].set(Ci0_0)

# # Save parameters into txt.
# with open('C:\Cristian\Postdoc\Madison\Code\Simulations\Two_stream_instability_1D_HF\S13\S13_Two_stream_instability_1D_HF.txt', 'w') as file:
#     file.write(f"Nx, Ny, Nz: {Nx}, {Ny}, {Nz}\n")
#     file.write(f"Nvx, Nvy, Nvz: {Nvx}, {Nvy}, {Nvz}\n")
#     file.write(f"Lx, Ly, Lz: {Lx}, {Ly}, {Lz}\n")
#     file.write(f"Nn, Nm, Np, Ns: {Nn}, {Nm}, {Np}, {Ns}\n")
#     file.write(f"mi_me: {mi_me}\n")
#     file.write(f"Omega_cs: {Omega_cs.tolist()}\n")
#     file.write(f"qs: {qs.tolist()}\n")
#     file.write(f"alpha_s: {alpha_s.tolist()}\n")
#     file.write(f"u_s: {u_s.tolist()}\n")
#     file.write(f"nu: {nu}\n")
#     file.write(f"t_steps, t_max: {t_steps}, {t_max}\n")
#     file.write(f"dt: {dt}\n")

# Run simulation.
start_time = time.time()
Ck, Fk, t = VM_simulation(Ck_0, Fk_0, qs, nu, Omega_cs, alpha_s, mi_me, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns, t_max, t_steps, dt)
end_time = time.time()
t.block_until_ready()
print(f"Runtime: {end_time - start_time} seconds")

# # Save results.
# jnp.save('C:\Cristian\Postdoc\Madison\Code\Simulations\Two_stream_instability_1D_HF\S13\Ck', Ck)
# jnp.save('C:\Cristian\Postdoc\Madison\Code\Simulations\Two_stream_instability_1D_HF\S13\Fk', Fk)
# jnp.save('C:\Cristian\Postdoc\Madison\Code\Simulations\Two_stream_instability_1D_HF\S13\\time', t)

####################################################################################################################################################
# Data analysis.

lambda_D = jnp.sqrt(1 / (2 * (1 / alpha_s[0] ** 2 + 1 / (mi_me * alpha_s[3] ** 2)))) # Debye length.
k_norm = jnp.sqrt(2) * jnp.pi * alpha_s[0] / Lx # Perturbation wavenumber normalized to the inverse of the Debye length.

# Set n = 0, k = 0 mode to zero to get array with time evolution of perturbation.
dCk = Ck.at[:, 0, 0, 1, 0].set(0)
dCk = dCk.at[:, Nn * Nm * Np, 0, 1, 0].set(0)

peaks, _ = find_peaks(jnp.abs(dCk[:, 0, 0, 0, 0].imag))
p = jnp.polyfit(t[peaks], jnp.log(jnp.abs(dCk[:, 0, 0, 0, 0].imag[peaks])), 1)

# Compute energy.
electron_energy_Ck = (0.5 * alpha_s[0] * alpha_s[1] * alpha_s[2]) * ((0.5 * (alpha_s[0] ** 2 + alpha_s[1] ** 2 + alpha_s[2] ** 2) + 
                                            (u_s[0] ** 2 + u_s[1] ** 2 + u_s[2] ** 2)) * Ck[:, 0, 0, int((Nx-1)/2), 0] + 
                                    jnp.sqrt(2) * alpha_s[0] * u_s[0] * Ck[:, 1, 0, int((Nx-1)/2), 0] + 
                                    (1 / jnp.sqrt(2)) * (alpha_s[0] ** 2) * Ck[:, 2, 0, int((Nx-1)/2), 0])
        
ion_energy_Ck = (0.5 * alpha_s[3] * alpha_s[4] * alpha_s[5]) * ((0.5 * (alpha_s[3] ** 2 + alpha_s[4] ** 2 + alpha_s[5] ** 2) + 
                                            (u_s[3] ** 2 + u_s[4] ** 2 + u_s[5] ** 2)) * Ck[:, Nn, 0, int((Nx-1)/2), 0] + 
                                    jnp.sqrt(2) * alpha_s[3] * u_s[3] * Ck[:, Nn + 1, 0, int((Nx-1)/2), 0] + 
                                    (1 / jnp.sqrt(2)) * (alpha_s[3] ** 2) * Ck[:, Nn + 2, 0, int((Nx-1)/2), 0])
                                                     
electric_energy_Fk = 0.5 * jnp.sum(jnp.abs(Fk[:, 0, 0, :, 0]) ** 2, axis=-1) * Omega_cs[0] ** 2



fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))

# Energy plots in linear scale.

# Electron energy.
axes1[0,0].plot(t, electron_energy_Ck + ion_energy_Ck, label='Plasma energy', linestyle='-', color='red', linewidth=3.0)
axes1[0,0].set_ylabel(r'plasma energy', fontsize=16)
axes1[0,0].set_xlabel(r'$t\omega_{pe}$', fontsize=16)
axes1[0,0].legend().set_draggable(True)


# Electric energy.
axes1[0,1].plot(t, electric_energy_Fk, label='electric energy', linestyle='-', color='red', linewidth=3.0)
axes1[0,1].set_ylabel(r'electric energy', fontsize=16)
axes1[0,1].set_xlabel(r'$t\omega_{pe}$', fontsize=16)
axes1[0,1].legend().set_draggable(True)


# Total energy.
axes1[0,2].plot(t, electron_energy_Ck + ion_energy_Ck + electric_energy_Fk, label='total energy', linestyle='-', color='red', linewidth=3.0)
axes1[0,2].set_ylabel(r'total energy', fontsize=16)
axes1[0,2].set_xlabel(r'$t\omega_{pe}$', fontsize=16)
axes1[0,2].legend().set_draggable(True)


# Energy plots in log-linear scale.

# Electron energy.
axes1[1,0].plot(t, jnp.log10(electron_energy_Ck + ion_energy_Ck), label='plasma energy', linestyle='-', color='red', linewidth=3.0)
axes1[1,0].set_ylabel(r'$\log_{10}(\text{plasma energy})$', fontsize=16)
axes1[1,0].set_xlabel(r'$t\omega_{pe}$', fontsize=16)
axes1[1,0].legend().set_draggable(True)


# Electric energy.
axes1[1,1].plot(t, jnp.log10(electric_energy_Fk), label='electric energy', linestyle='-', color='red', linewidth=3.0)
axes1[1,1].set_ylabel(r'$log_{10}(\text{electric energy})$', fontsize=16)
axes1[1,1].set_xlabel(r'$t\omega_{pe}$', fontsize=16)
axes1[1,1].legend().set_draggable(True)


# Total energy.
axes1[1,2].plot(t, jnp.log10(electron_energy_Ck + ion_energy_Ck + electric_energy_Fk), label='total energy', linestyle='-', color='red', linewidth=3.0)
axes1[1,2].set_ylabel(r'$log_{10}(\text{total energy})$', fontsize=16)
axes1[1,2].set_xlabel(r'$t\omega_{pe}$', fontsize=16)
axes1[1,2].legend().set_draggable(True)

fig1.suptitle(rf'$kv_{{th,e}}/\omega_{{pe}} = {k_norm:.2}, \nu = {nu}, u_e = {u_s[0]}, \alpha_e = {alpha_s[0]:.3}, N_x = {Nx}, N_n = {Nn}, \delta n = 0.001$', fontsize=14)



fig2, axes2 = plt.subplots(1, 2, figsize=(18, 10))

# Plot electron density fluctuation vs t.
axes2[0].plot(t, jnp.log(jnp.abs(dCk[:, 0, 0, 0, 0].imag)), label='$log(|\delta C^e_{000,k}|)$', linestyle='-', color='red', linewidth=3.0)
axes2[0].plot(t, p[0] * t + p[1], label=f'damping rate = {p[0]:.3}', linestyle='-', color='black', linewidth=3.0)
axes2[0].set_ylabel(r'$log(|\delta C^e_{000,k}|)$', fontsize=16)
axes2[0].set_xlabel(r'$t\omega_{pe}$', fontsize=16)
axes2[0].set_xlim(0.0, t_max)
axes2[0].legend().set_draggable(True)

# Plot ion density fluctuation vs t.
axes2[1].plot(t, jnp.log(jnp.abs(dCk[:, Nn, 0, 0, 0].imag)), label='$log(|\delta C^i_{000,k}|)$', linestyle='-', color='red', linewidth=3.0)
axes2[1].set_ylabel(r'$log(|\delta C^i_{000,k}|)$', fontsize=16)
axes2[1].set_xlabel(r'$t\omega_{pe}$', fontsize=16)
axes2[1].set_xlim(0.0, t_max)
axes2[1].legend().set_draggable(True)

fig2.suptitle(rf'$kv_{{th,e}}/\omega_{{pe}} = {k_norm:.2}, u_e/c = {u_s[0]}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=14)

plt.show()