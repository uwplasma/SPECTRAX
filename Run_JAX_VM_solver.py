import json
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from JAX_VM_solver import VM_simulation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
from jax.numpy.fft import fft, ifftn, fftshift, ifftshift, fftfreq
from jax.scipy.optimize import minimize
import time



with open('plasma_parameters_Kelvin_Helmholtz_2D.json', 'r') as file:
        parameters = json.load(file)
    
# Unpack parameters.
Nx, Ny, Nz = parameters['Nx'], parameters['Ny'], parameters['Nz']
Nvx, Nvy, Nvz = parameters['Nvx'], parameters['Nvy'], parameters['Nvz']
Lx, Ly, Lz = parameters['Lx'], parameters['Ly'], parameters['Lz']
Nn, Nm, Np, Ns = parameters['Nn'], parameters['Nm'], parameters['Np'], parameters['Ns']
mi_me = parameters['mi_me']
Omega_cs = parameters['Omega_ce'] * jnp.array([1.0, 1.0 / mi_me])
qs = jnp.array(parameters['qs'])
alpha_s = jnp.concatenate([jnp.array(parameters['alpha_e']), (jnp.array(parameters['alpha_e']) / jnp.sqrt(mi_me))])
u_s = jnp.array(parameters['u_s'])
nu = parameters['nu']
t_steps, t_max = parameters['t_steps'], parameters['t_max']

# # Save parameters into txt.
# with open('C:\Cristian\Postdoc\Madison\Code\Simulations\Two_stream_instability_1D_HF_ini\S3\Two_stream_instability_1D_S3.txt', 'w') as file:
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

start_time = time.time()
Ck, Fk, t = VM_simulation(qs, nu, Omega_cs, alpha_s, mi_me, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns, t_max, t_steps)
end_time = time.time()

print(f"Runtime: {end_time - start_time} seconds")


####################################################################################################################################################
# 1D Landau damping/two-stream stability: data analysis.

lambda_D = jnp.sqrt(1 / (2 * (1 / alpha_s[0] ** 2 + 1 / (mi_me * alpha_s[3] ** 2))))
k_norm = jnp.sqrt(2) * jnp.pi * alpha_s[0] / Lx

dCk = Ck.at[:, 0, 1, 0, 0].set(0)
dCk = dCk.at[:, Nn * Nm * Np, 1, 0, 0].set(0)

C2 = jnp.mean(jnp.abs(dCk) ** 2, axis={-3, -2, -1})

plasma_energy_0_Ck = (0.5 * ((0.5 * (alpha_s[0] ** 2 + alpha_s[1] ** 2 + alpha_s[2] ** 2)) * 
                                        alpha_s[0] * alpha_s[1] * alpha_s[2] * Ck[:, 0, 1, 0, 0].real) + 
                                0.5 * mi_me * ((0.5 * (alpha_s[3] ** 2 + alpha_s[4] ** 2 + alpha_s[5] ** 2)) * 
                                                alpha_s[3] * alpha_s[4] * alpha_s[5] * Ck[:, Nn, 1, 0, 0].real))

plasma_energy_2_Ck = (0.5 * (1 / jnp.sqrt(2)) * (alpha_s[0] ** 2) * Ck[:, 2, 1, 0, 0].real * alpha_s[0] * alpha_s[1] * alpha_s[2] + 
                                0.5 * mi_me * (1 / jnp.sqrt(2)) * (alpha_s[3] ** 2) * Ck[:, Nn + 2, 1, 0, 0].real * alpha_s[3] * alpha_s[4] * alpha_s[5])

electric_energy_Fk = 0.5 * jnp.mean(Fk[:, 0, :, 0, 0] ** 2, axis=-1) * Omega_cs[0] ** 2

# plasma_energy_mov_avg = moving_average(plasma_energy_2_Ck / 3, 101)
# electric_energy_mov_avg = moving_average(electric_energy_Fk, 101)

# Define the function to fit
def model_function(params, t):
    A, B, omega, gamma = params
    return A * jnp.cos(omega * t) * jnp.exp(-gamma * t) + B

# Define the loss function (mean squared error)
def loss_function(params, t, data):
    predictions = model_function(params, t)
    return jnp.mean((predictions - data) ** 2)

# Initialize parameters (A, B, omega, gamma) with some reasonable guesses
initial_params = jnp.array([5.0, 0.0, 1.0, 0.2])

# Example time array and data array (replace 'data' with your actual data array)

# Minimize the loss function to find the best-fit parameters
optimized_result = minimize(lambda params: loss_function(params, t, dCk[:, 0, 0, 0, 0].imag), initial_params, method='BFGS', tol=1e-9)

# Extract the best-fit parameters
best_fit_params = optimized_result.x
A, B, omega, gamma = best_fit_params


dCek_freq = fftshift(fft(dCk[:, 0, 0, 0, 0].imag))
cos_exp_freq = fftshift(fft(jnp.cos(omega * t) * jnp.exp(-gamma * t)))
freq = fftshift(fftfreq(len(dCk[:, 0, 0, 0, 0].imag), 0.1))
max_index = jnp.argmax(dCek_freq.real)


peaks, _ = find_peaks(jnp.abs(dCk[:, 0, 0, 0, 0].imag))
p = jnp.polyfit(t[peaks], jnp.log(jnp.abs(dCk[:, 0, 0, 0, 0].imag[peaks])), 1)


# Plot |C000| vs t.

plt.figure(figsize=(8, 6))
plt.plot(t[:2000], jnp.log10(jnp.abs(dCk[:2000, 1, 0, 0, 0].imag)), label='$log_{10}(|\delta C_{e000,k}|)$', linestyle='-', color='red', linewidth=3.0)
# plt.plot(t, p[0] * t + p[1], label='$log_{10}(|\delta C_{e00}|^2)$', linestyle='-', color='black', linewidth=3.0)
# plt.plot(t[peaks], jnp.log(jnp.abs(dCek[:, 0, 0, 0, 0].imag[peaks])), label='$log_{10}(|\delta C_{e00}|^2)$', linestyle='None', marker='x', color='blue', linewidth=3.0)
# plt.plot(t, A * jnp.cos(omega * t) * jnp.exp(-gamma * t) + B, label='$A\cos(\omega t)e^{-\gamma t}+B$', linestyle='-', color='blue', linewidth=3.0)
plt.ylabel(r'$log_{10}(|\delta C_{e000,k}|)$', fontsize=16)
# plt.plot(t[:100], jnp.log10(Ci0002[:100]), label='$log_{10}(|\delta C_{i00}|^2)$', linestyle='-', color='red', linewidth=3.0)
plt.xlabel(r'$t\omega_{pe}$', fontsize=16)
# plt.xlim(0.0, t_max)
plt.title(rf'$kv_{{th,e}}/\omega_{{pe}} = {k_norm:.2}, u_e/c = {u_s[0]}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=14)
# plt.legend().set_draggable(True)
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(t[:1000], jnp.log10(jnp.abs(dCk[:1000, Nn * Nm * Np, 0, 0, 0].imag)), label='$log_{10}(|\delta C_{i000,k}|)$', linestyle='-', color='red', linewidth=3.0)
# plt.plot(t, p[0] * t + p[1], label='$log_{10}(|\delta C_{e00}|^2)$', linestyle='-', color='black', linewidth=3.0)
# plt.plot(t[peaks], jnp.log(jnp.abs(dCek[:, 0, 0, 0, 0].imag[peaks])), label='$log_{10}(|\delta C_{e00}|^2)$', linestyle='None', marker='x', color='blue', linewidth=3.0)
# plt.plot(t, A * jnp.cos(omega * t) * jnp.exp(-gamma * t) + B, label='$A\cos(\omega t)e^{-\gamma t}+B$', linestyle='-', color='blue', linewidth=3.0)
plt.ylabel(r'$log_{10}(|\delta C_{i000,k}|)$', fontsize=16)
# plt.plot(t[:100], jnp.log10(Ci0002[:100]), label='$log_{10}(|\delta C_{i00}|^2)$', linestyle='-', color='red', linewidth=3.0)
plt.xlabel(r'$t\omega_{pe}$', fontsize=16)
# plt.xlim(0.0, t_max)
plt.title(rf'$kv_{{th,e}}/\omega_{{pe}} = {k_norm:.2}, u_e/c = {u_s[0]}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=14)
# plt.legend().set_draggable(True)
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(t[:1000], jnp.log10(jnp.abs(dCk[:1000, Nn * Nm * Np, 0, 0, 0].imag * (alpha_s[3] * alpha_s[4] * alpha_s[5]) - 
                                     dCk[:1000, 0, 0, 0, 0].imag * (alpha_s[0] * alpha_s[1] * alpha_s[2]))), 
         label='$log_{10}(|\rho_k|)$', linestyle='-', color='red', linewidth=3.0)
# plt.plot(t, p[0] * t + p[1], label='$log_{10}(|\delta C_{e00}|^2)$', linestyle='-', color='black', linewidth=3.0)
# plt.plot(t[peaks], jnp.log(jnp.abs(dCek[:, 0, 0, 0, 0].imag[peaks])), label='$log_{10}(|\delta C_{e00}|^2)$', linestyle='None', marker='x', color='blue', linewidth=3.0)
# plt.plot(t, A * jnp.cos(omega * t) * jnp.exp(-gamma * t) + B, label='$A\cos(\omega t)e^{-\gamma t}+B$', linestyle='-', color='blue', linewidth=3.0)
plt.ylabel(r'$log_{10}(|\rho_k|)$', fontsize=16)
# plt.plot(t[:100], jnp.log10(Ci0002[:100]), label='$log_{10}(|\delta C_{i00}|^2)$', linestyle='-', color='red', linewidth=3.0)
plt.xlabel(r'$t\omega_{pe}$', fontsize=16)
# plt.xlim(0.0, t_max)
plt.title(rf'$kv_{{th,e}}/\omega_{{pe}} = {k_norm:.2}, u_e/c = {u_s[0]}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=14)
# plt.legend().set_draggable(True)
plt.show()


# Plot C000 frequency spectrum.

plt.figure(figsize=(8, 6))
# plt.plot(2 * jnp.pi * freq, jnp.abs(dCek_freq), label='$|\delta C_{e000,\omega}|$', linestyle='None', marker='x', color='green', linewidth=3.0)
# plt.plot(2 * jnp.pi * freq, dCek_freq.real, label='$Re[\delta C_{e000,\omega}]$', linestyle='None', marker='x', color='blue', linewidth=3.0)
plt.plot(2 * jnp.pi * freq, dCek_freq.imag, label='$Im[\delta C_{e000,\omega}]$', linestyle='None', marker='x', color='red', linewidth=3.0)
# plt.axvline(x = 2 * jnp.pi * 0.24465855, color = 'r', label = '')
plt.ylabel(r'$\hat{\delta C}_{e000,\omega}$', fontsize=16)
# plt.plot(t[:100], jnp.log10(Ci0002[:100]), label='$log_{10}(|\delta C_{i00}|^2)$', linestyle='-', color='red', linewidth=3.0)
plt.xlabel(r'$\omega/\omega_{pe}$', fontsize=16)
# plt.xlim(0.0, t_max)
plt.title(rf'$\nu ={nu}, kv_{{th,e}}/\omega_{{pe}} = {k_norm:.2}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=14)
plt.legend().set_draggable(True)
plt.show()


plt.figure(figsize=(8, 6))
# plt.plot(2 * jnp.pi * freq, jnp.abs(cos_exp_freq), label='$|\delta C_{e000,\omega}|$', linestyle='None', marker='x', color='green', linewidth=3.0)
# plt.plot(2 * jnp.pi * freq, cos_exp_freq.real, label='$Re[\delta C_{e000,\omega}]$', linestyle='None', marker='x', color='blue', linewidth=3.0)
plt.plot(2 * jnp.pi * freq[400:600], cos_exp_freq[400:600].real, label='$Im[\delta C_{e000,\omega}]$', linestyle='None', marker='x', color='red', linewidth=3.0)
# plt.axvline(x = gamma, color = 'r', label = '')
plt.ylabel(r'$Real[\hat{\delta C}_{e000,\omega}]$', fontsize=16)
# plt.plot(t[:100], jnp.log10(Ci0002[:100]), label='$log_{10}(|\delta C_{i00}|^2)$', linestyle='-', color='red', linewidth=3.0)
plt.xlabel(r'$\omega/\omega_{pe}$', fontsize=16)
# plt.xlim(0.0, t_max)
plt.title(rf'$\nu ={nu}, kv_{{th,e}}/\omega_{{pe}} = {k_norm:.2}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=14)
plt.legend().set_draggable(True)
plt.show()


# Plot |C|^2 vs n vs t.
plt.figure(figsize=(8, 6))
plt.imshow(jnp.log10(C2[:1000, :Nn * Nm * Np]), aspect='auto', cmap='viridis', 
           interpolation='none', origin='lower', extent=(0, Nn, 0, 100), vmin=-10, vmax=10)
plt.colorbar(label=r'$log_{10}(\langle |C_{e,n}|^2\rangle (t))$').ax.yaxis.label.set_size(16)

# plt.imshow(C2[:10000, :Nn * Nm * Np], aspect='auto', cmap='viridis', 
# interpolation='none', origin='lower', extent=(0, Nn, 0, 1000))
# plt.colorbar(label=r'$\langle |C_{e,n}|^2\rangle (t)$').ax.yaxis.label.set_size(16)

# plt.plot(jnp.arange(Nn) + 0.5, 3.6*jnp.sqrt(jnp.arange(Nn)), label='$3.60\sqrt{n}$', linestyle='-', color='black', linewidth=3.0)
plt.xlabel('n', fontsize=16)
plt.ylabel('t', fontsize=16)
plt.title(rf'$\nu ={nu}, L_x/d_e = {Lx}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=14)
# plt.legend()
plt.show()


# Plot energy.
plt.figure(figsize=(8, 6))
# plt.yscale("log")
# plt.plot(t, plasma_energy_2_Ck / 3, label='Plasma energy ($C_{200}$)', linestyle='-', color='red', linewidth=3.0)
# plt.plot(t, (plasma_energy_0_Ck) / 3, label='Plasma energy ($C_{000}$)', linestyle='-', color='red', linewidth=3.0)
# plt.plot(t[9:992], plasma_energy_mov_avg, label='mov_avg(Plasma energy)', linestyle='-', color='black', linewidth=3.0)
# plt.plot(t, electric_energy_Fk, label='Electric energy', linestyle='-', color='blue', linewidth=3.0)
plt.plot(t[:], electric_energy_Fk[:] + plasma_energy_2_Ck[:] / 3, label='Total energy in fluctuations', linestyle='-', color='red', linewidth=3.0)
plt.xlabel(r'$t\omega_{pe}$', fontsize=16)
plt.ylabel(r'Energy', fontsize=16)
plt.xlim((0,t_max))
# plt.ylim((4,12))
# plt.title(rf'$\nu = {nu}, N_x = {Nx}, N_n = {Nn}$', fontsize=16)
plt.title(rf'$\nu ={nu}, L_x/d_e = {Lx}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=16)
plt.legend().set_draggable(True)

plt.show()


####################################################################################################################################################
# Kelvin-Helmholtz instability.

F = ifftn(ifftshift(Fk, axes=(-3, -2, -1)), axes=(-3, -2, -1))
E, B = F[:, :3, ...].real, F[:, 3:, ...].real
    
C = ifftn(ifftshift(Ck, axes=(-3, -2, -1)), axes=(-3, -2, -1))

Ce = C[:, :(Nn * Nm * Np), ...].real
Ci = C[:, (Nn * Nm * Np):, ...].real

Uex = (alpha_s[0] / jnp.sqrt(2)) * Ce[:, 1, ...] / Ce[:, 0, ...]
Uey = (alpha_s[1] / jnp.sqrt(2)) * Ce[:, Nn, ...] / Ce[:, 0, ...]
Uix = (alpha_s[3] / jnp.sqrt(2)) * Ci[:, 1, ...] / Ci[:, 0, ...]
Uiy = (alpha_s[4] / jnp.sqrt(2)) * Ci[:, Nn, ...] / Ci[:, 0, ...]

We = jnp.gradient(Uey, Lx / Nx, axis=-3) - jnp.gradient(Uex, Ly / Ny, axis=-2)
Wi = jnp.gradient(Uiy, Lx / Nx, axis=-3) - jnp.gradient(Uix, Ly / Ny, axis=-2)

electron_energy_dens = 0.5 * alpha_s[0] * alpha_s[1] * alpha_s[2] * ((0.5 * (alpha_s[0] ** 2 + alpha_s[1] ** 2 + alpha_s[2] ** 2) + 
                                         (u_s[0] ** 2 + u_s[1] ** 2 + u_s[2] ** 2)) * Ce[:, 0, ...] + 
                                  jnp.sqrt(2) * (alpha_s[0] * u_s[0] * Ce[:, 1, ...] * jnp.sign(Nn - 1) + 
                                                 alpha_s[1] * u_s[1] * Ce[:, Nn, ...] * jnp.sign(Nm - 1) + 
                                                 alpha_s[2] * u_s[2] * Ce[:, Nn * Nm, ...] * jnp.sign(Np - 1)) + 
                                  (1 / jnp.sqrt(2)) * ((alpha_s[0] ** 2) * Ce[:, 2, ...] * jnp.sign(Nn - 1) * jnp.sign(Nn - 2) + 
                                                       (alpha_s[1] ** 2) * Ce[:, 2 * Nn, ...] * jnp.sign(Nm - 1) * jnp.sign(Nm - 2) + 
                                                       (alpha_s[2] ** 2) * Ce[:, 2 * Nn * Nm, ...] * jnp.sign(Np - 1) * jnp.sign(Np - 2)))
    
ion_energy_dens = 0.5 * mi_me * alpha_s[3] * alpha_s[4] * alpha_s[5] * ((0.5 * (alpha_s[3] ** 2 + alpha_s[4] ** 2 + alpha_s[5] ** 2) + 
                                        (u_s[3] ** 2 + u_s[4] ** 2 + u_s[5] ** 2)) * Ci[:, 0, ...] + 
                                jnp.sqrt(2) * (alpha_s[3] * u_s[3] * Ci[:, 1, ...] * jnp.sign(Nn - 1) + 
                                               alpha_s[4] * u_s[4] * Ci[:, Nn, ...] * jnp.sign(Nm - 1) + 
                                               alpha_s[5] * u_s[5] * Ci[:, Nn * Nm, ...] * jnp.sign(Np - 1)) + 
                                (1 / jnp.sqrt(2)) * ((alpha_s[3] ** 2) * Ci[:, 2, ...] * jnp.sign(Nn - 1) * jnp.sign(Nn - 2) + 
                                                     (alpha_s[4] ** 2) * Ci[:, 2 * Nn, ...] * jnp.sign(Nm - 1) * jnp.sign(Nm - 2) + 
                                                     (alpha_s[5] ** 2) * Ci[:, 2 * Nn * Nm, ...] * jnp.sign(Np - 1) * jnp.sign(Np - 2)))
                                

    
plasma_energy = jnp.mean(electron_energy_dens[:, ...], axis=(-3, -2, -1)) + jnp.mean(ion_energy_dens[:, ...], axis=(-3, -2, -1))

EM_energy = (jnp.mean((E[:, 0, ...] ** 2 + E[:, 1, ...] ** 2 + E[:, 2, ...] ** 2 + 
                       B[:, 0, ...] ** 2 + B[:, 1, ...] ** 2 + B[:, 2, ...] ** 2), axis=(-3, -2, -1)) * Omega_cs[0] ** 2 / 2)



plt.figure(figsize=(8, 6))
plt.imshow(We[0, ...], aspect='auto', cmap='viridis', 
           interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))#, vmin=-10, vmax=10)
plt.colorbar(label=r'$U_{ex}$').ax.yaxis.label.set_size(16)

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
im = ax.imshow(We[0], cmap='viridis', interpolation='nearest')

# Add a color bar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("$U_{ex}/c$") 

# Set up the plot aesthetics
title = ax.set_title("Frame 0")
ax.axis('off')  # Optional: turn off axes for a cleaner look

# Update function for the animation
def update(frame):
    im.set_array(We[frame])
    title.set_text(f"Frame {frame}")
    return [im, title]

# Create the animation
anim = FuncAnimation(
    fig, update, frames=We.shape[0], interval=50, blit=True  # Adjust interval as needed
)

# Display the animation
plt.show()



plt.figure(figsize=(8, 6))
plt.plot(t, plasma_energy, label='plasma energy', linestyle='-', color='red', linewidth=3.0)
# plt.plot(t, p[0] * t + p[1], label='$log_{10}(|\delta C_{e00}|^2)$', linestyle='-', color='black', linewidth=3.0)
# plt.plot(t[peaks], jnp.log(jnp.abs(dCek[:, 0, 0, 0, 0].imag[peaks])), label='$log_{10}(|\delta C_{e00}|^2)$', linestyle='None', marker='x', color='blue', linewidth=3.0)
# plt.plot(t, A * jnp.cos(omega * t) * jnp.exp(-gamma * t) + B, label='$A\cos(\omega t)e^{-\gamma t}+B$', linestyle='-', color='blue', linewidth=3.0)
# plt.ylabel(r'$log_{10}(|\delta C_{e000,k}|)$', fontsize=16)
# plt.plot(t[:100], jnp.log10(Ci0002[:100]), label='$log_{10}(|\delta C_{i00}|^2)$', linestyle='-', color='red', linewidth=3.0)
# plt.xlabel(r'$t\omega_{pe}$', fontsize=16)
# plt.xlim(0.0, t_max)
# plt.title(rf'$kv_{{th,e}}/\omega_{{pe}} = {k_norm:.2}, u_e/c = {u_s[0]}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=14)
# plt.legend().set_draggable(True)
plt.show()