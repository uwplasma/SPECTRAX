import json
import jax.numpy as jnp
from JAX_VM_solver import VM_simulation
import matplotlib.pyplot as plt
import time


with open('plasma_parameters_Landau_damping_HF_1D.json', 'r') as file:
        parameters = json.load(file)
    
# Unpack parameters.
Nx, Ny, Nz = parameters['Nx'], parameters['Ny'], parameters['Nz']
Nvx, Nvy, Nvz = parameters['Nvx'], parameters['Nvy'], parameters['Nvz']
Lx, Ly, Lz = parameters['Lx'], parameters['Ly'], parameters['Lz']
Nn, Nm, Np, Ns = parameters['Nn'], parameters['Nm'], parameters['Np'], parameters['Ns']
mi_me = parameters['mi_me']
Omega_cs = parameters['Omega_ce'] * jnp.array([1.0, 1.0 / mi_me])
qs = jnp.array(parameters['qs'])
alpha_s = jnp.array(parameters['alpha_s'])
u_s = jnp.array(parameters['u_s'])
nu = parameters['nu']
t_steps, t_max = parameters['t_steps'], parameters['t_max']
    
# Save parameters into txt.
with open('C:\Cristian\Postdoc\Madison\Code\Simulations\Landau_damping_1D_HF_ini\Landau_damping_1D_S28.txt', 'w') as file:
    file.write(f"Nx, Ny, Nz: {Nx}, {Ny}, {Nz}\n")
    file.write(f"Nvx, Nvy, Nvz: {Nvx}, {Nvy}, {Nvz}\n")
    file.write(f"Lx, Ly, Lz: {Lx}, {Ly}, {Lz}\n")
    file.write(f"Nn, Nm, Np, Ns: {Nn}, {Nm}, {Np}, {Ns}\n")
    file.write(f"mi_me: {mi_me}\n")
    file.write(f"Omega_cs: {Omega_cs.tolist()}\n")
    file.write(f"qs: {qs.tolist()}\n")
    file.write(f"alpha_s: {alpha_s.tolist()}\n")
    file.write(f"u_s: {u_s.tolist()}\n")
    file.write(f"nu: {nu}\n")
    file.write(f"t_steps, t_max: {t_steps}, {t_max}\n")

start_time = time.time()
Ck, Fk, t = VM_simulation(qs, nu, Omega_cs, alpha_s, mi_me, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns, t_max, t_steps)
end_time = time.time()

print(f"Runtime: {end_time - start_time} seconds")


####################################################################################################################################################
# Data analysis.

lambda_D = jnp.sqrt(1 / (2 * (1 / alpha_s[0] ** 2 + 1 / (mi_me * alpha_s[3] ** 2))))
k_norm = jnp.sqrt(2) * jnp.pi * alpha_s[0] / Lx

dCek = Ck.at[:, 0, 1, 0, 0].set(0)  
dCik = Ck.at[:, Nn * Nm * Np, 1, 0, 0].set(0)

Ce0002 = jnp.mean(jnp.abs(dCek[:, 0, ...]) ** 2, axis=(-3, -2, -1))
Ci0002 = jnp.mean(jnp.abs(dCik[:, Nn * Nm * Np, ...]) ** 2, axis=(-3, -2, -1))

C2 = jnp.mean(jnp.abs(Ck) ** 2, axis=(-3, -2, -1))
C2 = C2.at[:, 0].set(Ce0002)
C2 = C2.at[:, Nn * Nm * Np].set(Ci0002)

plasma_energy_0_Ck = (0.5 * ((0.5 * (alpha_s[0] ** 2 + alpha_s[1] ** 2 + alpha_s[2] ** 2)) * 
                                        alpha_s[0] * alpha_s[1] * alpha_s[2] * Ck[:, 0, 1, 0, 0].real) + 
                                0.5 * mi_me * ((0.5 * (alpha_s[3] ** 2 + alpha_s[4] ** 2 + alpha_s[5] ** 2)) * 
                                                alpha_s[3] * alpha_s[4] * alpha_s[5] * Ck[:, Nn, 1, 0, 0].real))

plasma_energy_2_Ck = (0.5 * (1 / jnp.sqrt(2)) * (alpha_s[0] ** 2) * Ck[:, 2, 1, 0, 0].real * alpha_s[0] * alpha_s[1] * alpha_s[2] + 
                                0.5 * mi_me * (1 / jnp.sqrt(2)) * (alpha_s[3] ** 2) * Ck[:, Nn + 2, 1, 0, 0].real * alpha_s[3] * alpha_s[4] * alpha_s[5])

electric_energy_Fk = 0.5 * jnp.mean(Fk[:, 0, :, 0, 0] ** 2, axis=-1) * Omega_cs[0] ** 2

# plasma_energy_mov_avg = moving_average(plasma_energy_2_Ck / 3, 101)
# electric_energy_mov_avg = moving_average(electric_energy_Fk, 101)

# Plot |C000|^2 vs t.

plt.figure(figsize=(8, 6))
plt.plot(t[:200], jnp.log10(Ce0002[:200]), label='$log_{10}(|\delta C_{e00}|^2)$', linestyle='-', color='red', linewidth=3.0)
plt.ylabel(r'$\langle|\delta C_{e, 0}|^2\rangle$', fontsize=16)

# plt.plot(t[:100], jnp.log10(Ci0002[:100]), label='$log_{10}(|\delta C_{i00}|^2)$', linestyle='-', color='red', linewidth=3.0)
# plt.ylabel(r'$\langle|\delta C_{i, 0}|^2\rangle$', fontsize=16)

plt.xlabel(r'$t\omega_{pe}$', fontsize=16)

plt.title(rf'$\nu ={nu}, L_x/d_e = {Lx}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=16)
# plt.legend().set_draggable(True)

plt.show()


# Plot |C|^2 vs n vs t.

plt.figure(figsize=(8, 6))
plt.imshow(jnp.log10(C2[:1000, :Nn * Nm * Np]), aspect='auto', cmap='viridis', interpolation='none', origin='lower', extent=(0, Nn, 0, t_max))
plt.colorbar(label=r'$log_{10}(\langle |C_{e,n}|^2\rangle (t))$').ax.yaxis.label.set_size(16)

# plt.imshow(C2[:1000, :Nn * Nm * Np], aspect='auto', cmap='viridis', interpolation='none', origin='lower', extent=(0, Nn, 0, t_max / 10))
# plt.colorbar(label=r'$\langle |C_{e,n}|^2\rangle (t)$').ax.yaxis.label.set_size(16)

# plt.plot(jnp.arange(Nn) + 0.5, 3.6*jnp.sqrt(jnp.arange(Nn)), label='$3.60\sqrt{n}$', linestyle='-', color='black', linewidth=3.0)
plt.xlabel('n', fontsize=16)
plt.ylabel('t', fontsize=16)
plt.title(rf'$\nu ={nu}, L_x/d_e = {Lx}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=16)
# plt.legend()
plt.show()

# Plot energy.

plt.figure(figsize=(8, 6))
# plt.yscale("log")
# plt.plot(t, plasma_energy_2_Ck / 3, label='Plasma energy ($C_{200}$)', linestyle='-', color='red', linewidth=3.0)
# plt.plot(t, (plasma_energy_0_Ck) / 3, label='Plasma energy ($C_{000}$)', linestyle='-', color='red', linewidth=3.0)
# plt.plot(t[9:992], plasma_energy_mov_avg, label='mov_avg(Plasma energy)', linestyle='-', color='black', linewidth=3.0)
# plt.plot(t, electric_energy_Fk, label='Electric energy', linestyle='-', color='blue', linewidth=3.0)
plt.plot(t[:10000], electric_energy_Fk[:10000] + plasma_energy_2_Ck[:10000] / 3, label='Total energy in fluctuations', linestyle='-', color='red', linewidth=3.0)
plt.xlabel(r'$t\omega_{pe}$', fontsize=16)
plt.ylabel(r'Energy', fontsize=16)
# plt.xlim((0,t_max))
# plt.ylim((4,12))
# plt.title(rf'$\nu = {nu}, N_x = {Nx}, N_n = {Nn}$', fontsize=16)
plt.title(rf'$\nu ={nu}, L_x/d_e = {Lx}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=16)
plt.legend().set_draggable(True)

plt.show()


###################################################################################################################################