import sys

sys.path.append(r'/Users/csvega/Desktop/Madison/Code/Simulations')
sys.path.append(r'/Users/csvega/Desktop/Madison/Code/Vlasov-Maxwell_Spectral_Solver/Vlasov-MaxwellSpectralSolver')

import json
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from JAX_VM_solver import VM_simulation
from initialize_in_xv_space import initialize_system_xv
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time

   
# Simulation parameters.
Nx, Ny, Nz = 33, 33, 1
Lx, Ly, Lz = 50.0, 50.0, 1.0
Nn, Nm, Np, Ns = 4, 4, 4, 2 
mi_me, Ti_Te = 25.0, 1.0
Omega_cs = jnp.array([0.5, 0.5 / mi_me])
qs = jnp.array([-1, 1])
alpha_e = jnp.array([0.25, 0.25, 0.25])
alpha_s = jnp.concatenate([alpha_e, alpha_e * jnp.sqrt(Ti_Te / mi_me)])
u_s = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
nu = 1.0
t_steps, t_max, dt = 5, 0.4, 0.05



# Initial conditions

vte = alpha_s[0] / jnp.sqrt(2) # Electron thermal velocity.
vti = vte * jnp.sqrt(1 / mi_me) # Ion thermal velocity.
deltaB = 0.2 # In-plane magnetic field amplitude. 
U0 = deltaB * Omega_cs[0] / jnp.sqrt(mi_me) # Fluid velocity amplitude.

# Wavenumbers.
kx = 2 * jnp.pi / Lx
ky = 2 * jnp.pi / Ly

# Electron and ion fluid velocities.
Ue = lambda x, y, z: U0 * jnp.array([-jnp.sin(ky * y), jnp.sin(kx * x), -deltaB * Omega_cs[0] * (2 * kx * jnp.cos(2 * kx * x) + ky * jnp.cos(ky * y))])
Ui = lambda x, y, z: U0 * jnp.array([-jnp.sin(ky * y), jnp.sin(kx * x), jnp.zeros_like(x)])

# Magnetic and electric fields.
B = lambda x, y, z: jnp.array([-deltaB * jnp.sin(ky * y), deltaB * jnp.sin(2 * kx * x), jnp.ones_like(x)])
E = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)]) # Is this consistent with fe, fi?

# Electron and ion distribution functions.
fe = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * vte ** 3) * 
                                jnp.exp(-((vx - Ue(x, y, z)[0])**2 + (vy - Ue(x, y, z)[1])**2 + (vz - Ue(x, y, z)[2])**2) / (2 * vte ** 2))))
fi = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * vti ** 3) * 
                                jnp.exp(-((vx - Ui(x, y, z)[0])**2 + (vy - Ui(x, y, z)[1])**2 + (vz - Ui(x, y, z)[2])**2) / (2 * vti ** 2))))

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

Ck_0, Fk_0 = initialize_system_xv(B, E, fe, fi, Omega_cs[0], mi_me, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns)

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
