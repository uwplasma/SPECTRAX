"""Example: 1D bump-on-tail instability, with the beam expanded in the bulk's Hermite basis.

A bulk Maxwellian (density 0.9, thermal speed 1) and a warm beam (density 0.1, thermal speed 0.5,
drift 4.5) are shifted by -0.45 to the frame with no net current, so that SPECTRAX's Vlasov-Ampere
system has no uniform plasma oscillation. Each population is its own species.

The beam is expanded in the bulk's Hermite width (alpha = sqrt(2)), not its own (sqrt(2) * 0.5):
`compute_C_nmp(..., vth_s=...)` gives the closed-form coefficients of a Maxwellian narrower than its
basis. In its own narrow basis the trapped beam, which spreads over 2 < v < 6, has a divergent
expansion: its coefficients grow by 1e2-1e7 and the adaptive step collapses (near t = 28 with nu = 3).
Set BEAM_BASIS_VT = 0.5 to see it. Linear theory: omega = 0.866 + 0.198 i at k = 0.3.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from spectrax import compute_C_nmp, inverse_HF_transform, simulation

jax.config.update("jax_enable_x64", True)
BEAM_BASIS_VT = 1.0  # 0.5: the beam in its own basis (fails)
species = [(0.9, -0.45, 1.0), (0.1, 4.05, 0.5)]  # (density, drift, thermal speed)
k, eps, t_max, Nx, Nn = 0.3, 0.04, 100.0, 33, 128
alpha = np.array([np.sqrt(2.0), np.sqrt(2.0) * BEAM_BASIS_VT])

Ck_0 = []
for (n, u, vth), a in zip(species, alpha):
    C = n * compute_C_nmp(jnp.full((1, 3, 1, Nx, 1), u), [a] * 3, [0.0] * 3, Nn, 1, 1, 1,
                          vth_s=[vth, a / np.sqrt(2), a / np.sqrt(2)])[0, 0, 0]  # (Nn, 1, Nx//2+1, 1)
    Ck_0.append(C.at[:, :, 1].set(0.5 * eps * C[:, :, 0]))  # density n (1 + eps cos kx)
parameters = {
    "Lx": 2 * np.pi / k, "Ly": 1.0, "Lz": 1.0, "mi_me": 1.0, "Ti_Te": 1.0, "nx": 1, "t_max": t_max,
    "qs": jnp.array([-1.0, -1.0]), "Omega_cs": jnp.array([1.0, 1.0]), "nu": 3.0, "D": 0.0, "ode_tolerance": 1e-8,
    "alpha_s": jnp.repeat(jnp.asarray(alpha), 3), "u_s": jnp.array([0.0] * 6), "Ck_0": jnp.concatenate(Ck_0),
    # Gauss's law for the density perturbation: E = -(eps / k) sin(kx).
    "Fk_0": jnp.zeros((6, 1, Nx // 2 + 1, 1), dtype=jnp.complex128).at[0, 0, 1, 0].set(0.5j * eps / k),
}
output = simulation(parameters, Nx=Nx, Nn=Nn, Ns=2, timesteps=501, dt=0.01)

t, E1 = output["time"], 2 * jnp.abs(output["Fk"][:, 0, 0, 1, 0])
x, v = np.linspace(0, 2 * np.pi / k, Nx, endpoint=False), np.linspace(-8, 10, 241)
f = 0
for s, a in enumerate(alpha):  # 1D marginal: f(v_x, v_y = v_z = 0) times the transverse integral a^2 pi
    xi_y, xi_x, xi_z = np.meshgrid([0.0], v / a, [0.0], indexing="ij")
    f += np.pi * a**2 * inverse_HF_transform(output["Ck"][-1:, s * Nn:(s + 1) * Nn], Nn, 1, 1, Nx, 1, 1,
                                             xi_x, xi_y, xi_z)[0, 0, :, 0, 0, :, 0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
ax1.semilogy(t, E1, label="SPECTRAX")
linear = (t > 8) & (t < 20)
ax1.semilogy(t[linear], E1[linear][0] * np.exp(0.198 * (t[linear] - 8)), "k--", label=r"$\gamma = 0.198$")
ax1.set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$|\hat E_{k}|$", title="bump-on-tail, k = 0.3")
ax1.legend()
mesh = ax2.pcolormesh(x, v, np.asarray(f).T, shading="auto", cmap="magma")
ax2.set(xlabel=r"$x/\lambda_D$", ylabel=r"$v/v_t$", title=f"f(x, v, t = {t_max:g})")
fig.colorbar(mesh, ax=ax2)
plt.tight_layout()
plt.show()
