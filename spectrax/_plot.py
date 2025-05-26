import matplotlib.pyplot as plt
import jax.numpy as jnp
from ._inverse_transform import inverse_HF_transform
from matplotlib.animation import FuncAnimation
__all__ = ['plot']

def plot(output):
    time = output["time"]; k_norm = output["k_norm"]
    u_s = output["u_s"]; alpha_s = output["alpha_s"]; nu = output["nu"]
    Lx = output["Lx"]; Ly = output["Ly"]; Lz = output["Lz"]
    Nx = output["Nx"]; Ny = output["Ny"]; Nz = output["Nz"]
    Nn = output["Nn"]; Nm = output["Nm"]; Np = output["Np"]; Ns = output["Ns"]
    dn1 = output["dn1"]; Ck = output["Ck"]; dCk = output["dCk"]
    nx = output["nx"] if Nx > 1 else 0
    ny = output["ny"] if Ny > 1 else 0
    nz = output["nz"] if Nz > 1 else 0

    print(Ck.shape)
    if dCk is not None:
        any_zeros_in_dCk = jnp.any(dCk == 0)
        all_zeros_in_dCk = jnp.all(dCk == 0)
        num_total_elements_dCk = dCk.size
        num_zero_elements_dCk = jnp.sum(dCk == 0)

        print("\n--- dCk Zero Check ---")
        print(f"Shape of dCk: {dCk.shape}")
        print(f"Does dCk contain any zero elements? {any_zeros_in_dCk}")
        print(f"Are all elements in dCk zero? {all_zeros_in_dCk}")
        if any_zeros_in_dCk and not all_zeros_in_dCk:
            print(f"Number of zero elements in dCk: {num_zero_elements_dCk} / {num_total_elements_dCk}")
        elif all_zeros_in_dCk:
            print(f"All {num_total_elements_dCk} elements in dCk are zero.")
        print("--- End dCk Zero Check ---\n")


    ncps = jnp.array([Nn[s_idx] * Nm[s_idx] * Np[s_idx] for s_idx in range(Ns)])
    offsets = jnp.cumsum(jnp.concatenate([jnp.array([0]), ncps[:-1]]))

    # Setup plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    display_Nn_in_title = Nn[0] if isinstance(Nn, (tuple, list)) and Ns > 0 else Nn
    fig.suptitle(rf'$k \lambda_D \approx {k_norm:.2f}, '
                 + rf'\nu = {nu:.2e}, u_e = {u_s[0]:.2f}, \alpha_e = {alpha_s[0]:.3f}, '
                 + rf'N_x = {Nx}, N_n (sp0) = {display_Nn_in_title}, \delta n = {dn1}$', fontsize=14)

    # Energy plots
    axes[0, 0].plot(time, output["electric_field_energy"], label="Electric Field Energy")
    axes[0, 0].plot(time, output["kinetic_energy"], label="Kinetic Energy")
    axes[0, 0].plot(time, output["kinetic_energy_species1"], label="Kinetic Energy Species 1")
    axes[0, 0].plot(time, output["kinetic_energy_species2"], label="Kinetic Energy Species 2")
    axes[0, 0].plot(time, output["total_energy"], label="Total Energy")
    axes[0, 0].set(title="Energy", xlabel=r"Time ($\omega_{pe}^{-1}$)", ylabel="Energy (J)", yscale="log")#, ylim=[1e-5, None])
    axes[0, 0].legend()

    # Relative Energy Error
    axes[1, 0].plot(time[1:], jnp.abs(output["total_energy"][1:]-output["total_energy"][0])/(output["total_energy"][0]+1e-9), label="Relative energy error")
    axes[1, 0].set(xlabel=r"Time ($\omega_{pe}^{-1}$)", ylabel="Relative Energy Error", yscale="log")#, ylim=[1e-5, None])


    # Plot electron density fluctuation vs t.
    val = output["dCk"][:, 0, int((Ny - 1) / 2) + ny, int((Nx - 1) / 2) + nx, int((Nz - 1) / 2) + nz].imag
    dnk1 = jnp.abs(val * alpha_s[0] * alpha_s[1] * alpha_s[2])
    axes[1, 1].plot(time, dnk1, label='$|\delta n^{S1}_{k}|$', linestyle='-', linewidth=2.0)
    axes[1, 1].set(title='Species 1 density fluctuation', ylabel=r'$log(|\delta n^{s1}_{k}|)$', xlabel=r'$t\omega_{pe}$', yscale="log")#, ylim=[1e-20, None])

    # Plot ion density fluctuation vs t.
    val = output["dCk"][:, offsets[1], int((Ny - 1) / 2) + ny, int((Nx - 1) / 2) + nx, int((Nz - 1) / 2) + nz].imag
    dnk2 = jnp.abs(val * alpha_s[0] * alpha_s[1] * alpha_s[2])
    axes[1, 2].plot(time, dnk2, label='$|\delta n^{s2}_{k}|$', linestyle='-', linewidth=2.0)
    axes[1, 2].set(title='Species 2 density fluctuation', ylabel=r'$log(|\delta n^{s2}_{k}|)$', xlabel=r'$t\omega_{pe}$', yscale="log")#, ylim=[1e-20, None])

    # --- Electron Phase space plot (species i=0) ---
    i=0
    start = offsets[i]
    end = start + ncps[i]
    Ck_s = Ck[:, start:end, ...]
    vx = jnp.linspace(-4 * alpha_s[0], 4 * alpha_s[0], 201)
    Vx, Vy, Vz = jnp.meshgrid(vx, jnp.array([0.]), jnp.array([0.]), indexing='xy')
    f1 = inverse_HF_transform(Ck_s, int(Nn[i]), int(Nm[i]), int(Np[i]),
                                  (Vx - u_s[3*i]) / alpha_s[3*i],
                                  (Vy - u_s[3*i+1]) / alpha_s[3*i+1],
                                  (Vz - u_s[3*i+2]) / alpha_s[3*i+2])
    electron_phase_plot = axes[0, 1].imshow(jnp.transpose(f1[0, 0, :, 0, 0, :, 0]), extent=(0, Lx, vx[0], vx[-1]),
                                            cmap='jet', origin='lower', interpolation='sinc')
    plt.colorbar(electron_phase_plot, ax=axes[0, 1], label="$f_1$")
    axes[0, 1].set(xlabel="x/d_e", ylabel="v/c", title="Species 1 Phase Space")
    axes[0, 1].set_aspect('auto', adjustable='box')
    electron_phase_text = axes[0, 1].text(
        0.5, 0.9, "", transform=axes[0, 1].transAxes, ha="center", va="top",
        fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Ion Phase space plot
    i = 1
    start = offsets[i]
    end = start + ncps[i]
    Ck_s = Ck[:, start:end, ...]
    vx = jnp.linspace(-4 * alpha_s[0], 4 * alpha_s[0], 201)
    Vx, Vy, Vz = jnp.meshgrid(vx, jnp.array([0.]), jnp.array([0.]), indexing='xy')
    f2 = inverse_HF_transform(Ck_s, int(Nn[i]), int(Nm[i]), int(Np[i]),
                              (Vx - u_s[3 * i]) / alpha_s[3 * i],
                              (Vy - u_s[3 * i + 1]) / alpha_s[3 * i + 1],
                              (Vz - u_s[3 * i + 2]) / alpha_s[3 * i + 2])
    ion_phase_plot = axes[0, 2].imshow(jnp.transpose(f2[0, 0, :, 0, 0, :, 0]), extent=(0, Lx, vx[0], vx[-1]),
                                       cmap='jet', origin='lower', interpolation='sinc')
    plt.colorbar(ion_phase_plot, ax=axes[0, 2], label="$f_2$")
    axes[0, 2].set(xlabel="x/d_e", ylabel="v/c", title="Species 2 Phase Space")
    axes[0, 2].set_aspect('auto', adjustable='box')
    ion_phase_text = axes[0, 2].text(
        0.5, 0.9, "", transform=axes[0, 2].transAxes, ha="center", va="top",
        fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    def update(frame):
        electron_phase_plot.set_array(jnp.transpose(f1[frame, :, 0, 0, :, 0, 0]))
        electron_phase_plot.set_clim(vmin=f1[frame].min(), vmax=f1[frame].max())
        electron_phase_text.set_text(f"Time: {time[frame]:.1f} * ωₚ")

        ion_phase_plot.set_array(jnp.transpose(f2[frame, :, 0, 0, :, 0, 0]))
        ion_phase_plot.set_clim(vmin=f2[frame].min(), vmax=f2[frame].max())
        ion_phase_text.set_text(f"Time: {time[frame]:.1f} * ωₚ")

        return [electron_phase_plot, electron_phase_text, ion_phase_plot, ion_phase_text]
    global ani
    ani = FuncAnimation(fig, update, frames=len(time), blit=True, interval=1, repeat_delay=1000)

    dC2 = jnp.mean(jnp.abs(dCk) ** 2, axis=(-3, -2, -1))[:, offsets[0]:offsets[0] + Nn[0]]

    plt.figure(figsize=(8, 6))
    plt.imshow(jnp.log10(dC2), aspect='auto', cmap='viridis',
                   interpolation='none', origin='lower',
                   extent=(0, time[-1], 0, Nn[0]), vmin=-10, vmax=10)
    plt.colorbar(label=r'$log_{10}(\langle |C_{n,0,0}|^2\rangle)$').ax.yaxis.label.set_size(16)
    plt.title(r"$\langle |C_{n,0,0}|^2 \rangle$ vs Time and n", fontsize=16)
    plt.xlabel(r"Time ($\omega_{pe}^{-1}$)")
    plt.ylabel("n")
    plt.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()
