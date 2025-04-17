import matplotlib.pyplot as plt
import jax.numpy as jnp
from _inverse_transform import inverse_HF_transform
from matplotlib.animation import FuncAnimation

__all__ = ['plot']

def plot(output):
    time = output["time"]; k_norm = output["k_norm"]; Nm = output["Nm"]
    u_s = output["u_s"]; alpha_s = output["alpha_s"]; nu = output["nu"]
    Lx = output["Lx"]; Nx = output["Nx"]; Nn = output["Nn"]; 
    Np = output["Np"]; dn1 = output["dn1"]; Ck = output["Ck"]
    
    # Setup plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    fig.suptitle(rf'$kv_{{th,e}}/\omega_{{pe}} = {k_norm:.2},'
                 +rf'\nu = {nu}, u_e = {u_s[0]}, \alpha_e = {alpha_s[0]:.3},'
                 +rf'N_x = {Nx}, N_n = {Nn}, \delta n = {dn1}$', fontsize=14)
    
    # Energy plots
    axes[0, 0].plot(time, output["electric_field_energy"], label="Electric field energy")
    axes[0, 0].plot(time, output["kinetic_energy"], label="Kinetic energy")
    axes[0, 0].plot(time, output["kinetic_energy_species1"], label="Kinetic energy species 1")
    axes[0, 0].plot(time, output["kinetic_energy_species1"], label="Kinetic energy species 2")
    axes[0, 0].plot(time, output["total_energy"], label="Total energy")
    axes[0, 0].set(title="Energy", xlabel=r"Time ($\omega_{pe}^{-1}$)", ylabel="Energy (J)", yscale="log")#, ylim=[1e-5, None])
    axes[0, 0].legend()
    
    # Plot electron density fluctuation vs t.
    axes[1, 0].plot(time, (jnp.abs(output["dCk"][:, 0, 0, 0, 0].imag)), label='$|\delta C^e_{000,k}|$', linestyle='-', linewidth=2.0)
    axes[1, 0].set(title='Species 1 density fluctuation', ylabel=r'$log(|\delta C^e_{000,k}|)$', xlabel=r'$t\omega_{pe}$', yscale="log")
    
    # Plot ion density fluctuation vs t.
    axes[1, 1].plot(time, (jnp.abs(output["dCk"][:, Nn, 0, 0, 0].imag)), label='$|\delta C^i_{000,k}|$', linestyle='-', linewidth=2.0)
    axes[1, 1].set(title='Species 2 density fluctuation', ylabel=r'$log(|\delta C^i_{000,k}|)$', xlabel=r'$t\omega_{pe}$', yscale="log")
    
    # Phase space plot
    vx = jnp.linspace(-4 * alpha_s[0], 4 * alpha_s[0], 201)
    Vx, Vy, Vz = jnp.meshgrid(vx, jnp.array([0.]), jnp.array([0.]), indexing='xy')
    f = sum(inverse_HF_transform(Ck[:600, i*Nn:(i+1)*Nn, ...], Nn, Nm, Np, 
                                  (Vx - u_s[3*i]) / alpha_s[3*i], 
                                  (Vy - u_s[3*i+1]) / alpha_s[3*i+1], 
                                  (Vz - u_s[3*i+2]) / alpha_s[3*i+2]) for i in range(2))
    electron_phase_plot = axes[0, 1].imshow(jnp.transpose(f[0, 0, :, 0, 0, :, 0]), extent=(0, Lx, vx[0], vx[-1]),
                                            cmap='plasma', origin='lower', interpolation='sinc')
    plt.colorbar(electron_phase_plot, ax=axes[0, 1], label="$f$")
    axes[0, 1].set(xlabel="x/d_e", ylabel="v/c")
    electron_phase_text = axes[0, 1].text(
        0.5, 0.9, "", transform=axes[0, 1].transAxes, ha="center", va="top",
        fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    def update(frame):
        electron_phase_plot.set_array(jnp.transpose(f[frame, 0, :, 0, 0, :, 0]))
        electron_phase_plot.set_clim(vmin=f[frame].min(), vmax=f[frame].max())
        electron_phase_text.set_text(f"Time: {time[frame]:.1f} * ωₚ")
        return [electron_phase_plot, electron_phase_text]
    
    ani = FuncAnimation(fig, update, frames=len(time), blit=True, interval=1, repeat_delay=1000)

    plt.show()
