"""Example: 2D Orszag-Tang vortex on several local devices (spatial, species or hybrid sharding).

Same physics as ``2D_Orszag_Tang.py``; the device layout comes from the ``[parallel]`` table of the input TOML
(default ``input_OT_parallel.toml``, 32x32 cells, four CPU devices split 2x2 in space). Run it in a fresh process:

    python Examples/OT_parallel.py [input.toml] [--plot] [--output out.npz] [--no-progress]

``load_parameters`` configures the devices, so it must run before any JAX array is created. The simulation stays
distributed on the devices; ``--plot`` transfers only compact results (energies and the final total Jz slice) and
``--output`` gathers the saved histories to the host ONCE (host memory ~ the printed size).
"""
import argparse
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import factorial
from diffrax import NoProgressMeter
from spectrax import load_parameters, simulation, DG_project

# Global input parameters.
INPUT_TOML = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input_OT_parallel.toml')
deltaB = 0.2  # In-plane magnetic field amplitude (as in 2D_Orszag_Tang.py).


def initial_state(input_parameters, solver_parameters):
    """Orszag-Tang Hermite-DG initial state; sets ``Ck_0`` (Ns,Np,Nm,Nn,Ny,Nx,Nz,Nl) and ``Fk_0`` (6,Ny,Nx,Nz,Nl)."""
    Omega_ce, ms = input_parameters["Omega_ce"], input_parameters["ms"]
    Ns, Nn, Nm, Np = (solver_parameters[k] for k in ("Ns", "Nn", "Nm", "Np"))
    U0 = deltaB * Omega_ce / np.sqrt(ms[1] / ms[0])
    kx, ky = 2 * np.pi / input_parameters["Lx"], 2 * np.pi / input_parameters["Ly"]
    alpha = np.array(input_parameters["alpha_s"]).reshape(Ns, 3)
    alpha_x, alpha_y, alpha_z = (alpha[:, i, None, None, None, None, None, None, None, None, None] for i in range(3))

    # Electron (species 0) and ion (species 1) fluid velocities; the species dependence is BROADCAST (never stacked along
    # the species axis, which would make XLA insert all-to-alls into the species-sharded projection).
    electron = (np.arange(Ns) == 0).reshape(Ns, *[1] * 9)
    Uez = lambda x, y, z: -deltaB * Omega_ce * (2 * kx * jnp.cos(2 * kx * x) + ky * jnp.cos(ky * y))
    U_x = lambda x, y, z: (-U0 * jnp.sin(ky * y))[None, None, None, None]
    U_y = lambda x, y, z: (U0 * jnp.sin(kx * x))[None, None, None, None]
    U_z = lambda x, y, z: electron * Uez(x, y, z)[None, None, None, None]
    # Magnetic and electric fields.
    B = lambda x, y, z: jnp.array([-deltaB * jnp.sin(ky * y), deltaB * jnp.sin(2 * kx * x), jnp.ones_like(x)])
    E = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)])

    p = np.arange(Np)[None, :, None, None, None, None, None, None, None, None]
    m = np.arange(Nm)[None, None, :, None, None, None, None, None, None, None]
    n = np.arange(Nn)[None, None, None, :, None, None, None, None, None, None]

    C0 = lambda x, y, z: (jnp.sqrt(2 ** (n + m + p) / (factorial(n) * factorial(m) * factorial(p)))
         * (1 / (alpha_x ** (n + 1) * alpha_y ** (m + 1) * alpha_z ** (p + 1)))
         * U_x(x, y, z) ** n * U_y(x, y, z) ** m * U_z(x, y, z) ** p)
    F0 = lambda x, y, z: jnp.concatenate([E(x, y, z), B(x, y, z)])

    input_parameters["Ck_0"] = DG_project(C0, input_parameters, solver_parameters, species=True)  # species-sharded
    input_parameters["Fk_0"] = DG_project(F0, input_parameters, solver_parameters)  # spatially sharded


@partial(jax.jit, static_argnames=("Ns", "Nn", "Nm", "Np"))
def total_Jz(Ck, qs, alpha_s, u_s, Ns, Nn, Nm, Np):
    """Cell-mean Jz (Ny,Nx) at the last saved time from the saved Ck (Nt,H,Ny,Nx,Nz,Nl), summed over species ON DEVICE."""
    C = Ck[-1].reshape(Ns, Np * Nm * Nn, *Ck.shape[2:])[..., 0, 0]  # (Ns, Nv, Ny, Nx)
    a, u = alpha_s.reshape(Ns, 3), u_s.reshape(Ns, 3)
    C1 = C[:, Nn * Nm] if Np > 1 else 0.0
    return jnp.einsum('s,syx->yx', qs * a[:, 0] * a[:, 1] * a[:, 2], a[:, 2, None, None] * C1 / jnp.sqrt(2) + u[:, 2, None, None] * C[:, 0])


def describe(name, a):
    print(f"  {name:5s} global {tuple(a.shape)}  local {tuple(a.sharding.shard_shape(a.shape))}  spec {getattr(a.sharding, 'spec', '-')}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("input", nargs="?", default=INPUT_TOML, help="input TOML (default: input_OT_parallel.toml)")
    parser.add_argument("--plot", action="store_true", help="save OT_parallel_energy.png and OT_parallel_Jz.png (compact data only)")
    parser.add_argument("--output", metavar="PATH", help="gather the saved histories once and write them to this .npz")
    parser.add_argument("--no-progress", action="store_true", help="disable the progress bar (headless runs)")
    args = parser.parse_args(argv)

    print(f"Loading {args.input} and configuring devices...")
    input_parameters, solver_parameters = load_parameters(args.input)  # must precede any JAX array
    initial_state_start = time.time()
    initial_state(input_parameters, solver_parameters)
    jax.block_until_ready((input_parameters["Ck_0"], input_parameters["Fk_0"]))
    mesh = input_parameters["Ck_0"].sharding.mesh
    print(f"Backend {mesh.devices.flat[0].platform}, {mesh.devices.size} devices {[d.id for d in mesh.devices.flat]}; "
          f"mesh {dict(mesh.shape)} (axes {mesh.axis_names}), species_partitions ps={mesh.shape['species']}")
    describe("Ck_0", input_parameters["Ck_0"])
    describe("Fk_0", input_parameters["Fk_0"])
    print(f"Initial state projected in {time.time() - initial_state_start:.2f} s")

    print("Starting simulation (the first call includes JIT compilation; this can take a while)...")
    if args.no_progress:
        solver_parameters["progress_meter"] = NoProgressMeter()
    start = time.time()
    output = simulation(input_parameters, **solver_parameters)
    jax.block_until_ready((output["Ck"], output["Fk"]))
    print(f"Simulation finished in {time.time() - start:.2f} s with {int(output['num_steps'])} integration steps")
    describe("Ck", output["Ck"])
    describe("Fk", output["Fk"])
    E = np.asarray(output["total_energy"])  # (Nt,): compact diagnostics, computed on device
    print(f"Total energy {E[0]:.8e} -> {E[-1]:.8e} (relative change {abs(E[-1] - E[0]) / abs(E[0]):.2e}); "
          f"final field energy {float(output['EM_energy'][-1]):.8e}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.asarray(output["time"])
        Jz = np.asarray(total_Jz(output["Ck"], output["qs"], output["alpha_s"], output["u_s"], *(solver_parameters[k] for k in ("Ns", "Nn", "Nm", "Np"))))
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in ("EM_energy", "kinetic_energy", "total_energy"):
            ax.semilogy(t, np.asarray(output[key]), label=key)
        ax.set_xlabel(r"$t\,\omega_{pe}$")
        ax.legend()
        fig.savefig("OT_parallel_energy.png", dpi=150, bbox_inches="tight")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(Jz, origin="lower", cmap="jet", extent=(0, input_parameters["Lx"], 0, input_parameters["Ly"]))
        fig.colorbar(im, label=r"$J_z$")
        ax.set_title(f"t = {t[-1]:g}")
        fig.savefig("OT_parallel_Jz.png", dpi=150, bbox_inches="tight")
        print("Saved OT_parallel_energy.png and OT_parallel_Jz.png")

    if args.output:
        # One explicit gather. Ck_0/Fk_0 duplicate the first saved sample, so they are skipped; Fk is exported once.
        data = {k: np.asarray(v) for k, v in output.items()
                if k not in ("Ck_0", "Fk_0") and isinstance(v, (jax.Array, np.ndarray, int, float))}
        print(f"Writing {sum(v.nbytes for v in data.values()) / 1e6:.1f} MB gathered to host memory to {args.output}...")
        np.savez(args.output, **data)
    return output


if __name__ == "__main__":
    main()
