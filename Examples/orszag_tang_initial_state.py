"""Orszag-Tang vortex initial state: fluid velocities and fields, as Hermite moments and Fourier modes."""
import jax.numpy as jnp
from spectrax import compute_C_nmp


def initial_state(input_parameters, solver_parameters):
    deltaB, (Lx, Ly, Lz), Omega_ce = 0.2, (input_parameters[k] for k in ("Lx", "Ly", "Lz")), input_parameters["Omega_cs"][0]
    U0, kx, ky = deltaB * Omega_ce / jnp.sqrt(input_parameters["mi_me"]), 2 * jnp.pi / Lx, 2 * jnp.pi / Ly
    N = solver_parameters
    X, Y, Z = jnp.meshgrid(*(jnp.linspace(0, L, n, endpoint=False) for L, n in ((Lx, N["Nx"]), (Ly, N["Ny"]), (Lz, N["Nz"]))), indexing="xy")
    zero = jnp.zeros_like(X)
    Ue = jnp.array([-U0 * jnp.sin(ky * Y), U0 * jnp.sin(kx * X), -deltaB * Omega_ce * (2 * kx * jnp.cos(2 * kx * X) + ky * jnp.cos(ky * Y))])
    Ui = jnp.array([-U0 * jnp.sin(ky * Y), U0 * jnp.sin(kx * X), zero])
    B = jnp.array([-deltaB * jnp.sin(ky * Y), deltaB * jnp.sin(2 * kx * X), jnp.ones_like(X)])
    return {"Ck_0": compute_C_nmp(jnp.stack([Ue, Ui]), input_parameters["alpha_s"], input_parameters["u_s"], N["Nn"], N["Nm"], N["Np"], N["Ns"]),
            "Fk_0": jnp.fft.rfftn(jnp.concatenate((zero[None].repeat(3, 0), B)), axes=(-1, -3, -2), norm="forward")}
