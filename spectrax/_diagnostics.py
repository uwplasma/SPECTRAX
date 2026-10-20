"""Post-processing diagnostics for SPECTRAX simulation outputs.

The main entry point is :func:`diagnostics`, which takes the dictionary returned
by :func:`spectrax.simulation.simulation` and adds commonly used derived
quantities (energy diagnostics and normalization constants).

This implementation is *multi-species*: it computes kinetic energy per species
from low-order Hermite moments and stores the result in a single array
``kinetic_energy_species`` with shape ``(Nt, Ns)``.

For backwards compatibility with older plotting scripts, it also writes
``kinetic_energy_species1``, ``kinetic_energy_species2``, ... as separate keys
when possible.
"""

from __future__ import annotations

from typing import Any, Mapping

from functools import partial

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

__all__ = ["diagnostics"]


def _infer_Ns(output: Mapping[str, Any]) -> int:
    """Infer the number of species ``Ns`` from ``output["u_s"]``.

    Notes
    -----
    SPECTRAX packs species velocity drifts as a flat array:

        ``u_s = [u_x^0, u_y^0, u_z^0, u_x^1, u_y^1, u_z^1, ...]``

    so ``len(u_s) == 3 * Ns``.
    """
    u_s = output["u_s"]
    # `u_s.shape` is static information under JAX tracing.
    n = int(u_s.size)
    if (n % 3) != 0:
        raise ValueError(f"Cannot infer Ns from u_s of length {n}; expected multiple of 3.")
    return n // 3

@partial(jax.jit, static_argnames=("Ns", "Nn", "Nm", "Np"))
def _energies(Ck, Fk, alpha_s, u_s, ms, Omega_ce, basis_idx, Ns, Nn, Nm, Np):
    """Per-species kinetic energy ``(Nt, Ns)`` and field energy ``(Nt,)`` as global reductions.

    Only the cell-mean (DG mode 0) of a few low-order Hermite modes is read and reduced over space, so the sharded
    ``Ck`` history is never gathered: the compiler adds small all-reduces over the spatial mesh axes (``(Nt, Ns)``
    payload). ``Fk`` is replicated across species groups, and a global sum counts it once.
    """
    Nt, Htot, Ny, Nx, Nz, Nl = Ck.shape
    Ck_rs = Ck.reshape(Nt, Ns, Htot // Ns, Ny, Nx, Nz, Nl)  # whole species blocks, so ownership is preserved

    def _take_mode(n, m, p):
        """Domain average of the cell-mean coefficient ``C_{nmp}`` per species, ``(Nt, Ns)``; zero if not retained."""
        if n >= Nn or m >= Nm or p >= Np:
            return jnp.zeros((Nt, Ns), Ck.dtype)
        return jnp.sum(Ck_rs[:, :, n + Nn * m + Nn * Nm * p, :, :, :, 0], axis=(-3, -2, -1)) / (Nx * Ny * Nz)

    C000, C100, C010, C001 = _take_mode(0, 0, 0), _take_mode(1, 0, 0), _take_mode(0, 1, 0), _take_mode(0, 0, 1)
    C200, C020, C002 = _take_mode(2, 0, 0), _take_mode(0, 2, 0), _take_mode(0, 0, 2)

    a0, a1, a2 = alpha_s.reshape(Ns, 3).T
    u0, u1, u2 = u_s.reshape(Ns, 3).T
    pref = 0.5 * ms * a0 * a1 * a2  # (Ns,)

    term0 = (0.5 * (a0**2 + a1**2 + a2**2) + (u0**2 + u1**2 + u2**2)) * C000  # (Nt, Ns)
    term1 = jnp.sqrt(2.0) * (a0 * u0 * C100 + a1 * u1 * C010 + a2 * u2 * C001)  # (Nt, Ns)
    term2 = (1.0 / jnp.sqrt(2.0)) * (a0**2 * C200 + a1**2 * C020 + a2**2 * C002)  # (Nt, Ns)

    # Field energy, normalized by the global cell count.
    EM_energy = 0.5 * Omega_ce ** 2 * jnp.sum(jnp.abs(Fk) ** 2 / ((1 + 2 * basis_idx[:, 0]) * (1 + 2 * basis_idx[:, 1]) * (1 + 2 * basis_idx[:, 2])), axis=(-5, -4, -3, -2, -1)) / (Nx * Ny * Nz)
    return pref * (term0 + term1 + term2), EM_energy

def diagnostics(output: dict) -> None:
    """Compute and attach common diagnostics to ``output`` (mutates in-place).

    Parameters
    ----------
    output : dict
        Dictionary holding simulation results and parameters. This function
        expects at least the following keys:

        - ``alpha_s``: array of shape ``(3*Ns,)`` (thermal scales per species)
        - ``u_s``: array of shape ``(3*Ns,)`` (drift velocities per species)
        - ``qs``: species charges (normalized to electron charge)
        - ``ms`` for general per-species mass scaling
        - ``Ck``: Hermite-Fourier coefficients of shape ``(Nt, Ns*Hs, Ny, Nx, Nz, Nl)``
        - ``Fk``: field Fourier coefficients of shape ``(Nt, 6, Ny, Nx, Nz, Nl)``
        - ``Omega_ce``: electron cyclotron frequency
        - ``Lx``: domain length in x (used for ``k_norm``)

        Optionally, it may use:
        - ``Nn``, ``Nm``, ``Np`` to decode flattened Hermite indices

    Added keys
    ----------
    The following keys are added to ``output``:

    - ``Ns``: inferred number of species
    - ``lambda_D``: Debye length used for normalization (effective for ``Ns>2``)
    - ``k_norm``: normalized perturbation wavenumber (legacy convention)
    - ``kinetic_energy_species``: array of shape ``(Nt, Ns)``
    - ``kinetic_energy``: array of shape ``(Nt,)``
    - ``EM_energy``: array of shape ``(Nt,)``
    - ``total_energy``: array of shape ``(Nt,)``
    - ``kinetic_energy_species{j}`` for ``j=1..Ns`` (back-compat convenience keys)

    Returns
    -------
    None
        ``output`` is mutated in-place.
    """
    # Required arrays
    alpha_s = jnp.asarray(output["alpha_s"])
    u_s = jnp.asarray(output["u_s"])
    qs = jnp.asarray(output["qs"])
    ms = jnp.asarray(output["ms"])
    Omega_ce = output["Omega_ce"]
    Fk = jnp.asarray(output["Fk"])
    Ck = jnp.asarray(output["Ck"])
    Lx = output["Lx"]

    Ns = _infer_Ns(output)

    # Infer spatial grid sizes from shapes.
    Ny = int(Fk.shape[-4])
    Nx = int(Fk.shape[-3])
    Nz = int(Fk.shape[-2])
    Nl = int(Fk.shape[-1])

    # Basic sanity
    if alpha_s.size != 3 * Ns:
        raise ValueError(f"alpha_s has size {alpha_s.size}, expected 3*Ns={3*Ns}.")
    if u_s.size != 3 * Ns:
        raise ValueError(f"u_s has size {u_s.size}, expected 3*Ns={3*Ns}.")

    # Hermite layout sizes
    Htot = int(Ck.shape[1])
    if (Htot % Ns) != 0:
        raise ValueError(f"Ck.shape[1]={Htot} is not divisible by Ns={Ns}.")
    Hs = Htot // Ns  # Hermite DOFs per species (static)

    # Nn/Nm/Np define the flattening order for Hermite modes:
    #   idx = n + Nn*m + Nn*Nm*p
    # If missing, we fall back to a 1D Hermite layout (Nm=Np=1, Nn=Hs).
    Nn, Nm, Np = (int(output.get(k, d)) for k, d in (("Nn", Hs), ("Nm", 1), ("Np", 1)))

    # Debye length:
    # - For Ns==2, preserve the legacy electron/ion expression when mi_me is provided.
    # - For Ns>2, use an effective value based on the x-thermal scales.
    alpha_x = alpha_s.reshape(Ns, 3)[:, 0]
    lambda_D = jnp.sqrt(1.0 / (2.0 * jnp.sum((qs ** 2 / ms) / (alpha_x ** 2)))) # This assumes all species have the same average density.

    # Original k_norm convention (species 0, x thermal).
    k_norm = jnp.sqrt(2.0) * jnp.pi * alpha_s[0] / Lx

    kinetic_energy_species, EM_energy = _energies(Ck, Fk, alpha_s, u_s, ms, Omega_ce, output["basis_idx"], Ns, Nn, Nm, Np)
    kinetic_energy = jnp.sum(kinetic_energy_species, axis=1)  # (Nt,): species summed once, after the per-species energies
    total_energy = kinetic_energy + EM_energy

    output.update({
        "Ns": Ns,
        "lambda_D": lambda_D,
        "k_norm": k_norm,
        "kinetic_energy_species": kinetic_energy_species,
        "kinetic_energy": kinetic_energy,
        "EM_energy": EM_energy,
        "total_energy": total_energy,
    })

    # Convenience / back-compat keys (useful for existing plotting scripts).
    for s in range(Ns):
        output[f"kinetic_energy_species{s+1}"] = kinetic_energy_species[:, s]
