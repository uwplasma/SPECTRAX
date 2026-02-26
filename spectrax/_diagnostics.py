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

import jax.numpy as jnp

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
    n = int(u_s.shape[0])
    if (n % 3) != 0:
        raise ValueError(f"Cannot infer Ns from u_s of length {n}; expected multiple of 3.")
    return n // 3


def _infer_masses(output: Mapping[str, Any], Ns: int) -> jnp.ndarray:
    """Infer per-species mass multipliers for kinetic-energy normalization.

    The kinetic-energy diagnostic used here follows the convention in the legacy
    two-species code, where species 0 is normalized to mass 1 and species 1 is
    scaled by ``mi_me``. For ``Ns>2`` a user may provide:

    - ``output["ms"]``: array of shape ``(Ns,)``
    - ``output["masses"]``: array of shape ``(Ns,)``

    If neither is present, this returns ``ones((Ns,))``.
    """
    if "ms" in output:
        ms = jnp.asarray(output["ms"])
        if ms.size != Ns:
            raise ValueError(f'output["ms"] has size {ms.size}, expected {Ns}.')
        return ms
    if "masses" in output:
        ms = jnp.asarray(output["masses"])
        if ms.size != Ns:
            raise ValueError(f'output["masses"] has size {ms.size}, expected {Ns}.')
        return ms
    if Ns == 2 and "mi_me" in output:
        return jnp.array([1.0, output["mi_me"]])
    return jnp.ones((Ns,))


def diagnostics(output: dict) -> None:
    """Compute and attach common diagnostics to ``output`` (mutates in-place).

    Parameters
    ----------
    output : dict
        Dictionary holding simulation results and parameters. This function
        expects at least the following keys:

        - ``alpha_s``: array of shape ``(3*Ns,)`` (thermal scales per species)
        - ``u_s``: array of shape ``(3*Ns,)`` (drift velocities per species)
        - ``Ck``: Hermite-Fourier coefficients of shape ``(Nt, Ns*Hs, Ny, Nx, Nz)``
        - ``Fk``: field Fourier coefficients of shape ``(Nt, 6, Ny, Nx, Nz)``
        - ``Omega_cs``: array of shape ``(Ns,)`` (cyclotron frequencies)
        - ``Lx``: domain length in x (used for ``k_norm``)

        Optionally, it may use:
        - ``Nn``, ``Nm``, ``Np`` to decode flattened Hermite indices
        - ``mi_me`` for legacy 2-species Debye length and mass scaling
        - ``ms`` or ``masses`` for general per-species mass scaling

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
    Fk = jnp.asarray(output["Fk"])
    Ck = jnp.asarray(output["Ck"])
    Omega_cs = jnp.asarray(output["Omega_cs"])
    Lx = output["Lx"]

    Ns = _infer_Ns(output)

    # Infer spatial grid sizes from shapes.
    # Arrays are stored in fftshift ordering throughout the solver.
    Ny = int(Fk.shape[-3])
    Nx = int(Fk.shape[-2])
    Nz = int(Fk.shape[-1])

    # k=0 mode indices for fftshifted arrays.
    # - for odd N:  N//2 == (N-1)//2
    # - for even N: N//2 correctly points to the centered zero-frequency bin
    half_ny = Ny // 2
    half_nx = Nx // 2
    half_nz = Nz // 2

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
    Nn = jnp.asarray(output.get("Nn", Hs), dtype=jnp.int32)
    Nm = jnp.asarray(output.get("Nm", 1), dtype=jnp.int32)
    Np = jnp.asarray(output.get("Np", 1), dtype=jnp.int32)

    # Debye length:
    # - For Ns==2, preserve the legacy electron/ion expression when mi_me is provided.
    # - For Ns>2, use an effective value based on the x-thermal scales.
    if Ns == 2 and "mi_me" in output:
        mi_me = output["mi_me"]
        lambda_D = jnp.sqrt(1.0 / (2.0 * (1.0 / alpha_s[0] ** 2 + 1.0 / (mi_me * alpha_s[3] ** 2))))
    else:
        alpha_x = alpha_s.reshape(Ns, 3)[:, 0]
        lambda_D = jnp.sqrt(1.0 / (2.0 * jnp.sum(1.0 / (alpha_x ** 2))))

    # Original k_norm convention (species 0, x thermal).
    k_norm = jnp.sqrt(2.0) * jnp.pi * alpha_s[0] / Lx

    # Reshape Ck into species blocks along Hermite axis:
    # (Nt, Htot, Ny, Nx, Nz) -> (Nt, Ns, Hs, Ny, Nx, Nz)
    Ck_rs = Ck.reshape(Ck.shape[0], Ns, Hs, Ny, Nx, Nz)

    def _take_mode(offset: Any) -> jnp.ndarray:
        """Gather a Hermite coefficient at a given per-species flattened index.

        This helper is used to safely extract low-order Hermite moments such as
        ``C000`` and ``C100`` even when the Hermite basis is truncated so that
        some offsets fall out of range.

        Returns
        -------
        jnp.ndarray
            Array of shape ``(Nt, Ns)`` giving the coefficient evaluated at the
            k=0 Fourier mode.
        """
        offset = jnp.asarray(offset, dtype=jnp.int32)
        off_clip = jnp.clip(offset, 0, Hs - 1)
        gathered = jnp.take(Ck_rs, off_clip, axis=2, mode="clip")  # (Nt, Ns, Ny, Nx, Nz)
        k0 = gathered[:, :, half_ny, half_nx, half_nz]              # (Nt, Ns)
        mask = (offset >= 0) & (offset < Hs)
        return k0 * mask.astype(k0.dtype)

    # Low-order Hermite moments in flattened (n,m,p) indexing:
    # idx = n + Nn*m + Nn*Nm*p
    C000 = _take_mode(0)
    C100 = _take_mode(1)
    C010 = _take_mode(Nn)
    C001 = _take_mode(Nn * Nm)

    C200 = _take_mode(2)
    C020 = _take_mode(2 * Nn)
    C002 = _take_mode(2 * Nn * Nm)

    # Species parameters
    alpha = alpha_s.reshape(Ns, 3)
    u = u_s.reshape(Ns, 3)
    masses = _infer_masses(output, Ns)

    a0, a1, a2 = alpha[:, 0], alpha[:, 1], alpha[:, 2]
    u0, u1, u2 = u[:, 0], u[:, 1], u[:, 2]

    pref = 0.5 * masses * a0 * a1 * a2  # (Ns,)

    term0 = (0.5 * (a0**2 + a1**2 + a2**2) + (u0**2 + u1**2 + u2**2))  # (Ns,)
    term1 = jnp.sqrt(2.0) * (a0 * u0 * C100 + a1 * u1 * C010 + a2 * u2 * C001)  # (Nt, Ns)
    term2 = (1.0 / jnp.sqrt(2.0)) * (a0**2 * C200 + a1**2 * C020 + a2**2 * C002)  # (Nt, Ns)

    kinetic_energy_species = pref[None, :] * (term0[None, :] * C000 + term1 + term2)  # (Nt, Ns)
    kinetic_energy = jnp.sum(kinetic_energy_species, axis=1)  # (Nt,)
    # Field energy
    EM_energy = 0.5 * jnp.sum(jnp.abs(Fk) ** 2, axis=(-4, -3, -2, -1)) * Omega_cs[0] ** 2  # (Nt,)

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
