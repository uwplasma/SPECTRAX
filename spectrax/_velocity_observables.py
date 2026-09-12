"""Signed Hermite velocity diagnostics and a fixed lab-frame tail objective.

No space-by-velocity tensor is constructed. See docs/acceleration_objective.md
for conventions, preregistration, and the limits of spatial averaging.
"""

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class TailSpec:
    """Fixed physical energies and energy-density normalization (host scalars)."""

    threshold: float
    width: float
    normalization: float

    def __post_init__(self):
        if not all(np.isfinite(x) and x > 0 for x in
                   (self.threshold, self.width, self.normalization)):
            raise ValueError("tail threshold, width and normalization must be positive and finite")


def preregistered_tail_spec(initial_thermal_energy, initial_density=1.0):
    """Freeze E*=5 epsilon_th,0, width=epsilon_th,0/2, scale=n0 epsilon_th,0.

    epsilon_th,0 is the initial mean *internal* energy per electron, excluding
    local bulk flow. For the prescribed Maxwellian it is m_e sum(alpha_e**2)/4.
    Construct once outside AD, before comparing controls or evolved states.
    """
    energy, density = float(initial_thermal_energy), float(initial_density)
    if not np.isfinite(density) or density <= 0:
        raise ValueError("initial density must be positive and finite")
    return TailSpec(5 * energy, 0.5 * energy, density * energy)


def spatial_average_coefficients(Ck, *, Nn, Nm, Np, Ns, species=0):
    """Return real (Np,Nm,Nn) coefficients from the forward FFT's zero mode.

    Accept one flattened solver state (Ns*Np*Nm*Nn,Ny,Nkx,Nz), or its structured
    (Ns,Np,Nm,Nn,Ny,Nkx,Nz) form. Dimensions and species are static under JIT.
    """
    if min(Nn, Nm, Np, Ns) < 1 or not 0 <= species < Ns:
        raise ValueError("invalid mode dimensions or species")
    Ck = jnp.asarray(Ck)
    expected = (Ns * Np * Nm * Nn,) if Ck.ndim == 4 else (Ns, Np, Nm, Nn)
    if Ck.ndim not in (4, 7) or Ck.shape[:-3] != expected:
        raise ValueError("expected one flattened or structured Hermite-Fourier state")
    return Ck[..., 0, 0, 0].reshape(Ns, Np, Nm, Nn)[species].real


def _basis(count, x):
    """Physicists' H_n(x)/sqrt(2**n n!), without the Gaussian."""
    values = [jnp.ones_like(x)]
    if count > 1:
        values.append(jnp.sqrt(2.0) * x)
    for n in range(1, count - 1):
        values.append(jnp.sqrt(2.0 / (n + 1)) * x * values[-1]
                      - jnp.sqrt(n / (n + 1)) * values[-2])
    return jnp.stack(values)


def reconstruct_distribution(coefficients, velocities, alpha, u):
    """Evaluate signed f(v) at arbitrary lab velocities (...,3).

    coefficients is one real (Np,Nm,Nn) block, local or spatially averaged.
    alpha and u are length-three, spatially constant basis parameters; alpha>0.
    Coefficients already contain inverse-alpha factors: do not divide f by
    prod(alpha) again. Differentiable in coefficients, velocities, alpha and u.
    """
    c, v = jnp.asarray(coefficients), jnp.asarray(velocities)
    if c.ndim != 3 or v.shape[-1:] != (3,):
        raise ValueError("expected (Np,Nm,Nn) coefficients and (...,3) velocities")
    xi = (v - jnp.asarray(u)) / jnp.asarray(alpha)
    bx, by, bz = (_basis(size, xi[..., axis])
                  for size, axis in zip(c.shape[::-1], range(3)))
    polynomial = jnp.einsum('pmn,n...,m...,p...->...', c, bx, by, bz)
    return polynomial * jnp.exp(-jnp.sum(xi**2, axis=-1)) / jnp.pi**1.5


@lru_cache(maxsize=16)
def _hermgauss(order):
    if not isinstance(order, int) or order < 2:
        raise ValueError("quadrature_order must be an integer >= 2")
    return np.polynomial.hermite.hermgauss(order)


def _quadrature(shape, alpha, u, mass, order):
    if order < max(2, (max(shape) + 3) // 2):
        raise ValueError("quadrature_order too small for exact signed second moments")
    x, w = (jnp.asarray(a) for a in _hermgauss(order))
    alpha, u = jnp.asarray(alpha), jnp.asarray(u)
    if alpha.shape != (3,) or u.shape != (3,):
        raise ValueError("alpha and u must each have shape (3,)")
    bx, by, bz = (_basis(size, x) for size in shape[::-1])
    vx, vy, vz = (alpha[i] * x + u[i] for i in range(3))
    energy = 0.5 * mass * (vz[:, None, None]**2 + vy[None, :, None]**2
                           + vx[None, None, :]**2)
    # GH weights already include exp(-xi**2); only Jacobian/pi**(3/2) remains.
    weights = (jnp.prod(alpha) / jnp.pi**1.5
               * w[:, None, None] * w[None, :, None] * w[None, None, :])
    return bx, by, bz, energy, weights


def velocity_diagnostics(coefficients, alpha, u, *, spec, mass=1.0,
                         quadrature_order=48):
    """Signed integrals of one local/averaged block, with O(q**3) velocity storage.

    Energies and masses are densities per unit spatial volume, not domain totals.
    Negative mass is integral max(-f,0) dv, diagnostic only; no physics is clipped.
    Hard tails and negative mass are nonsmooth and usually converge more slowly.
    Averages cannot certify local positivity. Positive finite alpha/mass required.
    """
    c = jnp.asarray(coefficients)
    if c.ndim != 3:
        raise ValueError("expected (Np,Nm,Nn) coefficients")
    bx, by, bz, energy, weights = _quadrature(
        c.shape, alpha, u, mass, quadrature_order)
    polynomial = jnp.einsum('pmn,nx,my,pz->zyx', c, bx, by, bz)
    signed = polynomial * weights
    smooth = jax.nn.sigmoid((energy - spec.threshold) / spec.width)
    density = jnp.sum(signed)
    negative_mass = jnp.sum(jnp.maximum(-polynomial, 0) * weights)
    tail_energy = jnp.sum(signed * energy * smooth)
    kinetic_energy = jnp.sum(signed * energy)
    return dict(density=density, kinetic_energy=kinetic_energy,
                smooth_tail_number=jnp.sum(signed * smooth),
                smooth_tail_energy=tail_energy,
                hard_tail_number=jnp.sum(signed * (energy >= spec.threshold)),
                hard_tail_energy=jnp.sum(signed * energy * (energy >= spec.threshold)),
                negative_mass=negative_mass,
                negative_mass_fraction=negative_mass / density,
                tail_energy_fraction=tail_energy / kinetic_energy,
                objective=tail_energy / spec.normalization)


def make_tail_objective(alpha_s, u_s, *, Nn, Nm, Np, Ns, spec,
                        species=0, mass=1.0, quadrature_order=48):
    """Prepare a callable objective(Ck) -> real scalar JAX array to MAXIMIZE.

    Call once on the host. Basis parameters, mass, spec, sizes and quadrature are
    frozen; derivatives flow through Ck only. For minimizers use -objective(Ck).
    Setup contracts the velocity kernel to (Np,Nm,Nn); evaluation uses only the
    zero Fourier mode and this small kernel, including during reverse AD.
    """
    alpha = np.asarray(alpha_s, dtype=float).reshape(Ns, 3)
    u = np.asarray(u_s, dtype=float).reshape(Ns, 3)
    if min(Nn, Nm, Np, Ns) < 1 or not 0 <= species < Ns:
        raise ValueError("invalid mode dimensions or species")
    if (not np.all(np.isfinite(alpha)) or np.any(alpha <= 0)
            or not np.all(np.isfinite(u)) or not np.isfinite(mass) or mass <= 0):
        raise ValueError("finite positive alpha/mass and finite u required")
    bx, by, bz, energy, weights = _quadrature(
        (Np, Nm, Nn), alpha[species], u[species], mass, quadrature_order)
    weighted = (weights * energy * jax.nn.sigmoid(
        (energy - spec.threshold) / spec.width) / spec.normalization)
    kernel = jnp.einsum('zyx,nx,my,pz->pmn', weighted, bx, by, bz)

    def objective(Ck):
        c = spatial_average_coefficients(Ck, Nn=Nn, Nm=Nm, Np=Np,
                                         Ns=Ns, species=species)
        return jnp.sum(c * kernel)

    return objective


def quadrature_convergence(coefficients, alpha, u, *, spec, mass=1.0,
                           orders=(16, 24, 32)):
    """Return diagnostics and absolute successive changes; no automatic pass claim.

    This checks velocity integration at fixed Hermite resolution only. Compare
    objective gradients separately, with each order's make_tail_objective.
    """
    if len(orders) < 2 or any(a >= b for a, b in zip(orders, orders[1:])):
        raise ValueError("provide at least two strictly increasing quadrature orders")
    results = [velocity_diagnostics(coefficients, alpha, u, spec=spec, mass=mass,
                                    quadrature_order=q) for q in orders]
    return dict(orders=tuple(orders), diagnostics=results,
                absolute_changes=[{key: jnp.abs(b[key] - a[key]) for key in a}
                                  for a, b in zip(results, results[1:])])


def spatial_negative_mass_diagnostics(Ck, alpha_s, u_s, *, Nx, Nn, Nm, Np,
                                      Ns, species=0, quadrature_order=24):
    """Finite-quadrature local negativity, mapping one spatial cell at a time.

    Returns mean_negative_mass (per unit volume), max_cell_fraction, min_density.
    Nx must be the physical x-grid size, including for odd grids. Uses the
    solver's forward Fourier normalization. Stores local Hermite coefficients
    but never a full space-by-velocity tensor. Nonpositive cell density yields
    an infinite cell fraction so invalid densities cannot pass a negativity gate.
    This finite-node check is not continuous distribution positivity certification.
    """
    # Validate the same single-state layouts as the global diagnostic.
    spatial_average_coefficients(Ck, Nn=Nn, Nm=Nm, Np=Np, Ns=Ns, species=species)
    Ck = jnp.asarray(Ck)
    if Nx < 1 or Ck.shape[-2] != Nx // 2 + 1:
        raise ValueError("Nx must match the stored real-Fourier x axis")
    alpha = jnp.asarray(alpha_s).reshape(Ns, 3)[species]
    u = jnp.asarray(u_s).reshape(Ns, 3)[species]
    ck = Ck.reshape(Ns, Np, Nm, Nn, *Ck.shape[-3:])[species]
    ny, _, nz = ck.shape[-3:]
    local = jnp.fft.irfftn(ck, s=(nz, ny, Nx), axes=(-1, -3, -2), norm='forward')
    cells = jnp.moveaxis(local.reshape(Np, Nm, Nn, -1), -1, 0)
    bx, by, bz, _, weights = _quadrature(
        (Np, Nm, Nn), alpha, u, 1.0, quadrature_order)

    def integrate_cell(c):
        polynomial = jnp.einsum('pmn,nx,my,pz->zyx', c, bx, by, bz)
        negative = jnp.sum(jnp.maximum(-polynomial, 0) * weights)
        density = jnp.prod(alpha) * c[0, 0, 0]
        fraction = jnp.where(density > 0, negative / density, jnp.inf)
        return negative, fraction, density

    negative, fraction, density = jax.lax.map(integrate_cell, cells)
    return dict(mean_negative_mass=jnp.mean(negative),
                max_cell_fraction=jnp.max(fraction), min_density=jnp.min(density))
