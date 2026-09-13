"""Example-local signed tail energy, matched Gaussian, and sampled negativity.

Setup is host-only; AD flows through the single-state Hermite-Fourier input.
Finite tensor Gauss-Hermite quadrature requires value/gradient convergence checks;
negative mass at its nodes cannot certify continuous distribution positivity.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class TailSpec:
    threshold: float
    width: float
    normalization: float

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in
                   (self.threshold, self.width, self.normalization)):
            raise ValueError('tail threshold, width and normalization must be positive and finite')


def preregistered_tail_spec(initial_thermal_energy, initial_density=1.0):
    """Freeze (5 epsilon, epsilon/2, n0 epsilon), excluding initial bulk flow."""
    energy, density = float(initial_thermal_energy), float(initial_density)
    if not np.isfinite(density) or density <= 0:
        raise ValueError('initial density must be positive and finite')
    return TailSpec(5 * energy, .5 * energy, density * energy)


def _basis(count, x):
    """Physicists' H_n / sqrt(2**n n!), without the Gaussian."""
    values = [jnp.ones_like(x)]
    if count > 1:
        values.append(jnp.sqrt(2.) * x)
    for n in range(1, count - 1):
        values.append(jnp.sqrt(2. / (n+1)) * x * values[-1]
                      - jnp.sqrt(n / (n+1)) * values[-2])
    return jnp.stack(values)


def _setup(alpha_s, u_s, Nx, Nn, Nm, Np, Ns, species, quadrature_order):
    def integer(n):
        return isinstance(n, (int, np.integer)) and not isinstance(n, (bool, np.bool_))

    if (any(not integer(n) or n < 1 for n in (Nx, Nn, Nm, Np, Ns))
            or not integer(species) or not 0 <= species < Ns):
        raise ValueError('invalid grid/mode dimensions or species')
    if (not integer(quadrature_order)
            or quadrature_order < max(2, (max(Nn, Nm, Np) + 3) // 2)):
        raise ValueError('quadrature_order must be an integer resolving signed second moments')
    alpha = np.asarray(alpha_s, dtype=float).reshape(Ns, 3)
    u = np.asarray(u_s, dtype=float).reshape(Ns, 3)
    if (not np.all(np.isfinite(alpha)) or np.any(alpha <= 0)
            or not np.all(np.isfinite(u))):
        raise ValueError('finite positive alpha and finite u required')
    alpha, u = jnp.asarray(alpha[species]), jnp.asarray(u[species])
    nodes, w = map(jnp.asarray, np.polynomial.hermite.hermgauss(int(quadrature_order)))
    weights = w[:, None, None] * w[None, :, None] * w[None, None, :] / jnp.pi**1.5
    bx, by, bz = (_basis(size, nodes) for size in (Nn, Nm, Np))

    def polynomial(c):
        return jnp.einsum('pmn,nx,my,pz->zyx', c, bx, by, bz)

    def cells(Ck):
        Ck = jnp.asarray(Ck)
        expected = (Ns*Np*Nm*Nn,) if Ck.ndim == 4 else (Ns, Np, Nm, Nn)
        if (Ck.ndim not in (4, 7) or Ck.shape[:-3] != expected
                or min(Ck.shape[-3:]) < 1 or Ck.shape[-2] != Nx//2+1):
            raise ValueError('expected one flat or structured state matching the real-Fourier grid')
        ck = Ck.reshape(Ns, Np, Nm, Nn, *Ck.shape[-3:])[species]
        ny, _, nz = ck.shape[-3:]
        local = jnp.fft.irfftn(ck, s=(nz, ny, Nx), axes=(-1, -3, -2), norm='forward')
        return jnp.moveaxis(local.reshape(Np, Nm, Nn, -1), -1, 0)

    return alpha, u, nodes, weights, polynomial, cells


def _cell_moments(c, alpha, u):
    """Exact density, physical flow, and full central covariance; missing modes=0."""
    def mode(n, m, p):
        return c[p, m, n] if n < c.shape[2] and m < c.shape[1] and p < c.shape[0] else jnp.zeros_like(c[0, 0, 0])

    c0 = c[0, 0, 0]
    first = jnp.stack((mode(1, 0, 0), mode(0, 1, 0), mode(0, 0, 1)))
    density = jnp.prod(alpha) * c0
    flow = jnp.prod(alpha) * (u*c0 + alpha*first/jnp.sqrt(2.)) / density
    mean = first / (jnp.sqrt(2.) * c0)
    diagonal = .5 + jnp.stack((mode(2, 0, 0), mode(0, 2, 0), mode(0, 0, 2))) / (jnp.sqrt(2.)*c0)
    xy, xz, yz = mode(1, 1, 0)/(2*c0), mode(1, 0, 1)/(2*c0), mode(0, 1, 1)/(2*c0)
    second = jnp.stack((jnp.stack((diagonal[0], xy, xz)),
                        jnp.stack((xy, diagonal[1], yz)),
                        jnp.stack((xz, yz, diagonal[2]))))
    covariance = alpha[:, None]*alpha[None, :] * (second - jnp.outer(mean, mean))
    return density, flow, covariance


def _make_tail(alpha_s, u_s, Nx, Nn, Nm, Np, Ns, spec, species, mass,
               quadrature_order, diagnostic):
    alpha, u, nodes, weights, polynomial, cells = _setup(
        alpha_s, u_s, Nx, Nn, Nm, Np, Ns, species, quadrature_order)
    mass = float(mass)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('mass must be positive and finite')
    spec = TailSpec(*(float(getattr(spec, name)) for name in
                      ('threshold', 'width', 'normalization')))
    velocities = alpha[:, None]*nodes + u[:, None]
    signed_weights = jnp.prod(alpha)*weights

    def kernel(energy):
        return energy * jax.nn.sigmoid((energy-spec.threshold)/spec.width) / spec.normalization

    @jax.checkpoint
    def integrate(c):
        density, flow, covariance = _cell_moments(c, alpha, u)

        def tail():
            relative = velocities - flow[:, None]
            energy = mass/2 * (relative[2, :, None, None]**2
                               + relative[1, None, :, None]**2
                               + relative[0, None, None, :]**2)
            return jnp.sum(polynomial(c)*signed_weights*kernel(energy))

        if not diagnostic:
            return jnp.where(density > 0, tail(), jnp.nan)
        valid = jnp.isfinite(density) & (density > 0) & jnp.all(jnp.isfinite(covariance))
        minimum = jax.lax.cond(valid, lambda cov: jnp.linalg.eigvalsh(cov)[0],
                               lambda cov: jnp.asarray(jnp.nan, cov.dtype),
                               jax.lax.stop_gradient(covariance))

        def tails(_):
            chol = jnp.linalg.cholesky(covariance)
            energy = jnp.zeros_like(weights)
            for i in range(3):
                component = (chol[i, 0]*nodes[None, None, :]
                             + chol[i, 1]*nodes[None, :, None]
                             + chol[i, 2]*nodes[:, None, None])
                energy = energy + mass*component**2
            gaussian = density*jnp.sum(weights*kernel(energy))
            good = jnp.all(jnp.isfinite(chol)) & jnp.all(jnp.diag(chol) > 0)
            return jnp.where(good, jnp.stack((tail(), gaussian)), jnp.nan)

        pair = jax.lax.cond(valid & (minimum > 0), tails,
                            lambda _: jnp.full((2,), jnp.nan, c.dtype), None)
        return pair[0], pair[1], density, minimum

    def evaluate(Ck):
        result = jax.lax.map(integrate, cells(Ck), batch_size=4)
        if not diagnostic:
            return jnp.mean(result)
        tails, gaussians, densities, minima = result
        tail, gaussian = jnp.mean(tails), jnp.mean(gaussians)
        return dict(tail=tail, gaussian_tail=gaussian, excess=tail-gaussian,
                    min_density=jnp.min(densities), min_covariance_eigenvalue=jnp.min(minima))

    return evaluate


def make_local_tail_objective(alpha_s, u_s, *, Nx, Nn, Nm, Np, Ns, spec,
                              species=0, mass=1., quadrature_order=24):
    """Return spatial-mean signed E*sigmoid((E-threshold)/width)/normalization.

    E=mass*|v-U(x)|**2/2; differentiate both f and exact flux/density flow U.
    Nonpositive density yields NaN. Flat/structured solver RFFTs are accepted;
    Nx is the physical grid size, including odd grids. No distribution clipping.
    Four-cell map/checkpoint bounds velocity workspace to O(4*q**3), including AD;
    spatial Hermite coefficients and their adjoints still occupy O(Ncell*Nmode).
    """
    return _make_tail(alpha_s, u_s, Nx, Nn, Nm, Np, Ns, spec, species, mass,
                      quadrature_order, False)


def make_local_tail_diagnostics(alpha_s, u_s, *, Nx, Nn, Nm, Np, Ns, spec,
                                species=0, mass=1., quadrature_order=24):
    """Return signed tail, full-covariance matched Gaussian tail, and excess.

    Gaussian integration uses covariance-centered GH coordinates and n/pi**1.5
    product weights, with no extra determinant or Gaussian. Invalid density or
    covariance makes all tails NaN; raw minimum density/eigenvalue are retained.
    No floors or regularization. Tail/excess support AD through flow and Cholesky;
    the minimum eigenvalue is a nondifferentiated validity diagnostic.
    """
    return _make_tail(alpha_s, u_s, Nx, Nn, Nm, Np, Ns, spec, species, mass,
                      quadrature_order, True)


def spatial_negative_mass_diagnostics(Ck, alpha_s, u_s, *, Nx, Nn, Nm, Np,
                                      Ns, species=0, quadrature_order=24):
    """Sample local integral max(-f,0) dv with O(q**3) velocity workspace.

    Return mean_negative_mass per spatial volume, max_cell_fraction, min_density.
    Invalid density gives infinite fraction; finite-node sampling is diagnostic.
    """
    alpha, _, _, weights, polynomial, cells = _setup(
        alpha_s, u_s, Nx, Nn, Nm, Np, Ns, species, quadrature_order)

    def integrate(c):
        negative = jnp.prod(alpha)*jnp.sum(jnp.maximum(-polynomial(c), 0)*weights)
        density = jnp.prod(alpha)*c[0, 0, 0]
        valid = jnp.isfinite(density) & (density > 0)
        return negative, jnp.where(valid, negative/density, jnp.inf), density

    negative, fraction, density = jax.lax.map(integrate, cells(Ck), batch_size=4)
    return dict(mean_negative_mass=jnp.mean(negative), max_cell_fraction=jnp.max(fraction),
                min_density=jnp.min(density))
