"""Private local Gaussian-excess diagnostic; not an optimization target."""

import jax
import jax.numpy as jnp
import numpy as np

from ._velocity_observables import _hermgauss, _quadrature, spatial_average_coefficients


def _cell_moments(c, alpha, u):
    """Exact density, physical flow, and full central velocity covariance."""
    def mode(n, m, p):
        if n < c.shape[2] and m < c.shape[1] and p < c.shape[0]:
            return c[p, m, n]
        return jnp.zeros_like(c[0, 0, 0])

    c0 = c[0, 0, 0]
    first = jnp.stack((mode(1, 0, 0), mode(0, 1, 0), mode(0, 0, 1)))
    volume = jnp.prod(alpha)
    density = volume * c0
    # Identical flux/density convention to _local_tail, including basis shift.
    flow = volume * (u * c0 + alpha * first / jnp.sqrt(2.)) / density
    mean_xi = first / (jnp.sqrt(2.) * c0)
    diagonal = .5 + jnp.stack((mode(2, 0, 0), mode(0, 2, 0), mode(0, 0, 2))) / (jnp.sqrt(2.) * c0)
    xy, xz, yz = (mode(1, 1, 0) / (2*c0), mode(1, 0, 1) / (2*c0),
                  mode(0, 1, 1) / (2*c0))
    second = jnp.stack((jnp.stack((diagonal[0], xy, xz)),
                        jnp.stack((xy, diagonal[1], yz)),
                        jnp.stack((xz, yz, diagonal[2]))))
    covariance = alpha[:, None] * alpha[None, :] * (second - jnp.outer(mean_xi, mean_xi))
    return density, flow, covariance


def make_local_tail_diagnostics(alpha_s, u_s, *, Nx, Nn, Nm, Np, Ns, spec,
                                species=0, mass=1., quadrature_order=24):
    """Return callable(Ck) with spatial-mean normalized tail/Gaussian/excess.

    Accept flat or structured single-state RFFTs, as in _local_tail. Exact local
    moments include mixed Hermites; the reference matches the full covariance.
    Both tails use E=mass*|v-U|^2/2 and the fixed spec. The Gaussian integral uses
    covariance-centered GH coordinates, with n/pi**1.5 times product GH weights
    and no additional Gaussian or determinant. All tail integrals are signed.

    Nonpositive/nonfinite density or covariance makes all three tail outputs NaN.
    Minimum density and covariance eigenvalue remain diagnostic raw minima (NaN
    when undefined). No floors, clipping, or covariance regularization are used.
    lax.map/checkpoint bound per-cell velocity workspace to O(q**3); real-space
    Hermite coefficients still occupy O(Ncell*Np*Nm*Nn). Quadrature convergence
    and distribution validity must be assessed separately by the caller.
    """
    def integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))

    if (any(not integer(n) or n < 1 for n in (Nx, Nn, Nm, Np, Ns))
            or not integer(species) or not 0 <= species < Ns):
        raise ValueError("invalid grid/mode dimensions or species")
    if not integer(quadrature_order) or quadrature_order < 2:
        raise ValueError("quadrature_order must be an integer >= 2")
    alpha = np.asarray(alpha_s, dtype=float).reshape(Ns, 3)
    u = np.asarray(u_s, dtype=float).reshape(Ns, 3)
    mass = float(mass)
    if (not np.all(np.isfinite(alpha)) or np.any(alpha <= 0)
            or not np.all(np.isfinite(u)) or not np.isfinite(mass) or mass <= 0):
        raise ValueError("finite positive alpha/mass and finite u required")
    threshold, width, normalization = (float(getattr(spec, name)) for name in
                                        ('threshold', 'width', 'normalization'))
    if not all(np.isfinite(v) and v > 0 for v in (threshold, width, normalization)):
        raise ValueError("tail threshold, width and normalization must be positive and finite")
    alpha, u = jnp.asarray(alpha[species]), jnp.asarray(u[species])
    order = int(quadrature_order)
    bx, by, bz, _, weights = _quadrature((Np, Nm, Nn), alpha, u, mass, order)
    nodes, gh_weights = (jnp.asarray(a) for a in _hermgauss(order))
    velocities = alpha[:, None] * nodes + u[:, None]
    gaussian_weights = (gh_weights[:, None, None] * gh_weights[None, :, None]
                        * gh_weights[None, None, :] / jnp.pi**1.5)

    def kernel(energy):
        return energy * jax.nn.sigmoid((energy - threshold) / width)

    @jax.checkpoint
    def integrate_cell(c):
        density, flow, covariance = _cell_moments(c, alpha, u)
        density_valid = jnp.isfinite(density) & (density > 0)
        covariance_finite = jnp.all(jnp.isfinite(covariance))
        minimum = jax.lax.cond(density_valid & covariance_finite,
                               lambda cov: jnp.linalg.eigvalsh(cov)[0],
                               lambda cov: jnp.asarray(jnp.nan, cov.dtype), covariance)

        def valid_tail(_):
            relative = velocities - flow[:, None]
            energy = mass / 2 * (relative[2, :, None, None]**2
                                 + relative[1, None, :, None]**2
                                 + relative[0, None, None, :]**2)
            polynomial = jnp.einsum('pmn,nx,my,pz->zyx', c, bx, by, bz)
            tail = jnp.sum(polynomial * weights * kernel(energy)) / normalization
            chol = jnp.linalg.cholesky(covariance)
            # xi axes match the z,y,x tensor quadrature; no full Ncell*q^3 array.
            gaussian_energy = jnp.zeros_like(energy)
            for i in range(3):
                component = (chol[i, 0] * nodes[None, None, :]
                             + chol[i, 1] * nodes[None, :, None]
                             + chol[i, 2] * nodes[:, None, None])
                gaussian_energy = gaussian_energy + mass * component**2
            gaussian = density * jnp.sum(gaussian_weights * kernel(gaussian_energy)) / normalization
            chol_valid = jnp.all(jnp.isfinite(chol)) & jnp.all(jnp.diag(chol) > 0)
            return jnp.where(chol_valid, jnp.stack((tail, gaussian)), jnp.nan)

        tails = jax.lax.cond(density_valid & covariance_finite & (minimum > 0),
                             valid_tail, lambda _: jnp.full((2,), jnp.nan, c.dtype), None)
        return tails[0], tails[1], density, minimum

    def diagnostics(Ck):
        Ck = jnp.asarray(Ck)
        if (Ck.ndim not in (4, 7) or min(Ck.shape[-3:]) < 1
                or Ck.shape[-2] != Nx // 2 + 1):
            raise ValueError("Nx must match a nonempty stored real-Fourier grid")
        spatial_average_coefficients(Ck, Nn=Nn, Nm=Nm, Np=Np, Ns=Ns, species=species)
        ck = Ck.reshape(Ns, Np, Nm, Nn, *Ck.shape[-3:])[species]
        ny, _, nz = ck.shape[-3:]
        local = jnp.fft.irfftn(ck, s=(nz, ny, Nx), axes=(-1, -3, -2), norm='forward')
        cells = jnp.moveaxis(local.reshape(Np, Nm, Nn, -1), -1, 0)
        tails, gaussians, densities, minima = jax.lax.map(integrate_cell, cells)
        tail, gaussian = jnp.mean(tails), jnp.mean(gaussians)
        return dict(tail=tail, gaussian_tail=gaussian, excess=tail-gaussian,
                    min_density=jnp.min(densities), min_covariance_eigenvalue=jnp.min(minima))

    return diagnostics
