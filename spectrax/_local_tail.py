"""Private, signed tail energy measured relative to each spatial cell's flow."""

import jax
import jax.numpy as jnp
import numpy as np

from ._velocity_observables import _hermgauss, _quadrature, spatial_average_coefficients


def make_local_tail_objective(alpha_s, u_s, *, Nx, Nn, Nm, Np, Ns, spec,
                              species=0, mass=1., quadrature_order=24):
    """Freeze setup and return objective(Ck) -> spatial mean signed tail energy.

    The kernel is E*sigmoid((E-spec.threshold)/spec.width)/spec.normalization,
    E = mass*|v-U(x)|**2/2. U is exact particle flux / density, with absent
    first Hermite modes treated as zero. Both f and U are differentiated.
    Nonpositive local density returns NaN; neither f nor density is clipped.

    Accepts flat or structured single-state solver RFFTs, with physical Nx
    explicit for odd grids. Setup is host-only; basis parameters and spec stay
    fixed under AD. Each cell uses finite tensor Gauss-Hermite quadrature,
    so tail values AND gradients require convergence checks in applications.
    lax.map plus rematerialization bounds velocity workspace to O(q**3) in
    reverse AD, at the cost of recomputation. Spatial Hermite coefficients
    and their adjoints still require O(Nx*Ny*Nz*Nn*Nm*Np) storage; runtime
    includes a velocity contraction for every cell, unlike the lab-frame tail.
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
    nodes = jnp.asarray(_hermgauss(order)[0])
    velocities = alpha[:, None] * nodes + u[:, None]
    volume = jnp.prod(alpha)

    @jax.checkpoint
    def integrate_cell(c):
        c0 = c[0, 0, 0]
        # Flattened solver mapping: n + Nn*m + Nn*Nm*p (x, y, z).
        first = jnp.stack((c[0, 0, 1] if Nn > 1 else jnp.zeros_like(c0),
                           c[0, 1, 0] if Nm > 1 else jnp.zeros_like(c0),
                           c[1, 0, 0] if Np > 1 else jnp.zeros_like(c0)))
        density = volume * c0
        flux = volume * (u * c0 + alpha * first / jnp.sqrt(2.))
        flow = flux / density
        relative = velocities - flow[:, None]
        energy = mass / 2 * (relative[2, :, None, None]**2
                             + relative[1, None, :, None]**2
                             + relative[0, None, None, :]**2)
        polynomial = jnp.einsum('pmn,nx,my,pz->zyx', c, bx, by, bz)
        tail = jnp.sum(polynomial * weights * energy
                       * jax.nn.sigmoid((energy - threshold) / width))
        return jnp.where(density > 0, tail / normalization, jnp.nan)

    def objective(Ck):
        Ck = jnp.asarray(Ck)
        if (Ck.ndim not in (4, 7) or min(Ck.shape[-3:]) < 1
                or Ck.shape[-2] != Nx // 2 + 1):
            raise ValueError("Nx must match a nonempty stored real-Fourier grid")
        spatial_average_coefficients(Ck, Nn=Nn, Nm=Nm, Np=Np, Ns=Ns, species=species)
        ck = Ck.reshape(Ns, Np, Nm, Nn, *Ck.shape[-3:])[species]
        ny, _, nz = ck.shape[-3:]
        local = jnp.fft.irfftn(ck, s=(nz, ny, Nx), axes=(-1, -3, -2), norm='forward')
        cells = jnp.moveaxis(local.reshape(Np, Nm, Nn, -1), -1, 0)
        return jnp.mean(jax.lax.map(integrate_cell, cells))

    return objective
