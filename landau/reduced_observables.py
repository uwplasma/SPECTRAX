"""Electron 1D3V signed local tail with fixed Maxwellian transverse velocities."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_hermite, roots_laguerre
from scipy.linalg import eigh_tridiagonal


def local_tail(alpha_s, *, Nx, hermite, quadrature=128):
    """Build a differentiable observable for flat two-species RFFT states.

    Species 0 is electrons (mass 1, basis drift 0); Nm=Np=1. The kernel is
    E*sigmoid((E-5*eps0)/(.5*eps0))/eps0, with eps0=sum(alpha_e**2)/4
    frozen at factory construction. Outputs are spatial means except extrema.
    Gaussian covariance is diag(varx, alpha_y**2/2, alpha_y**2/2).
    Nonpositive/nonfinite density or variance invalidates tails and energies
    with NaN, without floors. Signed f is never clipped in the tail integral.
    Negativity uses longitudinal GH nodes only, integrating transverse factors
    exactly; finite-node sampling cannot certify positivity between nodes.
    Four-cell map/checkpoint bounds quadrature workspace to O(4*q**2).
    AD includes f, flow, variance and any initial reference formed by the caller.
    """
    for name, value in [('Nx', Nx), ('hermite', hermite), ('quadrature', quadrature)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if quadrature < max(2, (hermite + 3)//2):
        raise ValueError('quadrature must resolve signed second moments')
    alpha = np.asarray(alpha_s, dtype=float)
    if alpha.size != 6:
        raise ValueError('alpha_s must contain three scales for each of two species')
    alpha = alpha.reshape(2, 3)
    if not np.all(np.isfinite(alpha)) or np.any(alpha <= 0) or alpha[0, 1] != alpha[0, 2]:
        raise ValueError('positive finite scales and equal electron transverse scales required')
    ax, ay, az = alpha[0]
    volume, tperp, eps0 = ax*ay*az, ay**2/2, np.sum(alpha[0]**2)/4
    x, wx = (jnp.asarray(a) for a in roots_hermite(quadrature))
    with np.errstate(over='ignore', invalid='ignore'):
        s, ws = roots_laguerre(quadrature)
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(ws))):
        # Golub-Welsch fallback when SciPy's Laguerre polynomial overflows.
        k = np.arange(quadrature, dtype=float)
        s, vectors = eigh_tridiagonal(2*k+1, k[1:], lapack_driver='stev')
        ws = vectors[0]**2
    s, ws = jnp.asarray(s), jnp.asarray(ws)
    wx = wx/jnp.sqrt(jnp.pi)
    basis = [wx]  # Recur on weighted polynomials to avoid high-H overflow.
    if hermite > 1:
        basis.append(jnp.sqrt(2.)*x*wx)
    for k in range(1, hermite-1):
        basis.append(jnp.sqrt(2./(k+1))*x*basis[-1] - jnp.sqrt(k/(k+1))*basis[-2])
    basis = jnp.stack(basis)

    def kernel(parallel):
        energy = parallel[:, None] + tperp*s[None, :]
        return (energy/eps0*jax.nn.sigmoid((energy-5*eps0)/(.5*eps0))) @ ws

    @jax.checkpoint
    def cell(c):
        density = volume*c[0]
        mean = (c[1] if hermite > 1 else 0.)/(jnp.sqrt(2.)*c[0])
        second = (c[2] if hermite > 2 else 0.)/(jnp.sqrt(2.)*c[0])
        flow, variance = ax*mean, ax**2*(.5+second-mean**2)
        polynomial = c @ basis
        valid = jnp.isfinite(density) & (density > 0) & jnp.isfinite(variance) & (variance > 0)
        tail = volume*jnp.sum(polynomial*kernel(.5*(ax*x-flow)**2))
        gaussian = density*jnp.sum(wx*kernel(variance*x**2))
        negative = volume*jnp.sum(jnp.maximum(-polynomial, 0.))/density
        bulk = .5*density*flow**2
        internal = .5*density*(variance+2*tperp)
        values = jnp.where(valid, jnp.stack((tail, gaussian, negative, bulk, internal)), jnp.nan)
        return values, density, jnp.minimum(variance, tperp)

    def evaluate(Ck):
        Ck = jnp.asarray(Ck)
        if Ck.shape != (2*hermite, 1, Nx//2+1, 1):
            raise ValueError('expected Ck shape (2*hermite, 1, Nx//2+1, 1)')
        cells = jnp.fft.irfft(Ck[:hermite, 0, :, 0], n=Nx, axis=-1, norm='forward').T
        values, density, variance = jax.lax.map(cell, cells, batch_size=4)
        tail, gaussian, _, bulk, internal = jnp.mean(values, axis=0)
        return dict(tail=tail, gaussian_tail=gaussian, excess=tail-gaussian,
                    min_density=jnp.min(density), min_variance=jnp.min(variance),
                    max_negative_fraction=jnp.max(values[:, 2]),
                    bulk_energy=bulk, internal_energy=internal)

    return evaluate
