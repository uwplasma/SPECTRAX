"""Low-moment energization diagnostics for a single current-format RFFT state.

The spatial transform is ``rfftn(..., axes=(-1, -3, -2), norm='forward')``;
Hermite indices flatten as ``n + Nn*m + Nn*Nm*p`` within each species.
With A = prod(alpha), density is A*C000, particle flux component i is
A*(u_i*C000 + alpha_i*C1_i/sqrt(2)), and the second moment component is
A*((u_i**2 + alpha_i**2/2)*C000 + sqrt(2)*u_i*alpha_i*C1_i
   + alpha_i**2*C2_i/sqrt(2)). Missing Hermite modes are zero.

Collision audit: ``_initialization.initialize_simulation_parameters`` builds
col = h_Nn(n) + h_Nm(m) + h_Np(p), where
h_N(i) = i*(i-1)*(i-2)/((N-1)*(N-2)*(N-3)) for N>3, otherwise zero.
``_model.Hermite_Fourier_system`` adds exactly -nu*col*Ck. Thus all seven
coefficients entering these moments are undamped: density, momentum, total,
bulk and internal energy have zero direct collision rate, even for shifted,
anisotropic bases. Higher modes can damp and affect subsequent evolution;
this operator is not an interspecies thermal-energy exchange model.
``collision_power`` evaluates the supplied collision_matrix (including custom
overrides) rather than assuming the built-in zero-energy property.
"""

import jax.numpy as jnp

from ._diagnostics import _infer_masses
from ._simulation import _twothirds_mask


def _moments(Ck, alpha, u, masses, Nn, Nm, Np):
    """Return spectral number density, particle flux and kinetic energy."""
    C = Ck.reshape(alpha.shape[0], Np, Nm, Nn, *Ck.shape[-3:])
    c0 = C[:, 0, 0, 0]

    def mode(n, m, p):
        return C[:, p, m, n] if n < Nn and m < Nm and p < Np else jnp.zeros_like(c0)

    c1 = jnp.stack([mode(1, 0, 0), mode(0, 1, 0), mode(0, 0, 1)], axis=1)
    c2 = jnp.stack([mode(2, 0, 0), mode(0, 2, 0), mode(0, 0, 2)], axis=1)
    a, v = alpha[:, :, None, None, None], u[:, :, None, None, None]
    volume = jnp.prod(alpha, axis=1)[:, None, None, None]
    flux = volume[:, None] * (v * c0[:, None] + a * c1 / jnp.sqrt(2.0))
    second = ((v**2 + a**2 / 2) * c0[:, None]
              + jnp.sqrt(2.0) * v * a * c1 + a**2 * c2 / jnp.sqrt(2.0))
    energy = 0.5 * masses[:, None, None, None] * volume * jnp.sum(second, axis=1)
    return volume * c0, flux, energy


def species_energization(Ck, Fk, parameters, *, Nx, Nn, Nm, Np):
    """Return real JAX moment maps and spatially averaged species energies/rates.

    Ck has shape (Ns*Np*Nm*Nn, Ny, Nx//2+1, Nz), Fk has shape
    (6, Ny, Nx//2+1, Nz). Nx and Hermite sizes must be static under JIT;
    Nx is explicit to distinguish odd/even RFFTs. No time axis is accepted;
    use vmap for a trajectory. Only seven Hermite modes are transformed.

    Required parameters: alpha_s, u_s (3*Ns entries), qs, Omega_cs (Ns),
    nu, collision_matrix (Np, Nm, Nn). Masses follow existing diagnostics:
    ms, then masses, then [1, mi_me] for two species. One species defaults
    to mass 1; other cases require explicit masses to avoid silent unit errors.

    Outputs:
      density: number density, shape (Ns, Ny, Nx, Nz).
      momentum: mass times particle flux, shape (Ns, 3, Ny, Nx, Nz).
      kinetic_energy_density, bulk_energy_density, internal_energy_density:
        maps with density's shape; bulk = |momentum|^2/(2*mass*density),
        internal = kinetic - bulk. Nonpositive density yields NaN for bulk
        and internal; no floor or clipping conceals an invalid partition.
      kinetic_energy, bulk_energy, internal_energy: spatial means, (Ns,).
      j_dot_e: unfiltered <q_s * integral(v*f_s dv) dot E>, (Ns,).
      electric_power: mass_s*Omega_cs[s]*<J_s dot E> with BOTH spectra
        masked exactly as the Vlasov RHS, (Ns,). For Nn,Nm,Np>=3 this is
        the electric contribution to d<kinetic energy>/dt. Smaller bases
        reconstruct truncated moments but cannot guarantee that rate identity.
      collision_power: exact d<kinetic energy>/dt from -nu*col*Ck, (Ns,).

    Velocities, density and charge use the solver's dimensionless units;
    energy is in reference-mass density times reference-velocity squared,
    matching _diagnostics. These are volume AVERAGES, not domain integrals.
    Multiply by Lx*Ly*Lz for integrals. Field energy in these units is
    Omega_cs[0]**2 * <E**2+B**2>/2. Exchange cancels for resolved spectra
    when mass_s*Omega_cs[s] == Omega_cs[0]; arbitrary input frequencies or
    out-of-mask field/current modes need not satisfy that balance. Spatial
    diffusion has zero mean kinetic-energy rate because k2_grid(k=0)=0.
    Internal energy alone is not a measure of irreversible heating.

    For a streamed integrand returning an array, for example::

        def integrand(t, Ck, Fk):
            d = species_energization(Ck, Fk, parameters, Nx=Nx,
                                     Nn=Nn, Nm=Nm, Np=Np)
            return jnp.stack([d['electric_power'], d['collision_power']])
    """
    if min(Nx, Nn, Nm, Np) < 1:
        raise ValueError('Grid and Hermite sizes must be positive.')
    alpha = jnp.asarray(parameters['alpha_s']).reshape(-1, 3)
    Ns = alpha.shape[0]
    u = jnp.asarray(parameters['u_s']).reshape(Ns, 3)
    if not ('ms' in parameters or 'masses' in parameters
            or (Ns == 2 and 'mi_me' in parameters) or Ns == 1):
        raise ValueError('Specify ms/masses (or mi_me for two species).')
    masses = _infer_masses(parameters, Ns).reshape(Ns)
    q = jnp.asarray(parameters['qs']).reshape(Ns)
    omega = jnp.asarray(parameters['Omega_cs']).reshape(Ns)
    if Ck.ndim != 4 or Fk.ndim != 4:
        raise ValueError('Expected a single state with four array dimensions.')
    Ny, Nkx, Nz = Ck.shape[-3:]
    if (Nkx != Nx // 2 + 1 or Ck.shape[0] != Ns * Nn * Nm * Np
            or Fk.shape != (6, Ny, Nkx, Nz)):
        raise ValueError('State shapes do not match the supplied grid/species sizes.')
    col = jnp.asarray(parameters['collision_matrix'])
    if col.shape != (Np, Nm, Nn):
        raise ValueError('collision_matrix must have shape (Np, Nm, Nn).')

    def real_space(spectrum):
        return jnp.fft.irfftn(spectrum, s=(Nz, Ny, Nx),
                             axes=(-1, -3, -2), norm='forward')

    nk, fluxk, energyk = _moments(Ck, alpha, u, masses, Nn, Nm, Np)
    density, flux, kinetic = real_space(nk), real_space(fluxk), real_space(energyk)
    mass = masses[:, None, None, None]
    momentum = mass[:, None] * flux
    denominator = jnp.where(density > 0, density, jnp.nan)
    bulk = 0.5 * mass * jnp.sum(flux**2, axis=1) / denominator
    internal = kinetic - bulk
    currentk = q[:, None, None, None, None] * fluxk
    E = real_space(Fk[:3])
    j_dot_e = jnp.mean(jnp.sum(real_space(currentk) * E[None], axis=1), axis=(-3, -2, -1))
    mask = _twothirds_mask(Ny, Nx, Nz)
    rhs_work = jnp.mean(jnp.sum(real_space(currentk * mask)
                               * real_space(Fk[:3] * mask)[None], axis=1), axis=(-3, -2, -1))
    # Mean energy is linear in the zero spatial mode; no full-state collision FFT.
    c0 = Ck[:, :1, :1, :1].reshape(Ns, Np, Nm, Nn, 1, 1, 1)
    collision = -parameters['nu'] * col[None, :, :, :, None, None, None] * c0
    _, _, collision_energy = _moments(collision.reshape(-1, 1, 1, 1),
                                      alpha, u, masses, Nn, Nm, Np)
    return dict(
        density=density, momentum=momentum,
        kinetic_energy_density=kinetic, bulk_energy_density=bulk,
        internal_energy_density=internal,
        kinetic_energy=jnp.mean(kinetic, axis=(-3, -2, -1)),
        bulk_energy=jnp.mean(bulk, axis=(-3, -2, -1)),
        internal_energy=jnp.mean(internal, axis=(-3, -2, -1)),
        j_dot_e=j_dot_e, electric_power=masses * omega * rhs_work,
        collision_power=collision_energy[:, 0, 0, 0].real,
    )
