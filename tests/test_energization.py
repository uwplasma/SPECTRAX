"""Tiny analytic moment/RHS checks; no time integration or velocity quadrature."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectrax._diagnostics import diagnostics
from spectrax._energization import species_energization
from spectrax._initialization import initialize_simulation_parameters
from spectrax._initialize_maxwellian import compute_C_nmp
from spectrax._model import plasma_current
from spectrax._simulation import _solver_args, ode_system


def fft(x):
    return jnp.fft.rfftn(x, axes=(-1, -3, -2), norm='forward')


def case(nx=5, sizes=(3, 3, 3), spatial=True):
    nn, nm, np_ = sizes
    ny, nz = (3, 3) if spatial else (1, 1)
    alpha = jnp.array([[0.7, 0.9, 1.1], [0.2, 0.3, 0.4]])
    shift = jnp.array([[0.21, -0.13, 0.17], [-0.12, 0.18, -0.08]])
    p = initialize_simulation_parameters(
        dict(alpha_s=alpha.ravel(), u_s=shift.ravel(), mi_me=7.,
             qs=jnp.array([-1., 1.]), Omega_cs=jnp.array([0.5, 0.5 / 7]),
             nu=0.37, D=0.), Nx=nx, Ny=ny, Nz=nz, Nn=nn, Nm=nm, Np=np_, Ns=2)
    yy, xx, zz = jnp.meshgrid(2*jnp.pi*jnp.arange(ny)/ny,
                             2*jnp.pi*jnp.arange(nx)/nx,
                             2*jnp.pi*jnp.arange(nz)/nz, indexing='ij')
    rho = jnp.stack([1.3 + 0.1*jnp.cos(xx), 0.8 + 0.06*jnp.sin(xx)])
    perturb = jnp.stack([0.07*jnp.cos(xx), 0.04*jnp.sin(yy), 0.05*jnp.cos(zz)])
    velocity = shift[:, :, None, None, None] + jnp.stack([perturb, -0.7*perturb])
    ck_unit = compute_C_nmp(velocity, alpha.ravel(), shift.ravel(), nn, nm, np_, 2)
    # Apply nonuniform density in real space before the final transform.
    c = jnp.fft.irfftn(ck_unit, s=(nz, ny, nx), axes=(-1, -3, -2), norm='forward')
    ck = fft(c * rho[:, None, None, None]).reshape(-1, ny, nx//2+1, nz)
    e = jnp.stack([0.2 + 0.11*jnp.cos(xx), -0.1 + 0.03*jnp.sin(yy),
                   0.17 + 0.07*jnp.cos(zz)])
    fk = fft(jnp.concatenate([e, jnp.zeros_like(e)]))
    return p, ck, fk, rho, velocity, e


def measure(p, ck, fk, nx, sizes=(3, 3, 3)):
    return species_energization(ck, fk, p, Nx=nx, Nn=sizes[0], Nm=sizes[1], Np=sizes[2])


@pytest.mark.parametrize('nx', [5, 6])
def test_shifted_anisotropic_maxwellian_maps_and_averages(nx):
    p, ck, fk, rho, velocity, e = case(nx)
    d = measure(p, ck, fk, nx)
    mass = jnp.array([1., 7.])[:, None, None, None]
    thermal = 0.25*mass*rho*jnp.sum(p['alpha_s'].reshape(2, 3)**2, axis=1)[:, None, None, None]
    bulk = 0.5*mass*rho*jnp.sum(velocity**2, axis=1)
    np.testing.assert_allclose(d['density'], rho, atol=2e-14)
    np.testing.assert_allclose(d['momentum'], mass[:, None]*rho[:, None]*velocity, atol=2e-14)
    for key, expected in [('kinetic_energy', thermal+bulk), ('bulk_energy', bulk), ('internal_energy', thermal)]:
        np.testing.assert_allclose(d[key+'_density'], expected, atol=2e-14)
        np.testing.assert_allclose(d[key], expected.mean(axis=(-3, -2, -1)), atol=2e-14)
    work = (p['qs'][:, None, None, None]*rho*jnp.sum(velocity*e[None], axis=1)).mean(axis=(-3, -2, -1))
    np.testing.assert_allclose(d['j_dot_e'], work, atol=2e-14)
    np.testing.assert_allclose(d['kinetic_energy'], d['bulk_energy']+d['internal_energy'], atol=2e-14)
    current = plasma_current(p['qs'], p['alpha_s'], p['u_s'], ck, 3, 3, 3, 2)
    reconstructed = fft(jnp.sum(p['qs'][:, None, None, None, None]*d['momentum']/mass[:, None], axis=0))
    np.testing.assert_allclose(current, reconstructed, atol=2e-14)
    output = dict(p, Ck=ck[None], Fk=fk[None])
    diagnostics(output)
    np.testing.assert_allclose(d['kinetic_energy'], output['kinetic_energy_species'][0], atol=2e-14)


def test_basis_centered_maxwellian_and_missing_modes():
    # A single coefficient still includes thermal variance in all three axes.
    p, ck, fk, rho, _, _ = case(1, sizes=(1, 1, 1), spatial=False)
    d = measure(p, ck, fk, 1, (1, 1, 1))
    mass = jnp.array([1., 7.])
    velocity = p['u_s'].reshape(2, 3)
    expected_bulk = 0.5*mass*rho[:, 0, 0, 0]*jnp.sum(velocity**2, axis=1)
    expected_internal = 0.25*mass*rho[:, 0, 0, 0]*jnp.sum(p['alpha_s'].reshape(2, 3)**2, axis=1)
    np.testing.assert_allclose(d['bulk_energy'], expected_bulk, atol=2e-14)
    np.testing.assert_allclose(d['internal_energy'], expected_internal, atol=2e-14)


@pytest.mark.parametrize('nx', [5, 6])
def test_raw_work_at_highest_rfft_mode(nx):
    p, ck, fk, *_ = case(nx, spatial=False)
    ck = jnp.zeros_like(ck).at[0, 0, 0, 0].set(1.).at[27, 0, 0, 0].set(1.)
    # A unit-amplitude cosine has coefficient 1/2 on odd grids and 1 at
    # the even-grid Nyquist frequency. Its mean square is respectively 1/2, 1.
    amplitude = 0.5 if nx % 2 else 1.
    ck = ck.at[1, 0, -1, 0].set(amplitude)
    fk = jnp.zeros_like(fk).at[0, 0, -1, 0].set(amplitude)
    d = measure(p, ck, fk, nx)
    alpha = p['alpha_s'][:3]
    expected = p['qs'][0]*jnp.prod(alpha)*alpha[0]/jnp.sqrt(2.)*amplitude
    np.testing.assert_allclose(d['j_dot_e'], jnp.array([expected, 0.]), atol=2e-14)


@pytest.mark.parametrize('sizes', [(1, 1, 1), (3, 2, 1), (4, 4, 4)])
def test_collision_source_preserves_energy_but_damps_high_modes(sizes):
    p, ck, fk, *_ = case(1, sizes=sizes, spatial=False)
    # Exercise every Hermite mode, not only a Gaussian's near-zero high modes.
    ck = ck + jnp.arange(ck.size).reshape(ck.shape)*0.001
    fk = jnp.zeros_like(fk)
    state = jnp.concatenate([ck.ravel(), fk.ravel()])
    nn, nm, np_ = sizes
    args = _solver_args(p, 1, 1, 1, nn, nm, np_, 2)
    rhs = ode_system(1, 1, 1, nn, nm, np_, 2, 0., state, args)
    dck = rhs[:ck.size].reshape(ck.shape)
    expected = -p['nu']*p['collision_matrix'][None, :, :, :, None, None, None]*ck.reshape(2, np_, nm, nn, 1, 1, 1)
    np.testing.assert_allclose(dck, expected.reshape(ck.shape), atol=2e-14)
    _, rate = jax.jvp(lambda c: measure(p, c, fk, 1, sizes)['kinetic_energy'], (ck,), (dck,))
    np.testing.assert_allclose(rate, 0, atol=2e-14)
    np.testing.assert_array_equal(measure(p, ck, fk, 1, sizes)['collision_power'], jnp.zeros(2))
    if sizes == (4, 4, 4):
        assert float(jnp.max(jnp.abs(dck))) > 0
    # A user-supplied matrix need not conserve energy: uniform drag on f.
    custom = dict(p, collision_matrix=jnp.ones_like(p['collision_matrix']))
    custom_d = measure(custom, ck, fk, 1, sizes)
    np.testing.assert_allclose(custom_d['collision_power'], -p['nu']*custom_d['kinetic_energy'], atol=2e-14)


@pytest.mark.parametrize('unresolved', [False, True])
def test_electric_power_matches_instantaneous_rhs_energy_derivative(unresolved):
    # One RHS evaluation, no trajectory. Include a Nyquist mode to test masking.
    nx = 6
    p, ck, fk, *_ = case(nx, spatial=False)
    if unresolved:
        ck = ck.at[1, 0, 3, 0].add(0.13)
        fk = fk.at[0, 0, 3, 0].add(0.19)
    args = _solver_args(p, nx, 1, 1, 3, 3, 3, 2)
    rhs = ode_system(nx, 1, 1, 3, 3, 3, 2, 0., jnp.concatenate([ck.ravel(), fk.ravel()]), args)
    dck, dfk = rhs[:ck.size].reshape(ck.shape), rhs[ck.size:].reshape(fk.shape)
    d = measure(p, ck, fk, nx)
    _, rate = jax.jvp(lambda c: measure(p, c, fk, nx)['kinetic_energy'], (ck,), (dck,))
    np.testing.assert_allclose(rate, d['electric_power']+d['collision_power'], atol=2e-14)
    if unresolved:
        assert float(jnp.max(jnp.abs(d['electric_power']-0.5*d['j_dot_e']))) > 1e-4
    else:
        np.testing.assert_allclose(d['electric_power'], 0.5*d['j_dot_e'], atol=2e-14)
        # Independent field Parseval derivative, weights [1,2,2,1].
        weights = jnp.array([1., 2., 2., 1.])[None, None, :, None]
        field_rate = p['Omega_cs'][0]**2*jnp.sum(weights*jnp.real(jnp.conj(fk)*dfk))
        np.testing.assert_allclose(rate.sum()+field_rate, 0., atol=2e-14)


def test_jit_integrand_array_and_differentiable_bulk_partition():
    p, ck, fk, *_ = case(5)

    @jax.jit
    def integrand(scale):
        d = measure(p, ck*scale, fk, 5)
        return jnp.stack([d['bulk_energy'], d['internal_energy'],
                          d['electric_power'], d['collision_power']])

    value, derivative = jax.jvp(integrand, (jnp.array(1.),), (jnp.array(1.),))
    np.testing.assert_allclose(derivative, value, atol=2e-14)
    np.testing.assert_allclose(jax.jacrev(integrand)(1.), derivative, atol=2e-14)
    np.testing.assert_allclose((integrand(1.00001)-integrand(0.99999))/0.00002, derivative, atol=2e-10)


def test_invalid_partition_is_visible_and_mass_override_is_respected():
    p, ck, fk, *_ = case(1, spatial=False)
    base = measure(p, ck, fk, 1)
    changed = measure(dict(p, ms=jnp.array([2., 21.])), ck, fk, 1)
    np.testing.assert_allclose(changed['kinetic_energy'], base['kinetic_energy']*jnp.array([2., 3.]))
    np.testing.assert_allclose(changed['j_dot_e'], base['j_dot_e'])
    bad = measure(p, -ck, fk, 1)
    assert bool(jnp.all(jnp.isnan(bad['bulk_energy'])))
    assert bool(jnp.all(jnp.isnan(bad['internal_energy'])))
    with pytest.raises(ValueError, match='State shapes'):
        measure(p, ck, fk, 4)
    with pytest.raises(ValueError, match='Specify ms/masses'):
        measure({k: v for k, v in p.items() if k != 'mi_me'}, ck, fk, 1)
