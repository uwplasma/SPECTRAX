"""CPU-only signed global electron spectra from saved mechanism coefficients.

No spectrax/JAX imports. Fixed range E/epsilon_th,0=0..12, snapshots 0/50/60/70,
angular rules 24x48 and 48x96. Defaults alpha=(.25,.25,.25), u=0, mass=1
are explicit phase-example conventions because older archives omit basis metadata.
Global excess can reflect mixtures of local Maxwellians; it is not local
nonthermal proof. References match instantaneous moments, not fitted tail points.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import simpson

CAVEAT = ("Signed spatial-average lab-frame spectra; excess over a global Maxwellian "
          "or covariance-matched Gaussian can reflect spatial mixtures of local "
          "flows/temperatures. Not proof of local nonthermal acceleration. Finite "
          "energy range and angular quadrature do not certify positivity.")


def moments(coefficients, alpha, u, mass=1.0):
    """Exact signed low moments for one (p,m,n) Hermite block, no quadrature."""
    c = np.asarray(coefficients)
    a, u = np.asarray(alpha, float), np.asarray(u, float)
    if (c.ndim != 3 or min(c.shape) < 1 or np.iscomplexobj(c)
            or not np.all(np.isfinite(c)) or a.shape != (3,) or u.shape != (3,)
            or not np.all(np.isfinite(a)) or np.any(a <= 0)
            or not np.all(np.isfinite(u)) or not np.isfinite(mass) or mass <= 0):
        raise ValueError('Require real finite (p,m,n) coefficients, positive alpha/mass, finite u')

    def mode(indices):
        n, m, p = indices
        return c[p, m, n] if all(i < s for i, s in zip((p, m, n), c.shape)) else 0.

    c0 = c[0, 0, 0]
    if c0 <= 0:
        raise ValueError('Nonpositive density cannot define a matched Gaussian')
    first = np.array([mode(np.eye(3, dtype=int)[i]) for i in range(3)]) / (np.sqrt(2)*c0)
    second = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            index = np.eye(3, dtype=int)[i] + np.eye(3, dtype=int)[j]
            second[i, j] = (0.5 + mode(index)/(np.sqrt(2)*c0) if i == j
                            else mode(index)/(2*c0))
    covariance = a[:, None]*a[None, :]*(second - np.outer(first, first))
    if np.min(np.linalg.eigvalsh(covariance)) <= 0:
        raise ValueError('Nonpositive covariance cannot define a matched Gaussian')
    density, mean = np.prod(a)*c0, u+a*first
    kinetic = 0.5*mass*density*(np.trace(covariance)+mean@mean)
    return dict(density=float(density), mean=mean, covariance=covariance,
                kinetic_energy=float(kinetic))


def hermite_distribution(coefficients, velocities, alpha, u):
    """Signed reconstruction at an (N,3) list; separable contractions bound memory."""
    c = np.asarray(coefficients)
    xi = (np.asarray(velocities)-np.asarray(u))/np.asarray(alpha)
    basis = []
    for size, x in zip(c.shape[::-1], xi.T):
        b = np.empty((size, len(x)))
        b[0] = 1
        if size > 1:
            b[1] = np.sqrt(2)*x
        for n in range(1, size-1):
            b[n+1] = np.sqrt(2/(n+1))*x*b[n]-np.sqrt(n/(n+1))*b[n-1]
        basis.append(b)
    bx, by, bz = basis
    partial = np.einsum('pmn,nq->pmq', c, bx, optimize=True)
    partial = np.einsum('pmq,mq->pq', partial, by, optimize=True)
    polynomial = np.einsum('pq,pq->q', partial, bz, optimize=True)
    return polynomial*np.exp(-np.sum(xi*xi, axis=1))/np.pi**1.5


def gaussian_distribution(velocities, density, mean, covariance):
    """Gaussian with number density and full velocity covariance."""
    chol = np.linalg.cholesky(covariance)
    centered = np.asarray(velocities)-mean
    standardized = np.linalg.solve(chol, centered.T)
    return (density/((2*np.pi)**1.5*np.prod(np.diag(chol)))
            * np.exp(-0.5*np.sum(standardized**2, axis=0)))


def spherical_rule(n_mu, n_phi):
    """Gauss-Legendre in cos(theta), periodic trapezoid in azimuth; sum w=4pi."""
    if n_mu < 2 or n_phi < 4:
        raise ValueError('Angular rule needs n_mu>=2 and n_phi>=4')
    mu, weights = np.polynomial.legendre.leggauss(n_mu)
    phi = 2*np.pi*np.arange(n_phi)/n_phi
    radius = np.sqrt(1-mu**2)[:, None]
    directions = np.stack(np.broadcast_arrays(radius*np.cos(phi), radius*np.sin(phi),
                                               mu[:, None]), axis=-1).reshape(-1, 3)
    return directions, np.repeat(weights, n_phi)*(2*np.pi/n_phi)


def energy_spectrum(distribution, energy_ratio, thermal_mean, *, mass=1., angular=(24, 48)):
    """Return dN/dx, x=E/epsilon_th,0. One energy shell is held at a time.

    dN/dE = (v/m) integral f(v*direction) dOmega. Multiplication by
    epsilon_th,0 converts to dN/dx. The zero-energy value is zero for smooth f.
    """
    x = np.asarray(energy_ratio, float)
    if (x.ndim != 1 or not np.all(np.isfinite(x)) or np.any(x < 0)
            or not np.isfinite(thermal_mean) or thermal_mean <= 0
            or not np.isfinite(mass) or mass <= 0):
        raise ValueError('Require nonnegative energies and positive finite thermal mean/mass')
    directions, weights = spherical_rule(*angular)
    result = np.zeros_like(x)
    for i, ratio in enumerate(x):
        if ratio > 0:
            speed = np.sqrt(2*thermal_mean*ratio/mass)
            result[i] = thermal_mean*speed/mass * (weights @ distribution(speed*directions))
    return result


def calculate_snapshot(c, alpha, u, mass, thermal_mean, x):
    """Compute signed spectra, both references, and unhidden convergence residuals."""
    m = moments(c, alpha, u, mass)
    isotropic = np.eye(3)*np.trace(m['covariance'])/3
    distributions = dict(
        hermite=lambda v: hermite_distribution(c, v, alpha, u),
        maxwellian=lambda v: gaussian_distribution(v, m['density'], m['mean'], isotropic),
        gaussian=lambda v: gaussian_distribution(v, m['density'], m['mean'], m['covariance']))
    spectra = {}
    for angular in [(24, 48), (48, 96)]:
        curves = {key: energy_spectrum(fn, x, thermal_mean, mass=mass, angular=angular)
                  for key, fn in distributions.items()}
        curves['excess_maxwellian'] = curves['hermite']-curves['maxwellian']
        curves['excess_gaussian'] = curves['hermite']-curves['gaussian']
        spectra[f'{angular[0]}x{angular[1]}'] = curves
    coarse, fine = spectra.values()
    convergence = {}
    for key in fine:
        change = simpson(np.abs(fine[key]-coarse[key]), x=x)
        norm = simpson(np.abs(fine[key]), x=x)
        convergence[key] = dict(l1_absolute=float(change),
                                l1_relative=float(change/norm) if norm > 0 else None,
                                max_absolute=float(np.max(np.abs(fine[key]-coarse[key]))))
    integrals = {key: dict(number=float(simpson(y, x=x)),
                           energy=float(thermal_mean*simpson(x*y, x=x)))
                 for key, y in fine.items()}
    return dict(moments={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in m.items()},
                spectra={q: {k: y.tolist() for k, y in curves.items()} for q, curves in spectra.items()},
                angular_convergence=convergence, finite_range_integrals=integrals,
                finite_range_hermite_residuals=dict(
                    number=integrals['hermite']['number']-m['density'],
                    energy=integrals['hermite']['energy']-m['kinetic_energy']))


def calculate(results_path, coefficients_path, snapshots, alpha, u, mass):
    source = json.loads(results_path.read_text())
    spec = source['tail_spec']
    # This input format belongs to the fixed 5*mean / .5*mean mechanism study.
    thermal_mean = spec['threshold']/5
    if not np.isclose(spec['width'], thermal_mean/2, rtol=1e-12, atol=0):
        raise ValueError('Expected registered threshold=5*mean and width=.5*mean')
    x = 12*np.linspace(0, 1, 241)**2  # fixed speed-spaced grid, no result-dependent bounds
    report = dict(caveat=CAVEAT, thermal_mean=thermal_mean, energy_ratio=x.tolist(),
                  range=[0, 12], angular_rules=[[24, 48], [48, 96]],
                  spectrum_units='dN/dx, x=E/initial_thermal_mean; plotted x*dN/dx',
                  references=dict(
                      maxwellian='Same global density, mean velocity and total energy; isotropic covariance',
                      gaussian='Same global density, mean velocity and full covariance'),
                  integration_note=('Finite-range residuals combine omitted E>12*mean contributions '
                                    'and radial/angular discretization error; no automatic pass claim.'),
                  basis=dict(alpha=list(alpha), u=list(u), mass=mass, species=0),
                  tail_spec=spec, source_metadata={k: source.get(k) for k in
                      ['grid', 'hermite', 'nu', 'steps', 'time', 'controls_sha256', 'source_sha256']},
                  input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [results_path, coefficients_path]},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), runs={})
    with np.load(coefficients_path, allow_pickle=False) as archive:
        for label, run in source['runs'].items():
            coefficients = archive[label]
            rows = run['samples']
            if (coefficients.ndim != 5 or coefficients.shape[0] != len(rows)
                    or coefficients.shape[1] < 1
                    or coefficients.shape[2:] != (source['hermite'],)*3):
                raise ValueError(f'Coefficient archive shape mismatch for {label}')
            selected = []
            for time in snapshots:
                indices = [i for i, row in enumerate(rows) if abs(row['time']-time) < 1e-9]
                if len(indices) != 1:
                    raise ValueError(f'No unique saved snapshot at {time} for {label}')
                i = indices[0]
                c = coefficients[i, 0]
                m = moments(c, alpha, u, mass)
                velocity = rows[i]['velocity'][str(max(map(int, rows[i]['velocity'])))]
                if not np.allclose([m['density'], m['kinetic_energy']],
                                   [velocity['density'], rows[i]['kinetic_energy'][0]],
                                   rtol=1e-10, atol=1e-12):
                    raise ValueError('Basis/coefficients do not match saved density and energy')
                result = calculate_snapshot(c, alpha, u, mass, thermal_mean, x)
                result.update(time=rows[i]['time'], saved_velocity=velocity,
                              local_negativity=rows[i].get('local_negativity'))
                selected.append(result)
            report['runs'][label] = selected
    return report


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.asarray(report['energy_ratio'])
    for label, snapshots in report['runs'].items():
        fig, axes = plt.subplots(len(snapshots), 2, figsize=(9, 2.35*len(snapshots)),
                                 squeeze=False, layout='constrained')
        for row, data in zip(axes, snapshots):
            curves = data['spectra']['48x96']
            for key, style in [('hermite', '-'), ('maxwellian', '--'), ('gaussian', ':')]:
                row[0].plot(x, x*np.asarray(curves[key]), style, label=key)
            for key in ['excess_maxwellian', 'excess_gaussian']:
                row[1].plot(x, x*np.asarray(curves[key]), label=key.replace('excess_', 'vs '))
            for ax in row:
                ax.axhline(0, color='gray', lw=.5)
                ax.axvline(5, color='gray', lw=.7, ls=':')
                ax.set(xlim=(0, 12), xlabel=r'$E/\epsilon_{th,0}$')
                ax.legend(frameon=False, fontsize=7)
            row[0].set_yscale('symlog', linthresh=1e-8*report['tail_spec']['normalization']/report['thermal_mean'])
            row[0].set(title=f'{label}, t={data["time"]:g}', ylabel=r'$x\,dN/dx$ (signed symlog)')
            row[1].set(title='Global excess over matched reference', ylabel=r'$x\,\Delta(dN/dx)$')
        fig.suptitle('Global spectra: spatial mixtures can produce excess; not local nonthermal proof', fontsize=10)
        for extension in ['png', 'pdf']:
            fig.savefig(output/f'{label}.{extension}', dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, help='Mechanism results.json')
    parser.add_argument('--coefficients', type=Path, help='Default: mean_coefficients.npz beside results')
    parser.add_argument('--output', type=Path, default=Path('velocity-spectrum'))
    parser.add_argument('--snapshots', nargs='+', type=float, default=[0, 50, 60, 70])
    parser.add_argument('--alpha', nargs=3, type=float, default=[.25]*3)
    parser.add_argument('--u', nargs=3, type=float, default=[0.]*3)
    parser.add_argument('--mass', type=float, default=1.)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    target = args.output/'spectra.json'
    if args.plot_only:
        report = json.loads(target.read_text())
    else:
        if args.results is None:
            parser.error('--results is required unless --plot-only')
        if target.exists():
            parser.error('Use a fresh output directory; existing spectra.json is preserved')
        report = calculate(args.results, args.coefficients or args.results.with_name('mean_coefficients.npz'),
                           args.snapshots, args.alpha, args.u, args.mass)
        args.output.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    plot(report, args.output)


if __name__ == '__main__':
    main()
