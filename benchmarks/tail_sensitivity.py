"""CPU NumPy/SciPy sensitivity from saved spatial-mean Hermite coefficients.

All three thresholds are reported; none is selected or optimized. Sampled Simpson
integrals are not the RK-stage objective and need their own time-sampling check.
The uniform held-out gain is relative to each control's t=0 tail, not its t=60 tail.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import simpson
from scipy.special import expit, roots_hermite

LABELS = ('initial_phase', 'optimized_phase')
MULTIPLIERS = (4, 5, 6)
ORDERS = (48, 96)


def basis(count, nodes):
    """Normalized physicists' Hermite polynomials, without Gaussian weight."""
    out = np.empty((count, len(nodes)))
    out[0] = 1.
    if count > 1:
        out[1] = np.sqrt(2.) * nodes
    for n in range(1, count - 1):
        out[n+1] = np.sqrt(2./(n+1))*nodes*out[n] - np.sqrt(n/(n+1))*out[n-1]
    return out


def tail_kernel(shape, alpha, shift, epsilon0, normalization, multiplier, order):
    """Contract a lab-energy kernel to (p,m,n), electron mass one.

    C already carries inverse-alpha factors. Only the velocity Jacobian is
    applied here; GH weights already contain exp(-xi**2). Storage is O(q**3),
    with no time-by-velocity or space-by-velocity array.
    """
    nodes, weights = roots_hermite(order)
    bx, by, bz = (basis(n, nodes) for n in shape[::-1])
    vx, vy, vz = (alpha[i]*nodes + shift[i] for i in range(3))
    energy = .5*(vz[:, None, None]**2 + vy[None, :, None]**2 + vx[None, None, :]**2)
    measure = (np.prod(alpha)/np.pi**1.5 * weights[:, None, None]
               * weights[None, :, None]*weights[None, None, :])
    weighted = measure*energy*expit((energy-multiplier*epsilon0)/(.5*epsilon0))/normalization
    return np.einsum('zyx,nx,my,pz->pmn', weighted, bx, by, bz, optimize=True)


def sampled_windows(times, values):
    """Integrate saved values, requiring actual samples at window boundaries."""
    output = {}
    for name, lo, hi in (('training', 40., 60.), ('heldout', 60., 70.)):
        indices = []
        for boundary in (lo, hi):
            matches = np.flatnonzero(np.isclose(times, boundary, rtol=0, atol=1e-10))
            if len(matches) != 1:
                raise ValueError(f'Require one saved sample at t={boundary}; no interpolation is performed')
            indices.append(matches[0])
        left, right = indices
        if right-left < 2:
            raise ValueError(f'Require at least three samples in {name} window')
        t, v = times[left:right+1], values[left:right+1]
        if name == 'training':
            z = (2*t-lo-hi)/(hi-lo)
            w = 15/(8*(hi-lo))*np.maximum(1-z*z, 0)**2
        else:
            w = np.full(t.shape, 1/(hi-lo))
        output[name] = dict(window=[lo, hi], samples=len(t),
                            max_sample_spacing=float(np.max(np.diff(t))),
                            sampled_weight_integral=float(simpson(w, x=t)),
                            weighted_tail=float(simpson(w*v, x=t)),
                            gain=float(simpson(w*(v-values[0]), x=t)))
    return output


def discrepancy(low, high):
    absolute = abs(low-high)
    return dict(q48=low, q96=high, absolute_difference=absolute,
                relative_to_q96=absolute/abs(high) if high else None)


def analyze(coefficients, report, alpha, shift):
    """Evaluate all fixed thresholds at both quadratures; signed coefficients only."""
    alpha, shift = np.asarray(alpha, dtype=float), np.asarray(shift, dtype=float)
    if (alpha.shape != (3,) or shift.shape != (3,) or not np.all(np.isfinite(alpha))
            or not np.all(alpha > 0) or not np.all(np.isfinite(shift))):
        raise ValueError('Require finite positive alpha and finite shift, each length three')
    original = report['tail_spec']
    epsilon0 = float(original['threshold'])/5.
    normalization = float(original['normalization'])
    if (not np.isfinite(epsilon0) or epsilon0 <= 0 or not np.isfinite(normalization)
            or normalization <= 0 or not np.isclose(original['width'], .5*epsilon0, rtol=1e-12, atol=0)):
        raise ValueError('Expected original threshold=5 epsilon0, width=.5 epsilon0 and positive normalization')
    blocks, time_grids = {}, {}
    for label in LABELS:
        c = np.asarray(coefficients[label])
        times = np.array([r['time'] for r in report['runs'][label]['samples']], dtype=float)
        if (c.ndim != 5 or c.shape[1] < 1 or c.shape[0] != len(times)
                or min(c.shape[2:]) < 1 or not np.isrealobj(c) or not np.all(np.isfinite(c))
                or len(times) < 3 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)
                or not np.isclose(times[0], 0., rtol=0, atol=1e-10)):
            raise ValueError(f'Invalid {label} (time,species,p,m,n) coefficients or sample times')
        density0 = np.prod(alpha)*c[0, 0, 0, 0, 0]
        if not np.isclose(density0, normalization/epsilon0, rtol=1e-10, atol=1e-12):
            raise ValueError(f'{label} initial density disagrees with the fixed reference; check alpha/NPZ/report')
        blocks[label], time_grids[label] = c[:, 0], times
    shape = blocks[LABELS[0]].shape[1:]
    if blocks[LABELS[1]].shape[1:] != shape:
        raise ValueError('Control Hermite shapes must match')
    if not np.array_equal(time_grids[LABELS[0]], time_grids[LABELS[1]]):
        raise ValueError('Control sample times must match for benefit comparisons')
    result = dict(epsilon0=epsilon0, normalization=normalization, width=.5*epsilon0,
                  alpha=alpha.tolist(), shift=shift.tolist(), mass=1., hermite_shape=list(shape),
                  thresholds={}, times=time_grids[LABELS[0]].tolist())
    for multiplier in MULTIPLIERS:
        entry = dict(threshold=multiplier*epsilon0, quadratures={})
        for order in ORDERS:
            kernel = tail_kernel(shape, alpha, shift, epsilon0, normalization, multiplier, order)
            runs = {}
            for label in LABELS:
                values = np.einsum('tpmn,pmn->t', blocks[label], kernel)
                if not np.all(np.isfinite(values)):
                    raise ValueError('Nonfinite signed tail integral')
                runs[label] = dict(initial_tail=float(values[0]), sampled_tail=values.tolist(),
                                   windows=sampled_windows(time_grids[label], values))
            benefits = {w: runs['optimized_phase']['windows'][w]['gain']
                           - runs['initial_phase']['windows'][w]['gain'] for w in ('training', 'heldout')}
            entry['quadratures'][str(order)] = dict(runs=runs, benefits=benefits)
        low, high = (entry['quadratures'][str(q)] for q in ORDERS)
        entry['quadrature_differences'] = dict(
            benefits={w: discrepancy(low['benefits'][w], high['benefits'][w]) for w in ('training', 'heldout')},
            runs={label: dict(
                initial_tail=discrepancy(low['runs'][label]['initial_tail'], high['runs'][label]['initial_tail']),
                gains={w: discrepancy(low['runs'][label]['windows'][w]['gain'],
                                      high['runs'][label]['windows'][w]['gain']) for w in ('training', 'heldout')},
                maximum_sample_tail_absolute_difference=float(np.max(np.abs(
                    np.asarray(low['runs'][label]['sampled_tail'])-high['runs'][label]['sampled_tail']))))
                for label in LABELS})
        result['thresholds'][str(multiplier)] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coefficients', type=Path, required=True, help='Mechanism mean_coefficients.npz')
    parser.add_argument('--report', type=Path, required=True, help='Matching mechanism results.json or .json.gz')
    parser.add_argument('--output', type=Path, required=True, help='New output JSON; never overwrite')
    parser.add_argument('--alpha', type=float, nargs=3, default=[.25, .25, .25],
                        help='Electron basis widths; default prescribed phase-control setup')
    parser.add_argument('--shift', type=float, nargs=3, default=[0., 0., 0.],
                        help='Electron basis shift, not physical bulk velocity')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new JSON path')
    raw = args.report.read_bytes()
    report = json.loads(gzip.decompress(raw) if args.report.suffix == '.gz' else raw)
    with np.load(args.coefficients, allow_pickle=False) as archive:
        result = analyze(archive, report, args.alpha, args.shift)
    result.update(
        interpretation='All thresholds 4,5,6 epsilon0 are sensitivity results, with no selection or retraining. '
        'Signed lab-frame tail energy / fixed initial thermal energy density; own t=0 subtraction. '
        'Sampled Simpson windows are not RK-stage objectives; sample-time refinement is separate. '
        'Heldout [60,70] measures mean gain relative to t=0, not energy gained since t=60. '
        'No positivity, nonthermal-acceleration, gradient or physical-convergence certification.',
        basis_origin='Explicit CLI values; defaults match Examples/2D_phase_control.py',
        mechanism_settings={k: report.get(k) for k in ('grid', 'hermite', 'nu', 'steps', 'time', 'controls_sha256')},
        mechanism_source_sha256=report.get('source_sha256'),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.coefficients, args.report)},
        numpy_version=np.__version__, scipy_version=scipy.__version__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(str(args.output.resolve()))


if __name__ == '__main__':
    main()
