"""Publication panels from saved tail reports; NumPy/Matplotlib only, no solver.

Required: --coarse tail-control-coarse/results.json --validation tail-validation/results.json
Optional: --mechanism tail-mechanism-fine/results.json. Each input may be a local
path (.json or .json.gz) or HOST:/absolute/path for a read-only SSH cat. Running
validation reports are supported and visibly labeled incomplete. No remote code,
JAX import, optimization or simulation is performed. --output is a file prefix.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np

BLUE, ORANGE, GRAY = '#0072B2', '#D55E00', '#666666'


def read_report(location):
    """Read one explicit local/SSH file, retaining hashes for figure metadata."""
    match = re.fullmatch(r'([A-Za-z0-9_][A-Za-z0-9_.@-]*):(/.+)', location)
    if match:
        host, path = match.groups()
        raw = subprocess.check_output(
            ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
             'cat -- '+shlex.quote(path)], timeout=60)
    else:
        raw = Path(location).read_bytes()
    decoded = gzip.decompress(raw) if location.endswith('.gz') else raw
    return json.loads(decoded), dict(path=location, sha256=hashlib.sha256(decoded).hexdigest())


def resolution(settings, end):
    return (f'{settings["grid"]}² / H{settings["hermite"]} / '
            f'dt={end/settings["steps"]:g}')


def check_pair(coarse, validation, mechanism=None):
    """Reject mixed controls/physical objectives rather than silently combining them."""
    for key in ['initial_phase', 'optimized_phase', 'window']:
        if not np.array_equal(coarse[key], validation[key]):
            raise ValueError(f'Coarse and validation {key} differ')
    if coarse['nu'] != validation['nu']:
        raise ValueError('Coarse and validation nu differ')
    for case in validation['cases']:
        if 'tail_spec' in case and case['tail_spec'] != coarse['tail_spec']:
            raise ValueError('Fixed tail reference differs across reports')
        if case.get('status') == 'complete':
            for value in case['evaluations'].values():
                if not np.isfinite(value['window_tail_gain']):
                    raise ValueError('Nonfinite completed gain')
    if mechanism is not None:
        if mechanism['tail_spec'] != coarse['tail_spec'] or mechanism['nu'] != coarse['nu']:
            raise ValueError('Mechanism physical reference differs')
        for label in ['initial_phase', 'optimized_phase']:
            if not np.array_equal(mechanism['runs'][label]['phase'], coarse[label]):
                raise ValueError('Mechanism uses different saved controls')


def tolerance_ratio(reference, value, validation):
    error = abs(value-reference)
    tolerance = validation['atol']+validation['rtol']*abs(reference)
    if tolerance <= 0:
        raise ValueError('A positive comparison tolerance is required')
    return error/tolerance


def plot(coarse, validation, mechanism=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    check_pair(coarse, validation, mechanism)
    end = coarse['window'][1]
    cases = {c['name']: c for c in validation['cases']}
    primary = cases.get('primary', {})
    fine_complete = primary.get('status') == 'complete'
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.7))
    fig.subplots_adjust(left=.16, right=.97, bottom=.17, top=.88, hspace=.62, wspace=.37)
    status = validation.get('status', 'unknown')
    complete = status == 'complete' and all(c.get('status') == 'complete' for c in cases.values())
    title = 'Constrained control of signed electron tail-energy gain'
    if not complete:
        title += f'\nValidation incomplete ({status})'
    fig.suptitle(title, fontsize=12, y=.98)

    ax = axes[0, 0]
    history = coarse['history']
    gain = -np.array([row['objective'] for row in history])
    if not len(gain) or not np.all(np.isfinite(gain)):
        raise ValueError('Finite nonempty coarse training history required')
    ax.plot(np.arange(len(gain)), 1e4*gain, color=BLUE, lw=1.7)
    ax.scatter([0, len(gain)-1], 1e4*gain[[0, -1]], color=BLUE, s=20, zorder=3)
    ax.set(title='(a) Coarse training only\n'+resolution(coarse, end),
           xlabel='Optimizer iteration (initial state = 0)',
           ylabel=r'Window tail gain $\times 10^4$')
    ax.text(.34, .10, f'Optimizer success: {coarse.get("success", "unreported")}\n'
            'No fine optimization trajectory', transform=ax.transAxes, fontsize=8)

    ax = axes[0, 1]
    ax.set_title('(b) Fine frozen-control reevaluation\n'+(
        resolution(primary['settings'], end) if 'settings' in primary else 'Primary case unavailable'))
    ax.set_ylabel(r'Window tail gain $\times 10^4$')
    if fine_complete:
        baseline, best = [primary['evaluations'][label]['window_tail_gain']
                          for label in ['initial_phase', 'optimized_phase']]
        bars = ax.bar([0, 1], 1e4*np.array([baseline, best]), width=.55, color=[GRAY, BLUE])
        ax.set_xticks([0, 1], ['Baseline controls', 'Saved coarse optimum'])
        for bar, value in zip(bars, [baseline, best]):
            ax.annotate(f'{value*1e4:.6f}', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0, 4 if value >= 0 else -12), textcoords='offset points',
                        ha='center', fontsize=8)
        ax.margins(y=.22)
        ax.text(.03, .94, f'Added gain = {best-baseline:.6g}', transform=ax.transAxes,
                va='top', fontsize=8)
        ax.axhline(0, color=GRAY, lw=.5)
    else:
        ax.text(.5, .5, f'Primary case: {primary.get("status", "missing")}',
                transform=ax.transAxes, ha='center')
        ax.set_xticks([])

    ax = axes[1, 0]
    names = ['dt_halved', 'spatial_refined', 'velocity_refined']
    labels = []
    for i, name in enumerate(names):
        case = cases.get(name, {})
        label = {'dt_halved': 'Time', 'spatial_refined': 'Space + time',
                 'velocity_refined': 'Hermite'}[name]
        labels.append(label+'\n'+(resolution(case['settings'], end).replace(' / ', ' ') if 'settings' in case else 'unavailable'))
        if fine_complete and case.get('status') == 'complete':
            ratio = tolerance_ratio(primary['benefit'], case['benefit'], validation)
            ax.scatter(ratio, i, marker='o', color=BLUE, zorder=3)
            ax.annotate(f'{ratio:.2g}', (ratio, i), xytext=(4, 4), textcoords='offset points', fontsize=7)
            if 'directional_loss_derivative' in case and 'directional_loss_derivative' in primary:
                ratio = tolerance_ratio(primary['directional_loss_derivative'],
                                        case['directional_loss_derivative'], validation)
                ax.scatter(ratio, i+.20, marker='D', color=ORANGE, s=25, zorder=3)
                ax.annotate(f'{ratio:.2g}', (ratio, i+.20), xytext=(4, 4), textcoords='offset points', fontsize=7)
        else:
            ax.text(.04, i, case.get('status', 'missing'), transform=ax.get_yaxis_transform(),
                    va='center', color=GRAY, fontsize=8)
    ax.axvline(1, color=GRAY, ls='--', lw=1)
    ax.set_xscale('symlog', linthresh=1e-4)
    ax.set_xlim(left=0, right=max(3, ax.get_xlim()[1]*1.6))
    ax.set_yticks(range(3), labels, fontsize=7)
    ax.set_ylim(2.5, -.45)
    ax.set(title='(c) Agreement with fine primary case',
           xlabel='Absolute error / declared tolerance (pass ≤ 1)')
    ax.scatter([], [], color=BLUE, label='Control benefit', s=20)
    ax.scatter([], [], color=ORANGE, marker='D', label='One directional derivative', s=20)
    ax.legend(fontsize=7, frameon=False, loc='upper right')

    ax = axes[1, 1]
    if mechanism is not None:
        for label, color, text in [('initial_phase', GRAY, 'Baseline'),
                                    ('optimized_phase', BLUE, 'Saved coarse optimum')]:
            rows = mechanism['runs'][label]['samples']
            if not rows or rows[0]['time'] != 0:
                raise ValueError('Mechanism history must include the initial state')
            q = str(max(mechanism['quadratures']))
            values = np.array([r['velocity'][q]['objective'] for r in rows])
            times = np.array([r['time'] for r in rows])
            ax.plot(times, (values-values[0])*1e4, color=color, label=text)
        ax.axvspan(*coarse['window'], color=BLUE, alpha=.08)
        ax.axhline(0, color=GRAY, lw=.6)
        ax.set(title='(d) Frozen controls: sampled time history\n'+resolution(
            mechanism, mechanism['time']), xlabel='Time',
            ylabel=r'Instantaneous own-initial tail gain $\times10^4$')
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.set_title('(d) Fine directional derivative check')
        for i, name in enumerate(['primary', 'velocity_refined']):
            case = cases.get(name, {})
            if case.get('status') == 'complete' and 'directional_loss_derivative' in case:
                value = case['directional_loss_derivative']
                ax.scatter(i, value*1e8, color=BLUE if i == 0 else ORANGE, s=30)
                ax.annotate(f'{value:.6g}', (i, value*1e8), xytext=(0, 8),
                            textcoords='offset points', ha='center', fontsize=8)
            else:
                ax.text(i, .5, case.get('status', 'missing'), transform=ax.get_xaxis_transform(), ha='center')
        ax.set_xticks([0, 1], ['Primary (reverse AD)', 'Hermite-refined (JVP)'], fontsize=8)
        ax.set(xlim=(-.5, 1.5), ylabel=r'Directional loss derivative $\times10^8$')
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        limits = ax.get_ylim()
        ax.set_ylim(min(0, limits[0])*1.15, max(0, limits[1])*1.15)
        ax.axhline(0, color=GRAY, lw=.5)
        ax.text(.03, .83, 'Same fixed direction; loss = −gain\nNot a full-gradient convergence test',
                transform=ax.transAxes, fontsize=8)

    note = (f'Gain = weighted [{coarse["window"][0]:g}, {end:g}] tail(Ct) − tail(C0), '
            'with a fixed initial thermal reference.\n'
            f'Tolerance = {validation["atol"]:g} + {validation["rtol"]:g}|reference|. '
            'Refinement is reevaluation, not fine reoptimization.\n')
    if coarse.get('endpoint_local_negativity_pass') is False:
        note += 'Coarse endpoint local-negativity gate failed. '
    note += 'Signed tail gain alone does not establish nonthermal acceleration.'
    if validation.get('source_unchanged') is False:
        note += '\nWARNING: validator reports source changes during the run.'
    fig.text(.10, .035, note, fontsize=7.5, va='bottom', linespacing=1.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coarse', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--mechanism')
    parser.add_argument('--output', type=Path, default=Path('tail_control'))
    args = parser.parse_args()
    coarse, a = read_report(args.coarse)
    validation, b = read_report(args.validation)
    provenance = [a, b]
    mechanism = None
    if args.mechanism:
        mechanism, c = read_report(args.mechanism)
        provenance.append(c)
    fig = plot(coarse, validation, mechanism)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subject = json.dumps(provenance, separators=(',', ':'))
    for extension in ['png', 'pdf']:
        target = Path(str(args.output)+'.'+extension)
        metadata = {'Title': 'Signed electron tail-energy control', 'Subject': subject,
                    'Creator': 'benchmarks/tail_control_panels.py (saved data only)'}
        fig.savefig(target, dpi=240, metadata=metadata, bbox_inches='tight')
        print(target)
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == '__main__':
    main()
