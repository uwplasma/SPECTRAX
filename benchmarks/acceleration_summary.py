"""Summarize archived mechanism reports; window integrals use saved samples."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson


def summarize(report, name):
    result = dict(case=name, grid=report['grid'], hermite=report['hermite'],
                  nu=report['nu'], steps=report['steps'], runs={},
                  integration='Simpson integration of unit-spaced saved samples, not RK-stage objective')
    for label, run in report['runs'].items():
        rows = run['samples']
        t = np.array([r['time'] for r in rows])
        z = (t-50)/10
        weight = np.where(abs(z) < 1, 15/160*(1-z*z)**2, 0.)
        values = {}
        for key in ['kinetic_energy', 'bulk_energy', 'internal_energy']:
            delta = np.array([r[key][0] for r in rows])-rows[0][key][0]
            values[key] = float(simpson(weight*delta, x=t))
        values['tail_gain'] = float(simpson(weight*np.array([
            r['velocity']['96']['objective']-rows[0]['velocity']['96']['objective'] for r in rows]), x=t))
        values['maximum_work_residual'] = max(abs(r['kinetic_energy'][0]-rows[0]['kinetic_energy'][0]
            -np.array(r['integrated_power'])[:, 0].sum()) for r in rows)
        values['q48_q96_relative_change'] = max(abs(r['velocity']['48']['objective']/r['velocity']['96']['objective']-1) for r in rows)
        values['local_negativity'] = {str(r['time']): r['local_negativity'] for r in rows if r.get('local_negativity')}
        values['worst_local_fraction'] = max(r['local_negativity']['48']['max_cell_fraction'] for r in rows if r.get('local_negativity'))
        result['runs'][label] = values
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).parent/'results/acceleration')
    parser.add_argument('--output', type=Path, default=Path(__file__).parents[1]/'docs/figures')
    args = parser.parse_args()
    rows = []
    for p in sorted(args.directory.glob('mechanism-*.json.gz')):
        d = json.loads(gzip.decompress(p.read_bytes()))
        if d['time'] >= 70:
            rows.append(summarize(d, p.name.removesuffix('.json.gz')))
    (args.directory/'mechanism_summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), layout='constrained')
    coarse = next(r for r in rows if r['case'] == 'mechanism-16h4')
    base, opt = (coarse['runs'][k] for k in ('initial_phase', 'optimized_phase'))
    keys = ['kinetic_energy', 'bulk_energy', 'internal_energy']
    axes[0, 0].bar(range(3), [1e6*(opt[k]-base[k]) for k in keys], color=['#0072B2','#D55E00','#009E73'])
    axes[0, 0].set(xticks=range(3), xticklabels=['Total', 'Bulk', 'Internal'],
                   ylabel=r'Added averaged electron energy ($10^{-6}$)', title='(a) Bulk and internal energy gains')
    for mode, color, marker, label in [('fixed', '#777777','s','Fixed nu=1'),
                                      ('matched','#D55E00','^','Matched damping rates'),
                                      ('collisionless','#0072B2','o','Collisionless')]:
        subset = [r for r in rows if (mode == 'fixed' and r['nu'] == 1)
                  or (mode == 'matched' and (r['nu'] > 1 or r['case'] == 'mechanism-16h4'))
                  or (mode == 'collisionless' and r['nu'] == 0)]
        subset.sort(key=lambda r:r['hermite'])
        h = [r['hermite'] for r in subset]
        benefit = [r['runs']['optimized_phase']['tail_gain']-r['runs']['initial_phase']['tail_gain'] for r in subset]
        axes[0, 1].plot(h, np.array(benefit)*1e5, marker=marker, color=color, label=label)
        axes[1, 0].semilogy(h, [r['runs']['optimized_phase']['local_negativity']['70.0']['48']['max_cell_fraction'] for r in subset], marker=marker, color=color)
    axes[0, 1].set(xlabel='Hermite modes per axis', ylabel=r'Added averaged tail gain ($10^{-5}$)',
                   title='(b) Tail-control benefit')
    axes[0, 1].legend(frameon=False, fontsize=7)
    axes[1, 0].axhline(1e-8, color='black', ls=':', label='Pilot tolerance')
    axes[1, 0].set(xlabel='Hermite modes per axis', ylabel='Worst-cell negative mass fraction at T=70',
                   title='(c) Local distribution validity')
    axes[1, 0].legend(frameon=False)
    x = np.arange(len(rows))
    axes[1, 1].plot(x, [max(r['runs'][k]['maximum_work_residual'] for k in r['runs'])/1e-17 for r in rows], 'o', color='#0072B2')
    axes[1, 1].set(xlabel='Diagnostic case', ylabel=r'Maximum work residual ($10^{-17}$)', ylim=(0, 2.5),
                   title='(d) Work-balance residual', xticks=x+0, xticklabels=[str(i+1) for i in x])
    args.output.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf','png'):
        fig.savefig(args.output/f'energization_mechanism_comparison.{ext}', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
