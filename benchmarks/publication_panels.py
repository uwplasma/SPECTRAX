"""Render the two manuscript candidates from versioned data; no simulations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'benchmarks/results'
OUT = ROOT / 'docs/figures'


def read(name):
    return json.loads((DATA / name).read_text())


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'legend.fontsize': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42,
                         'lines.linewidth': 1.7})
    blue, orange, green = '#0072B2', '#D55E00', '#009E73'
    base = read('window_control_cpu.json')
    refined = read('window_refined_optimization_gpu.json')
    shifts = read('window_shifts_refined_gpu.json')
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8), layout='constrained')
    ax = axs[0, 0]
    ax.plot(-100 * np.asarray(base['history']), 'o-', ms=3, color=blue)
    ax.set(xlabel='L-BFGS iteration', ylabel=r'$100\,\overline{\Delta K_e}/W_{B,\perp}(0)$',
           title='(a) Fixed-energy phase control')
    ax.text(.97, .12, '+19.82% relative gain', ha='right', transform=ax.transAxes, color=blue)
    ax = axs[0, 1]
    baseline = -100 * np.array([base['initial_objective'], refined['initial_objective']])
    optimized = -100 * np.array([base['final_objective'], refined['coarse_controls_objective']])
    x = np.arange(2)
    ax.bar(x - .18, baseline, .36, color='#888888', label='Initial phases')
    ax.bar(x + .18, optimized, .36, color=blue, label='Frozen optimized phases')
    ax.set(xticks=x, xticklabels=['16² / 4³\n600 steps', '24² / 6³\n1200 steps'],
           ylabel=r'$100\,\overline{\Delta K_e}/W_{B,\perp}(0)$',
           title='(b) Benefit survives refinement', ylim=(0, 1.28))
    ax.legend(loc='upper left', frameon=False)
    ax = axs[1, 0]
    for seed, color, marker in zip((7, 11, 23), (blue, orange, green), ('o', 's', '^')):
        rows = [r for r in shifts if r['seed'] == seed]
        ax.plot([np.mean(r['time_window']) for r in rows],
                [100 * (r['ratio'] - 1) for r in rows], marker=marker, color=color,
                label=f'Seed {seed}', ms=4)
    ax.axvline(50, color='.65', ls=':', lw=1)
    ax.set(xlabel='Center of width-20 evaluation window', ylabel='Improvement over baseline (%)',
           title='(c) Shifted windows; refined grid', ylim=(0, 27))
    ax.legend(frameon=False, ncol=3, loc='lower left')
    ax = axs[1, 1]
    initial = np.asarray(base['initial_energies'])
    magnetic = initial[1] - .125
    for dx, field, color, label in [(-.18, 'baseline_final_energies', '#888888', 'Initial phases'),
                                   (.18, 'optimized_final_energies', blue, 'Optimized phases')]:
        ax.bar(np.arange(4) + dx, 100 * (np.asarray(base[field]) - initial) / magnetic,
               .36, color=color, label=label)
    ax.axhline(0, color='.25', lw=.8)
    ax.set(xticks=np.arange(4), xticklabels=[r'$W_E$', r'$W_B$', r'$K_e$', r'$K_i$'],
           ylabel=r'$100\,[W(60)-W(0)]/W_{B,\perp}(0)$',
           title='(d) Endpoint energy at T=60')
    ax.legend(frameon=False, loc='lower right')
    save(fig, 'publication_control')

    fresh = read('initial-vectorized/results.json')
    historical = read('gpu_scaling_results.json')
    long = read('gpu_long_rollout_results.json')
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8), layout='constrained')
    ax = axs[0, 0]
    for method, label, color, marker in [('checkpointed', 'Replay reverse AD', blue, 'o'),
                                         ('forward', 'Batched forward AD', green, '^'),
                                         ('fd', 'Centered finite differences', '#777777', 's')]:
        rows = sorted([r for r in fresh if r['method'] == method], key=lambda r: r['controls'])
        ax.loglog([r['controls'] for r in rows], [r['seconds'] for r in rows],
                  marker=marker, color=color, label=label, ms=4)
    ax.set(xticks=[8, 128], xticklabels=['8', '128'], xlabel='Control parameters',
           ylabel='Warm full-gradient time (s)', title='(a) Warm gradient cost')
    ax.minorticks_off()
    ax.legend(frameon=False, loc='upper left')
    ax.text(.97, .35, '60.4× at 128 controls', ha='right', transform=ax.transAxes, color=blue)
    ax = axs[0, 1]
    fd = np.array(base['finite_differences'])
    relative = np.abs(fd[:, 1] / base['directional_ad'] - 1)
    ax.loglog(fd[:, 0], relative, 'o-', color=blue, ms=4)
    ax.set(xlabel='Centered-difference step size', ylabel='Relative directional derivative error',
           title='(b) Directional accuracy (CPU)')
    ax = axs[1, 0]
    for method, label, color in [('checkpointed', 'Replay reverse AD', blue),
                                 ('taped', 'Taped reverse AD', orange)]:
        rows = [r for r in historical if r['axis'] == 'grid' and r['method'] == method]
        ax.semilogy([r['grid'] for r in rows],
                    [r['gpu_peak_bytes_in_use'] / 2**20 for r in rows], 'o-', color=color, label=label, ms=4)
    ax.set(xticks=[16, 32, 64], xticklabels=['16²', '32²', '64²'], xlabel='Spatial grid (4³ Hermite; 64 steps)',
           ylabel='Peak JAX allocator (MiB)', title='(c) Spatial memory scaling')
    ax.legend(frameon=False)
    ax = axs[1, 1]
    for method, label, color in [('checkpointed', 'Default replay', blue),
                                 ('budgeted', 'Fixed budget K=16', '#CC79A7')]:
        rows = [r for r in long if r['method'] == method]
        ax.plot([r['steps'] for r in rows], [r['gpu_peak_bytes_in_use'] / 2**20 for r in rows],
                'o-', color=color, label=label, ms=4)
    ax.set(xticks=[256, 1024], xlabel='RK4 steps (32² / 4³)', ylabel='Peak JAX allocator (MiB)',
           title='(d) Rollout memory at fixed budget', ylim=(0, 160))
    ax.legend(frameon=False)
    for ax in axs.flat:
        ax.grid(alpha=.12, which='major')
    save(fig, 'publication_autodiff')


if __name__ == '__main__':
    main()
