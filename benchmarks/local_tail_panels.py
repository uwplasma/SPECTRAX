"""Plot archived local-tail controls, with explicit coarse/fine distinctions."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(directory, name):
    p=directory/(name+'.json.gz')
    return json.loads(gzip.decompress(p.read_bytes()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path(__file__).parent/'results/local-tail')
    parser.add_argument('--output',type=Path,default=Path(__file__).parents[1]/'docs/figures/local_tail_control')
    args=parser.parse_args();root=args.directory
    base=read(root,'local-tail-q48');old=read(root,'local-tail-old-lab-opt')
    opt=read(root,'local-tail-optimization');fine=read(root,'local-tail-fine-q48')
    best=read(root,'local-tail-optimized-fine')
    for report in [base,old,opt,fine,best]:
        if report.get('status')!='complete' or not report.get('source_unchanged'):
            raise ValueError('Require complete reports with unchanged sources')
    if not np.array_equal(base['initial_phase'],opt['initial_phase']) or not np.array_equal(fine['initial_phase'],base['initial_phase']):
        raise ValueError('Baseline controls differ')
    if not np.array_equal(best['initial_phase'],opt['optimized_phase']):
        raise ValueError('Fine report does not evaluate saved local controls')
    if not all(np.isfinite(r['initial_loss']) for r in [base,old,fine,best]):
        raise ValueError('Nonfinite result')
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(7.5,5.8),layout='constrained')
    colors=['#777777','#D55E00','#0072B2']
    axes[0,0].bar(range(3),-1e4*np.array([base['initial_loss'],old['initial_loss'],opt['final_loss']]),color=colors)
    axes[0,0].set(xticks=range(3),xticklabels=['Baseline','Lab-tail\ncontrol','Local-tail\ncontrol'],
                  title='(a) Local-frame objective, coarse grid',ylabel=r'Window gain ($10^{-4}$)')
    hist=opt['history']
    axes[0,1].plot([r['seconds'] for r in hist],[-1e4*r['loss'] for r in hist],color=colors[-1])
    axes[0,1].set(title=f"(b) Coarse search ({opt['iterations']} iterations)",xlabel='Warm optimization time (s)',ylabel=r'Window gain ($10^{-4}$)')
    axes[1,0].bar([0,1],-1e4*np.array([fine['initial_loss'],best['initial_loss']]),color=[colors[0],colors[-1]])
    benefit=fine['initial_loss']-best['initial_loss']
    axes[1,0].set(xticks=[0,1],xticklabels=['Baseline','Saved control'],ylabel=r'Window gain ($10^{-4}$)',
                  title='(c) Fine frozen-control evaluation')
    axes[1,0].text(.03,.96,f'Added gain: {benefit:.3g}',transform=axes[1,0].transAxes,va='top',fontsize=8)
    ax=axes[1,1]
    for name,label,color in [('local-tail-q24','q24',colors[1]),('local-tail-q48','q48',colors[-1]),('local-tail-q96','q96',colors[0])]:
        r=read(root,name);ax.scatter(r['first_seconds'],-1e4*r['initial_loss'],label=label,color=color)
    ax.set(title='(d) Baseline velocity-quadrature cost',xlabel='First synchronized value/gradient (s)',ylabel=r'Window gain ($10^{-4}$)')
    ax.legend(frameon=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(str(args.output)+'.'+ext,dpi=300)
    plt.close(fig)


if __name__=='__main__':main()
