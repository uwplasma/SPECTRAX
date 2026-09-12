"""Scalar fine-grid held-out local-tail evaluations of frozen controls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
module=importlib.util.spec_from_file_location('local_control',ROOT/'Examples/2D_local_tail_control.py')
local=importlib.util.module_from_spec(module);module.loader.exec_module(local)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quadratures',type=int,nargs='+',default=[48,96])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    controls=json.loads(args.controls_file.read_text());initial=np.asarray(controls['initial_phase'])
    settings=dict(grid=24,hermite=16,steps=1400,window=[60.,70.],intervals=40)
    paths=['benchmarks/local_tail_holdout.py','Examples/2D_local_tail_control.py',
        'Examples/2D_phase_control.py','spectrax/_local_tail.py','spectrax/_autodiff.py',
        'spectrax/_model.py','spectrax/_initialization.py','spectrax/_velocity_observables.py']
    hashes=lambda:{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    report=dict(status='running',settings=settings,source_sha256=hashes(),
        controls_sha256=hashlib.sha256(args.controls_file.read_bytes()).hexdigest(),
        device=jax.devices()[0].device_kind,jax_version=jax.__version__,quadratures={},
        interpretation='Frozen controls; smooth normalized sampled average over held-out [60,70], gain relative to own T0 tail; no gradient or time-sampling convergence claim')
    def save():
        temp=args.output/'results.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.output/'results.json')
    save()
    for q in args.quadratures:
        loss,_,_=local.problem(initial,quadrature=q,**settings)
        start=perf_counter();fn=jax.jit(loss).lower(initial).compile()
        row=dict(compile_seconds=perf_counter()-start,runs={});report['quadratures'][str(q)]=row;save()
        for label in ['initial_phase','optimized_phase']:
            start=perf_counter();value=float(jax.block_until_ready(fn(np.asarray(controls[label]))))
            if not np.isfinite(value):raise ValueError('Nonfinite held-out value')
            row['runs'][label]=dict(gain=-value,seconds=perf_counter()-start,phase=controls[label]);save()
        row['benefit']=row['runs']['optimized_phase']['gain']-row['runs']['initial_phase']['gain'];save()
        print(json.dumps(dict(quadrature=q,benefit=row['benefit'])),flush=True)
        del fn; jax.clear_caches()
    report['source_unchanged']=hashes()==report['source_sha256'];report['status']='complete';save()


if __name__=='__main__':main()
