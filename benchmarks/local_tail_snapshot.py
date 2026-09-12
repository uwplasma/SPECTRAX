"""One fine snapshot of fixed controls: local tail versus moment-matched Gaussian.

This diagnoses an existing lab-frame optimum; no optimization or trajectory is
saved. Global mixtures cannot substitute for this cellwise matched reference.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from spectrax import simulation_final
from spectrax._initialization import initialize_simulation_parameters
from spectrax._energization import species_energization
from spectrax._local_tail_diagnostics import make_local_tail_diagnostics
from spectrax._velocity_observables import preregistered_tail_spec, spatial_negative_mass_diagnostics

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('phase_control',ROOT/'Examples/2D_phase_control.py')
phase=importlib.util.module_from_spec(spec);spec.loader.exec_module(phase)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--grid',type=int,default=24)
    parser.add_argument('--hermite',type=int,default=16)
    parser.add_argument('--time',type=float,default=50.)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--quadratures',nargs='+',type=int,default=[24,48,96])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    controls=json.loads(args.controls_file.read_text());g,h=args.grid,args.hermite
    base=phase.setup(np.asarray(controls['initial_phase']),g,h,args.time);base['nu']=0.
    base=initialize_simulation_parameters(base,g,g,1,h,h,h,2)
    ref=preregistered_tail_spec(float(jnp.sum(base['alpha_s'][:3]**2)/4))
    diagnostics={q:jax.jit(make_local_tail_diagnostics(base['alpha_s'],base['u_s'],Nx=g,
        Nn=h,Nm=h,Np=h,Ns=2,spec=ref,quadrature_order=q)) for q in args.quadratures}
    negativity=jax.jit(lambda C:spatial_negative_mass_diagnostics(C,base['alpha_s'],base['u_s'],
        Nx=g,Nn=h,Nm=h,Np=h,Ns=2,quadrature_order=48))
    energy=jax.jit(lambda C,F:species_energization(C,F,base,Nx=g,Nn=h,Nm=h,Np=h))
    def forward(C,F):
        p=dict(base,Ck_0=C,Fk_0=F)
        return simulation_final(p,steps=args.steps,Nx=g,Ny=g,Nn=h,Nm=h,Np=h,checkpointing=False)
    paths=['benchmarks/local_tail_snapshot.py','spectrax/_local_tail_diagnostics.py',
           'spectrax/_velocity_observables.py','spectrax/_energization.py','spectrax/_autodiff.py',
           'spectrax/_model.py','spectrax/_initialization.py','Examples/2D_phase_control.py']
    hashes=lambda:{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    report=dict(status='running',grid=g,hermite=h,steps=args.steps,time=args.time,nu=0.,
        source_sha256=hashes(),controls_sha256=hashlib.sha256(args.controls_file.read_bytes()).hexdigest(),
        device=jax.devices()[0].device_kind,jax_version=jax.__version__,runs={},
        interpretation='Cellwise flow-relative signed tail and local covariance-matched Gaussian excess; one snapshot does not establish acceleration')
    def save():
        temp=args.output/'results.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.output/'results.json')
    save();start=perf_counter();advance=jax.jit(forward).lower(base['Ck_0'],base['Fk_0']).compile()
    report['forward_compile_seconds']=perf_counter()-start
    for label in ['initial_phase','optimized_phase']:
        p=phase.setup(np.asarray(controls[label]),g,h,args.time)
        start=perf_counter();C,F=jax.block_until_ready(advance(p['Ck_0'],p['Fk_0']))
        row=dict(forward_seconds=perf_counter()-start,phase=controls[label],snapshots={})
        report['runs'][label]=row
        for when,c,f in [('initial',p['Ck_0'],p['Fk_0']),('final',C,F)]:
            values={}
            for q,fn in diagnostics.items():
                start=perf_counter();value=jax.block_until_ready(fn(c))
                values[str(q)]={k:float(v) for k,v in value.items()}
                values[str(q)]['seconds_including_first_compile']=perf_counter()-start
            d=energy(c,f)
            row['snapshots'][when]=dict(quadratures=values,
                energies={k:float(d[k][0]) for k in ['kinetic_energy','bulk_energy','internal_energy']},
                local_negativity={k:float(v) for k,v in negativity(c).items()})
            save()
        print(json.dumps(dict(run=label,status='complete')),flush=True)
    report['source_unchanged']=report['source_sha256']==hashes();report['status']='complete';save()


if __name__=='__main__':
    main()
