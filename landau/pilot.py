import json,sys
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from spectrax import simulation_final
from Examples._particle_observables import make_local_tail_diagnostics,preregistered_tail_spec,spatial_negative_mass_diagnostics
h=int(sys.argv[1]) if len(sys.argv)>1 else 128;g=int(sys.argv[2]) if len(sys.argv)>2 else 32;q=int(sys.argv[3]) if len(sys.argv)>3 else 128;v=.02;L=4*np.pi*v
alpha=jnp.array([np.sqrt(2)*v]*3+[np.sqrt(2/1836)*v]*3)
x=2*jnp.pi*jnp.arange(g)/g
phase=np.random.default_rng(7).uniform(-np.pi,np.pi,8)
m=jnp.arange(1,9);a=.45/m**2
n=1+jnp.sum(a[:,None]*jnp.cos(m[:,None]*x+phase[:,None]),axis=0)
C=jnp.zeros((2*h,1,g//2+1,1),dtype=complex).at[0,0,:,0].set(jnp.fft.rfft(n,norm='forward')/jnp.prod(alpha[:3])).at[h,0,0,0].set(1/jnp.prod(alpha[3:]))
E=-jnp.sum((a/(m*2*jnp.pi/L))[:,None]*jnp.sin(m[:,None]*x+phase[:,None]),axis=0)
F=jnp.zeros((6,1,g//2+1,1),dtype=complex).at[0,0,:,0].set(jnp.fft.rfft(E,norm='forward'))
p=dict(Ck_0=C,Fk_0=F,Lx=L,Ly=1.,Lz=1.,mi_me=1836.,qs=jnp.array([-1.,1.]),Omega_cs=jnp.array([1.,1/1836]),alpha_s=alpha,u_s=jnp.zeros(6),nu=0.,D=0.,t_max=1.)
k=dict(Nx=g,Nn=h,Nm=1,Np=1,Ns=2,quadrature_order=q)
d=jax.jit(make_local_tail_diagnostics(alpha,jnp.zeros(6),spec=preregistered_tail_spec(1.5*v*v),**k))
negative=jax.jit(lambda C:spatial_negative_mass_diagnostics(C,alpha,np.zeros(6),**k))
f=jax.jit(lambda C,F:simulation_final(dict(p,Ck_0=C,Fk_0=F),steps=round(g/32*200),Nx=g,Nn=h,Nm=1,Np=1,checkpointing=False))
rows=[]
for t in range(5):
 if t:C,F=jax.block_until_ready(f(C,F))
 row=dict(time=t,**{k:float(v) for k,v in d(C).items()});row.update({k:float(v) for k,v in negative(C).items()});rows.append(row);print(row,flush=True)
Path(f'results/landau-g{g}-h{h}-q{q}.json').write_text(json.dumps(rows,indent=2))
