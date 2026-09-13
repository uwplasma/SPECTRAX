"""Experiment E: forward cost and adaptive step counts of the OT case at moderate resolutions (CPU)."""
import os, sys, time, json
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
sys.path.insert(0, sys.argv[1])
from spectrax import simulation, compute_C_nmp
from diffrax import Dopri8, Tsit5, NoProgressMeter, RecursiveCheckpointAdjoint

def ot(Nx, Nn, t_max, deltaB=0.2):
    Ny = Nx; Nm = Np = Nn; Ns = 2; Lx = Ly = 50.0; Omega_ce = 0.5; mi_me = 25.0
    x = jnp.arange(Nx) * Lx / Nx; y = jnp.arange(Ny) * Ly / Ny
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    kx = 2*jnp.pi/Lx; ky = 2*jnp.pi/Ly
    U0 = deltaB * Omega_ce / jnp.sqrt(mi_me)
    Ue = jnp.stack([-U0*jnp.sin(ky*Y), U0*jnp.sin(kx*X), -deltaB*Omega_ce*(2*kx*jnp.cos(2*kx*X) + ky*jnp.cos(ky*Y))])
    Ui = jnp.stack([-U0*jnp.sin(ky*Y), U0*jnp.sin(kx*X), jnp.zeros_like(X)])
    B = jnp.stack([-deltaB*jnp.sin(ky*Y), deltaB*jnp.sin(2*kx*X), jnp.ones_like(X)])
    F = jnp.concatenate([jnp.zeros_like(B), B])[..., None]
    alpha_s = jnp.array([0.25]*3 + [0.05]*3); u_s = jnp.zeros(6)
    Us = jnp.stack([Ue, Ui])[..., None]  # (Ns, 3, Ny, Nx, Nz)
    Ck_0 = compute_C_nmp(Us, alpha_s, u_s, Nn, Nm, Np, Ns).reshape(Ns*Np*Nm*Nn, Ny, Nx//2+1, 1)
    Fk_0 = jnp.fft.rfftn(F, axes=(-1, -3, -2), norm="forward")
    return dict(Lx=Lx, Ly=Ly, Lz=1.0, mi_me=mi_me, qs=jnp.array([-1.0, 1.0]), Omega_cs=jnp.array([Omega_ce, Omega_ce/mi_me]),
                alpha_s=alpha_s, u_s=u_s, nu=1.0, D=0.0, t_max=t_max, ode_tolerance=1e-7, Ck_0=Ck_0, Fk_0=Fk_0)

recs = []
for Nx, Nn, t_max in [(16, 4, 10.0), (32, 4, 10.0), (32, 6, 10.0), (64, 6, 5.0)]:
    p = ot(Nx, Nn, t_max)
    kw = dict(Nx=Nx, Ny=Nx, Nz=1, Nn=Nn, Nm=Nn, Np=Nn, Ns=2, timesteps=3, progress_meter=NoProgressMeter())
    f = jax.jit(lambda p: simulation(p, dt=1e-3, solver=Dopri8(), **kw))
    t0 = time.time(); out = f(p); jax.block_until_ready(out["Fk"]); tc = time.time()-t0
    t0 = time.time(); out = f(p); jax.block_until_ready(out["Fk"]); tr = time.time()-t0
    st = {k: int(v) for k, v in out["solver_stats"].items()}
    S = (2*Nn**3 + 6) * Nx * (Nx//2+1) * 16 / 1e6
    E0 = float(jnp.real(out["total_energy"][0])); E1 = float(jnp.real(out["total_energy"][-1]))
    rec = dict(Nx=Nx, Nn=Nn, t_max=t_max, state_MB=S, compile_s=tc, run_s=tr, steps=st["num_accepted_steps"],
               rejected=st["num_rejected_steps"], mean_dt=t_max/st["num_accepted_steps"], s_per_step=tr/st["num_steps"],
               rel_energy_err=abs(E1-E0)/abs(E0), UB0=float(out["EM_energy"][0]), UB1=float(out["EM_energy"][-1]))
    print(json.dumps(rec), flush=True); recs.append(rec)
json.dump(recs, open("exp_e_calibration.json", "w"), indent=1)
