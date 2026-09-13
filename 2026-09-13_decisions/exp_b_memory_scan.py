"""Experiment B: what bounds gradient memory? Scan max_steps, checkpoints, N steps, grid."""
import os, sys, time, json, resource
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
sys.path.insert(0, sys.argv[1])
from spectrax import simulation
from diffrax import Tsit5, RecursiveCheckpointAdjoint, DirectAdjoint, NoProgressMeter

def ot_params(Nx, Ny, Nn, Nm, Np, deltaB, nu, t_max):
    Ns = 2; Lx = Ly = 50.0
    x = jnp.arange(Nx) * Lx / Nx; y = jnp.arange(Ny) * Ly / Ny
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    kx = 2*jnp.pi/Lx; ky = 2*jnp.pi/Ly
    B = jnp.stack([-deltaB*jnp.sin(ky*Y), deltaB*jnp.sin(2*kx*X), jnp.ones_like(X)])
    F = jnp.concatenate([jnp.zeros_like(B), B])[..., None]
    Fk_0 = jnp.fft.rfftn(F, axes=(-1, -3, -2), norm="forward")
    Ck_0 = jnp.zeros((Ns*Np*Nm*Nn, Ny, Nx//2+1, 1), dtype=jnp.complex128)
    Ck_0 = Ck_0.at[0, 0, 0, 0].set(1/(0.25**3)).at[Nn*Nm*Np, 0, 0, 0].set(1/(0.05**3))
    return dict(Lx=Lx, Ly=Ly, Lz=1.0, mi_me=25.0, qs=jnp.array([-1.0, 1.0]), Omega_cs=jnp.array([0.5, 0.02]),
                alpha_s=jnp.array([0.25]*3+[0.05]*3), u_s=jnp.zeros(6), nu=nu, D=0.0, t_max=t_max,
                ode_tolerance=1e-7, Ck_0=Ck_0, Fk_0=Fk_0)

def run(Nx=8, Nn=3, steps=100, dt=0.02, max_steps=None, adjoint=None, tape=False, label=""):
    Ny, Nm, Np = Nx, Nn, Nn
    t_max = steps*dt
    if adjoint is None:
        adjoint = DirectAdjoint() if tape else RecursiveCheckpointAdjoint()
    ms = max_steps if max_steps is not None else steps + 8
    def loss(theta):
        out = simulation(ot_params(Nx, Ny, Nn, Nm, Np, theta[0], theta[1], t_max), Nx=Nx, Ny=Ny, Nz=1,
                         Nn=Nn, Nm=Nm, Np=Np, Ns=2, timesteps=2, dt=dt, solver=Tsit5(),
                         adaptive_time_step=False, max_steps=ms, adjoint=adjoint, progress_meter=NoProgressMeter())
        return jnp.real(out["EM_energy"][-1] + 0.1*out["kinetic_energy"][-1])
    theta = jnp.array([0.2, 1.0])
    S = (2*Nn*Nm*Np + 6) * Ny * (Nx//2+1) * 16  # state bytes (complex128)
    f = jax.jit(loss); vg = jax.jit(jax.value_and_grad(loss))
    t0 = time.time(); cf = f.lower(theta).compile(); tcf = time.time()-t0
    t0 = time.time(); cg = vg.lower(theta).compile(); tcg = time.time()-t0
    v = cf(theta).block_until_ready()
    t0 = time.time(); v = cf(theta).block_until_ready(); tf = time.time()-t0
    v2, g = cg(theta); g.block_until_ready()
    t0 = time.time(); v2, g = cg(theta); g.block_until_ready(); tg = time.time()-t0
    mf, mg = cf.memory_analysis(), cg.memory_analysis()
    rec = dict(label=label, Nx=Nx, Nn=Nn, steps=steps, max_steps=ms, adjoint=type(adjoint).__name__,
               checkpoints=getattr(adjoint, "checkpoints", None), state_MB=S/1e6,
               fwd_temp_MB=mf.temp_size_in_bytes/1e6, grad_temp_MB=mg.temp_size_in_bytes/1e6,
               fwd_s=tf, grad_s=tg, compile_fwd_s=tcf, compile_grad_s=tcg, value=float(v), grad=np.asarray(g).tolist())
    print(json.dumps(rec), flush=True)
    return rec

which = sys.argv[2]
recs = []
if which == "maxsteps":
    for ms in [200, 2000, 20000, 200000, 1000000]:
        recs.append(run(steps=100, max_steps=ms, label="max_steps scan"))
elif which == "checkpoints":
    for K in [1, 2, 4, 8, 16, 64]:
        recs.append(run(steps=200, adjoint=RecursiveCheckpointAdjoint(checkpoints=K), label="K scan"))
elif which == "nsteps":
    for n in [50, 100, 200, 400, 800, 1600]:
        recs.append(run(steps=n, adjoint=RecursiveCheckpointAdjoint(checkpoints=8), label="N scan K=8"))
    for n in [50, 100, 200, 400, 800]:
        recs.append(run(steps=n, tape=True, label="N scan tape"))
elif which == "tape":
    for n in [50, 100, 200, 400, 800]:
        recs.append(run(steps=n, adjoint=RecursiveCheckpointAdjoint(checkpoints=n), label="N scan tape(K=N)"))
    for n in [50, 100, 200, 400, 800, 1600]:
        recs.append(run(steps=n, adjoint=RecursiveCheckpointAdjoint(checkpoints=32), label="N scan K=32"))
elif which == "grid":
    for Nx, Nn in [(8, 3), (16, 3), (32, 3), (16, 4), (16, 6), (32, 6)]:
        recs.append(run(Nx=Nx, Nn=Nn, steps=50, adjoint=RecursiveCheckpointAdjoint(checkpoints=8), label="grid scan"))
json.dump(recs, open(f"exp_b_{which}.json", "w"), indent=1)
print("peak RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6)
