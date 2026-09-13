"""Experiment A: does jax.grad flow through the unmodified simulation() on main?"""
import os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
sys.path.insert(0, sys.argv[1])  # worktree root
from spectrax import simulation, initialize_simulation_parameters
from diffrax import Dopri8, Tsit5

Nx = Ny = 8; Nn = Nm = Np = 3; Ns = 2

def ot_params(deltaB, nu, t_max=2.0):
    Lx = Ly = 50.0
    x = jnp.arange(Nx) * Lx / Nx; y = jnp.arange(Ny) * Ly / Ny
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    kx = 2*jnp.pi/Lx; ky = 2*jnp.pi/Ly
    B = jnp.stack([-deltaB*jnp.sin(ky*Y), deltaB*jnp.sin(2*kx*X), jnp.ones_like(X)])
    E = jnp.zeros_like(B)
    F = jnp.concatenate([E, B])[..., None]
    Fk_0 = jnp.fft.rfftn(F, axes=(-1, -3, -2), norm="forward")
    alpha_s = jnp.array([0.25]*3 + [0.05]*3)
    Ck_0 = jnp.zeros((Ns*Np*Nm*Nn, Ny, Nx//2+1, 1), dtype=jnp.complex128)
    Ck_0 = Ck_0.at[0, 0, 0, 0].set(1/(0.25**3)).at[Nn*Nm*Np, 0, 0, 0].set(1/(0.05**3))
    return dict(Lx=Lx, Ly=Ly, Lz=1.0, mi_me=25.0, qs=jnp.array([-1.0, 1.0]),
                Omega_cs=jnp.array([0.5, 0.02]), alpha_s=alpha_s, u_s=jnp.zeros(6),
                nu=nu, D=0.0, t_max=t_max, ode_tolerance=1e-7, Ck_0=Ck_0, Fk_0=Fk_0)

def make_loss(**kw):
    def loss(theta):
        out = simulation(ot_params(theta[0], theta[1]), Nx=Nx, Ny=Ny, Nz=1, Nn=Nn, Nm=Nm, Np=Np, Ns=Ns,
                         timesteps=3, **kw)
        return jnp.real(out["EM_energy"][-1] + 0.1*out["kinetic_energy"][-1])
    return loss

theta = jnp.array([0.2, 1.0])
for name, kw in [("adaptive Dopri8", dict(dt=0.01)),
                 ("fixed Dopri8 dt=0.02", dict(dt=0.02, adaptive_time_step=False)),
                 ("fixed Tsit5 dt=0.02", dict(dt=0.02, adaptive_time_step=False, solver=Tsit5()))]:
    print("==", name, flush=True)
    loss = make_loss(**kw)
    try:
        f = jax.jit(loss); t0 = time.time(); v = float(f(theta)); tc = time.time()-t0
        t0 = time.time(); v = float(f(theta)); tf = time.time()-t0
        vg = jax.jit(jax.value_and_grad(loss))
        t0 = time.time(); v2, g = vg(theta); g.block_until_ready(); tgc = time.time()-t0
        t0 = time.time(); v2, g = vg(theta); g.block_until_ready(); tg = time.time()-t0
        eps = 1e-5
        fd = [float((f(theta + eps*e) - f(theta - eps*e))/(2*eps)) for e in jnp.eye(2)]
        ma_f = f.lower(theta).compile().memory_analysis(); ma_g = vg.lower(theta).compile().memory_analysis()
        print(f"value {v:.10e}  grad {np.asarray(g)}  fd {np.asarray(fd)}")
        print(f"  rel err {np.abs(np.asarray(g)-np.asarray(fd))/np.abs(np.asarray(fd))}")
        print(f"  forward: compile {tc:.1f}s run {tf*1e3:.0f}ms | value_and_grad: compile {tgc:.1f}s run {tg*1e3:.0f}ms | grad/forward {tg/tf:.1f}x")
        print(f"  compiled temp bytes: forward {ma_f.temp_size_in_bytes/1e6:.2f} MB, grad {ma_g.temp_size_in_bytes/1e6:.2f} MB", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(limit=3)
        print("FAILED:", type(e).__name__, str(e)[:800], flush=True)
