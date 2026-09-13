"""Experiment C: prototype of the showcase — optimize the shape of the initial in-plane magnetic field
(stream-function Fourier modes at fixed in-plane magnetic energy, with the consistent electron current)
for (a) maximal magnetic-energy conversion and (b) maximal peak out-of-plane current at time T.
Checks AD gradient vs FD, timing AD vs FD for P controls, and a few L-BFGS-B iterations."""
import os, sys, time, json
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, sys.argv[1])
from spectrax import simulation, compute_C_nmp, plasma_current
from diffrax import Dopri8, NoProgressMeter, RecursiveCheckpointAdjoint

Nx = Ny = int(os.environ.get("NX", 16)); Nn = Nm = Np = int(os.environ.get("NH", 4)); Ns = 2
T = float(os.environ.get("T", 200.0)); M = int(os.environ.get("M", 4))
Lx = Ly = 50.0; Omega_ce = 0.5; mi_me = 25.0; deltaB = 0.2; nu = 1.0
alpha_s = jnp.array([0.25]*3 + [0.05]*3); u_s = jnp.zeros(6)
x = jnp.arange(Nx) * Lx / Nx; y = jnp.arange(Ny) * Ly / Ny
X, Y = jnp.meshgrid(x, y, indexing="xy")
kx = 2*jnp.pi/Lx; ky = 2*jnp.pi/Ly
# lowest |k| wavevectors (m, n) with m>=0 (cosine with phase covers +-k), excluding (0,0)
modes = sorted([(m, n) for m in range(0, 4) for n in range(-3, 4) if (m, n) != (0, 0) and (m > 0 or n > 0)],
               key=lambda mn: (mn[0]**2 + mn[1]**2, mn))[:M]
KX = jnp.array([m*kx for m, n in modes]); KY = jnp.array([n*ky for m, n in modes])
K2 = KX**2 + KY**2

def initial_condition(theta):
    """theta = (log-weights w_k, phases phi_k). psi = sum a_k cos(k.x + phi_k), <|B_perp|^2> = deltaB^2 fixed."""
    w, phi = theta[:M], theta[M:]
    a = jnp.exp(w)
    a = a * deltaB / jnp.sqrt(0.5*jnp.sum(a**2 * K2))      # fixed in-plane magnetic energy
    phase = KX[:, None, None]*X + KY[:, None, None]*Y + phi[:, None, None]
    s, c = jnp.sin(phase), jnp.cos(phase)
    Bx = -jnp.sum(a[:, None, None]*KY[:, None, None]*s, axis=0)
    By = jnp.sum(a[:, None, None]*KX[:, None, None]*s, axis=0)
    Jz = jnp.sum(a[:, None, None]*K2[:, None, None]*c, axis=0)   # (curl B)_z = -lap psi
    U0 = deltaB * Omega_ce / jnp.sqrt(mi_me)
    Ue = jnp.stack([-U0*jnp.sin(ky*Y), U0*jnp.sin(kx*X), -Omega_ce*Jz])   # electrons carry the current
    Ui = jnp.stack([-U0*jnp.sin(ky*Y), U0*jnp.sin(kx*X), jnp.zeros_like(X)])
    B = jnp.stack([Bx, By, jnp.ones_like(X)])
    F = jnp.concatenate([jnp.zeros_like(B), B])[..., None]
    Us = jnp.stack([Ue, Ui])[..., None]
    Ck_0 = compute_C_nmp(Us, alpha_s, u_s, Nn, Nm, Np, Ns).reshape(Ns*Np*Nm*Nn, Ny, Nx//2+1, 1)
    Fk_0 = jnp.fft.rfftn(F, axes=(-1, -3, -2), norm="forward")
    return dict(Lx=Lx, Ly=Ly, Lz=1.0, mi_me=mi_me, qs=jnp.array([-1.0, 1.0]), Omega_cs=jnp.array([Omega_ce, Omega_ce/mi_me]),
                alpha_s=alpha_s, u_s=u_s, nu=nu, D=0.0, t_max=T, ode_tolerance=1e-7, Ck_0=Ck_0, Fk_0=Fk_0)

def run(theta):
    return simulation(initial_condition(theta), Nx=Nx, Ny=Ny, Nz=1, Nn=Nn, Nm=Nm, Np=Np, Ns=Ns, timesteps=2, dt=0.01,
                      solver=Dopri8(), max_steps=100000, adjoint=RecursiveCheckpointAdjoint(checkpoints=32),
                      progress_meter=NoProgressMeter())

wts = jnp.where(jnp.arange(Nx//2+1) == 0, 1.0, 2.0).at[-1].set(1.0 if Nx % 2 == 0 else 2.0)[None, None, :, None]
def inplane_magnetic_energy(out, i=-1):
    return 0.5*Omega_ce**2*jnp.sum(jnp.abs(out["Fk"][i, 3:5])**2 * wts)
def current_density(out, i=-1):
    Jk = plasma_current(out["qs"], out["alpha_s"], out["u_s"], out["Ck"][i], Nn, Nm, Np, Ns)
    return jnp.fft.irfftn(Jk, s=(1, Ny, Nx), axes=(-1, -3, -2), norm="forward")[2, :, :, 0]
def peak_current(out, p=8):
    Jz = current_density(out)
    return jnp.mean(jnp.abs(Jz)**p)**(1/p)

objectives = {"conversion": lambda out: inplane_magnetic_energy(out) / inplane_magnetic_energy(out, 0),   # minimize -> convert
              "peak_current": lambda out: -peak_current(out)}
rng = np.random.default_rng(0)
theta0 = jnp.concatenate([jnp.zeros(M), jnp.asarray(rng.uniform(-np.pi, np.pi, M))])
results = {"settings": dict(Nx=Nx, Nn=Nn, T=T, M=M, P=2*M, modes=modes)}
for name, obj in objectives.items():
    loss = lambda th: obj(run(th))
    f = jax.jit(loss); vg = jax.jit(jax.value_and_grad(loss))
    t0 = time.time(); v0 = float(f(theta0)); tcf = time.time()-t0
    t0 = time.time(); v0 = float(f(theta0)); tf = time.time()-t0
    t0 = time.time(); v, g = vg(theta0); g.block_until_ready(); tcg = time.time()-t0
    t0 = time.time(); v, g = vg(theta0); g.block_until_ready(); tg = time.time()-t0
    eps = 1e-4; t0 = time.time()
    fd = np.array([(float(f(theta0+eps*e)) - float(f(theta0-eps*e)))/(2*eps) for e in jnp.eye(2*M)]); tfd = time.time()-t0
    g = np.asarray(g)
    rel = np.linalg.norm(g-fd)/np.linalg.norm(fd)
    steps = int(run(theta0)["solver_stats"]["num_accepted_steps"])
    print(f"[{name}] value {v0:.6g} steps {steps} | fwd {tf:.2f}s (compile {tcf:.1f}s) | AD grad {tg:.2f}s (compile {tcg:.1f}s) = {tg/tf:.1f}x fwd | FD {2*M} solves {tfd:.1f}s = {tfd/tg:.1f}x AD | grad rel err vs FD {rel:.2e}", flush=True)
    hist = []
    def fg(th):
        v, g = vg(jnp.asarray(th)); hist.append(float(v)); return float(v), np.asarray(g, dtype=float)
    t0 = time.time()
    res = minimize(fg, np.asarray(theta0), jac=True, method="L-BFGS-B", options=dict(maxiter=int(os.environ.get("ITERS", 10))))
    print(f"[{name}] L-BFGS-B: {res.nit} iterations, {res.nfev} evaluations, {time.time()-t0:.1f}s, objective {v0:.6g} -> {float(res.fun):.6g}  ({res.message})", flush=True)
    results[name] = dict(value0=v0, value_opt=float(res.fun), nit=int(res.nit), nfev=int(res.nfev), fwd_s=tf, grad_s=tg, fd_s=tfd,
                         grad_rel_err_fd=float(rel), history=hist, theta_opt=np.asarray(res.x).tolist(), steps=steps)
json.dump(results, open(f"exp_c_showcase_N{Nx}_H{Nn}_T{int(T)}_M{M}.json", "w"), indent=1)
