"""Experiment D: (1) continuous adjoint (BacksolveAdjoint) vs discrete checkpointed adjoint;
(2) McCallum-Foster reversible RK4 with O(1)-in-N memory: reconstruction stability and gradient accuracy
for collisionless (nu=0) and collisional (nu>0) tiny Orszag-Tang runs."""
import os, sys, time, json
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
sys.path.insert(0, sys.argv[1])
from spectrax import initialize_simulation_parameters
from spectrax._simulation import ode_system
import diffrax

Nx = Ny = 8; Nn = Nm = Np = 4; Ns = 2; Nz = 1  # Nn>=4 so the collision operator is nonzero

def params(deltaB, nu, t_max, cast=False):
    Lx = Ly = 50.0
    x = jnp.arange(Nx) * Lx / Nx; y = jnp.arange(Ny) * Ly / Ny
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    kx = 2*jnp.pi/Lx; ky = 2*jnp.pi/Ly
    B = jnp.stack([-deltaB*jnp.sin(ky*Y), deltaB*jnp.sin(2*kx*X), jnp.ones_like(X)])
    F = jnp.concatenate([jnp.zeros_like(B), B])[..., None]
    Fk_0 = jnp.fft.rfftn(F, axes=(-1, -3, -2), norm="forward")
    Ck_0 = jnp.zeros((Ns*Np*Nm*Nn, Ny, Nx//2+1, 1), dtype=jnp.complex128)
    Ck_0 = Ck_0.at[0, 0, 0, 0].set(1/(0.25**3)).at[Nn*Nm*Np, 0, 0, 0].set(1/(0.05**3))
    p = dict(Lx=Lx, Ly=Ly, Lz=1.0, mi_me=25.0, qs=jnp.array([-1.0, 1.0]), Omega_cs=jnp.array([0.5, 0.02]),
             alpha_s=jnp.array([0.25]*3+[0.05]*3), u_s=jnp.zeros(6), nu=nu, D=0.0, t_max=t_max, Ck_0=Ck_0, Fk_0=Fk_0)
    p = initialize_simulation_parameters(p, Nx, Ny, Nz, Nn, Nm, Np, Ns)
    args = (Nx, Ny, Nz, Nn, Nm, Np, Ns, p["qs"], p["nu"], p["D"], p["Omega_cs"], p["alpha_s"], p["u_s"],
            p["Lx"], p["Ly"], p["Lz"], p["kx_grid"], p["ky_grid"], p["kz_grid"], p["k2_grid"], p["nabla"],
            p["collision_matrix"], p["sqrt_n_plus"], p["sqrt_n_minus"], p["sqrt_m_plus"], p["sqrt_m_minus"],
            p["sqrt_p_plus"], p["sqrt_p_minus"])
    y0 = jnp.concatenate([p["Ck_0"].ravel(), p["Fk_0"].ravel()])
    if cast:  # BacksolveAdjoint needs one dtype for the differentiated leaves (qs, nu) of the augmented state
        args = args[:7] + (args[7].astype(jnp.complex128), args[8].astype(jnp.complex128)) + args[9:]
    return y0, args

rhs = lambda t, y, args: ode_system(Nx, Ny, Nz, Nn, Nm, Np, Ns, t, y, args)
nF = 6*Ny*(Nx//2+1)*Nz
def objective(y):  # in-plane magnetic energy (dynamics-sensitive, not dominated by constant modes)
    F = y[-nF:].reshape(6, Ny, Nx//2+1, Nz)
    return jnp.sum(jnp.abs(F[3:5])**2)

# ---------- (1) continuous vs discrete adjoint ----------
def solve(theta, adjoint, tol, T, solver=diffrax.Dopri5()):
    y0, args = params(theta[0], theta[1], T, cast=True)
    sol = diffrax.diffeqsolve(diffrax.ODETerm(rhs), solver, t0=0.0, t1=T, dt0=0.01, y0=y0, args=args,
                              stepsize_controller=diffrax.PIDController(rtol=tol, atol=tol),
                              saveat=diffrax.SaveAt(t1=True), max_steps=100000, adjoint=adjoint)
    return objective(sol.ys[0])

theta = jnp.array([0.2, 1.0])
out = {"continuous_vs_discrete": []}
for nu in []:
    th = theta.at[1].set(nu)
    for T in [2.0, 10.0]:
        for tol in [1e-7, 1e-10]:
            rec = dict(nu=nu, T=T, tol=tol)
            try:
                gd = jax.grad(lambda t: solve(t, diffrax.RecursiveCheckpointAdjoint(), tol, T))(th)
                rec["discrete"] = np.asarray(gd).tolist()
                gc = jax.grad(lambda t: solve(t, diffrax.BacksolveAdjoint(), tol, T))(th)
                rec["continuous"] = np.asarray(gc).tolist()
                eps = 1e-5
                fd = [float((solve(th+eps*e, diffrax.RecursiveCheckpointAdjoint(), tol, T) - solve(th-eps*e, diffrax.RecursiveCheckpointAdjoint(), tol, T))/(2*eps)) for e in jnp.eye(2)]
                rec["fd"] = fd
                rec["rel_err_continuous_vs_discrete"] = float(np.linalg.norm(np.asarray(gc)-np.asarray(gd))/np.linalg.norm(np.asarray(gd)))
                rec["rel_err_discrete_vs_fd"] = float(np.linalg.norm(np.asarray(gd)-np.asarray(fd))/np.linalg.norm(np.asarray(fd)))
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            print(json.dumps(rec), flush=True); out["continuous_vs_discrete"].append(rec)

# ---------- (2) reversible RK4 (McCallum & Foster 2024) ----------
def rk4_increment(t, y, h, args):
    k1 = rhs(t, y, args); k2 = rhs(t+h/2, y+h*k1/2, args); k3 = rhs(t+h/2, y+h*k2/2, args); k4 = rhs(t+h, y+h*k3, args)
    return (h/6)*(k1 + 2*k2 + 2*k3 + k4)

def make_reversible(h, N, lam, args):
    def step(n, y, z):
        t = n*h
        y1 = lam*y + (1-lam)*z + rk4_increment(t, z, h, args)
        z1 = z - rk4_increment(t+h, y1, -h, args)
        return y1, z1
    def reconstruct(n, y1, z1):
        t = n*h
        z = z1 + rk4_increment(t+h, y1, -h, args)
        y = (y1 - (1-lam)*z - rk4_increment(t, z, h, args))/lam
        return y, z

    @jax.custom_vjp
    def solve_rev(y0):
        def body(n, carry):
            y, z = carry
            return step(n, y, z)
        return jax.lax.fori_loop(0, N, body, (y0, y0))
    def fwd(y0):
        yN, zN = solve_rev(y0)
        return (yN, zN), (yN, zN)
    def bwd(res, cot):
        yN, zN = res
        ybar, zbar = cot
        def body(i, carry):
            n = N - 1 - i
            y1, z1, ybar, zbar = carry
            y, z = reconstruct(n, y1, z1)
            _, vjp_fn = jax.vjp(lambda yy, zz: step(n, yy, zz), y, z)
            ybar_n, zbar_n = vjp_fn((ybar, zbar))
            return y, z, ybar_n, zbar_n
        y0r, z0r, ybar0, zbar0 = jax.lax.fori_loop(0, N, body, (yN, zN, ybar, zbar))
        return (ybar0 + zbar0,)   # y0 = z0
    solve_rev.defvjp(fwd, bwd)

    def solve_ref(y0):  # same scheme, plain AD with per-step rematerialization (memory O(N) checkpoints)
        def body(n, carry):
            y, z = carry
            return step(n, y, z)
        return jax.lax.fori_loop(0, N, jax.checkpoint(body), (y0, y0))
    def solve_rk4(y0):
        return jax.lax.fori_loop(0, N, jax.checkpoint(lambda n, y: y + rk4_increment(n*h, y, h, args)), y0)
    def reconstruct_all(yN, zN):
        def body(i, carry):
            y1, z1 = carry
            return reconstruct(N-1-i, y1, z1)
        return jax.lax.fori_loop(0, N, body, (yN, zN))
    return solve_rev, solve_ref, solve_rk4, reconstruct_all

out["reversible"] = []
h = 0.02
for nu in [0.0, 1.0, 3.0]:
    for N in [100, 400, 1600]:
        for lam in [0.999, 0.99]:
            rec = dict(nu=nu, N=N, h=h, T=N*h, lam=lam)
            try:
                y0, args = params(0.2, nu, N*h)
                solve_rev, solve_ref, solve_rk4, reconstruct_all = make_reversible(h, N, lam, args)
                loss_rev = lambda y: objective(solve_rev(y)[0])
                loss_ref = lambda y: objective(solve_ref(y)[0])
                loss_rk4 = lambda y: objective(solve_rk4(y))
                t0 = time.time(); v_rev, g_rev = jax.jit(jax.value_and_grad(loss_rev))(y0); g_rev.block_until_ready(); t_rev = time.time()-t0
                t0 = time.time(); v_ref, g_ref = jax.jit(jax.value_and_grad(loss_ref))(y0); g_ref.block_until_ready(); t_ref = time.time()-t0
                v_rk4, g_rk4 = jax.jit(jax.value_and_grad(loss_rk4))(y0)
                yN, zN = solve_rev(y0)
                y0r, z0r = reconstruct_all(yN, zN)
                nrm = lambda a: float(jnp.linalg.norm(a))
                rec.update(value_rev=float(v_rev), value_ref=float(v_ref), value_rk4=float(v_rk4),
                           grad_rel_err_rev_vs_ref=nrm(g_rev-g_ref)/nrm(g_ref),
                           grad_rel_diff_rev_vs_plainrk4=nrm(g_rev-g_rk4)/nrm(g_rk4),
                           y0_reconstruction_rel_err=nrm(y0r-y0)/nrm(y0),
                           yN_minus_zN=nrm(yN-zN)/nrm(yN), t_rev=t_rev, t_ref=t_ref)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            print(json.dumps(rec), flush=True); out["reversible"].append(rec)
json.dump(out, open("exp_d_reversible_H4.json", "w"), indent=1)
