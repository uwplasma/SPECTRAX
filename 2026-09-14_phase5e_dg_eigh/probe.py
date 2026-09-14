"""Phase 5e probe: upwind matrices, a short DG simulation, and the gradient with respect to thermal speeds."""
import sys
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from diffrax import NoProgressMeter
from spectrax import simulation
from spectrax._initialization import compute_A_pm_matrices
from tests.test_autodiff import orszag_tang, theta0, mass, Nx, Ny, Nh, Ns, N_DG, dims
alpha = jnp.array([[0.25, 0.3, 0.35], [0.05, 0.06, 0.07]]); u = jnp.array([[0.1, -0.2, 0.0], [0.0, 0.03, -0.01]])
out = {f"A_{N}_{k}": np.asarray(M) for N in (1, 2, 3, 5, 8) for k, M in enumerate(compute_A_pm_matrices(N, N, N, alpha, u))}
base = orszag_tang(theta0)
kw = dict(Nx=Nx, Ny=Ny, Nz=1, Nn=Nh, Nm=Nh, Np=Nh, Ns=Ns, N_DG=N_DG, dims=dims, timesteps=3, dt=0.01, progress_meter=NoProgressMeter())
sim = simulation(base, **kw)
out["Ck"], out["Fk"] = np.asarray(sim["Ck"]), np.asarray(sim["Fk"])
np.savez(sys.argv[1], **out)
loss = lambda s: 0.5 * jnp.sum(simulation(dict(base, alpha_s=base["alpha_s"] * s), **kw)["Fk"][-1] ** 2 / mass)
try:
    print("gradient w.r.t. thermal-speed scale:", float(jax.grad(loss)(1.0)))
except Exception as error:
    print("gradient w.r.t. thermal-speed scale FAILS:", type(error).__name__, str(error)[:110])
