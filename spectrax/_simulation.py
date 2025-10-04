import jax.numpy as jnp
from jax import jit, config, vmap
config.update("jax_enable_x64", True)
from jax.debug import print as jprint
from functools import partial
from diffrax import (diffeqsolve, Tsit5, Dopri5, ODETerm,
                     SaveAt, PIDController, TqdmProgressMeter, NoProgressMeter, ConstantStepSize)
from ._initialization import initialize_simulation_parameters
from ._model import plasma_current, Hermite_Fourier_system, _twothirds_mask
from ._diagnostics import diagnostics

import equinox as eqx

# Parallelize the simulation using JAX
from jax.sharding import PartitionSpec as P
from jax import devices, make_mesh, NamedSharding, device_put
mesh = make_mesh((len(devices()),), ("x",))
# spec = P("batch")
# sharding = NamedSharding(mesh, spec)

# __all__ = ["cross_product", "ode_system", "simulation"]

class ODEVecField(eqx.Module):
    # static config (kept out of JIT traces)
    Nx: int = eqx.field(static=True)
    Ny: int = eqx.field(static=True)
    Nz: int = eqx.field(static=True)
    Nn: int = eqx.field(static=True)
    Nm: int = eqx.field(static=True)
    Np: int = eqx.field(static=True)
    Ns: int = eqx.field(static=True)

    # dynamic arrays (traced once, then re-used)
    qs: jnp.ndarray
    nu: jnp.ndarray
    D: jnp.ndarray
    Omega_cs: jnp.ndarray
    alpha_s: jnp.ndarray
    u_s: jnp.ndarray
    Lx: float
    Ly: float
    Lz: float
    kx_grid: jnp.ndarray
    ky_grid: jnp.ndarray
    kz_grid: jnp.ndarray
    k2_grid: jnp.ndarray
    nabla: jnp.ndarray
    col: jnp.ndarray
    sqrt_n_plus: jnp.ndarray
    sqrt_n_minus: jnp.ndarray
    sqrt_m_plus: jnp.ndarray
    sqrt_m_minus: jnp.ndarray
    sqrt_p_plus: jnp.ndarray
    sqrt_p_minus: jnp.ndarray
    mask23: jnp.ndarray

    def __call__(self, t, y):
        Ck, Fk = y  # PyTree state: (Ns*Nn*Nm*Np, Ny, Nx, Nz), (6, Ny, Nx, Nz)

        # IFFTs
        F = jnp.fft.ifftn(jnp.fft.ifftshift(Fk, axes=(-3, -2, -1)), axes=(-3, -2, -1))
        C = jnp.fft.ifftn(jnp.fft.ifftshift(Ck, axes=(-3, -2, -1)), axes=(-3, -2, -1))

        dCk_s_dt = Hermite_Fourier_system(
            Ck, C, F,
            self.kx_grid, self.ky_grid, self.kz_grid, self.k2_grid, self.col,
            self.sqrt_n_plus, self.sqrt_n_minus,
            self.sqrt_m_plus, self.sqrt_m_minus,
            self.sqrt_p_plus, self.sqrt_p_minus,
            self.Lx, self.Ly, self.Lz, self.nu, self.D,
            self.alpha_s, self.u_s, self.qs, self.Omega_cs,
            self.Nn, self.Nm, self.Np, self.Ns,
            mask23=self.mask23
        )

        dBk_dt = -1j * cross_product(self.nabla, Fk[:3])
        current = plasma_current(self.qs, self.alpha_s, self.u_s, Ck, self.Nn, self.Nm, self.Np, self.Ns)
        dEk_dt =  1j * cross_product(self.nabla, Fk[3:]) - current / self.Omega_cs[0]
        dFk_dt = jnp.concatenate([dEk_dt, dBk_dt], axis=0)
        dCk_s_dt = dCk_s_dt.reshape(
            self.Ns * self.Np * self.Nm * self.Nn, self.Ny, self.Nx, self.Nz
        )
        return (dCk_s_dt, dFk_dt)

@eqx.filter_jit
def _rhs(t, y, vf: ODEVecField):
    return vf(t, y)

def _shard_along_x(a):
    # (Ny, Nx, Nz)
    if a.ndim == 3:
        spec = P(None, "x", None)
    # (6 or N, Ny, Nx, Nz)
    elif a.ndim == 4:
        spec = P(None, None, "x", None)
    # (Ns, Np, Nm, Nn, Ny, Nx, Nz)
    elif a.ndim == 7:
        spec = P(None, None, None, None, None, "x", None)
    else:
        return a
    return device_put(a, NamedSharding(mesh, spec))

@jit
def cross_product(k_vec, F_vec):
    """
    Computes the cross product of two 3D vectors.
    Args:
        k_vec (array-like): A 3-element array representing the first vector (k).
        F_vec (array-like): A 3-element array representing the second vector (F).
    Returns:
        jnp.ndarray: A 3-element array representing the cross product k x F.
    """
    kx, ky, kz = k_vec
    Fx, Fy, Fz = F_vec
    return jnp.array([ky * Fz - kz * Fy, kz * Fx - kx * Fz, kx * Fy - ky * Fx])

# @partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns'])
# def ode_system(Nx, Ny, Nz, Nn, Nm, Np, Ns, t, Ck_Fk, args):

#     (qs, nu, D, Omega_cs, alpha_s, u_s,
#      Lx, Ly, Lz, kx_grid, ky_grid, kz_grid, k2_grid, nabla, col,
#      sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus
#     ) = args[7:]

#     total_Ck_size = Nn * Nm * Np * Ns * Nx * Ny * Nz
#     Ck = Ck_Fk[:total_Ck_size].reshape(Nn * Nm * Np * Ns, Ny, Nx, Nz)
#     Fk = Ck_Fk[total_Ck_size:].reshape(6, Ny, Nx, Nz)


#     F = jnp.fft.ifftn(jnp.fft.ifftshift(Fk, axes=(-3, -2, -1)), axes=(-3, -2, -1))
#     C = jnp.fft.ifftn(jnp.fft.ifftshift(Ck, axes=(-3, -2, -1)), axes=(-3, -2, -1))

#     # Build the 2/3 mask once per call (JIT will constant-fold it since Nx/Ny/Nz are static)
#     mask23 = _twothirds_mask(Ny, Nx, Nz)
#     dCk_s_dt = Hermite_Fourier_system(Ck, C, F, kx_grid, ky_grid, kz_grid, k2_grid, col, 
#                                       sqrt_n_plus, sqrt_n_minus, sqrt_m_plus, sqrt_m_minus, sqrt_p_plus, sqrt_p_minus, 
#                                       Lx, Ly, Lz, nu, D, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, Ns, mask23=mask23)

#     # nabla = jnp.array([kx_grid / Lx, ky_grid / Ly, kz_grid / Lz])
#     dBk_dt = -1j * cross_product(nabla, Fk[:3])
    
#     current = plasma_current(qs, alpha_s, u_s, Ck, Nn, Nm, Np, Ns)
#     dEk_dt = 1j * cross_product(nabla, Fk[3:]) - current / Omega_cs[0]

#     dFk_dt = jnp.concatenate([dEk_dt, dBk_dt], axis=0)
#     dy_dt  = jnp.concatenate([dCk_s_dt.reshape(-1), dFk_dt.reshape(-1)])
#     return dy_dt

# @partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'timesteps', 'solver'])
def simulation(input_parameters={}, Nx=33, Ny=1, Nz=1, Nn=20, Nm=1, Np=1, Ns=2, timesteps=200, dt=0.01, solver=Dopri5()):
    parameters = initialize_simulation_parameters(input_parameters, Nx, Ny, Nz, Nn, Nm, Np, Ns, timesteps, dt)

    # normalize Ck_0 to 4D so sharding kicks in
    if parameters["Ck_0"].ndim == 7:
        parameters["Ck_0"] = parameters["Ck_0"].reshape(Ns * Np * Nm * Nn, Ny, Nx, Nz)


    # --- SHARD once along the FFT x-axis (optional but recommended)
    for k in ["Ck_0","Fk_0","kx_grid","ky_grid","kz_grid","k2_grid","nabla","mask23"]:
        parameters[k] = _shard_along_x(parameters[k])

    # PyTree state -> keep spatial axes visible for JAX (FFT-friendly)
    y0 = (parameters["Ck_0"], parameters["Fk_0"])

    time = jnp.linspace(0, parameters["t_max"], timesteps)

    # Build vector field as an Equinox Module (static config, dynamic arrays)
    vf = ODEVecField(
        Nx=Nx, Ny=Ny, Nz=Nz, Nn=Nn, Nm=Nm, Np=Np, Ns=Ns,
        qs=parameters["qs"], nu=parameters["nu"], D=parameters["D"],
        Omega_cs=parameters["Omega_cs"], alpha_s=parameters["alpha_s"], u_s=parameters["u_s"],
        Lx=parameters["Lx"], Ly=parameters["Ly"], Lz=parameters["Lz"],
        kx_grid=parameters["kx_grid"], ky_grid=parameters["ky_grid"], kz_grid=parameters["kz_grid"],
        k2_grid=parameters["k2_grid"], nabla=parameters["nabla"], col=parameters["collision_matrix"],
        sqrt_n_plus=parameters["sqrt_n_plus"], sqrt_n_minus=parameters["sqrt_n_minus"],
        sqrt_m_plus=parameters["sqrt_m_plus"], sqrt_m_minus=parameters["sqrt_m_minus"],
        sqrt_p_plus=parameters["sqrt_p_plus"], sqrt_p_minus=parameters["sqrt_p_minus"],
        mask23=parameters["mask23"]
    )

    sol = diffeqsolve(
        ODETerm(_rhs),                           # NEW: jitted Equinox wrapper
        solver=solver,
        stepsize_controller=ConstantStepSize(),
        t0=0, t1=parameters["t_max"], dt0=dt,
        y0=y0, args=vf,                          # pass the Module via args
        saveat=SaveAt(ts=time),
        max_steps=1_000_000,
        progress_meter=TqdmProgressMeter()
    )

    # Unpack PyTree solution
    Ck, Fk = sol.ys              # shapes: (T, ...)

    if Ck.ndim == 8:  # (T, Ns, Np, Nm, Nn, Ny, Nx, Nz)
        T, Ns_, Np_, Nm_, Nn_, Ny_, Nx_, Nz_ = Ck.shape
        Ck = Ck.reshape(T, Ns_ * Np_ * Nm_ * Nn_, Ny_, Nx_, Nz_)

    # Your existing “remove k=0 mode” logic (unchanged)
    dCk = Ck.at[:, 0, 0, 1, 0].set(0)
    dCk = dCk.at[:, Nn * Nm * Np, 0, 1, 0].set(0)

    temporary_output = {"Ck": Ck, "Fk": Fk, "time": time, "dCk": dCk}
    output = {**temporary_output, **parameters}
    diagnostics(output)
    return output