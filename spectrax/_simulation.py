import jax.numpy as jnp
from jax import jit, config, vmap
config.update("jax_enable_x64", True)
from jax.debug import print as jprint
from functools import partial
from diffrax import (diffeqsolve, Tsit5, Dopri5, ODETerm,
                     SaveAt, PIDController, TqdmProgressMeter, NoProgressMeter, ConstantStepSize)
from ._initialization import initialize_simulation_parameters
from ._model import plasma_current, Hermite_Fourier_system, _twothirds_mask, _best_pencil, _ifft2_mpi, set_mpi_comms
from ._diagnostics import diagnostics

import equinox as eqx

import os

# Optional MPI-based distributed FFT path (multi-CPU clusters).
USE_MPI = os.environ.get("SPECTRAX_MPI", "0") == "1"
if USE_MPI:
    from mpi4py import MPI
    _COMM = MPI.COMM_WORLD
    _RANK = _COMM.Get_rank()
    _SIZE = _COMM.Get_size()
else:
    _COMM = None; _RANK = 0; _SIZE = 1

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
    use_mpi: bool = eqx.field(static=True)
    Py: int = eqx.field(static=True)
    Px: int = eqx.field(static=True)
    Ny_loc: int = eqx.field(static=True)
    Nx_loc: int = eqx.field(static=True)
    rx: int = eqx.field(static=True)
    ry: int = eqx.field(static=True)

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

        # IFFTs (physical space)
        if self.use_mpi:
            # state slices (k-space) are y-slabs: (.., Ny_loc, Nx, Nz)
            F = _ifft2_mpi(Fk, self.Py, self.Px, self.rx, self.ry)
            C = _ifft2_mpi(Ck, self.Py, self.Px, self.rx, self.ry)
        else:
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
            mask23=self.mask23,
            use_mpi=self.use_mpi, Py=self.Py, Px=self.Px,
            rx=self.rx, ry=self.ry
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


# @partial(jit, static_argnames=['Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'timesteps', 'solver'])
def simulation(input_parameters={}, Nx=33, Ny=1, Nz=1, Nn=20, Nm=1, Np=1, Ns=2, timesteps=200, dt=0.01, solver=Dopri5()):
    parameters = initialize_simulation_parameters(input_parameters, Nx, Ny, Nz, Nn, Nm, Np, Ns, timesteps, dt)

    # 7D -> 4D so all downstream code (and sharding) sees (N, Ny, Nx, Nz)
    if parameters["Ck_0"].ndim == 7:
        parameters["Ck_0"] = parameters["Ck_0"].reshape(Ns * Np * Nm * Nn, Ny, Nx, Nz)

    # ----- Distributed CPU path (MPI pencil FFT) -----
    if USE_MPI:
        Py, Px = _best_pencil(Ny, Nx, _SIZE)
        if _SIZE != Py * Px:
            raise ValueError(f"World size {_SIZE} != Py*Px ({Py}*{Px}).")
        if (Ny % Py) or (Nx % Px):
            raise ValueError(f"Ny={Ny} must be divisible by Py={Py} and Nx={Nx} by Px={Px}.")
        
        ry = _RANK // Px
        rx = _RANK % Px
        ROW_COMM = _COMM.Split(color=ry, key=rx)   # size = Px, fixed row (y-slab peers)
        COL_COMM = _COMM.Split(color=rx, key=ry)   # size = Py, fixed column (x-pencil peers)
        set_mpi_comms(ROW_COMM, COL_COMM)

        Ny_loc = Ny // Py
        Nx_loc = Nx // Px   # only needed for metadata and forward FFT

        # y-slab slicing for this rank-row (ry); ranks are [0..SIZE-1] laid out row-major
        y0 = ry * Ny_loc
        y1 = (ry + 1) * Ny_loc
        def _yslab(x):
            # slice Ny
            if x.ndim == 3:                  # (Ny, Nx, Nz)
                return x[y0:y1, :, :]
            elif x.ndim == 4:                # (C or N, Ny, Nx, Nz)
                return x[:, y0:y1, :, :]
            elif x.ndim == 7:                # (Ns,Np,Nm,Nn,Ny,Nx,Nz) - not used anymore here
                return x[..., y0:y1, :, :]
            else:
                return x

        parameters["Ck_0"] = _yslab(parameters["Ck_0"])
        parameters["Fk_0"] = _yslab(parameters["Fk_0"])
        parameters["kx_grid"] = _yslab(parameters["kx_grid"])
        parameters["ky_grid"] = _yslab(parameters["ky_grid"])
        parameters["kz_grid"] = _yslab(parameters["kz_grid"])
        parameters["k2_grid"] = _yslab(parameters["k2_grid"])
        parameters["nabla"]   = jnp.stack([parameters["kx_grid"]/parameters["Lx"],
                                           parameters["ky_grid"]/parameters["Ly"],
                                           parameters["kz_grid"]/parameters["Lz"]], axis=0)
        parameters["mask23"]  = _yslab(parameters["mask23"])

    # State
    y0 = (parameters["Ck_0"], parameters["Fk_0"])
    time = jnp.linspace(0, parameters["t_max"], timesteps)

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
        mask23=parameters["mask23"],
        use_mpi=USE_MPI,
        Py=(Py if USE_MPI else 1),
        Px=(Px if USE_MPI else 1),
        Ny_loc=(Ny_loc if USE_MPI else Ny),
        Nx_loc=(Nx_loc if USE_MPI else Nx),
        rx=(rx if USE_MPI else 0),
        ry=(ry if USE_MPI else 0),
    )

    sol = diffeqsolve(
        ODETerm(_rhs),
        solver=solver,
        stepsize_controller=ConstantStepSize(),
        t0=0, t1=parameters["t_max"], dt0=dt,
        y0=y0, args=vf,
        saveat=SaveAt(ts=time),
        max_steps=1_000_000,
        progress_meter=TqdmProgressMeter()
    )

    Ck, Fk = sol.ys
    if Ck.ndim == 8:
        T, Ns_, Np_, Nm_, Nn_, Ny_, Nx_, Nz_ = Ck.shape
        Ck = Ck.reshape(T, Ns_ * Np_ * Nm_ * Nn_, Ny_, Nx_, Nz_)

    # Remove k=0 as before (these indices refer to local tile; for diagnostics/gather you can MPI collect later)
    dCk = Ck.at[:, 0, 0, 1, 0].set(0)
    dCk = dCk.at[:, Nn * Nm * Np, 0, 1, 0].set(0)

    temporary_output = {
        "Ck": Ck, "Fk": Fk, "time": time, "dCk": dCk,
        "USE_MPI": USE_MPI,
    }
    if USE_MPI:
        temporary_output.update({"Py": Py, "Px": Px, "rank": _RANK, "size": _SIZE,
                                 "Ny_loc": Ny_loc, "Nx_loc": Nx_loc})

    output = {**temporary_output, **parameters}
    diagnostics(output)
    return output