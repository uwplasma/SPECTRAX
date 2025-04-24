import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap, jit
from jax.numpy.fft import fftn, fftshift
from jax.scipy.special import factorial
from jax.scipy.integrate import trapezoid
from functools import partial

__all__ = ['Hermite', 'generate_Hermite_function', 'compute_C_nmp', 'initialize_xv']

# Parallelize the simulation using JAX
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax import devices, make_mesh, NamedSharding, device_put
mesh = make_mesh((len(devices()),), ("batch"))
spec = P("batch")
sharding = NamedSharding(mesh, spec)

@jit
def Hermite(n, x):
    def body(m, acc):
        coeff = (-1.0)**m / (factorial(m) * factorial(n - 2 * m))
        term = coeff * (2 * x)**(n - 2 * m)
        return acc + term
    return factorial(n) * jax.lax.fori_loop(0, (n // 2) + 1, body, jnp.zeros_like(x))

@jit
def generate_Hermite_function(xi_x, xi_y, xi_z, Nn, Nm, indices):
    p = indices // (Nn * Nm)
    m = (indices % (Nn * Nm)) // Nn
    n = indices % Nn
    norm = jnp.sqrt(jnp.pi**3 * 2**(n + m + p) * factorial(n) * factorial(m) * factorial(p))
    return (
        Hermite(n, xi_x) * Hermite(m, xi_y) * Hermite(p, xi_z) *
        jnp.exp(-(xi_x**2 + xi_y**2 + xi_z**2)) / norm
    )

@partial(jit, static_argnames=['f', 'Nn', 'Nm', 'Np', 'Nx', 'Ny', 'Nz', 'Nv', 'nvxyz', 'max_min_v_factor'])
def compute_C_nmp(f, alpha, u, Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, Nv, nvxyz, max_min_v_factor, indices):
    # Indices below represent order of Hermite polynomials.
    p = jnp.floor(indices / (Nn * Nm)).astype(int)
    m = jnp.floor((indices - p * Nn * Nm) / Nn).astype(int)
    n = (indices - p * Nn * Nm - m * Nn).astype(int)
    # Generate 6D space for particle distribution function f.
    x = jnp.linspace(0, Lx, Nx)
    y = jnp.linspace(0, Ly, Ny)
    z = jnp.linspace(0, Lz, Nz)
    vx = jnp.linspace(-max_min_v_factor * alpha[0] + u[0], max_min_v_factor * alpha[0] + u[0], nvxyz)
    vy = jnp.linspace(-max_min_v_factor * alpha[1] + u[1], max_min_v_factor * alpha[1] + u[1], nvxyz)
    vz = jnp.linspace(-max_min_v_factor * alpha[2] + u[2], max_min_v_factor * alpha[2] + u[2], nvxyz)
    # Precompute all 8x8x8 velocity subgrids
    def slice_1d(v):
        return jnp.stack([v[i * 8:(i + 1) * 8] for i in range(5)])
    vx_chunks = slice_1d(vx)  # shape (5, 8)
    vy_chunks = slice_1d(vy)
    vz_chunks = slice_1d(vz)
    
    xv, yv, zv = jnp.meshgrid(x, y, z, indexing='xy')  # shape (Ny, Nx, Nz)
    X = xv[..., None, None, None]
    Y = yv[..., None, None, None]
    Z = zv[..., None, None, None]
  
    @jit
    def single_C_nmp(i):
        ivx = jnp.floor(i / (5 ** 2)).astype(int)
        ivy = jnp.floor((i - ivx * 5 ** 2) / 5).astype(int)
        ivz = (i - ivx * 5 ** 2 - ivy * 5).astype(int)
        
        vx_slice = vx_chunks[ivx]
        vy_slice = vy_chunks[ivy]
        vz_slice = vz_chunks[ivz]
        
        # X, Y, Z, Vx, Vy, Vz = jnp.meshgrid(x, y, z, vx_slice, vy_slice, vz_slice, indexing='xy')
        vxv, vyv, vzv = jnp.meshgrid(vx_slice, vy_slice, vz_slice, indexing='ij')  # shape (8, 8, 8)
        Vx = vxv[None, None, None, ...]
        Vy = vyv[None, None, None, ...]
        Vz = vzv[None, None, None, ...]

        # Compute coefficients of Hermite decomposition of 3D velocity space.
        xi_x = (Vx - u[0]) / alpha[0]
        xi_y = (Vy - u[1]) / alpha[1]
        xi_z = (Vz - u[2]) / alpha[2]
        Hx = Hermite(n, xi_x)
        Hy = Hermite(m, xi_y)
        Hz = Hermite(p, xi_z)
        hermite_norm = jnp.sqrt(factorial(n) * factorial(m) * factorial(p) * 2 ** (n + m + p))
        
        # Compute the integrand for the distribution function.
        integrand = f(X, Y, Z, Vx, Vy, Vz) * Hx * Hy * Hz / hermite_norm
        
        # Compute the contribution to C_nmp from the velocity space.
        contribution = trapezoid(trapezoid(trapezoid(integrand,
                    (vy_slice - u[0]) / alpha[0], axis=-3),
                    (vx_slice - u[1]) / alpha[1], axis=-2),
                    (vz_slice - u[2]) / alpha[2], axis=-1)
        return contribution
    C_nmp_all = vmap(single_C_nmp)(jnp.arange(Nv))  # shape (Nv, Ny, Nx, Nz)
    return jnp.sum(C_nmp_all, axis=0)

@partial(jit, static_argnames=['B', 'E', 'f1', 'f2', 'Nv', 'nvxyz', 'max_min_v_factor', 'Nx', 'Ny', 'Nz', 'Nn', 'Nm', 'Np', 'Ns', 'timesteps', 'solver'])
def initialize_xv(B, E, f1, f2, Nv=55, nvxyz=40, max_min_v_factor=5, input_parameters={},
                  Nx=33, Ny=33, Nz=1, Nn=4, Nm=4, Np=4, Ns=None, timesteps=None, solver=None):
    alpha_s = input_parameters["alpha_s"]
    u_s = input_parameters["u_s"]
    Lx = input_parameters["Lx"]
    Ly = input_parameters["Ly"]
    Lz = input_parameters["Lz"]
    
    indices_sharded = jnp.arange(Nn * Nm * Np)
    indices_sharded = device_put(indices_sharded, sharding)
    
    partial_C1_0 = partial(compute_C_nmp, f1, alpha_s[:3], u_s[:3], Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, Nv, nvxyz, max_min_v_factor)
    sharded_C10_fun = jit(shard_map(vmap(partial_C1_0), mesh, in_specs=spec, out_specs=spec, check_rep=False))
    C1_0 = sharded_C10_fun(indices_sharded)
    
    partial_C2_0 = partial(compute_C_nmp, f2, alpha_s[3:], u_s[3:], Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, Nv, nvxyz, max_min_v_factor)
    sharded_C20_fun = jit(shard_map(vmap(partial_C2_0), mesh, in_specs=spec, out_specs=spec, check_rep=False))
    C2_0 = sharded_C20_fun(indices_sharded)

    # Combine Ce_0 and Ci_0 into single array and compute the fast Fourier transform.
    C1k_0 = fftshift(fftn(C1_0, axes=(-3, -2, -1)), axes=(-3, -2, -1))
    C2k_0 = fftshift(fftn(C2_0, axes=(-3, -2, -1)), axes=(-3, -2, -1))
    Ck_0 = jnp.concatenate([C1k_0, C2k_0])
    
    # Define 3D grid for functions E(x, y, z) and B(x, y, z).
    x = jnp.linspace(0, Lx, Nx)
    y = jnp.linspace(0, Ly, Ny)
    z = jnp.linspace(0, Lz, Nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='xy')
    
    # Combine E and B into single array and compute the fast Fourier transform.
    Ek_0 = fftshift(fftn(E(X, Y, Z), axes=(-3, -2, -1)), axes=(-3, -2, -1))
    Bk_0 = fftshift(fftn(B(X, Y, Z), axes=(-3, -2, -1)), axes=(-3, -2, -1))
    Fk_0 = jnp.concatenate([Ek_0, Bk_0])
    
    return Ck_0, Fk_0