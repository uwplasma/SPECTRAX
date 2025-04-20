import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy.fft import ifftn, ifftshift
from jax.scipy.special import factorial
from orthax.hermite import hermval3d, hermval
from jax import debug

def Hermite(n, x):
    """
    I have to add docstrings!
    """
    
    c = jnp.zeros(n+1)
    c = c.at[-1].set(1)
    
    return hermval(x, c)
    
    
def generate_C_term(X, Y, Z, kx, ky, kz, Ck, Nx, Ny, index):
   
    # Indices below represent order of Hermite polynomials.
    nz = jnp.floor(index / (Nx * Ny)).astype(int)
    ny = jnp.floor((index - nz * Nx * Ny) / Nx).astype(int)
    nx = (index - nz * Nx * Ny - ny * Nx).astype(int)
    
    expkx = jnp.exp(1j * (kx[nx] * X + ky[ny] * Y + kz[nz] * Z))
    expkx_extended = expkx[jnp.newaxis, jnp.newaxis, :, :, :]
    
    return Ck[:, :, ny, nx, nz, jnp.newaxis, jnp.newaxis, jnp.newaxis] * expkx_extended


def generate_Hermite_term(xi_x, xi_y, xi_z, C, Nn, Nm, index):
    """
    I have to add docstrings!
    """
    
    # Indices below represent order of Hermite polynomials.
    p = jnp.floor(index / (Nn * Nm)).astype(int)
    m = jnp.floor((index - p * Nn * Nm) / Nn).astype(int)
    n = (index - p * Nn * Nm - m * Nn).astype(int)
    
    C_expanded = C[:, index, :, :, :, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    Herm_x_expanded = Hermite(n, xi_x)[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]
    Herm_y_expanded = Hermite(m, xi_y)[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]
    Herm_z_expanded = Hermite(p, xi_z)[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]
    exp_expanded = jnp.exp(-(xi_x**2 + xi_y**2 + xi_z**2))[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]
    
    # Generate element of AW Hermite basis in 3D space.
    Hermite_term = C_expanded * (Herm_x_expanded * Herm_y_expanded * Herm_z_expanded * exp_expanded / 
                (jnp.sqrt(jnp.pi**3) * 2**((n + m + p) / 2) * jnp.sqrt(factorial(n)) * jnp.sqrt(factorial(m)) * jnp.sqrt(factorial(p))))
    
    return Hermite_term

# def rescale_C(C,Nn,Nm,index):
    
#     p = jnp.floor(index / (Nn * Nm)).astype(int)
#     m = jnp.floor((index - p * Nn * Nm) / Nn).astype(int)
#     n = (index - p * Nn * Nm - m * Nn).astype(int)
#     # debug.print('{},{},{},{},{},{}', p, m, n, factorial(n) * factorial(m) * factorial(p), jnp.abs(jnp.power(2,(n + m + p)/1)), index)
#     return C[:,index,...] / (2**((n + m + p)/2) * jnp.sqrt(factorial(n)) * jnp.sqrt(factorial(m)) * jnp.sqrt(factorial(p)))
    

def inverse_HF_transform(Ck, Lx, Ly, Lz, Nn, Nm, Np, xi_x, xi_y, xi_z):
    
    C = ifftn(ifftshift(Ck, axes=(-3, -2, -1)), axes=(-3, -2, -1))
    
    Nt, Nx, Ny, Nz = Ck.shape[0], Ck.shape[-2], Ck.shape[-3], Ck.shape[-1]
    Nvx, Nvy, Nvz = xi_x.shape[1], xi_x.shape[0], xi_x.shape[2]
    
    f = jnp.zeros((Nt, Ny, Nx, Nz, Nvy, Nvx, Nvz))
    for index in jnp.arange(Nn * Nm * Np):  
        f = f + generate_Hermite_term(xi_x, xi_y, xi_z, C.real, Nn, Nm, index)
    
    return f