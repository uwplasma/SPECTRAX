import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy.fft import ifftn, ifftshift
from jax.scipy.special import factorial
from orthax.hermite import hermval
from jax import jit, vmap
from functools import partial

__all__ = ['generate_Hermite_basis', 'generate_Hermite_term_vmap', 'inverse_HF_transform']

# Generate Hermite polynomial values for all orders from 0 to N-1
def generate_Hermite_basis(N, x):
    def compute_herm(n):
        c = jnp.zeros((N,))
        c = c.at[n].set(1.0)
        return hermval(x, c)
    return vmap(compute_herm)(jnp.arange(N))  # Shape: (N, ..., ...)

@partial(jit, static_argnames=["Nn", "Nm", "Np"])
def generate_Hermite_term_vmap(C, Herm_x, Herm_y, Herm_z, Nn, Nm, Np, xi_x, xi_y, xi_z):
    N_modes = Nn * Nm * Np
    index = jnp.arange(N_modes)

    # Precompute all combinations (n, m, p)
    p = index // (Nn * Nm)
    m = (index % (Nn * Nm)) // Nn
    n = index % Nn

    def compute_term(i, n, m, p):
        C_exp = C[:, i, :, :, :, None, None, None]  # [Nt, Ny, Nx, Nz, 1, 1, 1]
        Hx = Herm_x[n]  # [Nvy, Nvx, Nvz]
        Hy = Herm_y[m]
        Hz = Herm_z[p]
        Herm = Hx * Hy * Hz * jnp.exp(-(xi_x**2 + xi_y**2 + xi_z**2))
        norm = jnp.sqrt(jnp.pi**3) * (2 ** ((n + m + p) / 2)) * jnp.sqrt(factorial(n) * factorial(m) * factorial(p))
        return C_exp * Herm[None, None, None, None, :, :, :] / norm

    f_terms = vmap(compute_term, in_axes=(0, 0, 0, 0))(index, n, m, p)
    return jnp.sum(f_terms, axis=0)

@partial(jit, static_argnames=['Nn', 'Nm', 'Np'])
def inverse_HF_transform(Ck, Nn, Nm, Np, xi_x, xi_y, xi_z):
    C = ifftn(ifftshift(Ck, axes=(-3, -2, -1)), axes=(-3, -2, -1)).real

    # Precompute Hermite functions up to desired order
    Herm_x = generate_Hermite_basis(Nn, xi_x)  # [Nn, Nvy, Nvx, Nvz]
    Herm_y = generate_Hermite_basis(Nm, xi_y)  # [Nm, Nvy, Nvx, Nvz]
    Herm_z = generate_Hermite_basis(Np, xi_z)  # [Np, Nvy, Nvx, Nvz]

    f = generate_Hermite_term_vmap(C, Herm_x, Herm_y, Herm_z, Nn, Nm, Np, xi_x, xi_y, xi_z)
    return f