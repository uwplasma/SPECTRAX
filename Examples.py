import jax
import jax.numpy as jnp

def Orszag_Tang(Lx, Ly, Omega_ce, mi_me):
    """
    I have to add docstrings!
    """
    
    vte = jnp.sqrt(0.25 / 2) * Omega_ce # Electron thermal velocity.
    vti = vte * jnp.sqrt(1 / mi_me) # Ion thermal velocity.
    deltaB = 0.2 # In-plane magnetic field amplitude. 
    U0 = deltaB * Omega_ce / jnp.sqrt(mi_me) # Fluid velocity amplitude.
    
    # Wavenumbers.
    kx = 2 * jnp.pi / Lx
    ky = 2 * jnp.pi / Ly
    
    # Electron and ion fluid velocities.
    Ue = lambda x, y, z: U0 * jnp.array([-jnp.sin(ky * y), jnp.sin(kx * x), -deltaB * Omega_ce * (2 * kx * jnp.cos(2 * kx * x) + ky * jnp.cos(ky * y))])
    Ui = lambda x, y, z: U0 * jnp.array([-jnp.sin(ky * y), jnp.sin(kx * x), jnp.zeros_like(x)])
    
    # Magnetic and electric fields.
    B = lambda x, y, z: jnp.array([-deltaB * jnp.sin(ky * y), deltaB * jnp.sin(2 * kx * x), jnp.ones_like(x)])
    E = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)]) # Is this consistent with fe, fi?
    
    # Electron and ion distribution functions.
    fe = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * vte ** 3) * 
                                        jnp.exp(-((vx - Ue(x, y, z)[0])**2 + (vy - Ue(x, y, z)[1])**2 + (vz - Ue(x, y, z)[2])**2) / (2 * vte ** 2))))
    fi = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * vti ** 3) * 
                                        jnp.exp(-((vx - Ui(x, y, z)[0])**2 + (vy - Ui(x, y, z)[1])**2 + (vz - Ui(x, y, z)[2])**2) / (2 * vti ** 2))))
    
    return B, E, fe, fi


def simple_example(Lx, Ly):
    """
    I have to add docstrings!
    """
    
    vte = 0.4 # Electron thermal velocity.
    vti = 0.4 # Ion thermal velocity.
    deltaB = 0.2 # In-plane magnetic field amplitude.
    
    # Wavenumbers.
    kx = 2 * jnp.pi / Lx
    ky = 2 * jnp.pi / Ly
    
    # Define elements of 3D Hermite basis.
    Hermite_000 = lambda xi_x, xi_y, xi_z: generate_Hermite_basis(xi_x, xi_y, xi_z, 1, 1, 1, 0)
    Hermite_100 = lambda xi_x, xi_y, xi_z: generate_Hermite_basis(xi_x, xi_y, xi_z, 1, 1, 1, 1)
    
    # Magnetic and electric fields.
    B = lambda x, y, z: jnp.array([-deltaB * jnp.sin(ky * y), deltaB * jnp.sin(2 * kx * x), jnp.ones_like(x)])
    E = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)])
    
    # Electron and ion distribution functions.
    fe = (lambda x, y, z, vx, vy, vz: 3 * jnp.sin(kx * x) * Hermite_000(vx/vte, vy/vte, vz/vte) + 
          2 * jnp.sin(2 * ky * y) * Hermite_100(vx/vte, vy/vte, vz/vte))
    fi = (lambda x, y, z, vx, vy, vz: 3 * jnp.sin(kx * x) * Hermite_000(vx/vti, vy/vti, vz/vti) + 
          2 * jnp.sin(2 * ky * y) * Hermite_100(vx/vti, vy/vti, vz/vti))
    
    return B, E, fe, fi


def density_perturbation(Lx, Omega_ce, mi_me):
    """
    I have to add docstrings!
    """
    
    vte = jnp.sqrt(0.25 / 2) # Electron thermal velocity.
    vti = vte * jnp.sqrt(1 / mi_me) # Ion thermal velocity.
    
    # Wavenumbers.
    kx = 2 * jnp.pi / Lx
    
    # Magnetic and electric fields.
    B = lambda x, y, z: jnp.array([Omega_ce * jnp.ones_like(x), jnp.zeros_like(y), jnp.zeros_like(z)])
    E = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(y), jnp.zeros_like(z)]) # Is this consistent with fe, fi?
    
    # Electron and ion distribution functions.
    fe = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * vte ** 3) * 
                                        jnp.exp(-(vx ** 2 + vy ** 2 + vz ** 2) / (2 * vte ** 2))) * 
                                        (1 + 0.3 * jnp.sin(kx * x)))
    fi = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * vti ** 3) * 
                                        jnp.exp(-(vx ** 2 + vy ** 2 + vz ** 2) / (2 * vti ** 2))) * 
                                        (1 + 0.3 * jnp.sin(kx * x)))
    
    return B, E, fe, fi


def density_perturbation_solution(Lx, Omega_ce, mi_me):
    """
    I have to add docstrings!
    """
    
    vte = jnp.sqrt(0.25 / 2) # Electron thermal velocity.
    vti = vte * jnp.sqrt(1 / mi_me) # Ion thermal velocity.
    
    # Wavenumbers.
    kx = 2 * jnp.pi / Lx
    
    # Magnetic and electric fields.
    B = lambda x, y, z: jnp.array([Omega_ce * jnp.ones_like(x), jnp.zeros_like(y), jnp.zeros_like(z)])
    E = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(y), jnp.zeros_like(z)]) # Is this consistent with fe, fi?
    
    dn = 0.3
    
    # Electron and ion distribution functions.
    fe_exact_0 = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * vte ** 3) * 
                                        jnp.exp(-(vx ** 2 + vy ** 2 + vz ** 2) / (2 * vte ** 2))) * 
                                        (1 + dn * jnp.sin(kx * (x - vx * 0.0))))
    
    fe_exact_2 = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * vte ** 3) * 
                                        jnp.exp(-(vx ** 2 + vy ** 2 + vz ** 2) / (2 * vte ** 2))) * 
                                        (1 + dn * jnp.sin(kx * (x - vx * 2.0))))
    
    fe_exact_5 = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * vte ** 3) * 
                                        jnp.exp(-(vx ** 2 + vy ** 2 + vz ** 2) / (2 * vte ** 2))) * 
                                        (1 + dn * jnp.sin(kx * (x - vx * 5.0))))
    # fi_exact = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * vti ** 3) * 
    #                                     jnp.exp(-(vx ** 2 + vy ** 2 + vz ** 2) / (2 * vti ** 2))) * 
    #                                     (1 + dn * jnp.sin(kx * (x - vx * 0.0))))
    
    C0_exact = (lambda t, x: (1 / (jnp.sqrt(2) * vte) ** 3) * (1 + dn * jnp.sin(kx * x) * jnp.exp(-(kx * vte * t) ** 2 / 2)))
    # C1_exact = (lambda t, x: -(1 / (jnp.sqrt(2) * vte) ** 3) * (kx * t * vte * dn * jnp.cos(kx * x) * jnp.exp(-(kx * vte * t) ** 2 / 2)))
    
    return B, E, fe_exact_0, fe_exact_2, fe_exact_5, C0_exact