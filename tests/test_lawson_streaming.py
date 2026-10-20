"""Exact Hermite streaming integrating factor: generator vs the real RHS, unitarity, analytic Hermite front at large
step, order (alone and with a diagonal rate in the factor), k=0 density, gradient, default bitwise."""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import diffrax
from scipy.linalg import expm
from spectrax import LawsonSolver, simulation, hermite_streaming_factor
from spectrax._initialization import initialize_simulation_parameters
from spectrax._lawson_streaming import StreamingFactor, hermite_axis_eig
from spectrax._simulation import ode_system

jax.config.update("jax_enable_x64", True)


def _setup(Nx=5, Ny=3, Nz=3, Nn=7, Nm=3, Np=2, Ns=2, params=None):
    p = initialize_simulation_parameters(params or {}, Nx, Ny, Nz, Nn, Nm, Np, Ns, 3, 0.01)
    args = (Nx, Ny, Nz, Nn, Nm, Np, Ns, p["qs"], p["nu"], p["D"], p["Omega_cs"], p["alpha_s"], p["u_s"],
            p["Lx"], p["Ly"], p["Lz"], p["kx_grid"], p["ky_grid"], p["kz_grid"], p["k2_grid"], p["nabla"],
            p["collision_matrix"], p["sqrt_n_plus"], p["sqrt_n_minus"], p["sqrt_m_plus"], p["sqrt_m_minus"],
            p["sqrt_p_plus"], p["sqrt_p_minus"])
    shape = (Ns, Np, Nm, Nn, Ny, Nx // 2 + 1, Nz)
    return p, args, shape


def _rand(shape, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))


def _factor(args, shape, rate=None, layout="first"):
    st = (jnp.zeros(shape, dtype=complex),)
    return hermite_streaming_factor(layout, shape, None if rate is None else (lambda t, y, a: rate))(0.0, st, args)


def test_generator_equals_streaming_part_of_the_rhs():
    """G Ck equals dCk/dt of the real ode_system with Fk = 0, nu = D = 0 (3-D, drifts, unequal widths, Nm,Np > 1)."""
    p, args, shape = _setup(params={"nu": 0.0, "alpha_s": jnp.array([0.7, 0.9, 1.1, 0.5, 0.6, 0.8])})
    Ck = _rand(shape)
    Fk0 = jnp.zeros((6, *shape[-3:]), dtype=complex)
    try:  # flat [Ck, Fk] state (main)
        dy = ode_system(*args[:7], 0.0, jnp.concatenate([Ck.reshape(-1), Fk0.reshape(-1)]), args)
        dck = dy[:Ck.size].reshape(shape)
    except Exception:  # tuple state (integration lineage)
        dck = ode_system(*args[:7], 0.0, (Ck, Fk0), args)[0].reshape(shape)
    fac = _factor(args, shape)
    # the 2/3 mask only enters the force terms (all multiplied by F = 0)
    np.testing.assert_allclose(dck, fac.streaming(Ck), rtol=1e-12, atol=1e-12)


def test_B_is_unitary_semigroup_and_matches_expm():
    p, args, shape = _setup(Nx=5, Ny=1, Nz=1, Nn=9, Nm=1, Np=1)
    fac = _factor(args, shape)
    Ck = _rand(shape, 1)
    h = 3.7
    out = fac.B(h, Ck)
    assert abs(float(jnp.linalg.norm(out) / jnp.linalg.norm(Ck)) - 1.0) < 1e-13
    np.testing.assert_allclose(fac.B(-h, out), Ck, atol=1e-12)
    np.testing.assert_allclose(fac.B(h / 2, fac.B(h / 2, Ck)), out, atol=1e-12)
    lam, V = hermite_axis_eig(9)
    T = np.diag(np.sqrt(np.arange(1, 9) / 2), 1) + np.diag(np.sqrt(np.arange(1, 9) / 2), -1)
    assert np.abs(V @ np.diag(lam) @ V.T - T).max() < 1e-14
    s, ik = 1, 2  # species 1, k index 2
    kap = float(args[20][0][0, ik, 0]); a = float(args[11].reshape(2, 3)[s, 0]); u = float(args[12].reshape(2, 3)[s, 0])
    M = expm(-1j * h * kap * (a * T + u * np.eye(9)))
    np.testing.assert_allclose(out[s, 0, 0, :, 0, ik, 0], M @ np.asarray(Ck[s, 0, 0, :, 0, ik, 0]), atol=1e-12)


def test_analytic_hermite_front_at_a_step_far_beyond_the_explicit_limit():
    """Free streaming of a Maxwellian: c_n(t) = (-i x)^n / sqrt(n!) exp(-x^2/2), x = k a t / sqrt(2), in one Lawson step
    of x = 3 (h k a sqrt(2 Nn) ~ 48, ~13x the Dopri8 imaginary-axis limit)."""
    Nn = 64
    p, args, shape = _setup(Nx=5, Ny=1, Nz=1, Nn=Nn, Nm=1, Np=1, Ns=1, params={"u_s": jnp.zeros(3), "alpha_s": jnp.ones(3),
                                                                                 "qs": jnp.array([-1]), "Omega_cs": jnp.array([1.0])})
    kap = float(args[20][0][0, 1, 0])
    x = 3.0
    h = x * np.sqrt(2) / kap  # a = 1
    ck0 = jnp.zeros(shape, dtype=complex).at[0, 0, 0, 0, 0, 1, 0].set(1.0)
    fac_fn = hermite_streaming_factor("first", shape)
    ls = LawsonSolver(diffrax.Dopri8(), factor_fn=fac_fn)
    fac = fac_fn(0.0, (ck0,), args)
    f = lambda t, y, a: (fac.streaming(y[0]),)
    out = diffrax.diffeqsolve(diffrax.ODETerm(f), ls, 0.0, h, h, (ck0,), args=args, max_steps=10,
                              stepsize_controller=diffrax.ConstantStepSize()).ys[0][0]
    n = np.arange(Nn)
    from scipy.special import gammaln
    ref = (-1j * x) ** n / np.exp(0.5 * gammaln(n + 1)) * np.exp(-x ** 2 / 2)
    np.testing.assert_allclose(np.asarray(out[0, 0, 0, :, 0, 1, 0]), ref, atol=1e-12)
    # the plain explicit solver blows up at the same step
    plain = diffrax.diffeqsolve(diffrax.ODETerm(f), diffrax.Dopri8(), 0.0, h, h, (ck0,), args=args, max_steps=10,
                                stepsize_controller=diffrax.ConstantStepSize()).ys[0][0]
    assert float(jnp.abs(plain).max()) > 50  # exact |c_n| <= 1


def _forced_problem(rate_scale):
    p, args, shape = _setup(Nx=5, Ny=1, Nz=1, Nn=10, Nm=1, Np=1, Ns=1, params={"u_s": jnp.array([0.3, 0, 0]),
                            "alpha_s": jnp.ones(3), "qs": jnp.array([-1]), "Omega_cs": jnp.array([1.0])})
    n = jnp.arange(10)[None, None, None, :, None, None, None]
    rate = rate_scale * (n / 9.0) ** 2 * jnp.ones(shape)
    fac_fn = hermite_streaming_factor("first", shape, (lambda t, y, a: rate) if rate_scale else None)
    G = _factor(args, shape)
    def f(t, y, a):
        c = y[0]
        return (G.streaming(c) - rate * c + jnp.sin(2 * t) * 0.3 + 0.05 * c * jnp.abs(c),)
    y0 = (_rand(shape, 3) * 0.3,)
    return args, f, y0, fac_fn


def test_order_with_streaming_and_with_rate_in_the_factor():
    for scale in (0.0, 40.0):
        args, f, y0, fac_fn = _forced_problem(scale)
        def run(solver, h=None, rtol=None):
            ctrl = diffrax.ConstantStepSize() if h else diffrax.PIDController(rtol=rtol, atol=rtol)
            return diffrax.diffeqsolve(diffrax.ODETerm(f), solver, 0.0, 1.0, h or 1e-3, y0, args=args,
                                       stepsize_controller=ctrl, max_steps=100000).ys[0][0]
        ref = run(diffrax.Dopri8(), rtol=1e-13)
        for base, p_, hs in ((diffrax.Heun(), 2, (1e-2, 5e-3, 2.5e-3)), (diffrax.Dopri8(), 4.5, (0.1, 0.05, 0.025))):
            errs = [float(jnp.abs(run(LawsonSolver(base, factor_fn=fac_fn), h=h) - ref).max()) for h in hs]
            slopes = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
            # Dopri8 is pre-asymptotic (slope 4-5, identical to the plain solver on this nonanalytic |c| problem): only check >= 4
            assert min(slopes) > min(p_ - 0.7, 3.8) or max(errs) < 1e-12, (scale, base, errs, slopes)


def test_simulation_streaming_matches_plain_and_keeps_k0_density():
    kw = dict(Nx=9, Nn=16, Ns=2, timesteps=5, input_parameters={"t_max": 2.0, "nu": 0.5})
    plain = simulation(**kw)
    for mode in ("streaming", "streaming+diagonal"):
        lw = simulation(**kw, integrating_factor=mode)
        a, b = np.asarray(plain["Ck"]), np.asarray(lw["Ck"])
        assert np.all(np.isfinite(b))
        assert np.abs(a - b).max() < 1e-6 * np.abs(a).max(), mode
        Nn = 16
        np.testing.assert_allclose(b[:, [0, Nn], 0, 0, 0], np.broadcast_to(b[0, [0, Nn], 0, 0, 0], (b.shape[0], 2)), atol=1e-12)  # k=0 density (G = 0)
    again = simulation(**kw, integrating_factor=False)
    assert np.array_equal(np.asarray(plain["Ck"]), np.asarray(again["Ck"]))  # default untouched, bitwise


def test_gradient_wrt_width_matches_finite_difference():
    p, args, shape = _setup(Nx=5, Ny=1, Nz=1, Nn=8, Nm=1, Np=1, Ns=1, params={"u_s": jnp.array([0.3, 0, 0]),
                            "alpha_s": jnp.ones(3), "qs": jnp.array([-1]), "Omega_cs": jnp.array([1.0])})
    y0 = (_rand(shape, 5) * 0.5,)
    def loss(w):
        a = list(args); a[11] = w * jnp.ones(3)
        ls = LawsonSolver(diffrax.Dopri8(), factor_fn=hermite_streaming_factor("first", shape))
        G = hermite_streaming_factor("first", shape)(0.0, y0, tuple(a))
        f = lambda t, y, ar: (G.streaming(y[0]) + 0.1 * y[0] * jnp.abs(y[0]),)
        out = diffrax.diffeqsolve(diffrax.ODETerm(f), ls, 0.0, 1.0, 0.5, y0, args=tuple(a), max_steps=100,
                                  stepsize_controller=diffrax.ConstantStepSize()).ys[0][0]
        return jnp.sum(jnp.abs(out) ** 2)
    g = jax.grad(loss)(1.2)
    eps = 1e-6
    fd = (loss(1.2 + eps) - loss(1.2 - eps)) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-5)


def test_basis_fn_overrides_args_basis():
    p, args, shape = _setup(Nx=5, Ny=1, Nz=1, Nn=6, Nm=1, Np=1, Ns=1, params={"u_s": jnp.zeros(3), "alpha_s": jnp.ones(3),
                            "qs": jnp.array([-1]), "Omega_cs": jnp.array([1.0])})
    st = (jnp.zeros(shape, dtype=complex),)
    f1 = hermite_streaming_factor("first", shape)(0.0, st, args)
    f2 = hermite_streaming_factor("first", shape, basis_fn=lambda t, y, a: (2.0 * jnp.ones(3), 0.5 * jnp.ones(3)))(0.0, st, args)
    z = _rand(shape, 7)
    assert float(jnp.abs(f1.B(1.0, z) - f2.B(1.0, z)).max()) > 1e-3
    np.testing.assert_allclose(f2.a, 2.0)
    np.testing.assert_allclose(f2.u, 0.5)
