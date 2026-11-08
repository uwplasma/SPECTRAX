"""Lawson (integrating-factor) wrapper: frozen diagonal exact, order, zero rate bitwise, conservation, gradient."""
import numpy as np
import jax
import jax.numpy as jnp
import diffrax
from scipy.linalg import expm
from scipy.special import gammaln
from spectrax import LawsonSolver, simulation, hermite_streaming_factor
from spectrax._initialization import initialize_simulation_parameters
from spectrax._simulation import ode_system

jax.config.update("jax_enable_x64", True)


def _solve(f, y0, solver, t1=1.0, h=None, rtol=None, saveat=None, adjoint=None, args=None, max_steps=100000):
    ctrl = diffrax.ConstantStepSize() if h else diffrax.PIDController(rtol=rtol, atol=rtol)
    return diffrax.diffeqsolve(diffrax.ODETerm(f), solver, 0.0, t1, h or 1e-3, y0, stepsize_controller=ctrl,
                               saveat=saveat or diffrax.SaveAt(t1=True), max_steps=max_steps, args=args,
                               **({} if adjoint is None else {"adjoint": adjoint}))


def test_frozen_diagonal_is_exact_at_stiff_step():
    """y' = -r y with the same rate: exact for any h, even r h = 200 where plain Dopri8 is unstable."""
    r = jnp.array([1.0, 50.0, 2000.0])
    f = lambda t, y, a: -r * y
    y0 = jnp.ones(3)
    ls = LawsonSolver(diffrax.Dopri8(), lambda t, y, a: r)
    out = _solve(f, y0, ls, t1=0.1, h=0.1).ys[0]
    np.testing.assert_allclose(out, np.exp(-0.1 * np.asarray(r)), rtol=1e-13, atol=0)
    plain = _solve(f, y0, diffrax.Dopri8(), t1=0.1, h=0.1).ys[0]
    assert float(jnp.abs(plain).max()) > 1e3  # explicit step far outside the stability region


def test_order_with_state_dependent_rate():
    """Heun-Lawson (order 2) and Dopri8-Lawson (order 8) on y' = -c(1+y_0^2) y + sin(t) (rate frozen at step start)."""
    c = 30.0
    f = lambda t, y, a: -c * (1 + y ** 2) * y + jnp.sin(t) * jnp.array([1.0, 0.5])
    rate = lambda t, y, a: c * (1 + y ** 2)
    y0 = jnp.array([0.7, -0.3])
    ref = _solve(f, y0, diffrax.Dopri8(), t1=1.0, rtol=1e-14).ys[0]
    for base, p, hs in ((diffrax.Heun(), 2, (2e-2, 1e-2, 5e-3)), (diffrax.Dopri8(), 8, (0.1, 0.05, 0.025))):
        errs = [float(jnp.abs(_solve(f, y0, LawsonSolver(base, rate), h=h).ys[0] - ref).max()) for h in hs]
        slopes = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
        assert min(slopes) > p - 0.6, (base, errs, slopes)


def test_zero_rate_bitwise_equal_and_dense_output():
    A = jnp.array([[0.0, 1.0], [-4.0, 0.0]])
    f = lambda t, y, a: A @ y + jnp.array([0.0, jnp.cos(t)])
    y0 = jnp.array([1.0, 0.0])
    ts = jnp.linspace(0.0, 2.0, 11)
    sv = diffrax.SaveAt(ts=ts)
    ref = _solve(f, y0, diffrax.Dopri8(), t1=2.0, rtol=1e-9, saveat=sv)
    got = _solve(f, y0, LawsonSolver(diffrax.Dopri8(), lambda t, y, a: 0.0), t1=2.0, rtol=1e-9, saveat=sv)
    assert np.array_equal(np.asarray(ref.ys), np.asarray(got.ys))
    assert int(ref.stats["num_steps"]) == int(got.stats["num_steps"])


def test_dense_output_matches_exact_decay():
    r = jnp.array([3.0, 40.0])
    f = lambda t, y, a: -r * y
    ts = jnp.linspace(0.0, 1.0, 7)
    ls = LawsonSolver(diffrax.Dopri8(), lambda t, y, a: r)
    out = _solve(f, jnp.ones(2), ls, t1=1.0, rtol=1e-8, saveat=diffrax.SaveAt(ts=ts)).ys
    np.testing.assert_allclose(out, np.exp(-np.asarray(ts)[:, None] * np.asarray(r)[None]), rtol=1e-7)


def test_pytree_state_with_rows_outside_exponent_untouched():
    """A ledger-like state with rate 0 integrates exactly as in the wrapped solver."""
    def f(t, y, a):
        return {"C": -5.0 * y["C"] + y["L"], "L": jnp.sum(y["C"]) * 0.1 + 0.0 * y["L"]}
    y0 = {"C": jnp.array([1.0, 2.0]), "L": jnp.array(0.0)}
    rate = lambda t, y, a: {"C": jnp.full(2, 5.0), "L": 0.0}
    out = _solve(f, y0, LawsonSolver(diffrax.Dopri8(), rate), t1=1.0, rtol=1e-11).ys
    ref = _solve(f, y0, diffrax.Dopri8(), t1=1.0, rtol=1e-12).ys
    for k in out:
        np.testing.assert_allclose(out[k], ref[k], rtol=1e-9, atol=1e-11)


def test_simulation_default_unchanged_and_low_moments_conserved():
    kw = dict(Nx=9, Nn=12, Ns=2, timesteps=5, input_parameters={"t_max": 0.5, "nu": 5.0})
    base = simulation(**kw)
    again = simulation(**kw, integrating_factor=False)
    assert np.array_equal(np.asarray(base["Ck"]), np.asarray(again["Ck"]))
    lw = simulation(**kw, integrating_factor=True)
    Ck0 = np.asarray(base["Ck"]); Ck1 = np.asarray(lw["Ck"])
    assert np.all(np.isfinite(Ck1))
    Nn = 12
    rows = np.array([s * Nn + n for s in range(2) for n in range(3)])  # n <= 2 rows of each species
    scale = np.abs(Ck0).max()
    assert np.abs(Ck1[:, rows] - Ck0[:, rows]).max() < 1e-6 * scale  # same physics at the solver tolerance


def test_gradient_smoke():
    r = lambda t, y, a: a[0] * jnp.ones_like(y)
    def loss(p):
        f = lambda t, y, a: -p * y + jnp.sin(y)
        ls = LawsonSolver(diffrax.Dopri8(), lambda t, y, a: p * jnp.ones_like(y))
        return _solve(f, jnp.ones(2), ls, t1=1.0, rtol=1e-8).ys[0].sum()
    g = jax.grad(loss)(3.0)
    eps = 1e-5
    fd = (loss(3.0 + eps) - loss(3.0 - eps)) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-5)


def test_simulation_stiff_hypercollision_matches_plain_with_fewer_steps():
    kw = dict(Nx=9, Nn=12, Ns=2, timesteps=5, input_parameters={"t_max": 2.0, "nu": 4000.0})
    plain = simulation(**kw)
    lw = simulation(**kw, integrating_factor=True)
    scale = np.abs(np.asarray(plain["Ck"])).max()
    assert np.abs(np.asarray(lw["Ck"]) - np.asarray(plain["Ck"])).max() < 1e-6 * scale
    assert int(lw["solver_stats"]["num_steps"]) * 2 < int(plain["solver_stats"]["num_steps"])


def test_simulation_rate_layout_matches_state():
    """The rate is a (Ck, Fk) pytree like the state, equals nu*col + D*k2 on Ck and exactly zero on the field."""
    from spectrax import simulation_rate
    from spectrax._initialization import initialize_simulation_parameters
    Nx, Ny, Nz, Nn, Nm, Np, Ns = 9, 1, 1, 6, 1, 1, 2
    p = initialize_simulation_parameters({"nu": 5.0, "D": 0.3}, Nx, Ny, Nz, Nn, Nm, Np, Ns, 3, 0.01)
    args = (0,) * 8 + (p["nu"], p["D"]) + (0,) * 9 + (p["k2_grid"], 0, p["collision_matrix"])
    rC, rF = simulation_rate(args, Ns, Nn, Nm, Np, Ny, Nx, Nz)
    r7 = np.asarray(rC)
    assert r7.shape == (Ns, Np, Nm, Nn, Ny, Nx // 2 + 1, Nz) and np.all(r7 >= 0.0)
    assert float(rF) == 0.0  # the six field rows are never damped
    col = np.asarray(p["collision_matrix"])
    k2 = np.asarray(p["k2_grid"])[0, :, 0]
    np.testing.assert_allclose(r7[0, 0, 0, :, 0, :, 0], 5.0 * col[0, 0, :, None] + 0.3 * k2[None, :])


def test_exp_overflow_step_is_rejected_not_accepted():
    """Adaptive stepping past r h ~ 700 used to accept an inf step (scale = inf => zero scaled error) and then reject
    forever; now the overflowing step is rejected, shrunk, and the solve finishes with a finite result."""
    r = jnp.array([1.0e4, 3.0])
    f = lambda t, y, a: -r * y + jnp.array([0.0, 1.0])
    ls = LawsonSolver(diffrax.Dopri8(), lambda t, y, a: r)
    sol = diffrax.diffeqsolve(diffrax.ODETerm(f), ls, 0.0, 2.0, 1e-3, jnp.ones(2),
                              stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9), max_steps=5000)
    out = np.asarray(sol.ys[0])
    assert np.all(np.isfinite(out)) and abs(out[0]) < 1e-12
    np.testing.assert_allclose(out[1], 1.0 / 3.0 + (2.0 / 3.0) * np.exp(-6.0), rtol=1e-4)


# --- exact Hermite streaming factor ---------------------------------------------------------------------------

def _setup(Nx=5, Ny=1, Nz=1, Nn=7, Nm=1, Np=1, Ns=1, **params):
    """Parameters, SPECTRAX ``args`` tuple and the ``Ck`` shape; single Maxwellian species with a drift unless overridden."""
    if Ns == 1:
        params = {"u_s": jnp.array([0.3, 0, 0]), "alpha_s": jnp.ones(3), "qs": jnp.array([-1]),
                  "Omega_cs": jnp.array([1.0]), **params}
    p = initialize_simulation_parameters(params, Nx, Ny, Nz, Nn, Nm, Np, Ns, 3, 0.01)
    args = (Nx, Ny, Nz, Nn, Nm, Np, Ns, p["qs"], p["nu"], p["D"], p["Omega_cs"], p["alpha_s"], p["u_s"],
            p["Lx"], p["Ly"], p["Lz"], p["kx_grid"], p["ky_grid"], p["kz_grid"], p["k2_grid"], p["nabla"],
            p["collision_matrix"], p["sqrt_n_plus"], p["sqrt_n_minus"], p["sqrt_m_plus"], p["sqrt_m_minus"],
            p["sqrt_p_plus"], p["sqrt_p_minus"])
    return args, (Ns, Np, Nm, Nn, Ny, Nx // 2 + 1, Nz)


def _rand(shape, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))


def _factor(args, shape, rate_fn=None, **kw):
    return hermite_streaming_factor(rate_fn, **kw)(0.0, (jnp.zeros(shape, dtype=complex),), args)


def _hermite_T(N):
    off = np.sqrt(np.arange(1, N) / 2)
    return np.diag(off, 1) + np.diag(off, -1)


def test_generator_equals_streaming_part_of_the_rhs():
    """G Ck equals dCk/dt of the real ode_system with Fk = 0, nu = D = 0 (3-D, drifts, unequal widths, Nm,Np > 1)."""
    args, shape = _setup(Ny=3, Nz=3, Nm=3, Np=2, Ns=2, nu=0.0, alpha_s=jnp.array([0.7, 0.9, 1.1, 0.5, 0.6, 0.8]))
    Ck = _rand(shape)
    dy = ode_system(*args[:7], 0.0, (Ck, jnp.zeros((6, *shape[-3:]), dtype=complex)), args)
    np.testing.assert_allclose(dy[0].reshape(shape), _factor(args, shape).streaming(Ck), rtol=1e-12, atol=1e-12)


def test_B_is_unitary_semigroup_and_matches_expm():
    args, shape = _setup(Nn=9, Ns=2)
    fac = _factor(args, shape)
    Ck, h = _rand(shape, 1), 3.7
    out = fac.B(h, Ck)
    assert abs(float(jnp.linalg.norm(out) / jnp.linalg.norm(Ck)) - 1.0) < 1e-13
    np.testing.assert_allclose(fac.B(-h, out), Ck, atol=1e-12)
    np.testing.assert_allclose(fac.B(h / 2, fac.B(h / 2, Ck)), out, atol=1e-12)
    s, ik = 1, 2
    kap, a, u = (float(args[20][0][0, ik, 0]), float(args[11].reshape(2, 3)[s, 0]), float(args[12].reshape(2, 3)[s, 0]))
    M = expm(-1j * h * kap * (a * _hermite_T(9) + u * np.eye(9)))
    np.testing.assert_allclose(out[s, 0, 0, :, 0, ik, 0], M @ np.asarray(Ck[s, 0, 0, :, 0, ik, 0]), atol=1e-12)


def test_analytic_hermite_front_at_a_step_far_beyond_the_explicit_limit():
    """Free streaming of a Maxwellian: c_n(t) = (-i x)^n / sqrt(n!) exp(-x^2/2), x = k a t / sqrt(2), in one Lawson step
    of x = 3 (h k a sqrt(2 Nn) ~ 48, ~13x the Dopri8 imaginary-axis limit)."""
    Nn, x = 64, 3.0
    args, shape = _setup(Nn=Nn, u_s=jnp.zeros(3))
    h = x * np.sqrt(2) / float(args[20][0][0, 1, 0])  # a = 1
    ck0 = jnp.zeros(shape, dtype=complex).at[0, 0, 0, 0, 0, 1, 0].set(1.0)
    fac = _factor(args, shape)
    f = lambda t, y, a: (fac.streaming(y[0]),)
    run = lambda solver: _solve(f, (ck0,), solver, t1=h, h=h, args=args, max_steps=10).ys[0][0]
    n = np.arange(Nn)
    ref = (-1j * x) ** n / np.exp(0.5 * gammaln(n + 1)) * np.exp(-x ** 2 / 2)
    np.testing.assert_allclose(np.asarray(run(LawsonSolver(diffrax.Dopri8(), factor_fn=hermite_streaming_factor()))[0, 0, 0, :, 0, 1, 0]),
                               ref, atol=1e-12)
    assert float(jnp.abs(run(diffrax.Dopri8())).max()) > 50  # the plain explicit solver blows up (exact |c_n| <= 1)


def test_order_with_streaming_and_with_rate_in_the_factor():
    args, shape = _setup(Nn=10)
    G = _factor(args, shape)
    for scale in (0.0, 40.0):
        rate = scale * (jnp.arange(10)[None, None, None, :, None, None, None] / 9.0) ** 2 * jnp.ones(shape)
        fac_fn = hermite_streaming_factor((lambda t, y, a: rate) if scale else None)
        f = lambda t, y, a: (G.streaming(y[0]) - rate * y[0] + jnp.sin(2 * t) * 0.3 + 0.05 * y[0] * jnp.abs(y[0]),)
        y0 = (_rand(shape, 3) * 0.3,)
        ref = _solve(f, y0, diffrax.Dopri8(), rtol=1e-13, args=args).ys[0][0]
        for base, p, hs in ((diffrax.Heun(), 2, (1e-2, 5e-3, 2.5e-3)), (diffrax.Dopri8(), 4.5, (0.1, 0.05, 0.025))):
            errs = [float(jnp.abs(_solve(f, y0, LawsonSolver(base, factor_fn=fac_fn), h=h, args=args).ys[0][0] - ref).max())
                    for h in hs]
            slopes = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
            # Dopri8 is pre-asymptotic (slope 4-5, identical to the plain solver on this nonanalytic |c| problem)
            assert min(slopes) > min(p - 0.7, 3.8) or max(errs) < 1e-12, (scale, base, errs, slopes)


def test_simulation_streaming_matches_plain_and_keeps_k0_density():
    kw = dict(Nx=9, Nn=16, Ns=2, timesteps=5, input_parameters={"t_max": 2.0, "nu": 0.5})
    plain = simulation(**kw)
    lw = simulation(**kw, integrating_factor="streaming")
    a, b = np.asarray(plain["Ck"]), np.asarray(lw["Ck"])
    assert np.all(np.isfinite(b)) and np.abs(a - b).max() < 1e-6 * np.abs(a).max()
    np.testing.assert_allclose(b[:, [0, 16], 0, 0, 0], np.broadcast_to(b[0, [0, 16], 0, 0, 0], (b.shape[0], 2)), atol=1e-12)
    assert np.array_equal(a, np.asarray(simulation(**kw, integrating_factor=False)["Ck"]))  # default bitwise unchanged


def test_streaming_gradient_wrt_width_matches_finite_difference():
    args, shape = _setup(Nn=8)
    y0 = (_rand(shape, 5) * 0.5,)

    def loss(w):
        a = (*args[:11], w * jnp.ones(3), *args[12:])
        G = _factor(a, shape)
        f = lambda t, y, ar: (G.streaming(y[0]) + 0.1 * y[0] * jnp.abs(y[0]),)
        out = _solve(f, y0, LawsonSolver(diffrax.Dopri8(), factor_fn=hermite_streaming_factor()), t1=1.0, h=0.5,
                     args=a, max_steps=100).ys[0][0]
        return jnp.sum(jnp.abs(out) ** 2)

    eps = 1e-6
    np.testing.assert_allclose(jax.grad(loss)(1.2), (loss(1.2 + eps) - loss(1.2 - eps)) / (2 * eps), rtol=1e-5)


def test_basis_fn_overrides_args_basis():
    args, shape = _setup(Nn=6, u_s=jnp.zeros(3))
    f1 = _factor(args, shape)
    f2 = _factor(args, shape, basis_fn=lambda t, y, a: (2.0 * jnp.ones(3), 0.5 * jnp.ones(3)))
    z = _rand(shape, 7)
    assert float(jnp.abs(f1.B(1.0, z) - f2.B(1.0, z)).max()) > 1e-3
    np.testing.assert_allclose(f2.a, 2.0)
    np.testing.assert_allclose(f2.u, 0.5)
