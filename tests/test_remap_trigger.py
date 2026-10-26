"""Remap triggers with hysteresis, a recorded schedule and its frozen replay (for gradients)."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from diffrax import NoProgressMeter
from spectrax import simulation
from spectrax._remap import remap, remap_event, remap_trigger
from tests.test_remap import _maxwellian

jax.config.update("jax_enable_x64", True)
N = 16
B0 = jnp.array([[0.0, 0, 0], [1.0, 1, 1]])


def _state(d, r):
    return jnp.zeros((1, 1, 1, N, 1, 1, 1)).at[0, 0, 0, :, 0, 0, 0].set(_maxwellian(N, d, r))


@pytest.mark.parametrize("d,r,fires", [(0.2, 1.0, False), (0.35, 1.0, True), (-0.35, 1.0, True),
                                        (0.0, 1.03, False), (0.0, 1.08, True), (0.0, 0.93, True)])
def test_manufactured_shifted_and_heated_maxwellians(d, r, fires):
    fire, new, info = remap_trigger(_state(d, r), B0, N, 1, 1, 1)
    assert fire is fires
    if fires:  # after the remap the same state does not fire again
        C2 = remap(_state(d, r), B0, jnp.asarray(new), N, 1, 1, 1)
        assert remap_trigger(C2, jnp.asarray(new), N, 1, 1, 1)[0] is False


def test_slow_drift_does_not_chatter():
    """A Maxwellian drifting by 0.05 a per check fires every ~7 checks, never on consecutive checks."""
    basis, events = B0, []
    for j in range(1, 60):
        C = remap(_state(0.05 * j, 1.0), B0, basis, N, 1, 1, 1)  # same f, current basis
        fire, new, _ = remap_trigger(C, basis, N, 1, 1, 1)
        if fire:
            events.append(j)
            basis = jnp.asarray(new)
    gaps = np.diff(events)
    assert len(events) >= 7 and gaps.min() >= 6


def test_tail_trigger_needs_growth_and_a_move():
    C = _state(0.1, 1.0).at[0, 0, 0, 14, 0, 0, 0].set(1e-2)  # a tail, small drift
    assert remap_trigger(C, B0, N, 1, 1, 1, tail_on=1e-6)[0] is True
    assert remap_trigger(C, B0, N, 1, 1, 1, tail_on=1e-6, last_tail=[1e-4])[0] is False  # not doubled
    still = _state(0.0, 1.0).at[0, 0, 0, 14, 0, 0, 0].set(1e-2)
    assert remap_trigger(still, B0, N, 1, 1, 1, tail_on=1e-6)[0] is False  # nothing to move


def _segments(U, schedule=None, n_seg=6, dt_seg=2.0):
    """Two-species uniform plasma oscillation in a fixed basis, remapped between segments either live
    (trigger) or from a frozen schedule [(segment, new_basis), ...]."""
    Nn, Nx, a_e, a_i, Om_i = 10, 2, 0.2, 0.05, 0.1
    Ck = (jnp.zeros((2 * Nn, 1, Nx // 2 + 1, 1), complex).at[0, 0, 0, 0].set(a_e ** -3)
          .at[1, 0, 0, 0].set(np.sqrt(2) * U / a_e * a_e ** -3).at[Nn, 0, 0, 0].set(a_i ** -3))
    Fk = jnp.zeros((6, 1, Nx // 2 + 1, 1), complex)
    basis = jnp.array([[0.0] * 6, [a_e] * 3 + [a_i] * 3])
    events = []
    for k in range(n_seg):
        p = {"Ck_0": Ck, "Fk_0": Fk, "qs": jnp.array([-1.0, 1.0]), "alpha_s": basis[1], "u_s": basis[0],
             "Omega_cs": jnp.array([1.0, Om_i]), "mi_me": 1 / Om_i, "nu": 0.0, "t_max": dt_seg, "ode_tolerance": 1e-11}
        out = simulation(p, Nx=Nx, Nn=Nn, Ns=2, timesteps=2, progress_meter=NoProgressMeter())
        Ck, Fk = out["Ck"][-1], out["Fk"][-1]
        if schedule is None:
            fire, new, _ = remap_trigger(Ck, basis, Nn, 1, 1, 2)
            if fire:
                Ck, rec = remap_event(Ck, basis, jnp.asarray(new), Nn, 1, 1, 2, t=(k + 1) * dt_seg)
                events.append((k, jnp.asarray(new), rec))
                basis = jnp.asarray(new)
        else:
            for seg, new in schedule:
                if seg == k:
                    Ck, basis = remap(Ck, basis, new, Nn, 1, 1, 2), new
    return Ck, Fk, events


def test_live_schedule_replays_bitwise_and_is_differentiable():
    U = 0.08
    Ck, Fk, events = _segments(U)
    assert len(events) >= 2 and all(ev[2]["moment_defect"] < 1e-13 for ev in events)
    schedule = [(k, new) for k, new, _ in events]
    Ck2, Fk2, _ = _segments(U, schedule)
    assert jnp.array_equal(Ck, Ck2) and jnp.array_equal(Fk, Fk2)

    def objective(U):
        return jnp.real(_segments(U, schedule)[1][0, 0, 0, 0])  # final uniform E_x

    g = jax.grad(objective)(U)
    h = 1e-5
    fd = (objective(U + h) - objective(U - h)) / (2 * h)
    assert np.isfinite(g) and abs(g - fd) < 1e-6 * abs(fd)
