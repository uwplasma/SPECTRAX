"""Independent CPU algebra for saved-coefficient sensitivity; no solver imports."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad, simpson
from scipy.special import expit, gamma

spec = importlib.util.spec_from_file_location(
    'tail_sensitivity', Path(__file__).parents[1]/'benchmarks/tail_sensitivity.py')
sensitivity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sensitivity)


@pytest.mark.parametrize('multiplier', [4, 5, 6])
def test_maxwellian_tail_against_independent_radial_integral(multiplier):
    alpha = np.full(3, .25)
    epsilon = 3*alpha[0]**2/4
    # Dimensionless E/T has Gamma(3/2,1) density; epsilon=3T/2.
    exact = quad(lambda x: x**1.5*np.exp(-x)/gamma(1.5)/1.5
                 * expit((x-1.5*multiplier)/.75), 0, np.inf, epsabs=1e-12)[0]
    kernel = sensitivity.tail_kernel((1, 1, 1), alpha, np.zeros(3), epsilon,
                                     epsilon, multiplier, 96)
    value = kernel[0, 0, 0]/np.prod(alpha)
    # Fixed q96 integrates the nonpolynomial sigmoid approximately (largest
    # relative error here is 3.8e-6). This is not a physical acceptance gate.
    np.testing.assert_allclose(value, exact, rtol=1e-5, atol=1e-11)
    low = sensitivity.tail_kernel((1, 1, 1), alpha, np.zeros(3), epsilon,
                                  epsilon, multiplier, 48)[0, 0, 0]/np.prod(alpha)
    assert abs(value-exact) < abs(low-exact)


def test_shifted_anisotropic_signed_energy_normalization():
    alpha, shift = np.array([.2, .3, .4]), np.array([.13, -.17, .21])
    c = np.zeros((3, 3, 3))
    c[0, 0, 0] = 2.
    first, second = np.array([.1, -.2, .3]), np.array([-.4, .5, -.6])
    for axis, index in enumerate(((0, 0, 1), (0, 1, 0), (1, 0, 0))):
        c[index] = first[axis]
    for axis, index in enumerate(((0, 0, 2), (0, 2, 0), (2, 0, 0))):
        c[index] = second[axis]
    # A far-negative threshold saturates the sigmoid to unity: exact second moment.
    kernel = sensitivity.tail_kernel(c.shape, alpha, shift, 1., 1., -100, 48)
    expected = .5*np.prod(alpha)*np.sum(
        (shift**2+alpha**2/2)*2 + np.sqrt(2)*shift*alpha*first + alpha**2*second/np.sqrt(2))
    np.testing.assert_allclose(np.sum(c*kernel), expected, atol=1e-14)


def test_own_initial_subtraction_and_uniform_heldout():
    t, alpha = np.arange(71.), np.full(3, .25)
    epsilon = 3*alpha[0]**2/4
    arrays = {}
    for label, offset, slope in [('initial_phase', .2, .001), ('optimized_phase', .5, .002)]:
        c = np.zeros((len(t), 1, 3, 3, 3))
        c[:, 0, 0, 0, 0] = 1/np.prod(alpha)
        c[:, 0, 0, 0, 2] = offset+slope*t
        arrays[label] = c
    report = dict(tail_spec=dict(threshold=5*epsilon, width=.5*epsilon, normalization=epsilon),
                  runs={label: dict(samples=[dict(time=float(x)) for x in t]) for label in arrays})
    result = sensitivity.analyze(arrays, report, alpha, np.zeros(3))
    assert set(result['thresholds']) == {'4', '5', '6'}
    for mult in (4, 5, 6):
        for order in (48, 96):
            entry = result['thresholds'][str(mult)]['quadratures'][str(order)]
            k = sensitivity.tail_kernel((3, 3, 3), alpha, np.zeros(3), epsilon, epsilon, mult, order)[0, 0, 2]
            np.testing.assert_allclose(entry['benefits']['heldout'], .001*65*k, atol=1e-14)
            tt = t[40:61]
            w = 15/160*(1-((tt-50)/10)**2)**2
            np.testing.assert_allclose(entry['benefits']['training'], .001*k*simpson(w*tt, x=tt), atol=1e-14)
    with pytest.raises(ValueError, match='sample at t=40'):
        sensitivity.sampled_windows(np.array([0., 41., 50., 60., 65., 70.]), np.ones(6))
