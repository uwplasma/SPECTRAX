# Local flow-relative tail: implementation and pilot evidence

This extends the lab-frame result in `acceleration_results.md`. The old tail
optimum increased bulk energy while decreasing internal energy; its positive
lab-frame tail benefit therefore does not establish improved random-particle
energization. The new instantaneous observable subtracts each cell's exact bulk
velocity, differentiating both the represented distribution and that bulk velocity.
Its threshold, width and normalization remain fixed to the common initial state.

`make_local_tail_objective` is a private host factory returning a scalar JAX
callable. `Examples/2D_local_tail_control.py` composes it with the existing public
`simulation_final` API. No plasma adjoint or different time integrator is added.
Cellwise rematerialization limits velocity workspace; nested SOLVAX replay limits
retained time states. State and adjoint memory still grow with phase-space DOFs.

A cellwise tail is too expensive to insert at every RK stage without measurement.
The example instead uses a fixed normalized Simpson sum of the same smooth time
window. With 20 intervals, only 20 post-advance tail evaluations and the own-initial
term are needed, versus 2400 RK-stage evaluations in a 600-step integrand.
The window endpoints have zero weight. Both quadrature in time and velocity
must be refined; gradients are exact derivatives of this declared discrete
sampled objective, not of an assumed continuous-time integral.

## Algebra and CPU/GPU verification

31 CPU tests cover drift invariance, moment mapping, layouts, invalid densities,
and JVP/VJP/finite differences including the derivative of the local mean.
A 12²/H3, four-step rollout through T=0.2 with two sampled intervals and q16
agrees with directional centered differences to 1.98e-15 absolute error at h=1e-3.
The matched GPU run differs from CPU by 1.32e-17 in loss and 9.90e-21 in the
largest gradient component. These are plumbing checks, not resolved physics.
The rejected 8² smoke configuration had too many phase controls for its
spatial grid; it was replaced by 12² without changing the eight-control problem.

## Baseline pilots

At 16²/H8, nu=0, 600 RK4 steps and 20 sampled intervals over [40,60], the
baseline local-tail gain is 0.000217764031 at q24 and 0.000217043085 at q48.
The change is 0.332% in gain and 0.469% in the full phase gradient. The q24
result is retained; it does not establish sufficient value convergence.
A q48 full value/gradient takes 6.39 s on the RTX A4000 (first execution after
explicit compilation). Further quadrature, sampled-time and Hermite checks are
recorded with their source-versioned reports before a search is interpreted.

The q96 gain is 0.000217069986. Relative q48→q96 changes are 1.239e-4
in gain and 1.746e-4 in the full gradient, passing the pilot tolerances. q96
first execution takes 10.20 s. Full-horizon centered directional FD agrees to
1.27e-14 absolute at step1e-4. Doubling time intervals from20 to40 at q48 changes
gain by8.58e-6 and gradient by1.03e-5 relatively. These checks establish a usable
baseline configuration, not convergence at every possible optimized control.

## Scientific interpretation

Removing local flow excludes instantaneous bulk drift from the kernel, but
heating, anisotropy, density weighting and transport can still change the tail.
A companion diagnostic compares it with a Gaussian matching the *local* density,
flow and full covariance. Its signed excess tests shape beyond second moments;
it is not a positive-definite distance or a unique acceleration diagnostic.
See `local_tail_plan.md` for equations, primary literature and prospective gates.

## Reproduction

```sh
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
python Examples/2D_local_tail_control.py --quadrature 48 --evaluate-only --output results/local-tail-q48
python Examples/2D_local_tail_control.py --quadrature 48 --intervals 40 --evaluate-only --output results/local-tail-q48-i40
```

All numerical evidence in this iteration still uses released SOLVAX 0.20.0.
SOLVAX PR103 is merged at917d1a6; switching dependencies is a separate measured
change. SPECTRAX PR42 and PR43 remain open for other collaborators to merge.
