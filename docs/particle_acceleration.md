# Differentiable electron-tail control

`simulation_final` returns the final Hermite/Fourier state using the same fixed
RK4 plasma RHS as the forward calculation. Compose a scalar observable with
`jax.value_and_grad`; JVP and VJP work with the default SOLVAX replay. For example:

```python
from spectrax import simulation_final
import jax
import jax.numpy as jnp

def loss(initial_coefficients):
    C, F = simulation_final(dict(parameters, Ck_0=initial_coefficients),
                            steps=600, Nx=16, Ny=16, Nn=8, Nm=8, Np=8)
    return jnp.sum(jnp.abs(F[3:])**2)  # Replace with a physical scalar observable.
value, gradient = jax.value_and_grad(loss)(parameters['Ck_0'])
```

Default replay retains O(S sqrt(N)) state for N steps and state size S.
`checkpoints=K` uses a fixed O(K S) retained-state budget and supports reverse
mode only. Both include RHS workspace; memory still grows with phase-space
resolution. `integrand(t,C,F)` optionally accumulates a scalar/vector RK-stage
integral without retaining a trajectory. This differentiates the discrete RK4
solver; it is not a continuous adjoint or a block-implicit reverse sweep.

## One reproducible example

[2D_particle_acceleration.py](../Examples/2D_particle_acceleration.py) controls eight
phases at fixed initial magnetic spectrum and integrated field/particle energies
in a collisionless 2D3V Orszag–Tang plasma. It optimizes a smooth electron tail
above five initial thermal energies, subtracting each cell's exact flow. The
threshold and normalization stay fixed, and each control's own initial value is
subtracted. Twenty normalized Simpson intervals over [40,60] define the sampled
objective; both the distribution and local flow are differentiated.

`--objective excess` additionally subtracts the tail of a Gaussian matching each
cell's density, flow and full covariance. This removes changes attributable to
local second moments, but a positive residual is not a unique acceleration
mechanism. Signed Hermite distributions are never clipped. Refine dynamics,
quadrature and sampling, and check local negativity before physical interpretation.
Four-cell quadrature batches and rematerialization bound velocity workspace;
only the generic solver API enters the library itself.

```sh
export PYTHONPATH="$PWD"
# GPU: CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
# CPU smoke: add --grid 12 --hermite 3 --steps 4 --window 0 .2 --intervals 2 --quadrature 16
python Examples/2D_particle_acceleration.py --objective excess --method ad --output results/ad
python Examples/2D_particle_acceleration.py --objective excess --method fd --output results/fd
python Examples/2D_particle_acceleration.py --validate --controls-file results/ad/results.json --grid 24 --hermite 16 --steps 1200 --quadrature 128 --output results/validation
python Examples/2D_particle_acceleration.py --plot results/ad/results.json results/fd/results.json results/validation/results.json --output results/figures
```

AD and centered FD use identical initial controls, loss scaling and L-BFGS-B
settings. FD runs no AD transformation. Reports include compilation, warm and
full elapsed times, evaluation counts, source hashes and dependency versions.
A factor 1e4 rescales the loss for optimizer tolerances; reported physical gains
undo that factor. `--fd-check` checks two finite-difference steps separately;
exclude that optional diagnostic from matched optimization timing runs.
`--evaluate-only`, `--controls-key optimized_phase`, refined grids/Hermites/steps,
`--intervals` and `--quadrature` support frozen-control validation. Generated
reports and both PNG/vector-PDF figures stay in ignored `results/`, outside Git.

## Interpretation and provenance

The earlier local-tail study demonstrated increased energization, not nonthermal
acceleration. Its 10.28% increase was relative to a small baseline gain: the added
energy was only 0.002235% of initial electron thermal energy. The earlier 3.01×
AD/FD speedup used a different energy objective and must not be attributed to this
tail example. Complete exploratory evidence is preserved at historical commit
`adaf5423f3d670dd1e2e6781a0321ded317b194f`; it is intentionally outside this PR's
review diff. New matched-objective results and figures belong in the PR comment.

This builds on [Joglekar–Thomas differentiable kinetic optimization](https://doi.org/10.1017/S0022377822000939),
[spectral adjoints](https://arxiv.org/abs/2506.14792), and
[local moment-matched non-Maxwellianity diagnostics](https://doi.org/10.1017/S0022377820001270).
The contribution is a small composable SPECTRAX API and a measured constrained
kinetic-control example, not priority for differentiable plasma optimization.
SOLVAX 0.20.0 is the numerical-study dependency; its merged PR103 does not change
that provenance. SPECTRAX PR42 and PR43 remain open for collaborators to merge.
