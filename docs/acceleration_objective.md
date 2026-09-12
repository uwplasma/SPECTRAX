# Velocity diagnostics and predeclared tail objective

This implements the velocity-observable lane of [the execution plan](acceleration_plan.md).
It supplies algebra and quadrature helpers, not an optimization or an acceleration
result. The convention below is fixed before new tail measurements and agrees with
the parent plan. Existing phase-example behavior is unchanged.

## Physical definition fixed before evaluation

Let epsilon_th,0 be the initial **mean internal kinetic energy per electron**,
excluding the local bulk velocity, and n0 the initial spatial-mean electron number
density. For the prescribed drifting Maxwellian with fixed thermal widths,
`epsilon_th,0 = m_e * sum(alpha_e**2) / 4`; isotropically this is `3*T_e,0/2`
with Boltzmann's constant absorbed into T. An arbitrary non-Maxwellian initial state
requires its actual internal energy per particle, not an inference from basis alpha.
The reference is a single shared initial physical value across all phase controls.

The preregistered choices are:

- Lab-frame particle energy `E(v) = m_e*|v|**2/2`, with electron mass 1 in solver units.
- Fixed threshold `E_star = 5 * epsilon_th,0` (isotropically `7.5*T_e,0`).
- Fixed sigmoid width `Delta_E = 0.5 * epsilon_th,0` (`0.75*T_e,0`).
- Smooth tail weight `S(E) = sigmoid((E-E_star)/Delta_E)`.
- Signed tail energy density `K_tail = integral E(v)*S(E(v))*fbar_e(v) d^3v`.
- Objective to maximize `J = K_tail/(n0*epsilon_th,0)`.

J measures tail energy as a fraction of **initial thermal energy**, not the fraction
of current total energy and not a probability bounded by one. The diagnostics also
return `tail_energy_fraction = K_tail/K_lab` for descriptive use, keeping its
changing denominator separate from the objective. Neither ratio is clipped or
regularized: zero denominators yield nonfinite values, and invalid signed
representations can give negative values or fractions outside [0,1]. Report these.
The parent tail optimization maximizes the smooth time average over [40,60] of
`J(Ct(theta))-J(C0(theta))`, using the parent's normalized time-window weight.
Differentiate **both** the evolved and initial-state terms; do not stop the
initial-tail gradient. This removes a control-dependent initial-tail offset from
the gain. Report initial tail versus controls as well as final/averaged tails and
their gain, because fixed integrated initial energies and spectra do not fix
higher velocity moments or tails. Initial-tail variation alone must not be called
acceleration. The factory itself remains an instantaneous linear functional.

Construct the spec once on the host, outside differentiation. Do not recompute the
threshold, width, or normalization from evolved states or choose them after viewing
control benefits. Lower/higher threshold sensitivity is a separately labeled study.

## Representation, quadrature, and memory

The solver stores modes in `(species,p,m,n,ky,kx,kz)` order, often flattening the
first four axes. With `xi_i=(v_i-u_i)/alpha_i`, the actual basis is

`f(v) = exp(-|xi|²)/pi^(3/2) * sum C[p,m,n] h_n(xi_x) h_m(xi_y) h_p(xi_z)`,

where `h_n=H_n/sqrt(2^n*n!)` uses physicists' Hermite polynomials. There is no extra
inverse-alpha factor multiplying this reconstruction: the Maxwellian coefficient
is already `C000=n/prod(alpha)`. Integration does require the velocity Jacobian
`d^3v=prod(alpha)*d^3xi`. The basis shift u is a representation parameter and need
not equal the physical bulk velocity encoded by the first Hermite coefficients.

The solver's Fourier transform uses `norm="forward"`. Therefore the spatial-average
Hermite coefficients are exactly `real(Ck[...,0,0,0])`, without a grid-size factor,
Parseval weights, inverse FFT, or domain volume. A global lab-frame tail has a
position-independent energy kernel, so spatial averaging commutes with its linear
velocity integral. These helpers require spatially constant alpha/u. A basis whose
alpha/u varies in space must be transformed to a common velocity representation
before averaging; averaging its coefficients directly is incorrect.

Tensor Gauss-Hermite quadrature integrates against `exp(-|xi|²)`. We contract the
normalized polynomials against product GH weights and `prod(alpha)/pi^(3/2)`.
Multiplying another Gaussian into those weights would count it twice. Density and
second moments are polynomial integrals; order q integrates each degree through
`2*q-1` exactly in exact arithmetic. The helper rejects q too small for the retained
modes' signed second moments. Smooth sigmoid tails are nonpolynomial and require
convergence checks; hard indicator tails and negative mass converge more slowly.

Diagnostics construct only a q³ velocity array for one Hermite block, with
separable mode contractions. Reconstruction accepts an arbitrary list/grid of
velocity points. The objective factory contracts the quadrature kernel once into
an `(Np,Nm,Nn)` array. Each objective call contracts this with the selected zero
Fourier mode. Neither path creates a space×velocity tensor or a modes×q³ basis
array; the factory freezes alpha/u and propagates derivatives through Ck only.
JAX's state-gradient output still has the input state's size.

## Saved-coefficient and objective APIs

```python
from spectrax._velocity_observables import (
    preregistered_tail_spec, velocity_diagnostics, quadrature_convergence,
    make_tail_objective, reconstruct_distribution,
)

spec = preregistered_tail_spec(epsilon_th0, n0)
# Saved Cbar is (Ns,Np,Nm,Nn) for ONE time. Iterate over sampled times.
c = Cbar[0].real
report = velocity_diagnostics(c, alpha_e, u_e, spec=spec,
                             mass=1.0, quadrature_order=24)
checks = quadrature_convergence(c, alpha_e, u_e, spec=spec,
                               mass=1.0, orders=(16, 24, 32))
fbar_at_points = reconstruct_distribution(c, velocities_xyz, alpha_e, u_e)

# Construct once, outside jit/grad; alpha_s/u_s have 3*Ns elements.
objective = make_tail_objective(
    alpha_s, u_s, Nn=Nn, Nm=Nm, Np=Np, Ns=Ns, species=0,
    spec=spec, mass=1.0, quadrature_order=24,
)
J = objective(Ck)  # scalar real JAX array; use -J for minimization
# Equivalently, report['objective'] is the scalar for saved mean coefficients.
```

`spatial_average_coefficients(Ck, *, Nn, Nm, Np, Ns, species=0)` accepts one flattened
four-dimensional or structured seven-dimensional Fourier state. It does not accept
a trajectory axis. The parent can directly use its saved `(p,m,n)` electron block.
`reconstruct_distribution` accepts velocities of shape `(...,3)` and differentiates
through coefficients, velocity, alpha and u. Supply real coefficients, positive
finite length-three alpha, finite u and positive finite mass to diagnostics.

Reports contain `density`, `kinetic_energy`, `smooth_tail_number`,
`smooth_tail_energy`, `hard_tail_number`, `hard_tail_energy`, `negative_mass`,
`negative_mass_fraction`, `tail_energy_fraction`, and `objective`. All integrals
are per unit spatial volume. Multiply by domain volume only for domain totals.
`quadrature_convergence` returns `orders`, a list of `diagnostics`, and successive
`absolute_changes` dictionaries. It makes no automatic convergence or validity claim.

## Local snapshot negativity with bounded velocity memory

```python
from spectrax._velocity_observables import spatial_negative_mass_diagnostics
local = spatial_negative_mass_diagnostics(
    Ck, alpha_s, u_s, Nx=Nx, Nn=Nn, Nm=Nm, Np=Np, Ns=Ns,
    species=0, quadrature_order=24,
)
# Repeat at q48 on selected initial/final snapshots; report changes.
# Keys: mean_negative_mass, max_cell_fraction, min_density.
```

This helper accepts the same flattened/structured single-state Fourier layouts.
Nx is explicit to disambiguate odd/even real transforms. It inverse-transforms
only the selected species using `irfftn(..., norm="forward")`, then `lax.map`
integrates one spatial cell at a time with the same GH polynomial/weight convention
as `velocity_diagnostics`. Storage includes the real-space Hermite state, one q³
velocity workspace and scalar outputs per cell, never full space×velocity storage.
The mass is the unweighted spatial mean of cell negative masses; `max_cell_fraction`
is the maximum of each cell's negative mass divided by its own density.
`min_density` is the minimum signed density from the local zeroth moment.
Nonpositive density produces an infinite cell fraction to prevent a false pass.
Alpha/u must be finite spatially constant basis parameters with positive alpha.
No mass/spec is required: negativity is a number-density diagnostic independent of
the energy threshold. The operation does not modify Ck or clip the tail objective.

## Acceptance and interpretation

For cheap pilot algebra, compare q=16,24,32 on saved coefficients. The parent
measurement runner uses q=24,48,96 for each saved global coefficient and q=24,48
for local checks at selected initial/final snapshots; increase q if necessary. Require relative successive smooth-tail-energy change
below 1e-3 for sufficiently resolved quadratures; report absolute changes near
zero, where a relative criterion is ill-conditioned. Require averaged negative
mass/density below 1e-8. A failed gate is a reportable result, not authorization to
optimize an invalid representation or alter this preregistered threshold.
Independently refine Hermite resolution and check tail benefit and directional
derivatives (parent's 5% gate). Quadrature refinement alone cannot repair velocity
truncation. Compare gradients from objectives prepared at each q as well as values.

Negative mass is `integral max(-fbar,0) dv`, with the maximum used **only for the
diagnostic**. Tail objectives always integrate signed f, without positivity
clipping. Globally averaged negativity is a lower bound on the spatial average of
local negative mass; cancellation between spatial regions can hide local negative
lobes entirely. The bounded-memory local helper below exposes that hidden negativity at selected
snapshots. Finite quadrature samples can miss negative lobes between nodes or in
the far tail; even passing local q24/q48 checks is not continuous certification.

Lab-frame tail growth can result from coherent bulk shifts, spatially varying
bulk flows, heating of a Maxwellian, or genuinely changed distribution shape. It
alone establishes none of those mechanisms, nonthermal acceleration, or irreversible
heating. Pair it with bulk/internal energy, mean-flow and local distribution
analysis. Subtracting one global mean velocity would still leave spatially varying
bulk flow and is not equivalent to the local internal-energy diagnostic.

## Algebra-only validation

`tests/test_velocity_observables.py` checks anisotropic shifted Maxwellian
reconstruction, density and energy normalization; isotropic analytic Gamma-law
Maxwellian tail integrals (including independent smooth one-dimensional integration);
q16/24/32 changes; signed negative mass; odd/even Fourier grids; spatial averaging
hiding local negativity; factory/saved-coefficient agreement; JIT, JVP, reverse AD
and centered FD; reconstruction derivatives through alpha/u; and invalid inputs. Local snapshot tests compare the mapped result with independent
cell integrations on odd/even grids, exercise JIT/flat layouts and nonpositive
densities, and verify local negativity hidden by spatial averaging.
The smooth-tail reference tolerance in the unit test is a numerical implementation
check, not an assertion that every physical state passes the stricter pilot gate.
No trajectory, optimization, GPU, SSH, or lengthy CPU simulation is part of this
validation. Higher-resolution tail physics and continuous local positivity remain unvalidated.
