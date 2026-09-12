# Local flow-relative suprathermal objective: research and validation plan

Maximize the gain in electron suprathermal energy relative to the local bulk
velocity, using a fixed initial-energy threshold and each control's own initial
subtraction. A matched local Gaussian excess diagnoses shape changes beyond
heating and anisotropy; the objective alone does not establish nonthermal acceleration.

Prospective validation gates cover time sampling, velocity quadrature, Hermite
resolution, local validity, AD/FD agreement, CPU/GPU parity, and memory cost.
Require 1e-3 quadrature agreement for tail values and 5% agreement for benefit
and gradient refinement, with absolute tolerance 1e-10 near zero. Retain failed
checks before proceeding to optimization or physical interpretation.

## Physical definition and own-initial subtraction

For electrons, let angle brackets denote spatial volume averaging, and define
the physical moments of the represented distribution at every cell and time:

```text
n(x,t) = integral f(x,v,t) d^3v
Gamma(x,t) = integral v f(x,v,t) d^3v
U(x,t) = Gamma(x,t)/n(x,t)
c = v - U(x,t)
Erel(x,v,t) = m_e |c|^2/2
Kint(x,t) = integral Erel f d^3v
```

Use the shared initial reference `n0 = <n(x,0)>` and
`eps0 = <Kint(x,0)>/n0`, the initial mean internal energy per electron. For the
prescribed Maxwellian with fixed thermal widths, `eps0 = m_e sum(alpha_e^2)/4`
(isotropically `3 T_e,0/2`). For another initial distribution use its exact moments.
Confirm that these references are common across admissible phase controls before
freezing them; do not silently derive a different scale for each candidate.

Retain the predeclared **fixed initial threshold `E_star = 5 eps0` and width
`Delta_E = 0.5 eps0`**, with the same sigmoid as the existing lab-frame study:

```text
S(E) = sigmoid((E - E_star)/Delta_E)
W(E) = E S(E)
Klocal(t;theta) = <integral W(Erel) f_theta(x,v,t) d^3v>
Jlocal(t;theta) = Klocal(t;theta)/(n0 eps0)
Gain(theta) = integral_[40,60] w(t) [Jlocal(t;theta)-Jlocal(0;theta)] dt
integral_[40,60] w(t) dt = 1
```

Use the parent's existing normalized quartic window. Maximize `Gain` (or minimize
its negative); retain [60,70] as held-out evaluation. Subtract **each candidate's
own initial value**, not the baseline control's initial value. Differentiate both
terms, including their local velocities. Report initial, window-averaged, gain,
and held-out values separately. This dimensionless objective is energy normalized
by initial internal energy, not a probability or a current-energy fraction.
Threshold, width, and denominator stay outside differentiation and never track
evolved temperature. Threshold sensitivity is a separate labeled study.

The parent is implementing `Examples/2D_local_tail_control.py` with sampled,
normalized composite Simpson weights: split [40,60] into 20 equal intervals,
evaluate at 21 endpoints, multiply the Simpson coefficients by the same quartic
window, and normalize their sum to one. If `a_j` denotes those normalized weights,
the implemented gain is `sum_j a_j Jlocal(t_j;theta)-Jlocal(0;theta)`.
This requires 21 local-tail calls in the window rather than 2400 RK-stage calls,
plus the own-initial evaluation. Both the own-initial term and local mean remain
differentiated. Use a shared RK grid and nested SOLVAX `checkpointed_fori_loop`
loops over sampling intervals and RK steps. Checkpointing changes storage and
recomputation, not the definition of the sampled objective. Implementation details are parent-reported; validation remains prospective.

Before optimization, compare 20 versus 40 sampling intervals (21 versus 41 nodes)
on the same RK grid and quartic window, renormalizing each set of Simpson weights.
This tests time aliasing separately from timestep convergence; refine further if
the comparison fails. Do not infer convergence from Simpson's formal order when
the local-tail signal may vary faster than the sampling interval.

Obtain `n` and `Gamma` from exact low-order Hermite moments, then inverse Fourier
transform and divide in real space. “Exact” means algebraically exact for the
retained representation, not physically converged. Follow `_energization.py`:
with `A=prod(alpha)` and normalized first Hermite coefficient `C1_i`,
`n=A C000` and `Gamma_i=A(u_i C000+alpha_i C1_i/sqrt(2))` in each cell.
The basis shift `u` is not generally the physical `U`. Do not estimate `U` from
tail quadrature, use one global mean, or stop its gradient. Nonpositive/nonfinite
density invalidates the evaluation; do not hide it with a density floor.

For a variation of the state at fixed velocity nodes,
`delta U=(delta Gamma-U delta n)/n` and
`delta Erel=-m_e c dot delta U`. Thus the tail variation contains both
`integral W delta f` and `integral f W'(Erel) delta Erel`. Unlike the unweighted
internal-energy moment, the latter does not generally vanish. It is essential
to the AD/FD gate.

## What removing local flow establishes

At a fixed state, a resolved velocity translation of a cell's distribution and
its mean by the same amount leaves `c` and this observable unchanged. It therefore
removes coherent local drift from the energy kernel, including spatially varying
flows that a single global subtraction misses. This is an observable identity,
not a claim that arbitrary local frame changes leave Vlasov dynamics unchanged.

It still **does not prove nonthermal acceleration**. A hotter local Maxwellian
places more energy above this fixed initial threshold. A Gaussian with a changed
pressure tensor can change the radial tail through anisotropy alone, even at
fixed trace. Reversible compression, unresolved streams, spatial transport,
and changes in density weighting also affect interpretations. Bulk motion may
feed internal energy dynamically even though its instantaneous drift energy is
excluded. Internal-energy growth alone does not establish irreversible heating.

Local moment-matched comparisons have established precedent. *Kinetic
entropy-based measures of distribution function non-Maxwellianity: theory and
simulations* (2020) compares a local distribution with a Maxwellian sharing its
density, flow, and temperature, and analyzes limitations of scalar measures.
That supports a local reference, not equating a suprathermal integral with
non-Maxwellianity. [Primary paper](https://doi.org/10.1017/S0022377820001270).

Graham et al. (2021) measure departures from a bi-Maxwellian in MMS data and find
that hot/cold population mixtures can dominate the signal. Thus even a local
shape departure need not identify newly accelerated particles or their mechanism.
[Primary paper and author abstract](https://arxiv.org/abs/2102.09639).

## Diagnostic: excess over a matched local Gaussian

At each cell construct the full central covariance, including off-diagonal terms:

```text
Sigma_ij = integral c_i c_j f d^3v / n
Glocal(v) = n exp[-c^T Sigma^-1 c/2] / [(2 pi)^(3/2) sqrt(det Sigma)]
Xlocal(t) = <integral W(Erel) [f-Glocal] d^3v> / (n0 eps0)
DeltaX(theta) = integral_[40,60] w(t) [Xlocal(t;theta)-Xlocal(0;theta)] dt
```

This is a proposed **diagnostic**, not a replacement optimization target in this
plan. Compute second moments algebraically, including mixed first-order modes
for off-diagonal raw moments. The current energy helper supplies diagonal moments
and scalar energy; it is not already a full covariance API. Require finite,
positive-definite covariance; report failure rather than clipping eigenvalues.

The Gaussian matches local density, flow, and the complete pressure tensor
`P=m_e n Sigma`. Its excess is zero for any exact local Gaussian, including
heated, drifting, anisotropic, and rotated cases. A scalar-temperature Maxwellian
reference would leave anisotropy in the residual. Fit before spatial averaging:
a Gaussian fitted to the global distribution cannot remove mixtures of local
flows and temperatures. The same fixed radial energy kernel is used on `f` and
`Glocal`; covariance is not used to rescale the threshold.

This signed weighted residual is not a positive-definite distance, and a zero
value does not imply Gaussianity. Positive/negative lobes may cancel. Report both
constituent integrals and their error estimates, local maps, covariance eigenvalues,
and representative resolved energy spectra. A small excess obtained by subtracting
two large uncertain integrals is not evidence. Matching covariance deliberately
removes second-moment changes, including any such changes driven by acceleration;
the residual is a conservative shape test, not a unique thermal/nonthermal split.

An isotropic Maxwellian comparison alongside the full Gaussian can help separate
anisotropy from higher-order shape. Density-normalized relative entropy against
the matched Gaussian is an optional complementary shape diagnostic for positive
distributions; signed Hermite reconstructions do not support `f log f` without a
validity check. Do not clip `f` to manufacture an entropy or an objective.

Mechanism claims additionally need resolved temporal/spectral evolution and energy
transfer evidence, not just `DeltaX>0`. Klein, Howes, and TenBarge (2017) demonstrate
velocity-resolved field-particle correlations in gyrokinetic turbulence as a way
to identify resonant energy-transfer signatures. Applying that approach here
would require the appropriate equations, frame terms, and time averaging; the
local-tail time derivative is not simply `J dot E`.
[Primary paper](https://arxiv.org/abs/1705.06385).

## Implementation tradeoff: direct cell quadrature is costly

The existing lab-frame factory precontracts one position-independent velocity
kernel and needs only the zero Fourier mode. Here `U(x,t)` makes the kernel
state-dependent and local; spatial averaging before integration is invalid.
Saved spatial-average coefficients alone cannot recover this observable. Evaluate
it during a streamed trajectory or from selected full spatial Hermite snapshots.

Use direct cell quadrature as the correctness reference: transform the selected
species to real space, reconstruct a cell using separable Hermite contractions,
evaluate the local kernel at tensor Gauss-Hermite nodes, integrate, and reduce.
Preserve the existing `norm="forward"` RFFT convention, explicit odd/even `Nx`,
Hermite axis order, velocity Jacobian `prod(alpha)`, and single Gaussian weight.
The low-order moment calculation stays independent of this nonpolynomial integral.

Direct evaluation visits `Ncell*q^3` velocity points per sample, plus reconstruction
and FFT work; doubling q multiplies the node work by eight. For illustration,
24^2 cells at q=48 visit 63,700,992 points per sample. A full float64 scalar array
at that shape already occupies about 486 MiB, before distributions, kernels,
intermediates, gradients, or time checkpoints. It must not become the default
space-times-velocity allocation.

Map cells or use a bounded cell chunk: storage includes `O(Ncell*H^3)` real-space
coefficients and `O(chunk*q^3)` velocity workspace, not an entire phase-space
trajectory. A serial map can underutilize a GPU; larger chunks trade memory for
throughput. Forward bounded memory does not guarantee bounded reverse-AD memory:
profile residual storage and use rematerialization/checkpointing as needed.

The Gaussian diagnostic can use a separate quadrature in covariance-scaled
coordinates with Cholesky factor `Sigma=L L^T` and `v=U+sqrt(2)L xi` under GH
weight. Cross-check it against common physical-node integration; do not accidentally
apply the solver-basis Gaussian weight to a different Gaussian twice. Interpolated
kernel tables or other approximations may later reduce cost, but must reproduce
direct-reference values and derivatives over the actual range of `U` before use.

## Gates before any local-tail optimization

These are prospective study acceptance criteria, not completed checks or universal
physical standards. Fix quadrature tail-value relative tolerance at 1e-3 and
benefit/directional-gradient refinement tolerance at 5%, with absolute tolerance
1e-10 near zero; retain the 1e-8 negativity criteria. For a comparison to the finer
reference b, use `abs(a-b) <= 1e-10 + rtol*abs(b)` and report both errors. For vector
gradients also report the norm comparison and fixed directional probes. The
absolute tolerance is in the dimensionless normalized objective/control units;
record control scaling. Declare memory and wall-time budgets before measurements.
Retain failed checks and their values; do not tune thresholds to a control benefit.

| Gate | Required evidence before optimization |
| --- | --- |
| Algebra and invariance | Small CPU fixtures recover exact density, momentum, and central second moments; verify uniform and spatially varying Gaussian drifts, isotropic heating, anisotropy with fixed trace, rotated covariance, and own-initial subtraction. Local Gaussian excess must vanish within quadrature error. Check flat/structured states and odd/even Fourier grids. |
| Velocity quadrature | Compare q24 versus q48, then q96 on representative states, increasing q if needed. Apply the 1e-3 tail-value and 5% benefit/gradient criteria with absolute 1e-10. Use an independent radial integral for an isotropic Gaussian. For excess and small gains, bound absolute error in both constituent integrals below the claimed signal; relative convergence of their difference is insufficient. |
| Sampling/time aliasing | Compare 20 versus 40 Simpson intervals with the same quartic window and shared RK grid. Require benefit and gradient agreement within 5% with absolute 1e-10, and report constituent window/initial values. Retain failures and refine sampling before any search; separately test the RK timestep. |
| Hermite and dynamics | Compare coarse H8 versus fine H16 before any physical claim; require benefit and gradient agreement within 5% with absolute 1e-10. Refine H independently of q and space/timestep to isolate truncation error; passing one H pair is not continuous convergence certification. Keep physics fixed: use nu=0 for collisionless comparisons, or match shared-mode damping rates when changing H. A larger q cannot repair truncated Hermite dynamics. |
| Local validity | Require mean local negative mass divided by species number and worst-cell negative-mass/density each below 1e-8 on declared initial/window/held-out snapshots, with q refinement. Require positive finite density and covariance. Global cancellation cannot pass this gate; finite-node checks do not certify continuous positivity. Never clip the signed objective. |
| AD versus FD | Check instantaneous JVP/reverse AD and complete sampled-trajectory directional derivatives through both nested checkpoint loops, including dU and the own-initial term. Sweep centered-FD steps to exhibit an agreement plateau with identical states, precision, RK grid, Simpson nodes/weights, and solver tolerances. Proposed relative agreement target: 1e-3 with absolute 1e-10. Include a controlled shift test that would fail if U were frozen. |
| CPU/GPU parity | After CPU algebra gates, the parent must compare the same small forward and value/gradient cases on CPU and GPU at matched precision. Proposed 1e-3 relative value/directional-derivative agreement (absolute checks near zero), tighter where established accuracy allows. Record source revision, backend, precision, device, compilation, and contention. |
| Memory and cost | Before a search, set a device-memory budget, measure compiled peak forward and reverse memory across cells/q/H and checkpoint counts, and verify chunking prevents full space-times-velocity or trajectory retention. Measure compilation separately and a representative full-horizon value/gradient wall time; estimate total search cost. An OOM or impractical budget is a failed gate, not permission to start optimization. |

Then evaluate fixed baseline controls through the training and held-out windows;
the existing lab-frame optimum is a diagnostic comparison, not a validated local
optimum. Only after all gates pass should the parent consider a bounded local-tail
search with energy partition, local Gaussian excess, and local validity recorded.
Any cheaper surrogate needs its own local-objective gradient/refinement validation;
agreement established for the previous lab-frame objective does not transfer.
Reapply the gates to candidate controls before interpreting a benefit.

## Prior art and claim boundary

Differentiable kinetic optimization is already published: *Unsupervised discovery
of nonlinear plasma physics using differentiable kinetic simulations* (2022)
uses a loss involving departure from a local Maxwell-Boltzmann distribution.
[Primary paper](https://doi.org/10.1017/S0022377822000939). This plan combines
established moment subtraction, smooth tail integration, matched references, and
numerical validation. No novelty or priority claim follows from this literature
check. A successful optimization would first establish increased **local
flow-relative suprathermal energy gain** under the specified numerical model;
nonthermal acceleration and irreversible heating require additional evidence.

## Dependency status

The parent confirms SOLVAX PR103 merged at
`917d1a679d8258bb7c9929e2320f4befbbe92f93`. SPECTRAX PR42 and PR43 remain
open for collaborators only.

## Execution limits for this iteration

Use only GPU0 on office (RTX A4000, 16 GiB). A search is limited to 20 L-BFGS
iterations and a planned ten-minute warm-optimization budget after measuring
representative gradient cost; do not launch multiple searches concurrently.
Target less than 12 GiB of device allocations, leaving space for runtime/context.
Compiler memory estimates and nvidia-smi snapshots are supporting evidence, not
measured allocator peaks. Small CPU/GPU smoke runs precede the full-horizon pilots.
Initial q24/q48/time-sampling pilots evaluate the same seed-7 baseline. The old
lab-tail optimum is separately evaluated and must not be described as a local-tail
optimum. All existing study archives retain their original SOLVAX version.
