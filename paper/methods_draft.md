# Draft text for the SPECTRAX paper: differentiability section (2026-09-13)

Numbers marked [GPU] are to be replaced by the RTX A4000 campaign in PR #44; the rest are measured
(CPU, float64/complex128) and reproducible from `Examples/2D_Orszag_Tang_optimization.py benchmark`.

## Methods paragraph

SPECTRAX is differentiable end to end. The right-hand side of the Hermite–Fourier system is composed
of real fast Fourier transforms, Hermite shift operators, a 2/3 de-aliasing mask and a diagonal
collision matrix, each of which has an exact built-in derivative in JAX, so no hand-written
vector–Jacobian product is required. Time integration is delegated to Diffrax, and the reverse pass
differentiates the executed discrete algorithm, adaptive step-size controller included, rather than a
continuous adjoint equation: with an explicit spectral integrator this is the direct analogue of the
"differentiate the solver, not the equation" principle. Memory is bounded by binomial (Revolve-type)
checkpointing of the time loop: for K checkpoints the retained state is O(K S), independent of the
number of steps N, at the price of O(N log N) recomputation. In the compiled programme the reverse-pass
workspace is 2.4–3.0× (fixed-step Tsit5) or ≈4× (adaptive Dopri8) the forward-solve workspace at every
resolution we tested, from 8²×3³ to 64²×6³, and is flat in N from 50 to 1600 steps, whereas storing the
full trajectory grows linearly (6 → 93 MB over N = 50 → 800 for a 38 KB state). One gradient of a scalar
objective costs 4–7 forward solves regardless of the number of controls P, against 2P solves for centred
finite differences; at P = 8 the finite-difference gradient is already 3.2× more expensive, and the
gradients agree to 1e-8 (adaptive stepping) or 1e-11 (fixed stepping). Forward mode
(`adjoint=ForwardMode()`) is available for few-parameter sensitivities. Two alternatives were measured and
rejected: the continuous adjoint is inexact by construction and cannot be assembled for the complex
spectral state in Diffrax 0.7.2, and algebraically reversible time stepping (McCallum & Foster 2024),
which would give O(1) memory in N, is exact for collisionless runs (gradient error 1e-17 … 1e-13 up to
1600 steps) but amplifies round-off by e^{νT} through the damped Hermite modes and fails at ν = 1,
T = 32 (NaN) and ν = 3, T = 8 (reconstruction error 1e106), which rules it out for collisional
turbulence.

## Resolution check of the optimised designs (Phase 2a, measured)

Optimised controls obtained at 32²×4³ were frozen and re-simulated at 32²×6³, 64²×4³, 64²×6³, 64²×8³
and 128²×6³. The conversion objective improved by 21.2 % at every resolution (0.938 → 0.739 at the
training resolution, 0.945 → 0.745 at 128²×6³), and the current-sheet objective by 57.1–59.5 %,
retaining 97 % of the gain found at the training resolution. The relative total-energy error at
t ω_pe = 200 is set by the Dopri8 tolerance and grows with the number of Fourier modes at a fixed
tolerance of 1e-7 (1e-9 at 32², 4e-6 at 64², 2e-4 at 128²); at a tolerance of 1e-9 it is 1e-10 at 64²,
while the objective values change by less than 0.05 %. Production optimisations therefore use a
tolerance of 1e-9.

## Phase-only control isolates the nonlinear effect (Phase 2c, measured, 32²×4³)

With the mode amplitudes fixed and only the eight phases optimised, every Fourier mode keeps its initial
magnetic energy, current and drift energy, and linear evolution on the uniform background cannot change
the energy objective. Phase-only control still raises the converted fraction of in-plane magnetic energy
at t ω_pe = 200 from 6.2 % to 23.9 %, 89 % of the improvement found when amplitudes are also free,
with the extra energy going mainly to the ions. For the current-sheet objective phase-only control
partly works by building a larger initial current peak, so the dynamical intensification (1.25×, against
1.04× at baseline) is smaller than with amplitude and phase control (1.65× with an unchanged initial peak).

## Production optimisation on one GPU (Phase 3a, measured)

At 64²×6³ Fourier–Hermite resolution (two species, 14.1 MiB state) with adaptive Dopri8 at tolerance 1e-9,
30 L-BFGS-B iterations over 16 controls took 34 gradient evaluations of 77 s each on one NVIDIA RTX A4000,
47.5 min in total including 95 s of compilation. The optimised initial field converts 26.0 % of the in-plane
magnetic energy by t ω_pe = 200, against 6.0 % for the reference field, and the extra energy goes mainly to
the ions (kinetic-energy gain 9.2e-4 against 4.2e-5). The optimum coincides with the one found at
32²×4³ on a laptop CPU (objective 0.740 against 0.739), and it is unchanged at 64²×8³ (−21.3 %) and
128²×6³ (−21.2 %), with relative energy errors of at most 1.4e-8. A centred finite-difference gradient
at this resolution would need 32 forward solves per evaluation, about 8× the cost of the adjoint gradient;
a full optimisation of this kind is then out of reach for a non-differentiable code of the same speed.
The magnetic energy is still decreasing at t ω_pe = 200; the objective is defined at this fixed horizon.

## Current-sheet objective on the GPU (Phase 3a, measured)

At 64²×6³ and tolerance 1e-9, 30 iterations (40 gradient evaluations, 51 min) raise the smooth peak of J_z
at t ω_pe = 200 from 0.0386 to 0.0640, an increase of 66 % that is unchanged at 64²×8³ and 128²×6³. Part of
the gain comes from rearranging the initial field: the initial peak rises from 0.0373 to 0.0416. The
dynamical intensification, the ratio of the peak at t ω_pe = 200 to its initial value, is 1.54 against 1.04
for the reference field. The in-plane magnetic energy grows by 21 % while the ions lose 1.5e-3 of kinetic
energy, consistent with flow-driven field amplification and current-sheet thinning.

## Gradient cost and memory on one GPU (Phase 3c, measured; RTX A4000, float64)

Cost at 32²×4³, T = 100, adaptive Dopri8, 8 checkpoints: one forward solve takes 0.43 s and one reverse-mode
gradient 2.8–3.0 s (6.5 forward solves) for P = 4 to 64 controls, while a centred finite-difference
gradient takes 3.6 s to 59 s, 1.3 to 19.6 times the adjoint cost. Finite differences agree with the
adjoint gradient to 2e-8 … 2e-7 at their best step.

Peak device memory, one process per configuration (T = 20):

| resolution | state | forward | gradient, K = 8 | ratio | gradient, K = 32 | ratio |
|---|---|---|---|---|---|---|
| 32²×4³ | 1.1 MiB | 64 MiB | 176 MiB | 2.75 | 258 MiB | 4.0 |
| 64²×6³ | 14.1 MiB | 311 MiB | 862 MiB | 2.77 | 1907 MiB | 6.1 |
| 64²×8³ | 33.2 MiB | 731 MiB | 2112 MiB | 2.89 | 4514 MiB | 6.2 |
| 128²×6³ | 55.6 MiB | 1226 MiB | 3408 MiB | 2.78 | 7523 MiB | 6.1 |

Compiled reverse-pass workspace against the number of fixed steps N (32²×4³, Tsit5): 54 MiB with 8
checkpoints and 149 MiB with 32 checkpoints for every N from 50 to 800, against 230 MiB to 3.57 GiB for a
full trajectory tape. The gradient-to-forward memory ratio is therefore a constant set by the checkpoint
budget, independent of both resolution and integration length; the price is recomputation, which raises
the gradient cost from 4.7 (K = 32) to 6.2 (K = 8) forward solves at 128²×6³.

## Time-window objectives (Phase 5b, measured)

Objectives may combine several saved states, for example a time average of the field energy, without any
change to the solver: the saved snapshots are stored once, outside the checkpointed loop state. At 32²×4³
(state S = 1.1 MiB) the compiled reverse-pass workspace grows by 10, 44 and 308 MiB for 11, 41 and 161
saved snapshots, about 0.8–1.7 K×S, and by the same amounts with 8 and 32 checkpoints. The memory of a
time-averaged objective therefore stays independent of the number of time steps, with an additive K×S for
the K saved states that define it.

## Sensitivity of the optimised designs to physical parameters (Phase 5c, measured)

Forward-mode derivatives, four tangents propagated through one adaptive Dopri8 solve at 64²×6³ and tolerance
1e-9, give the logarithmic sensitivities d ln|J| / d ln p of every objective with respect to the collision
frequency ν, the mass ratio m_i/m_e, the guide field B_z and the in-plane field amplitude δB. They agree with
centred differences to at most 4.4e-6 in the same units. At the conversion optimum, a 1 % increase of δB lowers
the magnetic-energy fraction remaining at t ω_pe = 200 by 0.26 %, a 1 % increase of m_i/m_e or B_z raises it
by 0.07 %, and a 1 % increase of ν changes it by only 0.001 %. At the current-sheet optimum the peak current
scales as δB^1.36, a super-linear response weaker at the reference field (δB^1.10). It decreases with mass
ratio (−0.22) and increases with guide field (+0.26), while its collisional sensitivity is 0.004. Both
optimised designs are therefore insensitive to collisionality but moderately sensitive to the mass ratio,
which bounds how far a design found at m_i/m_e = 25 transfers to larger mass ratios without re-optimisation.

## Figure: inverse design of the Orszag–Tang vortex

Caption. Gradient-based inverse design of the 2D Orszag–Tang vortex. The controls are the amplitudes and
phases of the M = 8 lowest Fourier modes of the in-plane magnetic stream function, constrained to a fixed
in-plane magnetic energy, with electrons carrying the consistent out-of-plane current. (a) Initial
out-of-plane current density J_z for the initial controls; (b) J_z at t ω_pe = 200 for the initial
controls; (c) J_z at the same time for the optimised controls that maximise the current-sheet intensity
(smooth p-norm of J_z, p = 8). (d) In-plane magnetic energy normalised to its initial value and (e)
electron and ion kinetic-energy change for initial and optimised controls. (f) Objective against L-BFGS-B
iteration; each iteration costs one forward solve plus one reverse pass. Resolution 32²×4³ Hermite modes,
two species, adaptive Dopri8 at tolerance 1e-7, ν = 1 [local run; replace by 64²×6³ GPU run].

## Figure: cost and memory of gradients

Caption. Cost and memory of reverse-mode gradients through SPECTRAX. (a) Wall time of one gradient of
the magnetic-energy-conversion objective against the number of controls P for reverse-mode AD (one
forward solve plus one checkpointed reverse pass) and for centred finite differences (2P forward
solves); the dashed line is one forward solve. (b) Compiled reverse-pass workspace against the number of
fixed time steps N for a full trajectory tape and for binomial checkpointing with 8 and 32 checkpoints;
the forward-solve workspace is dashed. (c) Forward and reverse-pass workspace against the state size for
resolutions from 16²×4³ to 64²×6³; the annotated ratio is the memory overhead of the gradient. (d)
Relative error of the centred finite-difference gradient with respect to the AD gradient as a function
of the finite-difference step, showing the truncation/cancellation trade-off that AD avoids. [GPU:
add peak device memory for 64²×6³ and 128²×6³ at T = 500.]

## Sentences for the positioning paragraph

Differentiable kinetic plasma optimisation is prior art (Joglekar & Thomas, JPP 2022), as are
automated adjoints for sparse spectral PDE solvers (Skene & Burns 2025); the contribution here is an
end-to-end differentiable Hermite–Fourier Vlasov–Maxwell solver whose gradient memory is a measured
constant multiple of the forward solve, exposed through the unchanged `simulation` interface, and a
closed inverse-design result on the Orszag–Tang vortex.
