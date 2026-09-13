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
