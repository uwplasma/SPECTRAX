# Reconnection follow-up: baseline before optimization

**Future work only — 2026-09-13.** This handoff proposes an experiment; no
reconnection implementation, simulation, or validation is reported here. Reviewed
`local_tail_results.md`, `acceleration_handoff.md`, the Orszag–Tang/phase-control
examples, and the initialization, current and Maxwell operators. No `AGENTS.md`
was found in this checkout or its ancestor directories.

## Recommended first baseline

Use collisionless **2D3V, doubly periodic double-Harris reconnection with uniform
guide field $B_g/B_0=1$**, equal electron/ion temperatures and mass ratio 25.
This is our choice for tractable equilibrium and mechanism validation, not a
prediction of the strongest tail. Only after it passes, compare $B_g/B_0=0.2$
at otherwise matched parameters. Dahlin, Drake and Swisdak's [2014 primary
paper, §§II–III](https://arxiv.org/pdf/1406.0831) supplies a double-Harris precedent
and parallel-electric, curvature/Fermi and betatron diagnostics; its PIC solver
is relativistic. Its parameters include background density $0.2n_0$,
$T_e=T_i=0.25m_i v_A^2$, half-width $0.25d_i$, and $c/v_A=15$.

Proposed economical pilot: $L_x\times L_y=32\times16d_i$, half-width
$w=0.5d_i$, $n_b/n_0=0.2$, the temperatures above, and **$c/v_A=100$**.
Here $v_A=B_0/\sqrt{\mu_0n_0m_i}$ and $d_i$ use sheet normalization $n_0$,
not upstream background density. These domain, width and light-speed choices
are deliberate adaptations, not a reproduction. Enlarge the box independently;
stop analysis before inter-sheet interaction or box-limited island growth.
Do not inherit the current 16²/H8 or 24²/H16 resolutions or time windows.

## Initialization contract

For orientation, let $y_1=L_y/4, y_2=3L_y/4, h_j=\mathrm{sech}^2[(y-y_j)/w]$.
The familiar separated-sheet target is

$$
B_x=B_0\{\tanh[(y-y_1)/w]-\tanh[(y-y_2)/w]-1\},\quad
B_y=0,\quad B_z=B_g,\quad n_e=n_i=n_b+n_0(h_1+h_2).
$$

For each *isolated* sheet, use a stationary background Maxwellian plus a
sheet Maxwellian with constant species drift along $z$. Temperatures are in
energy units. Enforce

$$
n_0(T_e+T_i)=B_0^2/(2\mu_0),\qquad
U_{zs,1}=-2T_s/(q_sB_0w),\qquad U_{zs,2}=-U_{zs,1}.
$$

These signs give $J_z=-\partial_yB_x/\mu_0$; both species carry current.
The [Harris original (1962)](https://link.springer.com/article/10.1007/BF02733547)
establishes the single-sheet Vlasov equilibrium. A constant guide field does not
remove its pressure-balance requirement. Check the full normal stress
$P_{yy}+B^2/(2\mu_0)$, including population moments, rather than assuming
uniform density is adequate.

**Periodicity is a separate construction problem.** A single tanh sheet is
incompatible with periodic Fourier boundaries. The finite double-tanh formula
matches endpoint values but generally not all derivatives; overlapping sheets
also spoil exact isolated-sheet pressure and Vlasov balance. Use a smooth
periodic construction and quantify the error against the separated-sheet target.
Neither periodization, setting density from magnetic pressure, nor matching
Ampère's law proves kinetic equilibrium. Require a positive distribution with
vanishing stationary Vlasov residual, or explicitly label the state approximate
and bound its relaxation below the seeded signal. Check seam derivatives,
Fourier convergence and zero mean current; reject a boundary current artifact.

Do not substitute a force-free magnetic profile with locally shifted Maxwellians
and call it equilibrium. The [Harrison–Neukirch exact force-free solution
(2009)](https://arxiv.org/pdf/0812.1240), especially Eq. (14), requires a specific
energy/canonical-momentum distribution and positivity constraints. Its single
sheet also needs a periodic-domain treatment. That is a separate follow-up,
with a more involved distribution projection.

Repository implications for later implementation:

- `compute_C_nmp` in `spectrax/_initialize_maxwellian.py` constructs unit-density
  Maxwellians with fixed species thermal scales. Build density-weighted sheet
  and background contributions in physical space before the Fourier transform,
  summing populations within each species. Do not replace their mixture with
  one Maxwellian at the mean drift. Verify reconstructed density, flow and full
  pressure tensor, including drift variance.
- Supply both `Ck_0` and `Fk_0` explicitly; `_initialization.py` otherwise creates
  a two-stream state before overrides. Preserve `(species,p,m,n,y,kx,z)` ordering
  and the velocity Jacobian. The existing phase example's electron-only current
  at uniform density is not this equilibrium.
- Audit normalized units against `_simulation.py`: $c=1$,
  $\partial_tE=\nabla\times B-J/\Omega_{ce,0}$. Thus the stationary code check
  is $J=\Omega_{ce,0}\nabla\times B$, with matching Gauss constraints.
  With $n_0=1,B_0=1,m_i/m_e=25,c/v_A=100$, the proposed mapping is
  `Omega_cs=[0.05,0.002]`, $d_i=5$, `Lx=160`, `Ly=80`, $w=2.5$,
  and isotropic `alpha_e=sqrt(2)*0.025`, `alpha_i=sqrt(2)*0.005`.
  Validate this mapping through moments and field/particle energy units.

Thermal speeds, sheet drifts, exhaust speeds **and the measured tail** must
remain nonrelativistic. The proposal has initial electron one-component thermal
speed $0.025c$; that alone does not bound evolved velocities. Preregister an
analyzed range below $0.2c$, and require absolute reconstructed mass and energy
outside it to be negligible relative to the claimed signal. Gaussian support is
unbounded: use integral bounds, not the largest quadrature node. If the tail
reaches this limit, rescale and reconverge or change physical model; clipping
cannot validate a nonrelativistic acceleration claim.

## Ordered validation and go/no-go

The following are proposed acceptance gates, not achieved results. Freeze them
and their dimensional normalizations before running.

1. **Unseeded initialization:** verify charge neutrality, divergence-free B,
   normalized Ampère balance, pressure balance and stationary kinetic residual
   on independently refined grids. Target normalized constraint residuals below
   $10^{-8}$; require kinetic residual convergence and an unseeded relaxation
   check over the intended analysis horizon. Its spurious energy/tail change
   must be below 1% of the seeded signal. Preserve failures; no optimization
   while initialization transients dominate.
2. **Seeded forward benchmark:** prescribe a smooth periodic $\delta A_z$
   with fixed mode, phase and peak $\delta B/B_0=10^{-3}$; derive B by its curl.
   Declare the intentional perturbation imbalance and energy. Compare half/double
   seed amplitude and an unseeded control. Track X/O points, reconnected flux
   $\Delta A_z$, its rate normalized by $B_0v_A$, outflows and both sheets.
   Compare with an independent kinetic calculation at the *adapted* parameters;
   published unmatched reconnection rates are context, not a pass criterion.
3. **Numerics and budgets:** independently refine space, all three Hermite axes,
   timestep, velocity quadrature, diagnostic sampling and box size. Resolve
   electron inertial/Debye/gyro scales and explicit electromagnetic stability;
   inspect spectra near truncation and recurrence. Start with `nu=D=0`; if
   regularization becomes necessary, quantify its effect and budget separately.
   Require positive density/covariance, worst-cell negative mass fraction below
   $10^{-8}$, energy drift below $10^{-5}$ of initial energy, and work-budget
   error below 1% of electron energy transfer. Errors must also be below 10% of
   any claimed tail excess; total-energy accuracy alone is insufficient.
4. **Physics go/no-go:** require converged topology change and electron
   energization. Close species-integrated $J_s\cdot E$ against kinetic energy
   and field loss; separate bulk/internal energy and spatial transport. Diagnose
   parallel work, curvature/Fermi and betatron contributions only where electrons
   are magnetized, retaining the discrepancy from exact work. Sample local
   distributions and covariance-matched Gaussian excess throughout the window,
   including their initial mixtures. Require positive excess and excess growth
   above combined numerical error before a nonthermal-tail optimization study.
   Heating alone is a valid baseline outcome, not that go decision.
5. **Only then control:** freeze threshold, normalization and training/held-out
   windows; subtract each control's own initial observable. Construct controls
   preserving particle numbers, initial energy and equilibrium quality. Old
   phase-only energy invariance does not automatically survive variable density
   and multiple populations. Require <5% changes in resolved values and full
   gradients, an absolute tolerance near zero gradients, directional FD sweeps,
   multiple starts and frozen-control held-out validation before interpretation.

## Limits that remain open

The current Orszag–Tang study's 10.28% training and 9.40% held-out local-tail
gain improvements do not establish reconnection or nonthermal acceleration.
At T=50 its candidate has a more negative local Gaussian excess. Whole-window
shape diagnostics, independent held-out time-sampling refinement, separate
discretization convergence and fine stationarity remain unclosed; q48's coarse
near-stationary gradient failure remains evidence despite later passing checks.

A successful future 2D pilot would still not establish physical-mass-ratio,
large-system or 3D acceleration, irreversible heating, or a power law.
[Dahlin, Drake and Swisdak (2015)](https://arxiv.org/abs/1503.02218) find enhanced
energetic-electron production in 3D associated with access to acceleration
regions unavailable to island-trapped electrons in 2D. All four primary sources
linked above were checked on the web for this handoff (Harris via the publisher's
summary; the equilibrium and 2014 papers via full text; 2015 via the abstract).
