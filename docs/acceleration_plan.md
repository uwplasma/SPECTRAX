# Acceleration follow-up: execution and acceptance plan

The user approved the scientific follow-up and parallel subagents on 2026-09-12. This branch starts at SPECTRAX 23102ab (PR42). SOLVAX PR103 and SPECTRAX PR42 are ready with passing checks, but GitHub requires review approval before merging; ordinary merge is blocked and auto-merge is disabled. No administrator bypass or branch-policy changes were made. Work proceeds on the reviewed source while that external requirement remains outstanding.

## Independent implementation lanes

1. Energy mechanism: species density/momentum, bulk/internal/total energy, correctly normalized field work, and the exact implemented collision contribution. Analytic and source-RHS checks.
2. Velocity observables: quadrature of spatially averaged Hermite distributions, signed tail-energy objective, negative mass, analytic Maxwellian and derivative checks. No clipping of negative distributions in the objective.
3. AD/FD comparison: identical L-BFGS objective/start/stopping rules at the actual time horizon; counted FD tuning, compilation and evaluations; objective versus total time and source/device provenance.
4. Parent integration: streaming sampled trajectories at bounded storage, CPU/GPU checks, serial GPU studies, scientific assessment and figures.

Agents edit disjoint files in the dedicated worktree. The parent alone schedules GPU jobs. GPU0 on office is used; other workloads and global Python are untouched. Local CPU runs are restricted to unit/small correctness checks.

## Predeclared diagnostic study

Re-evaluate saved seed-7 initial and optimized energy controls through T=70 (training average [40,60], held-out interval [60,70]). First use 16²/4³/dt=0.1, then 24²/6³/dt=0.05; add velocity refinement before tail claims. Record electron/ion bulk and internal energy, signed integrated field work, collision/diffusion contributions and energy-balance residuals. Save only sampled scalars and spatial-average Hermite coefficients, not the full phase-space trajectory.

For the acceleration objective use a fixed lab-frame energy threshold of five times the initial electron mean thermal energy, with sigmoid transition width half that energy. Report lower/higher threshold sensitivity separately; do not choose a threshold by maximizing the resulting benefit. This lab-frame tail includes bulk acceleration; bulk/internal and mean-flow diagnostics must be reported alongside it. Negative mass of the spatially averaged distribution is necessary but not sufficient for local distribution positivity.

Pilot acceptance gates, chosen before the new results: quadrature relative change below 1e-3 in tail energy for successive sufficiently resolved quadratures; velocity refinement changes the optimized-control tail benefit and directional derivative by less than 5% (with absolute differences reported near zero); negative averaged distribution mass below 1e-8 of species number, with a separate local positivity diagnostic required before an acceleration claim. These are study tolerances, not universal physical criteria. If a gate fails, retain/report the failure and improve resolution or the objective before interpreting an optimization as particle acceleration.

The initial mechanism runs diagnose the existing energy optimum. A tail-optimized result is a distinct experiment and must not inherit its success claim. Match AD and FD complete optimization at T=60 with compilation counted; T=2 gradient timings do not substitute for it. A small CPU/GPU benchmark smoke precedes the GPU physics run. Report any GPU contention rather than labeling affected timings uncontended.

## Collision-normalization audit

The built-in rate is `nu * i(i-1)(i-2)/[(H-1)(H-2)(H-3)]` summed over velocity axes. Therefore fixed nu across H changes the damping of shared modes. Earlier H4→H6 results demonstrate benefit across those settings, not pure fixed-collision-operator refinement. A matched-rate comparison to H4/nu1 uses H6/nu10 and H8/nu35. The largest corner rate at H8/nu35 is105; use dt=0.02 (3500 steps to T70) so its scalar damping stability parameter is2.1, then assess time refinement if results depend on it. This supplements rather than relabels the original fixed-nu measurements. The built-in operator directly preserves the first three moment orders; it can still change transfer indirectly through damped higher moments.

## Response to matched-rate validity failure

The matched H6/nu10 and H8/nu35 pilot converges to nonzero local negative mass at T70 (~2.95e-6 spatial mean for optimized controls). Increasing the Hermite cutoff does not remove this defect at fixed damping. Before optimizing a tail, test the distinct collisionless Vlasov case nu=0 at 24² with H8 and H12, dt=0.05, through T70; retain the same controls and threshold. This removes the non-positivity-preserving high-order filter rather than altering the threshold or tolerance. It is a new physics setting and requires its own tail optimization and resolution/validity checks if it passes. No tail claim is transferred from the damped case.

The collisionless H12 endpoint has mean local negative mass1.8e-9 but maximum cell fraction3.6e-8. Add H16 at the same grid/time step to check the stricter local-cell1e-8 gate before considering a tail optimization; do not relax the gate after observing this result. Physical continuous positivity is not certified by finite quadrature.

## Control-search cost decision

The collisionless baseline tail gradient changes only8.59e-5 relatively between H12 andH16 on24²/dt0.05, but one H16 value-and-gradient costs142s. Before a many-iteration search, test a16²/H8/dt0.1 proposal gradient against that saved fine gradient. If it meets the predeclared5% gradient gate, optimize that cheaper discrete surrogate, then evaluate the resulting controls, gradients, local negativity and held-out time on24²/H16/dt0.05. Coarse distribution positivity is not claimed and no coarse tail optimum is a physical result by itself; the final result must pass the fine-grid gates. This is a control-search acceleration choice, not relaxed physical validation.
