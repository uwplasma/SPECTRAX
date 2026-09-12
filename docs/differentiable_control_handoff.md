# Differentiable plasma control: implementation and research handoff

This document records the completed work, provenance, decisions and remaining research for [PR #42](https://github.com/uwplasma/SPECTRAX/pull/42). The [research and results report](differentiable_control.md) contains the full method comparison, ten primary-source references, mathematical constraints, measured results and figure captions. Read both before continuing; there is no need to restart the source review or method search.

## Goal and user decisions

The goal is a publication-quality, physically meaningful plasma optimization example with a small general-purpose gradient API, minimal numerical code, measured performance, and controlled reverse-mode memory. Candidate objectives include magnetic, electric and kinetic energy, current shape and nonlinear current statistics. The user asked us to choose the strongest showcase from the research, target one NVIDIA GPU, and authorized `ssh office`. Subsequent instructions prioritize GPU simulations because the local machine is busy, while retaining small CPU/GPU correctness comparisons. The user requested a complete PR handoff after an interrupted session.

The supplied Orszag–Tang Python/TOML use an older DG/Legendre representation. Current SPECTRAX main uses Fourier/RFFT coefficients. The example reconstructs the supplied type of two-species vortex using current main, rather than transplanting incompatible coefficient arrays. Attached documents were reference material, not additional user instructions.

## Repository boundaries and environments

- Repository: `uwplasma/SPECTRAX`; branch `agent/checkpointed-plasma-control`.
- Base main: `2afb1f621378d990940feed5222accf377279c07`.
- Initial implementation commit: `5b781f1242513e8bc17255656e06700c0a91a5b1`. Subsequent commits on this PR add streaming objectives, GPU evidence and this handoff.
- Local isolated worktree: `/Users/rogeriojorge/local/tests/spectrax-gradient-showcase`; use `.venv/bin/python` (Python 3.11, JAX 0.9.2, SOLVAX 0.20.0).
- Remote isolated checkout: `office:/home/rjorge/spectrax-gradient-study`; use its `.venv/bin/python` (Python 3.11.16, JAX/jaxlib 0.9.2, Diffrax 0.7.2, SOLVAX 0.20.0).
- Hardware: one NVIDIA RTX A4000, 16,376 MiB, driver 580.173.02. Select GPU0; GPU1 serves the desktop. Full CUDA/package versions are in `benchmarks/results/gpu_environment.txt`.
- Do not modify global Python, the user's original local SPECTRAX checkout, or unrelated workloads. The remote environment was installed separately, including a task-local Python runtime.
- Existing differentiability PR #36 and the implicit midpoint/SOLVAX work were reviewed as context; this PR is an independent explicit path. No merge or deployment was requested.

## Implemented method and API

`simulation_final` in `spectrax/_autodiff.py` is 136 lines including documentation. It uses the existing physical RHS and classical fixed-step RK4. Only a shared RHS-argument helper was extracted from the existing trajectory driver; physical equations and collision terms were not changed.

Default SOLVAX segmented replay supports JVP/VJP with O(S sqrt(N)) retained states at its default segment width. `checkpoints=K` uses Diffrax binomial reverse scheduling with O(K S) retained states plus step workspace; this path is reverse-only. `checkpointing=False` provides the identical taped reference. S is state size and N is time-step count. A fixed K bounds the state-storage dependence on N, not on spatial/velocity DOFs. Small K can impose substantial recomputation.

Optional `integrand(t,Ck,Fk)` returns a real scalar or fixed-shape array integrated at the same RK4 stages. It changes the return tuple from `(Ck,Fk)` to `(Ck,Fk,integral)`, adding only its accumulator to checkpoints. Initial conditions, physical parameters, terminal time and callback-captured parameters can be differentiated. No full trajectory, dense state Jacobian, custom plasma VJP or new SOLVAX method is required.

The research chose exact discrete replay over a continuous backsolve or unconverged implicit-root derivative. The block-local reverse-sweep paper is not a drop-in adjoint for the global spectral RHS. The report explains its applicable principle and its non-transferable assumptions, with source links. A new implicit/IMEX or block-local primal solver would need its own stability, convergence, residual and transpose-solve study.

## Physics and completed experiments

Phase-only control varies distinct oblique magnetic-potential modes at fixed amplitude. It preserves initial magnetic/current spectra and all integrated initial energy channels. Eight phases are optimized with L-BFGS. The objective is normalized electron kinetic-energy gain; guide-field energy is removed from its normalization.

1. Original endpoint study: T=50, 16²/4³, 500 steps; 15.52% improvement, preserving 15.63% at 24²/6³ with half the time step.
2. Seeds 7, 11 and 23 converge to the same endpoint objective. Nearby terminal times show oscillatory energy exchange, including a sign change in net electron gain. Ratios are omitted for nonpositive baselines.
3. The recommended showcase therefore uses a normalized smooth quartic weight over [40,60]. Its streamed average improves 19.815% on 16²/4³, 600 steps, converging in 29 iterations. Halving the step size changes its optimized value by 1.17e-9.
4. Five independent GPU refinements preserve the benefit. At 24²/6³ and 1,200 steps, saved controls improve the average 19.9668%. Warm-start re-optimization converges in 11 iterations to 19.9711%; gradient norm is below 1e-8. The baseline and refined optimum are 0.00791404248 and 0.00949456331.
5. CPU/GPU coarse objective agreement is within 3e-16; the baseline directional-gradient difference is about 2.7e-18. This comparison reuses completed CPU runs and avoids redundant local simulation load.
6. Fourteen original repository tests passed locally; the new current-inverse integration test is additional; eleven autodiff/planning tests passed on GPU. Coverage includes taped/replay/budgeted agreement, JVPs, initial and physical parameters, endpoint time, scalar/vector integrals, current/energy observables and RK4 convergence against Dopri8. Fatal lint and whitespace checks passed.
7. Thirty-eight isolated GPU benchmark cases passed matched-gradient checks. Compilation is separated from synchronized execution medians. The main matrix varies spatial resolution, Hermite resolution, controls, steps and checkpoint count independently, at final time T=2.

At 64²/4³ and 64 steps, replay uses 260 MiB peak JAX allocator memory versus 6457 MiB taped. At 128 controls, replay takes 0.768 s versus 17.568 s for serial centered differences (22.9x), with relative gradient-norm discrepancy 4.76e-7. At 1,024 steps, default replay uses 130 MiB/4.588 s; K=16 uses 66 MiB/6.740 s. The latter retains the same 66 MiB peak at 256 steps.

GPU peaks are allocator measurements, excluding driver/context memory. Compiler estimates and process RSS are separate fields. GPU scaling used the terminal-only core at the initial implementation commit, before the optional accumulator; exact file hashes are saved. The added integral path was validated separately on GPU. CPU before/after checks found identical default-path gradients and compiled buffer sizes. Do not relabel old measurements as a profile of a different source revision.

## Artifacts and reproducibility

`benchmarks/results/` contains original CPU evidence, `gpu_scaling_*`, `gpu_long_rollout_*`, environment freeze, seed/terminal-time robustness data, `window_control_cpu.json`, `window_resolution_gpu.json`, `window_refined_optimization_gpu.json`, `window_cpu_gpu_agreement.json` and the GPU validation manifest. JSON files retain phases needed for reproduction. PNG and vector PDF figures are in `docs/figures/`; the report supplies metric-specific captions.

Full local outputs/logs: `/Users/rogeriojorge/local/tests/output/spectrax-gradient/`. Remote outputs/logs: `/home/rjorge/spectrax-gradient-study/results/`. The `window-control-gpu/results.json` file is the original CPU input report copied for refinement; GPU results are in `resolution_validation.json`, `refined_optimization.json` and `manifest.json`. Do not mislabel the copied input as a GPU optimization run.

Small CPU check, from the local worktree:

```bash
.venv/bin/python -m pytest tests/test_autodiff.py tests/test_gradient_benchmark.py -q
```

Single-GPU execution, from the remote checkout:

```bash
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
.venv/bin/python Examples/2D_phase_control.py --grid 16 --hermite 4 --time 60 --window 40 60 --steps 600 --iterations 30 --output results/new-window
.venv/bin/python benchmarks/validate_phase_control.py results/new-window --reoptimize 20
STUDY_PYTHON=.venv/bin/python bash benchmarks/run_gpu_study.sh results/new-gpu-study
```

Use fresh output directories after source/environment changes. Benchmark `--resume` requires matching source, package, device and allocator manifests; `--plot-only` renders saved measurements without running simulations. Keep GPU simulation jobs serial for reliable timing. Check device occupancy before new runs and leave unrelated work alone.

## Remaining decisions and best next steps

1. Review the public API, explicit RK4 choice and dependency boundary against #36; retain the draft until maintainer review. CI status is visible on the PR.
2. Review the physics interpretation and choose the paper's target regime. The demonstrated result is finite-time, time-averaged energy control, not irreversible heating, solely magnetic-to-electron transfer, or certified resolved turbulence. Endpoint electron energy can be negative while the average is positive.
3. The three-seed, five-window check of the averaged objective is now complete at both 16²/4³ and 24²/6³: all 30 frozen-control cases show positive benefit. Broaden physical parameters or horizons only as required by the paper, and increase resolution to the chosen scientific tolerance. Current small Hermite tails and energy conservation do not certify distribution positivity or global optimality.
4. If a longer nonlinear regime is required, profile it on GPU before changing solvers. Add an implicit/IMEX method or specialized adjoint only for a measured advantage, and validate its executed numerical algorithm.
5. Select final panels/captions from the checked-in vector figures and report exact hardware, precision, controls, discretization, checkpoint budget and memory metric.

At this handoff all simulations, benchmarks and test jobs belonging to this task have completed; no local or remote simulation was left running. No automation or scheduled continuation exists. Other users' workloads were not stopped. No additional long local simulation is needed to reconstruct these results.

## Subsequent robustness checkpoint

Seeds 11 and 23 optimized the averaged [40,60] objective in 16 and 28 iterations and matched seed 7 within 3e-13. Five width-20 windows centered at 40,45,50,55,60 were specified before evaluation. Frozen controls improve every baseline: 13.748–21.929% at 16²/4³, and 13.813–22.085% at 24²/6³ with half dt. The 30 cases are overlapping-window evaluations, not independent statistical trials. All runs completed on office GPU0. A three-window small CPU/GPU smoke check agrees within 6.8e-21; resume and control reuse were exercised locally without long CPU runs.

The report contains reproduction commands and figures; raw data and manifests are `window_seed_optimizations_gpu.json`, `window_shifts_gpu.json`, `window_shifts_refined_gpu.json`, `window_robustness*_manifest.json`, and `window_robustness_cpu_gpu_smoke.json`. The exact coarse runner is in commit `c578481`; refinement/reuse is in `6baaf72`. Subsequent plotting changes only shorten a figure title. Full run directories are `results/window-robustness-gpu` and `results/window-robustness-refined-gpu` on office, with copies in the local output directory.

The user explicitly requested continued autonomous implementation of remaining goals, including measured performance bottlenecks and relevant SOLVAX solvers. The replay and initializer profiling is now complete, as recorded below. No new implicit method should be substituted without a measured benefit and correctness/stability evidence.

## Completed differentiability and performance continuation

- SPECTRAX `762746f` adds SOLVAX source hashes to benchmark manifests; `7ac745c` preserves the vectorized initializer and baseline matrix-free current inverse. Exact measured source hashes are in each report. Array initialization agrees with the original loop in primal/JVP to 6.5e-16 (0/8/128 controls).
- Paired 128-control GPU measurements show reverse compilation 29.31 → 18.46 s, forward execution 8.796 → 6.129 s, and essentially unchanged reverse execution 0.2833 → 0.2818 s and allocator peaks. Raw paired results/manifests are committed. The historical 0.768 s reverse measurement must not be used as the pre-vectorization baseline.
- `Examples/2D_current_inverse.py` uses released SOLVAX's matrix-free damped Gauss–Newton/PCG with JVP/VJP of the plasma residual. Eight controls at 24²/6³, T=20, 200 steps converge in four outer/16 PCG iterations, relative map error 1.67e-9. Thirty-two controls converge in five/27 iterations, relative error 9.92e-12. Small CPU/GPU runs match to roundoff. The synthetic target uses the same discretization: this is API/inverse verification, not experimental validation. `--plot-only` redraws saved results without simulation; checked vector/PNG figures and native maps are committed.
- A spectral diagonal preconditioner increased PCG counts in all three tested cases (8→12, 16→21, 27→40); it was rejected. Experimental source is preserved at `1a6171e` and all raw reports are committed. An unrelated process started on GPU0 during the 32-control comparison, so those times are not uncontended performance evidence. No unrelated processes were stopped.
- SOLVAX [PR #103](https://github.com/uwplasma/SOLVAX/pull/103), branch `agent/spectrax-replay-performance`, commit `cd67665`, replaces guarded full replay segments with unconditional segments and a checkpointed tail. Its isolated worktree is `/Users/rogeriojorge/local/tests/solvax-replay-performance`; remote candidate is `/home/rjorge/solvax-replay-performance`. The source is selected only per process using `PYTHONPATH`; SPECTRAX remains on released 0.20.x.
- SOLVAX tests: 44 autodiff cases pass on CPU and GPU, including six additional exact/tail/clamped/negative-bound and complex real-linear cases. Two SPECTRAX GPU integration cases pass with the candidate. Ten-sample interleaved measurements give 3.40% lower warm gradient time but ~2 s higher compile time (~210-call break-even); peak allocator use stays 66 MiB, with some static estimates increasing. The upstream PR includes raw measurements, script, source hashes and explicit tradeoffs. Keep it draft for performance review.

The implementation deliverables are covered: general terminal/streamed objectives, default JVP/VJP replay and bounded-checkpoint reverse mode, finite-difference/taped checks, fixed-energy phase optimization with refinement/seed/window evidence, matrix-free current inversion, measured bottleneck changes, raw provenance and publication figures. Remaining work is scientific/maintainer review: choose the paper's final regime/panels, reconcile API scope with PR #36, and decide whether the measured SOLVAX warm-runtime tradeoff warrants merging. Broader noise/model mismatch or long-time turbulent physics needs a separately specified scientific target; the present results do not certify those claims.

Validation checkpoint: the full experimental suite passed 16 tests locally in 156.26 s; after removing the rejected option, the retained inverse integration test passed in 71.72 s (local host was busy). Final suite has 15 tests. Fatal lint and whitespace checks pass. SOLVAX PR103 CI is fully green, including all five platform/dependency test configurations, build, lint, types and combined coverage. SPECTRAX final-source CI is available on PR42. All task GPU simulations finished; no further local/GPU simulation is scheduled. Remote result directories retain the raw logs. The remote checkouts are synchronized after publication, with prepublication source preserved in named git stashes; `results/` and `.bootstrap/` remain untouched.
