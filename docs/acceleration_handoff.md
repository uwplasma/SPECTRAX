# Scientific follow-up handoff

**Current integration policy:** SOLVAX PR103 was admin-squash-merged at `917d1a679d8258bb7c9929e2320f4befbbe92f93` with explicit user authorization. SPECTRAX PR42 and PR43 must remain open and be merged only by other collaborators. Earlier blocked-merge descriptions below record the previous state. Existing numerical archives still use released SOLVAX 0.20.0.

## Branches and merge order

SPECTRAX PR42 (`agent/checkpointed-plasma-control`, 23102ab) contains the small
public `simulation_final` API, discrete RK4 replay, gradient checks, constrained
energy/current examples and original performance evidence. SOLVAX PR103
(`agent/spectrax-replay-performance`, cd67665) contains the independent replay
performance change. Both are ready with passing CI; normal merge was rejected
because required reviews are missing, and repository auto-merge is disabled.
Do not bypass branch protection. This follow-up is SPECTRAX PR43
(`agent/acceleration-validation`), based on PR42 and kept draft for scientific
review. After PR42 merges, inspect the resulting PR43 diff before review.

All reported SPECTRAX results use released SOLVAX 0.20.0. They do not silently
use the candidate SOLVAX change. The latter measured a modest warm improvement
with a compilation tradeoff, not a new adjoint or a memory reduction.

## Reproduce the principal experiments

Use a fresh output path for each simulation. Install the repository's declared
dependencies in an isolated environment, record versions, and use float64.
The saved production environment is in individual JSON provenance. Run GPU
studies serially on a single device; local CPU tests are small correctness checks.
`office` used GPU0 (RTX A4000); do not terminate unrelated workloads.

```sh
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

# Fresh separate processes, actual T=60 horizon, identical eight-control problem.
python benchmarks/optimization_comparison.py --method fd --fd-reference none --output results/comparison
python benchmarks/optimization_comparison.py --method ad --fd-reference none --output results/comparison
python benchmarks/optimization_comparison.py --plot-only --output results/comparison

# Cheap candidate search; its coarse positivity failure is retained.
python Examples/2D_tail_control.py --grid 16 --hermite 8 --steps 600 --nu 0 --output results/tail-control-coarse
# Frozen-control time, space, Hermite and directional-gradient validation.
python benchmarks/validate_tail_control.py --controls-file results/tail-control-coarse/results.json --output results/tail-validation
# Fine sampled work, energy, tail and local negativity through held-out T=70.
python benchmarks/energization_mechanism.py --controls-file results/tail-control-coarse/results.json --grid 24 --hermite 16 --steps 1400 --nu 0 --output results/tail-mechanism-fine
# CPU-only postprocessing of saved spatial-average coefficients.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/velocity_spectrum_panels.py --results results/tail-mechanism-fine/results.json --output results/tail-spectrum
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/tail_sensitivity.py --coefficients results/tail-mechanism-fine/mean_coefficients.npz --report results/tail-mechanism-fine/results.json --output results/tail-sensitivity.json
```

The tail's threshold is fixed at five times initial mean thermal energy, width
half that energy; each phase configuration subtracts its own initial tail.
Initial integrated energy and magnetic/current spectra remain constrained.
Scalar benefit convergence and a single fixed-direction derivative check do not
prove full-gradient convergence or global optimality. Sampled finite-quadrature
negativity is not a continuous positivity proof. In particular, a global lab-frame
tail or global Gaussian excess can include bulk motion and spatial mixtures.

## Evidence and interpretation

`acceleration_results.md` reports the final numbers and figure captions.
`acceleration_plan.md` preserves prospective gates and responses to failed
experiments. `acceleration_objective.md` defines signed observables and limitations.
Compressed JSON under `benchmarks/results/acceleration` retains source hashes,
controls, timing details and failed coarse/filtered cases. `manifest.json` records
uncompressed checksums. Original runs and memory benchmarks remain in
`benchmarks/results` and `differentiable_control.md`.

The full raw working archives (including sampled mean Hermite coefficients) are
on `office:/home/rjorge/spectrax-acceleration-validation/results` and local collected
artifacts under `/Users/rogeriojorge/local/tests/output/spectrax-acceleration`.
Only compressed numerical reports and polished figures belong in Git; source
hashes identify different revisions within the study. No full phase-space
trajectory is needed to regenerate the scalar figures.

Literature positioning and links are in `publication_assessment.md` and
`differentiable_control.md`: Joglekar–Thomas differentiable kinetic optimization,
Skene–Burns spectral adjoints, checkpointing/Revolve, and kinetic turbulence
mechanism studies. Discrete RK4 checkpoint replay is the method implemented;
the block-implicit reverse sweep of “Differentiate the Solver, Not the Equation”
is not interchangeable with this global spectral explicit solver.

## Remaining publication decisions

Separate a methods-paper demonstration of constrained finite-time tail control
from a discovery claim about nonthermal acceleration. The latter needs local
flow/temperature/anisotropy-conditioned distributions and mechanism evidence.
Fixed threshold 4/5/6 sensitivity is reported from saved samples. Independent
tail-optimization starts and held-out time-sampling convergence remain additional
robustness studies. The direct AD/FD timing is one paired full-horizon energy
optimization; repeat pairs for timing uncertainty and do not transfer its ratio
to the more expensive tail objective without measurement. Fixed-budget memory
bounds apply to stored time states; memory still grows with phase-space DOFs.

Regenerate the tail figure entirely from committed JSON archives:

```sh
python benchmarks/tail_control_panels.py --coarse benchmarks/results/acceleration/tail-control-coarse-results.json.gz --validation benchmarks/results/acceleration/tail-validation-results.json.gz --mechanism benchmarks/results/acceleration/tail-mechanism-fine.json.gz --output docs/figures/tail_control
```

## Local-frame extension

`local_tail_results.md` and `local_tail_plan.md` record the extension motivated by
the old lab-frame optimum's bulk/internal partition. The private local objective
differentiates the cellwise mean; the Gaussian diagnostic matches full local
covariance. `Examples/2D_local_tail_control.py` uses sampled time weights with
nested replay, and `benchmarks/local_tail_snapshot.py` provides the matched
reference and validity checks. Raw evidence is under `benchmarks/results/local-tail`.
Local source and office remain on PR43; no SPECTRAX merge is authorized.
