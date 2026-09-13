# SPECTRAX differentiability: evidence branch

Generated results for uwplasma/SPECTRAX PR #44. Nothing here is library code. The PR description is the
handoff and states what each result supports. Every JSON records its settings; later reports also record
the commit, package versions and device.

| directory | contents |
|---|---|
| `2026-09-13_decisions/` | scripts and JSON behind the "Decisions" table in PR #44: gradient through the stock `simulation` (exp_a), memory vs max_steps / checkpoints / steps / resolution (exp_b), showcase prototype (exp_c), continuous adjoint and reversible RK4 (exp_d), forward cost calibration (exp_e) |
| `2026-09-13_cpu_32x4/` | `Examples/2D_Orszag_Tang_optimization.py` at commit 9600144 on an Apple M3 Max CPU: `optimize` for both objectives at 32²×4³, T=200, 16 controls, 20 iterations, and `benchmark` at 16²×4³, T=100 |
| `2026-09-13_phase2a_refinement/` | frozen-control refinement of the 32²×4³ optima: script, office launchers (tolerance 1e-7 refinement, then 64² at 1e-9), CPU results; GPU results are added when they finish |
| `2026-09-13_phase2c_phases_32x4/` | phase-only (`--fixed-amplitudes`) optimisations of both objectives at 32²×4³, T=200, 8 phase controls, commit 8380edc, Apple M3 Max CPU |
| `2026-09-13_phase3_gpu_64x6/` | Phase 3 on one RTX A4000 at commit 8380edc: 64²×6³, T=200, tolerance 1e-9, 16 controls, 30 iterations, with validation at 64²×8³ and 128²×6³; benchmark and peak memory are added when they finish |
| `scripts/` | office GPU launchers for Phase 3 (`run_phase3.sh`) and per-process peak device memory (`peak_memory.py`) |
| `paper/` | draft methods paragraph and figure captions |
