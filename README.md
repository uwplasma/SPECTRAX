# SPECTRAX differentiability: evidence branch

Generated results for uwplasma/SPECTRAX PR #44. Nothing here is library code. The PR description is the
handoff and states what each result supports. Every JSON records its settings; later reports also record
the commit, package versions and device.

| directory | contents |
|---|---|
| `2026-09-13_decisions/` | scripts and JSON behind the "Decisions" table in PR #44: gradient through the stock `simulation` (exp_a), memory vs max_steps / checkpoints / steps / resolution (exp_b), showcase prototype (exp_c), continuous adjoint and reversible RK4 (exp_d), forward cost calibration (exp_e) |
| `2026-09-13_cpu_32x4/` | `Examples/2D_Orszag_Tang_optimization.py` at commit 9600144 on an Apple M3 Max CPU: `optimize` for both objectives at 32²×4³, T=200, 16 controls, 20 iterations, and `benchmark` at 16²×4³, T=100 |
| `2026-09-13_phase2a_refinement/` | frozen-control refinement of the 32²×4³ optima: script, office launchers (tolerance 1e-7 refinement, then 64² at 1e-9), CPU results; GPU results are added when they finish |
| `scripts/` | office GPU launchers for Phase 3 (`run_phase3.sh`) and per-process peak device memory (`peak_memory.py`) |
| `paper/` | draft methods paragraph and figure captions |
