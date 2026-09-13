# SPECTRAX differentiability: evidence branch

Generated results for uwplasma/SPECTRAX PR #44. Nothing here is library code. The PR description is the
handoff and states what each result supports. Every JSON records its settings; later reports also record
the commit, package versions and device.

| directory | contents |
|---|---|
| `2026-09-13_decisions/` | scripts and JSON behind the "Decisions" table in PR #44: gradient through the stock `simulation` (exp_a), memory vs max_steps / checkpoints / steps / resolution (exp_b), showcase prototype (exp_c), continuous adjoint and reversible RK4 (exp_d), forward cost calibration (exp_e) |
| `2026-09-13_cpu_32x4/` | `Examples/2D_Orszag_Tang_optimization.py` at commit 9600144 on an Apple M3 Max CPU: `optimize` for both objectives at 32²×4³, T=200, 16 controls, 20 iterations, and `benchmark` at 16²×4³, T=100 |
| `paper/` | draft methods paragraph and figure captions |
