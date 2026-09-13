"""Phase 5b gate: does reverse-pass memory store K saved snapshots once, or once per checkpoint?

Compile-only (memory_analysis of the compiled programme) for the mean_conversion objective at 32²×4³, T=100,
adaptive Dopri8, for K snapshots and C checkpoints. Stored once predicts ~K*S growth; per checkpoint ~K*C*S.
"""
import importlib, json, os, sys
import jax
jax.config.update("jax_enable_x64", True)
root = os.environ["SPECTRAX_ROOT"]; sys.path.insert(0, root); sys.path.insert(0, os.path.join(root, "Examples"))
ex = importlib.import_module("2D_Orszag_Tang_optimization")
grid, hermite, t_max = 32, 4, 100.0
state_MiB = (2 * hermite**3 + 6) * grid * (grid // 2 + 1) * 16 / 2**20
theta = ex.initial_controls(8, 0)
rows = []
for C in (8, 32):
    for K in (2, 11, 41, 161):
        loss = lambda th: ex.OBJECTIVES["mean_conversion"](ex.solve(ex.setup(th, grid, hermite, t_max), grid, hermite, C, timesteps=K))
        fwd = jax.jit(loss).lower(theta).compile().memory_analysis().temp_size_in_bytes / 2**20
        rev = jax.jit(jax.value_and_grad(loss)).lower(theta).compile().memory_analysis().temp_size_in_bytes / 2**20
        row = dict(checkpoints=C, snapshots=K, state_MiB=state_MiB, forward_MiB=fwd, reverse_MiB=rev)
        rows.append(row); print(json.dumps(row), flush=True)
json.dump(dict(grid=grid, hermite=hermite, t_max=t_max, provenance=ex.provenance(), rows=rows),
          open(os.path.join(os.path.dirname(__file__), "snapshot_memory.json"), "w"), indent=1)
base = {r["checkpoints"]: r["reverse_MiB"] for r in rows if r["snapshots"] == 2}
for r in rows:
    extra = r["reverse_MiB"] - base[r["checkpoints"]]
    print(f"C={r['checkpoints']:2d} K={r['snapshots']:3d}: reverse {r['reverse_MiB']:8.1f} MiB, extra {extra:8.1f} MiB = "
          f"{extra / (r['snapshots'] * state_MiB):5.2f} K*S = {extra / (r['snapshots'] * r['checkpoints'] * state_MiB):5.3f} K*C*S")
