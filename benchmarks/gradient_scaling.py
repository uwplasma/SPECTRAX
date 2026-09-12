"""Isolated-process memory/timing comparison of identical plasma RK4 trajectories.

Run with an installed SPECTRAX: python benchmarks/gradient_scaling.py --output DIR
For a single GPU set JAX_PLATFORMS=cuda. No GPU numbers are inferred from CPU runs.
Each worker disables allocator preallocation before importing JAX. Compilation is
reported separately; synchronized medians exclude compilation. XLA buffer bytes
are compiler estimates, RSS is process-wide, and GPU allocator peaks are recorded
only when the backend supplies them. These are distinct memory measurements.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def metadata():
    """Record provenance before admitting resumed measurements."""
    import hashlib
    import importlib.metadata
    import platform
    import inspect
    import jax
    import solvax
    root = Path(__file__).parents[1]
    files = [str(p.relative_to(root)) for p in sorted((root / "spectrax").glob("*.py"))
             if p.name != "version.py"] + ["Examples/2D_phase_control.py", "benchmarks/gradient_scaling.py"]
    info = dict(source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
                packages={name: importlib.metadata.version(name) for name in
                          ("jax", "jaxlib", "diffrax", "solvax", "numpy", "scipy", "equinox")},
                solvax_autodiff_sha256=hashlib.sha256(Path(inspect.getsourcefile(solvax.checkpointed_fori_loop)).read_bytes()).hexdigest(),
                python=platform.python_version(), platform=platform.platform(),
                devices=[dict(kind=d.device_kind, platform=d.platform) for d in jax.devices()],
                allocator_environment={key: os.environ.get(key) for key in
                                       ("CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_PREALLOCATE",
                                        "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR")})
    if jax.default_backend() == "gpu":
        query = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                "--format=csv,noheader"], text=True, capture_output=True, timeout=10)
        info["nvidia_smi"] = query.stdout.strip() if query.returncode == 0 else query.stderr.strip()
    return info


def worker(config):
    import resource
    import hashlib
    import platform
    import jax
    import numpy as np
    import solvax
    spec = importlib.util.spec_from_file_location(
        "phase_control", Path(__file__).parents[1] / "Examples" / "2D_phase_control.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    objective, _ = example.problem(
        config["grid"], config["hermite"], 2.0, config["steps"],
        checkpointing=config["method"] != "taped",
        checkpoints=config.get("checkpoints", 8) if config["method"] == "budgeted" else None)
    x = np.random.default_rng(7).uniform(-np.pi, np.pi, config["controls"])
    if config["method"] == "forward":
        fn = jax.jacfwd(objective)
    elif config["method"] == "fd":
        fn = objective
    else:
        fn = jax.value_and_grad(objective)
    start = perf_counter()
    executable = jax.jit(fn).lower(x).compile()
    compile_seconds = perf_counter() - start
    memory_after_compile = jax.devices()[0].memory_stats()
    memory = executable.memory_analysis()
    epsilon = 1e-4

    def evaluate():
        if config["method"] == "fd":
            # Serial centered differences are a low-memory black-box baseline.
            gradient = np.empty(len(x))
            for i in range(len(x)):
                delta = np.zeros_like(x)
                delta[i] = epsilon
                gradient[i] = float((executable(x + delta) - executable(x - delta)) / (2 * epsilon))
            return gradient
        return jax.block_until_ready(executable(x))

    evaluate()
    memory_after_warmup = jax.devices()[0].memory_stats()
    elapsed = []
    for _ in range(3):
        start = perf_counter()
        result = evaluate()
        elapsed.append(perf_counter() - start)
    gradient = result if config["method"] in ("forward", "fd") else result[1]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes; Linux reports KiB.
    if sys.platform != "darwin":
        rss *= 1024
    device_stats = jax.devices()[0].memory_stats() or {}
    xla_bytes = (memory.argument_size_in_bytes + memory.output_size_in_bytes
                 + memory.temp_size_in_bytes - memory.alias_size_in_bytes)
    root = Path(__file__).parents[1]
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                     for name in ("spectrax/_autodiff.py", "spectrax/_model.py",
                                  "spectrax/_simulation.py", "Examples/2D_phase_control.py")}
    report = dict(config, source_sha256=source_hashes, seconds=float(np.median(elapsed)),
                  compile_seconds=compile_seconds, xla_buffer_bytes=xla_bytes,
                  xla_temporary_bytes=memory.temp_size_in_bytes,
                  peak_process_rss_bytes=rss,
                  gpu_peak_bytes_in_use=device_stats.get("peak_bytes_in_use"),
                  allocator_after_compile=memory_after_compile, allocator_after_warmup=memory_after_warmup,
                  state_bytes=(2 * config["hermite"] ** 3 + 6) * config["grid"]
                              * (config["grid"] // 2 + 1) * 16,
                  gradient=np.asarray(gradient).tolist(), device=str(jax.devices()[0]),
                  jax_version=jax.__version__, solvax_version=solvax.__version__,
                  platform=platform.platform(), machine=platform.machine(),
                  samples_seconds=elapsed, device_kind=jax.devices()[0].device_kind,
                  backend=jax.default_backend(), allocator_stats=device_stats)
    print(json.dumps(report), flush=True)


def build_cases(options):
    """Build independent sweeps with a configurable common reference problem."""
    cases = []
    for axis, values in [("grid", options.grids), ("steps", options.steps),
                         ("controls", options.controls), ("hermite", options.hermites),
                         ("checkpoints", options.checkpoint_counts)]:
        if axis not in options.axes:
            continue
        for value in values:
            methods = (["checkpointed", "forward", "fd"] if axis == "controls" else
                       ["checkpointed", "budgeted"] if axis == "checkpoints" else
                       ["checkpointed", "budgeted", "taped"])
            for method in methods:
                if method not in options.methods:
                    continue
                case = dict(axis=axis, method=method, grid=options.base_grid,
                            steps=options.base_steps, controls=options.base_controls,
                            hermite=options.hermite, checkpoints=options.checkpoints)
                case[axis] = value
                capacity = (case["grid"] // 3 - 1) * (len(range(-case["grid"] // 3 + 1, case["grid"] // 3)) - 1)
                if case["grid"] < 8 or case["hermite"] < 3 or min(case["steps"], case["controls"], case["checkpoints"]) < 1:
                    raise ValueError(f"Invalid benchmark resolution or count: {case}")
                if case["controls"] > capacity:
                    raise ValueError(f"{case['controls']} controls exceed grid {case['grid']} capacity {capacity}; increase --base-grid")
                cases.append(case)
    if not cases:
        raise ValueError("No benchmark cases selected")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--metadata", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Render saved measurements without rerunning workers")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse this output directory only on unchanged code/hardware")
    parser.add_argument("--output", type=Path, default=Path("gradient-scaling"))
    parser.add_argument("--grids", type=int, nargs="+", default=[12, 16, 24])
    parser.add_argument("--steps", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--controls", type=int, nargs="+", default=[2, 8, 16])
    parser.add_argument("--hermite", type=int, default=4)
    parser.add_argument("--hermites", type=int, nargs="+", default=[])
    parser.add_argument("--checkpoint-counts", type=int, nargs="+", default=[])
    parser.add_argument("--base-grid", type=int, default=16)
    parser.add_argument("--base-steps", type=int, default=64)
    parser.add_argument("--base-controls", type=int, default=8)
    parser.add_argument("--checkpoints", type=int, default=8)
    parser.add_argument("--methods", nargs="+", choices=["checkpointed", "budgeted", "taped", "forward", "fd"],
                        default=["checkpointed", "budgeted", "taped", "forward", "fd"])
    parser.add_argument("--axes", nargs="+", choices=["grid", "steps", "controls", "hermite", "checkpoints"],
                        default=["grid", "steps", "controls", "hermite", "checkpoints"])
    options = parser.parse_args()
    if options.plot_only:
        render(json.loads((options.output / "results.json").read_text()), options.output)
        return
    if options.metadata:
        print(json.dumps(metadata()))
        return
    if options.worker:
        worker(json.loads(options.worker))
        return
    options.output.mkdir(parents=True, exist_ok=True)
    cases = build_cases(options)
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    current_metadata = json.loads(subprocess.check_output(
        [sys.executable, __file__, "--metadata"], env=env, text=True))
    manifest_path = options.output / "manifest.json"
    if options.resume and (not manifest_path.exists() or json.loads(manifest_path.read_text()) != current_metadata):
        raise ValueError("Cannot resume across changed source, packages, hardware or allocator settings; use a new output directory")
    manifest_path.write_text(json.dumps(current_metadata, indent=2) + "\n")
    results_path = options.output / "results.json"
    results = json.loads(results_path.read_text()) if options.resume and results_path.exists() else []
    results = [row for row in results if "error" not in row
               and any(all(row.get(k) == v for k, v in case.items()) for case in cases)]
    for case in cases:
        if any(all(row.get(k) == v for k, v in case.items()) and "error" not in row for row in results):
            continue
        print(case, flush=True)
        env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
        process = subprocess.run([sys.executable, __file__, "--worker", json.dumps(case)],
                                  env=env, text=True, capture_output=True)
        if process.returncode:
            results.append(dict(case, error=process.stderr[-3000:], returncode=process.returncode))
        else:
            results.append(json.loads(process.stdout.strip().splitlines()[-1]))
        (options.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    render(results, options.output)


def render(results, output):
    """Render actual allocator peaks when available; otherwise compiler estimates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    checkpoint_count = results[0].get("checkpoints", 8)
    use_allocator = all(row.get("gpu_peak_bytes_in_use") is not None for row in results if "error" not in row)
    memory_key = "gpu_peak_bytes_in_use" if use_allocator else "xla_buffer_bytes"
    active_axes = list(dict.fromkeys(row["axis"] for row in results))
    fig, axes = plt.subplots(1, len(active_axes), figsize=(3.7 * len(active_axes), 3.4),
                             squeeze=False, layout="constrained")
    colors = dict(checkpointed="#0072B2", budgeted="#CC79A7", taped="#D55E00", forward="#009E73", fd="#777777")
    labels = dict(checkpointed="SOLVAX", budgeted=f"Fixed budget (K={checkpoint_count})",
                  taped="Taped AD", forward="Forward AD", fd="Finite differences")
    for ax, axis in zip(axes[0], active_axes):
        positions = []
        for method, color in colors.items():
            rows = [r for r in results if r["axis"] == axis and r["method"] == method and "error" not in r]
            if not rows:
                continue
            x = [r["state_bytes"] / 2 ** 20 if axis == "grid" else r[axis] for r in rows]
            y = [r["seconds"] if axis == "controls" else r[memory_key] / 2 ** 20 for r in rows]
            positions.extend(x)
            ax.loglog(x, y, "o-", label="Fixed budget" if axis == "checkpoints" and method == "budgeted" else labels[method], color=color, ms=4)
        ax.set_xlabel(dict(grid="Plasma state (MiB)", steps="RK4 steps", controls="Control parameters",
                                 hermite="Hermite modes per velocity axis", checkpoints="Checkpoint count")[axis])
        ax.set_ylabel("Gradient time (s)" if axis == "controls" else
                      "Peak JAX allocator (MiB)" if use_allocator else "XLA buffer estimate (MiB)")
        ax.set_xticks(sorted(set(positions)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.15)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"gradient_scaling.{extension}", dpi=300)
    # A different gradient is not an acceptable speedup. Compare every matched case.
    failures = []
    for axis in active_axes:
        for reference in [r for r in results if r["axis"] == axis and r["method"] == "checkpointed" and "error" not in r]:
            for row in [r for r in results if r["axis"] == axis and r[axis] == reference[axis] and "error" not in r]:
                rtol, atol = (2e-4, 2e-9) if row["method"] == "fd" else (1e-9, 1e-12)
                if not np.allclose(row["gradient"], reference["gradient"], rtol=rtol, atol=atol):
                    failures.append((axis, row[axis], row["method"]))
    if failures or any("error" in r for r in results):
        raise RuntimeError(f"Inspect results.json: worker/gradient failures {failures}")


if __name__ == "__main__":
    main()
