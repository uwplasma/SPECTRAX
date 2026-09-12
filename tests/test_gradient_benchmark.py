"""Validate experiment planning before allocating a GPU for a large sweep."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "gradient_scaling", Path(__file__).parents[1] / "benchmarks/gradient_scaling.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def options(**updates):
    config = dict(grids=[16, 32], steps=[64, 256], controls=[2, 128],
                  hermites=[4, 6], checkpoint_counts=[4, 16], base_grid=32,
                  base_steps=64, base_controls=8, hermite=4, checkpoints=8,
                  methods=["checkpointed", "budgeted", "taped", "forward", "fd"],
                  axes=["grid", "steps", "controls", "hermite", "checkpoints"])
    return SimpleNamespace(**(config | updates))


def test_independent_gpu_sweeps():
    cases = benchmark.build_cases(options())
    assert len(cases) == 28
    for case in cases:
        if case["axis"] != "grid":
            assert case["grid"] == 32
        if case["axis"] != "steps":
            assert case["steps"] == 64
        if case["axis"] != "controls":
            assert case["controls"] == 8
    assert any(c["controls"] == 128 for c in cases)
    assert {c["checkpoints"] for c in cases if c["axis"] == "checkpoints"} == {4, 16}


@pytest.mark.parametrize("controls", [37, 128])
def test_reject_unresolved_control_count(controls):
    with pytest.raises(ValueError, match="increase --base-grid"):
        benchmark.build_cases(options(base_grid=16, controls=[controls]))


def test_long_rollout_without_taped_allocation():
    cases = benchmark.build_cases(options(axes=["steps"], steps=[1024],
                                           methods=["checkpointed", "budgeted"]))
    assert len(cases) == 2
    assert all(c["steps"] == 1024 for c in cases)
