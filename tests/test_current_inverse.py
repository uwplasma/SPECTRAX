"""End-to-end test of SOLVAX's matrix-free inverse-current integration."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_small_current_inverse(tmp_path):
    example = Path(__file__).parents[1] / "Examples/2D_current_inverse.py"
    subprocess.run([sys.executable, str(example), "--grid", "12", "--hermite", "3",
                    "--time", "0.2", "--steps", "4", "--controls", "4", "--iterations", "12",
                    "--output", str(tmp_path)], check=True, capture_output=True, timeout=120)
    report = json.loads((tmp_path / "results.json").read_text())
    assert report["converged"]
    assert report["final_relative_current_error"] < 1e-6
    assert report["phase_rms_error"] < 1e-5
    assert report["initial_energy_constraint_error"] < 1e-13
    assert report["adjoint_dot_error"] < 1e-12
    np.testing.assert_allclose(report["directional_ad"], report["directional_fd"], rtol=1e-6, atol=1e-10)
