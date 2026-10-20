import sys
import numpy as np
import pytest
import unittest
from spectrax.__main__ import main
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_simulation():
    with patch('spectrax._simulation.simulation') as mock_sim:
        mock_sim.return_value = {
            "Ck": MagicMock(),
            "Fk": MagicMock(),
        }
        yield mock_sim

@pytest.fixture
def mock_plot():
    with patch('spectrax._plot.plot') as mock_plot:
        yield mock_plot

# This is a pure pytest-style test, not unittest-style
def test_main_function_runs(mock_simulation, mock_plot):
    test_args = ["__main__.py"]
    with patch.object(sys, 'argv', test_args):
        main(sys.argv[1:])

    mock_simulation.assert_called_once()
    mock_plot.assert_called_once()

def test_main_no_plot_no_progress(mock_simulation, mock_plot, capsys):
    ck = MagicMock()
    ck.sharding.mesh = SimpleNamespace(devices=np.array([SimpleNamespace(platform="cpu")] * 2), shape={"species": 1, "x": 2})
    mock_simulation.return_value = {"Ck": ck, "time": np.array([0., 1.]), "total_energy": np.array([3., 2.]), "num_steps": 7,
                                    "kinetic_energy": np.array([2., 1.]), "EM_energy": np.array([1., 1.])}
    main(["--no-plot", "--no-progress"])
    mock_plot.assert_not_called()
    assert type(mock_simulation.call_args.kwargs["progress_meter"]).__name__ == "NoProgressMeter"
    out = capsys.readouterr().out
    assert "Devices: 2 cpu" in out and "total energy = 2.000000e+00" in out

if __name__ == '__main__':
    unittest.main()
