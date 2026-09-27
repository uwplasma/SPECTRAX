import sys
import pytest
import unittest
from spectrax.__main__ import main
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_simulation():
    with patch('spectrax.__main__.simulation') as mock_sim:
        mock_sim.return_value = {
            "Ck": MagicMock(),
            "Fk": MagicMock(),
        }
        yield mock_sim

@pytest.fixture
def mock_plot():
    with patch('spectrax.__main__.plot') as mock_plot:
        yield mock_plot

# This is a pure pytest-style test, not unittest-style
def test_main_function_runs(mock_simulation, mock_plot):
    test_args = ["__main__.py"]
    with patch.object(sys, 'argv', test_args):
        main(sys.argv[1:])

    mock_simulation.assert_called_once()
    mock_plot.assert_called_once()

def test_main_runs_a_2d_toml_without_an_initial_state(tmp_path, mock_plot):
    """A 2D, multi-Hermite TOML with no Ck_0 / Fk_0 (like input_2D_orszag_tang.toml) used to crash on a 1D default state."""
    toml = tmp_path / "input.toml"
    toml.write_text("[input_parameters]\nt_max = 0.5\n[solver_parameters]\nNx = 9\nNy = 4\nNn = 4\nNm = 2\ntimesteps = 3\n")
    main([str(toml)])
    output = mock_plot.call_args[0][0]
    assert output["Ck"].shape == (3, 2 * 4 * 2, 4, 9, 1, 2)

if __name__ == '__main__':
    unittest.main()