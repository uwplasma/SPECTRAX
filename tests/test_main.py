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

if __name__ == '__main__':
    unittest.main()