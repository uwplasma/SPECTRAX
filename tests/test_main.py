import pytest
from unittest.mock import patch
from spectrax.__main__ import main


def test_main_without_a_file_prints_usage():
    with pytest.raises(SystemExit):
        main([])


def test_main_runs_a_2d_toml_without_an_initial_state(tmp_path):
    """A 2D, multi-Hermite TOML with no Ck_0 / Fk_0 (like input_2D_orszag_tang.toml) used to crash on a 1D default state."""
    toml = tmp_path / "input.toml"
    toml.write_text("[input_parameters]\nt_max = 0.5\n[solver_parameters]\nNx = 9\nNy = 4\nNn = 4\nNm = 2\ntimesteps = 3\n")
    with patch("spectrax._run.plot") as mock_plot:
        output = main([str(toml)])
    mock_plot.assert_called_once()
    assert output["Ck"].shape == (3, 2 * 4 * 2, 4, 5, 1)


def test_cli_runs_a_species_toml_and_saves_the_diagnostic_figure(tmp_path):
    """``spectrax input.toml --no-show`` runs any number of species and writes input.png with every panel."""
    import matplotlib
    matplotlib.use("Agg")
    toml = tmp_path / "input.toml"
    toml.write_text("[input_parameters]\nLx = 10.0\nLy = 10.0\nt_max = 0.5\n[solver_parameters]\nNx = 6\nNy = 5\nNn = 3\nNm = 2\ntimesteps = 4\n"
                    "[species.electrons]\ncharge = -1\nvth = 0.2\nperturbation_amplitude = 0.05\nperturbation_axis = 'y'\n"
                    "[species.ions]\ncharge = 1\nmass = 25\nvth = 0.04\n[species.beam]\ncharge = 1\nmass = 25\nvth = 0.04\ndrift = 0.1\n")
    output = main([str(toml), "--no-show"])
    assert (tmp_path / "input.png").stat().st_size > 0
    assert output["Ck"].shape[1] == 3 * 3 * 2 and "electric_energy" in output


def test_initial_state_hook_supplies_the_coefficients(tmp_path):
    """[initial_state] file = ... calls the file's initial_state(input_parameters, solver_parameters)."""
    import numpy as np
    from spectrax import load_parameters
    (tmp_path / "init.py").write_text("def initial_state(i, s):\n    return {'marker': s['Nx']}\n")
    toml = tmp_path / "case.toml"
    toml.write_text("[initial_state]\nfile = 'init.py'\n[input_parameters]\nt_max = 1.0\n[solver_parameters]\nNx = 7\n")
    assert load_parameters(toml)[0]["marker"] == 7


def test_run_applies_the_output_table_and_overrides(tmp_path):
    """``run`` honours [output], returns a Result, and key=value / keyword overrides reach the case."""
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from spectrax import run
    toml = tmp_path / "case.toml"
    toml.write_text("[input_parameters]\nLx = 10.0\nt_max = 0.5\n[solver_parameters]\nNx = 6\nNn = 3\ntimesteps = 4\n"
                    "[species.electrons]\ncharge = -1\nvth = 0.2\nperturbation_amplitude = 0.05\n[species.ions]\ncharge = 1\nmass = 25\nvth = 0.04\n"
                    "[output]\nfigure = 'fig.png'\nshow = false\nsave = 'out.npz'\nkeep = ['time', 'EM_energy']\n")
    result = run(toml, t_max=0.25, species={"electrons": {"vth": 0.3}})
    assert (tmp_path / "fig.png").stat().st_size > 0 and set(np.load(tmp_path / "out.npz")) == {"time", "EM_energy"}
    assert float(result["time"][-1]) == 0.25 and result.summary()["saved_times"] == 4
    main([str(toml), "species.electrons.vth=0.25", "figure=false", "save=all.npz", "keep=false", "--no-show"])
    assert "Ck" in np.load(tmp_path / "all.npz")
