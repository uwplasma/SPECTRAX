"""Landau damping of a Langmuir wave. The whole case is in input_1D_landau_damping.toml; this is the same as running

    spectrax input_1D_landau_damping.toml
"""
from pathlib import Path
from spectrax import load_parameters, plot, simulation

here = Path(__file__).parent
input_parameters, solver_parameters = load_parameters(here / "input_1D_landau_damping.toml")
output = simulation(input_parameters, **solver_parameters)
plot(output, save=here / "1D_Landau_damping.png")
