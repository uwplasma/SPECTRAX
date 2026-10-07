"""Command line interface: ``spectrax input.toml`` runs the case, saves ``input.png`` next to it and shows it."""
import sys
from pathlib import Path
from ._plot import plot
from ._simulation import simulation
from ._initialization import load_parameters


def main(cl_args=sys.argv[1:]):
    """Run SPECTRAX from the command line: ``spectrax [input.toml] [--no-show]``."""
    files = [a for a in cl_args if not a.startswith("--")]
    if not files:
        print("Using standard input parameters instead of an input TOML file.")
        output = simulation()
    else:
        input_parameters, solver_parameters = load_parameters(files[0])
        output = simulation(input_parameters, **solver_parameters)
    plot(output, save=Path(files[0]).with_suffix(".png") if files else None, show="--no-show" not in cl_args)
    return output


if __name__ == "__main__":
    main(sys.argv[1:])
