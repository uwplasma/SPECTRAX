"""Main command line interface to SPECTRAX."""
import argparse
import sys

def main(cl_args=sys.argv[1:]):
    """Run the main SPECTRAX code from the command line.

    ``spectrax [input.toml] [--no-plot] [--no-progress]``: reads the input (which configures the devices before
    JAX starts), runs the simulation, then plots it or, with ``--no-plot``, prints a short text summary.
    """
    parser = argparse.ArgumentParser(prog="spectrax", description=main.__doc__.split("\n")[0])
    parser.add_argument("input", nargs="?", help="TOML input file (default: standard input parameters)")
    parser.add_argument("--no-plot", action="store_true", help="print a text summary instead of plotting")
    parser.add_argument("--no-progress", action="store_true", help="disable the progress meter")
    args = parser.parse_args(cl_args)
    if args.input:
        from ._initialization import load_parameters
        input_parameters, solver_parameters = load_parameters(args.input)
    else:
        print("Using standard input parameters instead of an input TOML file.")
        input_parameters, solver_parameters = {}, {}
    if args.no_progress:
        import diffrax
        solver_parameters["progress_meter"] = diffrax.NoProgressMeter()
    from ._simulation import simulation
    output = simulation(input_parameters, **solver_parameters)
    if args.no_plot:
        mesh = output["Ck"].sharding.mesh
        print(f"Devices: {mesh.devices.size} {mesh.devices.flat[0].platform}, mesh {dict(mesh.shape)}, {int(output['num_steps'])} steps")
        print(f"t = {float(output['time'][-1]):.6g}: total energy = {float(output['total_energy'][-1]):.6e}, "
              f"kinetic = {float(output['kinetic_energy'][-1]):.6e}, EM = {float(output['EM_energy'][-1]):.6e}")
    else:
        from ._plot import plot
        plot(output)

if __name__ == "__main__":
    main(sys.argv[1:])
