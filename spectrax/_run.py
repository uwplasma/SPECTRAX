"""One-call interface: ``run("input.toml", **overrides)`` runs a case and acts on its ``[output]`` table.

The ``[output]`` table of the input file sets what happens to the result (all keys optional)::

    [output]
    figure = "case.png"   # diagnostic figure; default input.png next to the file; false for none
    show = true           # open the figure window
    save = "case.npz"     # write the numeric results (all of them, or only those in ``keep``)
    keep = ["time", "electric_energy"]
    summary = true        # print the one-screen summary

Any key, input or output, can be overridden by keyword (or ``key=value`` on the command line); a
parameter scan is a loop over ``run``.
"""
try: import tomllib
except ModuleNotFoundError: import tomli as tomllib
from pathlib import Path
import numpy as np
from ._initialization import load_parameters
from ._plot import plot
from ._simulation import simulation

__all__ = ["run", "Result"]


class Result(dict):
    """Simulation output (a dictionary of arrays) with ``plot``, ``save`` and ``summary`` methods."""

    def plot(self, save=None, show=True):
        return plot(self, save=save, show=show)

    def save(self, path, keep=None):
        """Write the numeric entries (or only those named in ``keep``) to the ``.npz`` file ``path``."""
        data = {}
        for key in keep or self:
            try:
                data[key] = np.asarray(self[key])
            except (TypeError, ValueError):
                continue
        np.savez(path, **{k: v for k, v in data.items() if v.dtype != object})

    def summary(self):
        """Print and return the headline numbers of the run."""
        W, t = np.asarray(self["total_energy"]), np.asarray(self["time"])
        numbers = {"t_max": float(t[-1]), "saved_times": len(t), "max_energy_error": float(np.max(np.abs(W / W[0] - 1))),
                   "peak_field_energy": float(np.max(np.asarray(self["EM_energy"]))), "solver_steps": int(self["solver_stats"]["num_steps"])}
        print("  ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in numbers.items()))
        return numbers


def run(input_file, **overrides):
    """Run the case in ``input_file`` and return a :class:`Result`; ``overrides`` replace file values.

    Output keys (``figure``, ``show``, ``save``, ``keep``, ``summary``) are overridden the same way as
    input ones, and species entries as ``species={"electrons": {"vth": 0.2}}``.
    """
    path = Path(input_file)
    with open(path, "rb") as f:
        out = {"figure": path.with_suffix(".png"), "show": True, "summary": True, **tomllib.load(f).get("output", {})}
    out.update({key: overrides.pop(key) for key in tuple(out) + ("keep",) if key in overrides})
    input_parameters, solver_parameters = load_parameters(path, **overrides)
    result = Result(simulation(input_parameters, **solver_parameters))
    if out["summary"]:
        result.summary()
    if out.get("save"):
        result.save(path.parent / out["save"], out.get("keep"))
    if out["figure"] or out["show"]:
        result.plot(save=path.parent / out["figure"] if out["figure"] else None, show=out["show"])
    return result
