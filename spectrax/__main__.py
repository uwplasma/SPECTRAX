"""Command line: ``spectrax input.toml [--no-show] [key=value ...]`` runs the case, saves the figure and prints a summary.

``key=value`` overrides a value of the file (TOML syntax), e.g. ``Lx=2.0 Nn=64 species.electrons.vth=0.2 save=out.npz``.
"""
import sys
try: import tomllib
except ModuleNotFoundError: import tomli as tomllib
from ._run import run


def main(cl_args=sys.argv[1:]):
    files = [a for a in cl_args if "=" not in a and not a.startswith("--")]
    if not files:
        sys.exit("usage: spectrax input.toml [--no-show] [key=value ...]")
    overrides = {}
    for argument in (a for a in cl_args if "=" in a):
        key, value = argument.split("=", 1)
        try:
            value = tomllib.loads(f"v = {value}")["v"]
        except tomllib.TOMLDecodeError:
            pass                                     # a bare word such as case.npz is a string
        *path, leaf = key.split(".")
        target = overrides
        for part in path:                            # species.electrons.vth=0.2 -> {"species": {"electrons": {"vth": 0.2}}}
            target = target.setdefault(part, {})
        target[leaf] = value
    return run(files[0], **({"show": False} if "--no-show" in cl_args else {}), **overrides)


if __name__ == "__main__":
    main(sys.argv[1:])
