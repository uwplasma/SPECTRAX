"""
Orszag–Tang analysis helpers for SPECTRAX example runs.

This module is intentionally small and example-oriented:
- energy time series (including total energy)
- relative total energy error
- Jz (out-of-plane current) slice plot at user-selected time
- Jz animation up to user-selected time (full series by default)

Assumptions (matching the current 2D_Orszag_Tang.py example):
- You saved the simulation output as a .npz containing keys like:
  "time", "Ck", "EM_energy", "kinetic_energy", "kinetic_energy_species1",
  "kinetic_energy_species2", "total_energy", ...
- Jz is computed from the real-space Hermite coefficients C via the same
  index convention used in the example script.

Typical usage from an example script:

    from orszag_tang_data_analysis import (
        load_run, plot_energy_timeseries, plot_relative_energy_error,
        plot_Jz_slice, animate_Jz
    )

    output, input_p, solver_p = load_run("output_orszag.npz", "input_2D_orszag_tang.toml")
    plot_energy_timeseries(output, savepath="energy.png")
    plot_relative_energy_error(output, savepath="energy_error.png")
    plot_Jz_slice(output, input_p, solver_p, t_query=output["time"][-1], savepath="Jz_slice.png")
    animate_Jz(output, input_p, solver_p, tmax=None, out_gif="Jz.gif")

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    # Preferred: reuse SPECTRAX's TOML parsing.
    from spectrax import load_parameters as _spectrax_load_parameters
except Exception:  # pragma: no cover
    _spectrax_load_parameters = None


Array = np.ndarray


def _to_numpy(x: Any) -> Any:
    """Best-effort conversion of JAX arrays / numpy scalars to numpy."""
    # JAX arrays typically have .__array__ implemented.
    try:
        return np.asarray(x)
    except Exception:
        return x


def load_output_npz(output_npz: str) -> Dict[str, Any]:
    """
    Load a SPECTRAX run output from a .npz (created via np.savez(**output)).

    Returns a plain dict with numpy arrays/scalars.
    """
    data = np.load(output_npz, allow_pickle=True)
    out: Dict[str, Any] = {}
    for k in data.files:
        out[k] = _to_numpy(data[k])
    return out


def load_run(output_npz: str, toml_file: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load output (.npz) and (optionally) input+solver parameters from the TOML file.

    Returns:
        output, input_parameters, solver_parameters

    If toml_file is None, returns empty dicts for parameters.
    """
    output = load_output_npz(output_npz)

    if toml_file is None:
        return output, {}, {}

    if _spectrax_load_parameters is None:
        raise ImportError(
            "Could not import spectrax.load_parameters. "
            "Either install/activate SPECTRAX or call load_run(..., toml_file=None)."
        )

    input_parameters, solver_parameters = _spectrax_load_parameters(toml_file)
    return output, input_parameters, solver_parameters


def _nearest_time_index(t: Array, t_query: float) -> int:
    """Return the index of t closest to t_query."""
    t = np.asarray(t)
    return int(np.argmin(np.abs(t - float(t_query))))


def _frame_stop_index(t: Array, tmax: Optional[float]) -> int:
    """
    Return exclusive stop index for frames up to tmax.
    If tmax is None -> full length.
    """
    t = np.asarray(t)
    if tmax is None:
        return int(t.shape[0])
    # include frames with t <= tmax
    return int(np.searchsorted(t, float(tmax), side="right"))


def plot_energy_timeseries(
    output: Dict[str, Any],
    *,
    ax: Optional[plt.Axes] = None,
    logy: bool = True,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot EM energy, kinetic energy (total + per species if available), and total energy vs time.
    """
    t = np.asarray(output["time"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Required series
    ax.plot(t, np.asarray(output["EM_energy"]), label="EM energy")
    ax.plot(t, np.asarray(output["kinetic_energy"]), label="Kinetic energy")
    # Optional series
    if "kinetic_energy_species1" in output:
        ax.plot(t, np.asarray(output["kinetic_energy_species1"]), label="Kinetic energy species 1")
    if "kinetic_energy_species2" in output:
        ax.plot(t, np.asarray(output["kinetic_energy_species2"]), label="Kinetic energy species 2")
    ax.plot(t, np.asarray(output["total_energy"]), label="Total energy")

    if logy:
        ax.set_yscale("log")

    ax.set_xlabel(r"$t\,\omega_{pe}$")
    ax.set_ylabel("Energy")
    if title is not None:
        ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def plot_relative_energy_error(
    output: Dict[str, Any],
    *,
    ax: Optional[plt.Axes] = None,
    logy: bool = True,
    eps: float = 1e-12,
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot relative total energy error vs time:
        |E(t) - E(0)| / (|E(0)| + eps)
    """
    t = np.asarray(output["time"])
    E = np.asarray(output["total_energy"])
    rel = np.abs(E - E[0]) / (np.abs(E[0]) + eps)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(t, rel, label="Relative total energy error")

    if logy:
        ax.set_yscale("log")

    ax.set_xlabel(r"$t\,\omega_{pe}$")
    ax.set_ylabel("Relative energy error")
    if title is not None:
        ax.set_title(title)

    ax.legend()
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def _ifft_Ck_frame(Ck_frame: Array) -> Array:
    """
    Inverse FFT (k->x) for one time frame.

    Matches the example:
        C = ifftn(ifftshift(Ck, axes=(-3,-2,-1)), axes=(-3,-2,-1), norm="forward").real
    """
    # Ensure complex dtype
    Ck_frame = np.asarray(Ck_frame)
    shifted = np.fft.ifftshift(Ck_frame, axes=(-3, -2, -1))
    C_frame = np.fft.ifftn(shifted, axes=(-3, -2, -1), norm="forward").real
    return C_frame


def _compute_Jz_from_C_frame(
    C_frame: Array,
    input_parameters: Dict[str, Any],
    solver_parameters: Dict[str, Any],
    *,
    iz: int = 0,
) -> Array:
    """
    Compute Jz(x,y) at a single time frame given C(x,y,z) Hermite coefficients.

    Uses the same coefficient indexing convention as the 2D_Orszag_Tang example.
    """
    alpha_s = np.asarray(input_parameters["alpha_s"])
    Nn = int(solver_parameters["Nn"])
    Nm = int(solver_parameters["Nm"])
    Np = int(solver_parameters["Np"])

    # Species layout (as used in the example):
    # - electrons occupy [0 : Nn*Nm*Np)
    # - ions occupy [Nn*Nm*Np : 2*Nn*Nm*Np)
    base_i = Nn * Nm * Np

    # C_frame shape: (ncoeff, Nx, Ny, Nz) or (ncoeff, Nx, Ny) depending on storage.
    # The example indexes C[:, coeff, :, :, 0] after building C with axes (-3,-2,-1),
    # so we handle both shapes.
    if C_frame.ndim == 4:
        # (ncoeff, Nx, Ny, Nz)
        Ce_p1 = C_frame[Nn * Nm, :, :, iz]
        Ci_p1 = C_frame[base_i + Nn * Nm, :, :, iz]
    elif C_frame.ndim == 3:
        # (ncoeff, Nx, Ny) - assume already 2D
        Ce_p1 = C_frame[Nn * Nm, :, :]
        Ci_p1 = C_frame[base_i + Nn * Nm, :, :]
    else:
        raise ValueError(f"Unexpected C_frame.ndim={C_frame.ndim}. Expected 3 or 4.")

    Jz = (1.0 / np.sqrt(2.0)) * (
        (alpha_s[3] * alpha_s[4] * (alpha_s[5] ** 2.0)) * Ci_p1
        - (alpha_s[0] * alpha_s[1] * (alpha_s[2] ** 2.0)) * Ce_p1
    )
    return Jz


def compute_Jz_slice(
    output: Dict[str, Any],
    input_parameters: Dict[str, Any],
    solver_parameters: Dict[str, Any],
    *,
    t_query: Optional[float] = None,
    index: Optional[int] = None,
) -> Tuple[Array, float, int]:
    """
    Compute Jz(x,y) at a single time slice.

    Provide either:
      - t_query (physical time): nearest time index will be used, or
      - index (int): direct time index

    Returns:
        Jz_2d, t_used, idx_used
    """
    t = np.asarray(output["time"])
    Ck = np.asarray(output["Ck"])

    if index is None and t_query is None:
        index = int(t.shape[0] - 1)
    elif index is None:
        index = _nearest_time_index(t, float(t_query))
    else:
        index = int(index)

    C_frame = _ifft_Ck_frame(Ck[index])
    Jz = _compute_Jz_from_C_frame(C_frame, input_parameters, solver_parameters)
    return Jz, float(t[index]), index


def plot_Jz_slice(
    output: Dict[str, Any],
    input_parameters: Dict[str, Any],
    solver_parameters: Dict[str, Any],
    *,
    t_query: Optional[float] = None,
    index: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "jet",
    title: Optional[str] = None,
    cbar_label: str = r"$J_z$",
    savepath: Optional[str] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot a 2D colormap of Jz at a user-selected time slice.
    """
    Jz, t_used, idx_used = compute_Jz_slice(
        output, input_parameters, solver_parameters, t_query=t_query, index=index
    )

    Lx = float(input_parameters["Lx"])
    Ly = float(input_parameters["Ly"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    im = ax.imshow(
        Jz,
        origin="lower",
        extent=(0.0, Lx, 0.0, Ly),
        interpolation="none",
        aspect="auto",
        cmap=cmap,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlabel(r"$x/d_e$")
    ax.set_ylabel(r"$y/d_e$")

    if title is None:
        title = rf"$J_z$ at $t={t_used:.3g}$ (index {idx_used})"
    ax.set_title(title)

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def animate_Jz(
    output: Dict[str, Any],
    input_parameters: Dict[str, Any],
    solver_parameters: Dict[str, Any],
    *,
    tmax: Optional[float] = None,
    out_gif: str = "Jz.gif",
    fps: int = 5,
    interval_ms: int = 50,
    cmap: str = "jet",
    fixed_clim: bool = True,
    precompute: bool = True,
) -> str:
    """
    Create an animation (GIF) of Jz(x,y) vs time.

    Args:
        tmax: animate up to this time (inclusive). If None, uses full series.
        out_gif: output path for GIF.
        fps: frames per second for PillowWriter.
        interval_ms: matplotlib animation interval (ms).
        fixed_clim: if True, fix color limits across frames (recommended for interpretability).
        precompute: if True, precompute Jz for all frames (up to tmax) to:
            - make animation smooth
            - enable fixed clim without a separate scan pass
          If False, Jz is computed on the fly per frame (lower memory, slower).

    Returns:
        Path to the written GIF (out_gif).
    """
    t = np.asarray(output["time"])
    Ck = np.asarray(output["Ck"])
    stop = _frame_stop_index(t, tmax)
    nframes = stop

    Lx = float(input_parameters["Lx"])
    Ly = float(input_parameters["Ly"])

    if nframes <= 0:
        raise ValueError("No frames selected for animation (check tmax).")

    # Precompute frames if requested
    Jz_series: Optional[Array] = None
    vmin = vmax = None

    if precompute:
        Jz_list = []
        for i in range(nframes):
            C_frame = _ifft_Ck_frame(Ck[i])
            Jz_i = _compute_Jz_from_C_frame(C_frame, input_parameters, solver_parameters)
            Jz_list.append(Jz_i)
        Jz_series = np.stack(Jz_list, axis=0)  # (t, Nx, Ny)

        if fixed_clim:
            vmin = float(np.min(Jz_series))
            vmax = float(np.max(Jz_series))

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel(r"$x/d_e$")
    ax.set_ylabel(r"$y/d_e$")

    # Initial frame
    if precompute:
        J0 = Jz_series[0]
    else:
        C0 = _ifft_Ck_frame(Ck[0])
        J0 = _compute_Jz_from_C_frame(C0, input_parameters, solver_parameters)

    im = ax.imshow(
        J0,
        origin="lower",
        extent=(0.0, Lx, 0.0, Ly),
        interpolation="nearest",
        aspect="auto",
        cmap=cmap,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$J_z$")

    title_obj = ax.set_title(f"t = {t[0]:.6g}")

    if fixed_clim and vmin is not None and vmax is not None:
        im.set_clim(vmin=vmin, vmax=vmax)

    def _get_frame(i: int) -> Array:
        if precompute:
            return Jz_series[i]
        C_frame = _ifft_Ck_frame(Ck[i])
        return _compute_Jz_from_C_frame(C_frame, input_parameters, solver_parameters)

    def update(i: int):
        nonlocal vmin, vmax   # ← must be first

        Ji = _get_frame(i)
        im.set_data(Ji)
        title_obj.set_text(f"t = {t[i]:.6g}")

        if fixed_clim and (vmin is None or vmax is None):
            jmin = float(np.min(Ji))
            jmax = float(np.max(Ji))
            vmin = jmin if vmin is None else min(vmin, jmin)
            vmax = jmax if vmax is None else max(vmax, jmax)
            im.set_clim(vmin=vmin, vmax=vmax)
            cbar.update_normal(im)

        return (im, title_obj)

    anim = FuncAnimation(fig, update, frames=nframes, interval=interval_ms, blit=False)

    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_gif


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Analyze a SPECTRAX Orszag–Tang run (.npz).")
    parser.add_argument("output_npz", help="Path to output_orszag.npz")
    parser.add_argument("--toml", required=True, help="Path to input_2D_orszag_tang.toml used for the run.")
    parser.add_argument("--t_slice", type=float, default=None, help="Time for Jz slice (nearest frame). Default: last frame.")
    parser.add_argument("--tmax", type=float, default=None, help="Max time for Jz animation. Default: full time series.")
    parser.add_argument("--energy_png", default="energy.png", help="Output path for energy time series plot.")
    parser.add_argument("--error_png", default="energy_error.png", help="Output path for relative energy error plot.")
    parser.add_argument("--Jz_png", default="Jz_slice.png", help="Output path for Jz slice plot.")
    parser.add_argument("--Jz_gif", default="Jz.gif", help="Output path for Jz animation GIF.")
    parser.add_argument("--fps", type=int, default=5, help="FPS for GIF.")
    args = parser.parse_args()

    out, inp, solv = load_run(args.output_npz, args.toml)
    plot_energy_timeseries(out, savepath=args.energy_png, show=False)
    plot_relative_energy_error(out, savepath=args.error_png, show=False)
    plot_Jz_slice(out, inp, solv, t_query=args.t_slice, savepath=args.Jz_png, show=False)
    animate_Jz(out, inp, solv, tmax=args.tmax, out_gif=args.Jz_gif, fps=args.fps)
    print("Wrote:", args.energy_png, args.error_png, args.Jz_png, args.Jz_gif)
