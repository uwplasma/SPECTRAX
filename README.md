<p align="center">
    <img src="https://raw.githubusercontent.com/uwplasma/SPECTRAX/refs/heads/main/docs/SPECTRAX_logo.png" align="center" width="30%">
</p>
<!-- <p align="center"><h3 align="center">SPECTRAX</h1></p> -->
<p align="center">
	<em><code>❯ SPECTRAX: Hermite-Fourier Vlasov-Maxwell solver in JAX for plasma physics simulations</code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/uwplasma/SPECTRAX?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/uwplasma/SPECTRAX?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/uwplasma/SPECTRAX?style=default&color=0080ff" alt="repo-top-language">
	<a href="https://github.com/uwplasma/SPECTRAX/actions/workflows/build_test.yml">
		<img src="https://github.com/uwplasma/SPECTRAX/actions/workflows/build_test.yml/badge.svg" alt="Build Status">
	</a>
	<a href="https://codecov.io/gh/uwplasma/SPECTRAX">
		<img src="https://codecov.io/gh/uwplasma/SPECTRAX/branch/main/graph/badge.svg" alt="Coverage">
	</a>
	<a href="https://spectrax.readthedocs.io/en/latest/?badge=latest">
		<img src="https://readthedocs.org/projects/spectrax/badge/?version=latest" alt="Documentation Status">
	</a>

</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>


##  Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Mathematical Method](#mathematical-method)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Command‑line Interface](#commandline-interface)
  - [Testing](#testing)
- [Input File Format](#input-file-format)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

##  Overview

**SPECTRAX** is an open-source spectral kinetic plasma solver written in Python with the [JAX](https://github.com/jax-ml/jax) ecosystem. It solves the collisionless Vlasov–Maxwell equations by evolving the Hermite–Fourier coefficients of the particle distribution function and the electromagnetic fields. This approach was previously implemented in *SpectralPlasmaSolver* (SPS), developed at Los Alamos National Laboratory, and described in [Delzanno (2025)](https://www.sciencedirect.com/science/article/pii/S0021999115004738), [Vencels et al. (2016)](https://iopscience.iop.org/article/10.1088/1742-6596/719/1/012022) and [Roytershteyn & Delzanno (2018)](https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2018.00027/full), where the one particle distribution is expanded in asymmetrically weighted (AW) Hermite functions in velocity space and Fourier modes in configuration space. By performing an AW Hermite expansion in velocity space, the method naturally couples fluid and kinetic physics: the lowest‐order Hermite coefficients correspond to fluid moments and higher modes capture kinetic corrections.

SPECTRAX implements a Hermite–Fourier spectral approach for Vlasov–Maxwell simulations in a JAX framework. It uses just‑in‑time compilation to run efficiently on CPUs and GPUs, adopts state‑of‑the‑art ODE solvers from the [Diffrax](https://github.com/patrick-kidger/diffrax) library, and includes utilities for diagnostics and plotting. The code supports multi‑species plasmas and arbitrary spatial dimensionality (1D to 3D) and can serve as a test bed for studying kinetic instabilities, turbulence, and the transition between fluid and kinetic regimes.


---

##  Mathematical Method

The Hermite–Fourier spectral method replaces the Vlasov equation in the 6-dimensional phase-space with a hierarchy of coupled ordinary differential equations (ODEs) for the Hermite–Fourier moments of the one-particle probability density function and the electric and magnetic fields. The Hermite-Fourier expansion is truncated for closure and a hypercollisional operator is introduced in velocity space to suppress recurrence. The resulting system of ODEs is integrated in time using solvers from the [Diffrax](https://github.com/patrick-kidger/diffrax) library. Diffrax's implementation of the Dormand-Prince’s 8/7 method, `Dopri8`, proved to be notoriously fast and stable, and is set as the default solver.

---

##  Features

* **JAX‑based spectral solver** – all core operations are implemented in JAX and compiled with `jit`, enabling efficient execution on CPUs and GPUs.

* **Efficient time integration** – SPECTRAX uses ODE solvers from the Diffrax library (e.g., `Dopri5`, `Dopri8`, `Tsit5`; a Diffrax-based, custom-made implicit midpoint solver is also available as `ImplicitMidpoint`) to advance the Hermite–Fourier coefficients in time. The `simulation` function assembles the right‑hand‑side, applies a 2⁄3 de‑aliasing mask on Fourier modes in the nonlinear term, and integrates the system until `t_max`, returning the time‑evolved coefficients.

* **Multi‑species and multi‑dimensional** – the code supports multiple particle species with distinct mass ratios, temperatures and drift velocities. Resolution in spatial dimensions is controlled via `Nx`, `Ny`, `Nz`, and velocity Hermite orders via `Nn`, `Nm`, `Np`.

* **Diagnostics** – after each simulation the `diagnostics` function computes the Debye length, kinetic energies of each species, electromagnetic energy and total energy and stores them in the output dictionary.

* **Plotting utilities** – the `plot` function produces a multi‑panel figure showing energy evolution,
relative energy error, density fluctuations and phase‑space distributions for each species. It reconstructs the distribution function by performing an inverse Fourier transform followed by an inverse Hermite transform. The phase‑space reconstruction uses the `inverse_HF_transform` function, which evaluates Hermite polynomials and sums over all modes. The phase-space plots assume a 1D simulation.

* **Flexible initialization** – simulation parameters may be provided through simple TOML files or directly in Python. The `load_parameters` function reads a TOML file and merges it with sensible defaults that initialize a two‑stream instability. Users can also initialize their own spectral coefficients, as shown in the example scripts.

* **Open source and extensible** – SPECTRAX is released under the MIT License. Its modular structure
allows researchers to experiment with new closures, collision operators or boundary conditions.

---


<!-- ##  Project Structure

```sh
└── SPECTRAX/
    ├── LICENSE
    ├── docs
    ├── examples
    │   ├── 1D_two-stream.py
    │   ├── 1D_landau_damping.py
    │   └── 2D_orszag_tang.py
    ├── spectrax
    │   ├── file1.py
    │   ├── file2.py
    │   └── file3.py
    └── tests
        └── test1.py
``` 

---
-->

##  Getting Started

###  Prerequisites

- **Programming Language:** Python

Besides Python, SPECTRAX has minimum requirements. These are stated in [requirements.txt](requirements.txt), and consist of the Python libraries `jax`, `jax_tqdm` and `matplotlib`.

### Installation

SPECTRAX is a standard Python package that may be installed from a local checkout. The project depends
on `jax`, `jaxlib`, `jax_tqdm`, `diffrax`, `orthax`, and `matplotlib`.

1. Clone this repository:

```sh
git clone https://github.com/uwplasma/SPECTRAX.git
cd SPECTRAX
```

2. (Optional) create a virtual environment and activate it.

3. Install the package in editable mode:

```sh
pip install -r requirements.txt
pip install -e .
```

JAX will automatically select the available hardware (CPU or GPU). For GPU support you may need the
appropriate CUDA-enabled version of `jaxlib`; consult the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

---

###  Usage

A case is one TOML file. Run it with

```sh
spectrax Examples/input_1D_two_stream.toml
```

This runs the simulation, saves `Examples/input_1D_two_stream.png` next to the input and shows it (add `--no-show` for batch runs). `spectrax` with no file runs a default two-stream case.

The same thing from Python, for scans or custom analysis:

```python
from spectrax import load_parameters, simulation, plot

input_parameters, solver_parameters = load_parameters("input.toml", Lx=2.0)   # optional overrides
output = simulation(input_parameters, **solver_parameters)
plot(output, save="run.png")
```

`output` is a dictionary with the Hermite–Fourier coefficients `Ck`, the field coefficients `Fk`, `time`, the energies (`electric_energy`, `magnetic_energy`, `kinetic_energy_species`, `total_energy`) and every input parameter.

### The automatic figure

`plot(output)` works for any number of species in 1D, 2D or 3D and shows:

* the electric and magnetic field energy and the change in kinetic energy of each species;
* energy conservation, |W(t)/W(0) − 1|;
* the amplitude of the strongest electric-field modes, labelled by their wavevector;
* the Hermite spectrum of each species against time, which shows phase mixing, recurrence and whether the closure absorbs what reaches the last Hermite mode;
* the phase-space perturbation f − ⟨f⟩ₓ of each species at the last saved time;
* in 2D and 3D, the strongest field component in the plane.

`spectrax.phase_space(output, species, time_index)` returns the reconstructed f(x, v) for custom figures.

###  Testing
Run the test suite using the following command:
```sh
pytest .
```

---

## Input File Format

Units: time in 1/ω_pe, length in c/ω_pe, velocity in c, density in the reference density that defines ω_pe.

```toml
[input_parameters]
Lx = 12.57            # box lengths Lx, Ly, Lz (periodic)
t_max = 50.0          # final time
nu = 3.0              # high-Hermite sink, closes the Hermite hierarchy (not a collision rate)
ode_tolerance = 1e-7  # adaptive time-step tolerance
Omega_ce = 1.0        # optional: electron gyrofrequency of the reference field over omega_pe (sets field units)
B0 = [0.0, 0.0, 0.0]  # optional: uniform background magnetic field in those units

[solver_parameters]
Nx = 33               # Fourier modes in x, y, z: Nx, Ny, Nz (default 1 for y and z)
Nn = 50               # Hermite modes in vx, vy, vz: Nn, Nm, Np
timesteps = 501       # saved times
solver = "Dopri8"     # any Diffrax solver, or ImplicitMidpoint

[species.electrons]   # one table per species; the first one must have unit mass
charge = -1           # in units of e
mass = 1              # in units of m_e
density = 1.0
vth = 0.707           # sqrt(2 T / m) / c, a number or [vx, vy, vz]; also the Hermite width
drift = 1.0           # mean velocity / c, a number (along x) or [ux, uy, uz]
perturbation_amplitude = 0.01   # density * (1 + A cos(2 pi m x / L))
perturbation_mode = 1           # m
perturbation_axis = "x"         # "x", "y" or "z"
```

The initial electric field is computed from Gauss's law, so the state is consistent at t = 0. Cases that are not described by species densities, such as the Orszag–Tang vortex, can still pass `qs`, `alpha_s`, `u_s`, `Omega_cs`, `Ck_0` and `Fk_0` directly, as in `Examples/2D_Orszag_Tang.py`.

**Choosing `nu` and `Nn`.** Landau damping moves free energy to high Hermite order. The sink must absorb it before it reaches the last mode, or it reflects as recurrence. Check the Hermite-spectrum panel: the energy should fall by many orders of magnitude before the last mode. For ion waves with hot electrons, the electron resonance lies deep inside the truncated spectrum, and the sink also supplies electron Landau damping; `nu` of order 1 ω_pe with 128 or more Hermite modes reproduces exact kinetic growth rates.

---


##  Contributing

- **💬 [Join the Discussions](https://github.com/uwplasma/SPECTRAX/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/uwplasma/SPECTRAX/issues)**: Submit bugs found or log feature requests for the `SPECTRAX` project.
- **💡 [Submit Pull Requests](https://github.com/uwplasma/SPECTRAX/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/uwplasma/SPECTRAX
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/uwplasma/SPECTRAX/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=uwplasma/SPECTRAX">
   </a>
</p>
</details>

---

##  License

This project is protected under the MIT License. For more details, refer to the [LICENSE](LICENSE) file.

---

##  Acknowledgments

- We acknowledge the help of the whole [UWPlasma](https://rogerio.physics.wisc.edu/) plasma group.

---





