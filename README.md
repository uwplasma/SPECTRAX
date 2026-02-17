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

- [ Overview](#-overview)
- [ Mathematical Method](#-background)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
    - [ Command‑line Interface](#-command-line-interface)
    - [ Running from Python](#-running-from-python)
  - [ Testing](#-testing)
- [ Input File Format](#-input-file-format)
- [ Project Roadmap](#-project-roadmap)
- [ How to Cite](#-how-to-cite)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

**SPECTRAX** is an open-source spectral kinetic plasma solver written in Python with the [JAX](https://github.com/jax-ml/jax) ecosystem. It solves the collisionless Vlasov–Maxwell equations by evolving the Hermite–Fourier coefficients of the particle distribution function and the electromagnetic fields. The approach builds on the *SpectralPlasmaSolver* (SPS) algorithm developed at Los Alamos National Laboratory and described in [Delzanno (2025)](https://www.sciencedirect.com/science/article/pii/S0021999115004738), [Vencels et al. (2016)](https://iopscience.iop.org/article/10.1088/1742-6596/719/1/012022) and [Roytershteyn & Delzanno (2018)](https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2018.00027/full), where the one particle distribution is expanded in Hermite functions in velocity space and Fourier modes in configuration space. By performing a Hermite expansion in velocity space, the method naturally couples fluid and kinetic physics—the lowest‐order Hermite coefficients correspond to fluid moments and higher modes capture kinetic corrections.

SPECTRAX re‑implements this algorithm in a JAX framework. It uses just‑in‑time compilation to run efficiently on CPUs, GPUs or TPUs, adopts state‑of‑the‑art ODE solvers from the [Diffrax](https://github.com/patrick-kidger/diffrax) library, and includes utilities for diagnostics and plotting. The code supports multi‑species plasmas and arbitrary spatial dimensionality (1D to 3D) and can serve as a test bed for studying kinetic instabilities, turbulence, and the transition between fluid and kinetic regimes.


---

##  Mathematical Method

The Hermite–Fourier spectral method replaces the Vlasov equation in the 6-dimensional phase-space with a hierarchy of coupled ordinary differential equations (ODEs) for the Hermite–Fourier moments of the one-particle probability density function and the electric and magnetic fields. The Hermite-Fourier expansion is truncated for closure and a hypercollisional operator is introduced in velocity space to suppress recurrence. The resulting system of ODEs is integrated in time using solvers from the [Diffrax](https://github.com/patrick-kidger/diffrax) library. Diffrax's implementation of the Dormand-Prince’s 8/7 method, `Dopri8`, proved to be notoriously fast and stable, and is set as the default solver.

---

##  Features

* **JAX‑based spectral solver** – all core operations are implemented in JAX and compiled with `jit`, enabling efficient execution on CPUs, GPUs, or TPUs.

* **Efficient time integration** – SPECTRAX uses ODE solvers from the Diffrax library (e.g., `Dopri5`, `Dopri8`, `Tsit5`; a Diffrax-based, custom-made implicit midpoint solver is also available as `ImplicitMidpoint`) to advance the Hermite–Fourier coefficients in time. The `simulation` function assembles the right‑hand‑side, applies a 2⁄3 de‑aliasing mask on Fourier modes in the nonlinear term, and integrates the system until `t_max`, returning the time‑evolved coefficients.

* **Multi‑species and multi‑dimensional** – the code supports multiple particle species with distinct mass ratios, temperatures and drift velocities. Resolution in spatial dimensions is controlled via `Nx`, `Ny`, `Nz`, and velocity Hermite orders via `Nn`, `Nm`, `Np`.

* **Diagnostics** – after each simulation the `diagnostics` function computes the Debye length, normalized wavenumber, kinetic energies of each species, electromagnetic energy and total energy and stores them in the output dictionary.

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

JAX will automatically select the available hardware (CPU, GPU or TPU). For GPU support you may need the
appropriate CUDA-enabled version of `jaxlib`; consult the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

---

###  Usage

SPECTRAX can be used either via a command‑line interface or directly as a Python module.

#### Command‑line Interface

After installation, the `spectrax` CLI entry point is available. You can run an instance of the 1D two-stream instability by simply calling `spectrax` from the terminal.

```sh
spectrax
```

To run it with different input parameters, us a TOML file like those in the `Examples` directory.

```sh
spectrax example_input.toml
```

Other examples written in Python scripts, like those in the `Examples` directory, can be executed from the terminal as follows:

```sh
python example_script.py
```


The `simulation` function returns a dictionary containing the evolved Hermite coefficients `Ck`, electromagnetic coefficients `Fk`, time array, the input parameters and diagnostic quantities.

<!--
#### Running from Python

You can import SPECTRAX in your own scripts. A typical workflow is:

```sh
from spectrax import load_parameters, simulation, plot

# Read parameters from a TOML file
input_params, solver_params = load_parameters('Examples/
input_1D_two_stream.toml')

# (Optional) modify or add initial spectral coefficients here
output = simulation(input_params, **solver_params)

plot(output)
```
-->



###  Testing
Run the test suite using the following command:
```sh
pytest .
```

---

## Input File Format

Input files are written in TOML and define both physical and solver parameters. Below is a summary of the
most important keys. Keys absent from the file fall back to sensible defaults specified in the code.

| Parameter | Description |
|---|---|
| `Lx, Ly, Lz` | Domain lengths in the spatial directions (periodic boundary condition). |
| `mi_me` | Ion‑to‑electron mass ratio. |
| `Ti_Te` | Ion‑to‑electron temperature ratio. |
| `qs` | Array of species charges. |
| `alpha_s` | Thermal scales for each species for Hermite basis. |
| `u_s` | Drift velocities for each species for Hermite basis (packed as `[u_x,u_y,u_z]` ). |
| `Omega_cs` | Cyclotron frequencies for each species. |
| `nu` | Hyper-collision frequency to damp recurrence. |
| `D` | Hyper‑diffusion coefficient. |
| `t_max` | Final simulation time. |
| `nx, ny, nz` | Mode numbers used to seed sinusoidal perturbations (see examples). |
| `dn1, dn2, dE` | Amplitudes of initial density or field perturbations (see examples). |
| `ode_tolerance` | Relative/absolute tolerance for adaptive solvers. |
| `Nx, Ny, Nz` | Number of retained Fourier modes per spatial dimension. |
| `Nn, Nm, Np` | Number of Hermite modes per velocity dimension. |
| `Ns` | Number of species (two by default). |
| `timesteps` | Number of solution snapshots to store between `t=0` and `t_max`. |
| `dt` | Initial step size provided to the ODE solver. |
| `solver` | Name of Diffrax solver (e.g., `Tsit5`, `Dopri5`, `Dopri8`, `ImplicitMidpoint`). |
| `adaptive_time_step` | Timestep adaptability (`true` by default). |

Many of these parameters can be arrays; for example `alpha_s` must contain three values per species (one
for each velocity dimension) and can be used to represent anisotropic plasmas.

---


<!-- ##  Project Roadmap

- [X] **`Task 1`**: <strike>Task 1.</strike>
- [ ] **`Task 2`**: Task 2.
- [ ] **`Task 3`**: Task 3.

---

## How to Cite



--- -->

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





