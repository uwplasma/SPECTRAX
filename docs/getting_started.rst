Getting started
===============

.. _installation:

Installation
------------

.. code-block:: console

    $ git clone https://github.com/uwplasma/SPECTRAX.git
    $ cd SPECTRAX
    $ pip install -e .

Run a case
----------

A case is one TOML file that describes the box, the resolution and each species physically:

.. code-block:: toml

    [input_parameters]
    Lx = 1.0
    t_max = 50.0
    nu = 1.0

    [solver_parameters]
    Nx = 4
    Nn = 40

    [species.electrons]
    charge = -1
    vth = 0.1
    perturbation_amplitude = 0.001

    [species.protons]
    charge = 1
    mass = 1836
    vth = 0.0023

.. code-block:: console

    $ spectrax input.toml

This runs the simulation and saves ``input.png`` with the automatic diagnostic figure: field and kinetic
energy exchange, energy conservation, the strongest field modes, the Hermite spectrum and the phase-space
perturbation of each species. Units are 1/omega_pe for time, c/omega_pe for length and c for velocity. The
full list of keys is in the README.

From Python
-----------

.. code-block:: python

    from spectrax import load_parameters, simulation, plot

    input_parameters, solver_parameters = load_parameters("input.toml", Lx=2.0)   # overrides for scans
    output = simulation(input_parameters, **solver_parameters)
    plot(output, save="run.png")

``simulation`` is differentiable with JAX, so ``jax.grad`` of any scalar built from ``output`` with
respect to inputs gives exact sensitivities.

Examples
--------

The ``Examples`` folder has Landau damping and the two-stream instability as TOML files, scans of their
damping and growth rates against kinetic theory, and the 2D Orszag-Tang vortex.
