Multi-device parallel execution
===============================

SPECTRAX can decompose one simulation over several devices in physical space, over species, or both. The same code
(state layout, sharding strategy, kernels and time integrator) runs on CPU and GPU, and on a single device.

Scope
-----

* **One Python process controlling several LOCAL JAX devices of one host.** The devices may be logical CPU devices or the
  GPUs visible to the process.
* ``num_devices`` is the **exact number of local devices of the selected backend** that the run uses. It is not a node
  count, a process count, a number of physical CPU cores or a scheduler resource request.
* Multi-host (multi-process) execution is **not implemented and not tested**. The design (one global array per state
  leaf, named mesh axes) is meant not to prevent such an extension, but nothing here claims it works.
* CPU logical devices validate correctness and communication structure. They do not demonstrate GPU speedups, and no
  GPU benchmark is reported here unless it appears in the measured-results section below.

The ``[parallel]`` table
------------------------

Add an optional top-level table to the run TOML:

.. code-block:: toml

   [parallel]
   device = "cpu"            # "cpu", "gpu" or "auto"
   num_devices = 4           # TOTAL selected local devices
   species_partitions = 1    # ps, optional (default 1)
   mesh_shape = [2, 2, 1]    # optional SPATIAL factors [px, py, pz]

Let ``ps = species_partitions``. Omitted keys take their defaults (``device="auto"``, ``num_devices=1``, ``ps=1``); with no
table at all, one device of JAX's preferred backend is used and nothing is sharded. The configuration is rejected (with an
error naming the failed rule, never a silent adjustment) when:

* a key is not one of the four above, a value has the wrong type (booleans are not accepted as counts or factors), or a
  count/factor is not positive;
* ``device`` is not ``"cpu"``, ``"gpu"`` or ``"auto"``, the explicit backend is unavailable (an explicit ``"gpu"`` never
  falls back to CPU), or fewer than ``num_devices`` local devices exist;
* ``num_devices`` is not divisible by ``ps``, ``ps`` does not divide ``Ns`` (so ``ps <= Ns``), or
  ``num_devices != ps*px*py*pz``;
* ``px`` does not divide ``Nx``, ``py`` does not divide ``Ny`` or ``pz`` does not divide ``Nz`` (singleton axes need factor
  one).

SPECTRAX never selects fewer devices, changes the species count, enlarges the grid or pads species/cells. Whole species
are partitioned: each shard owns ``Ns/ps`` species with **all** their Hermite and DG coefficients for its spatial cells;
velocity/Hermite and DG modes are never split.

**Automatic mesh.** If ``mesh_shape`` is omitted, all valid triples with ``px*py*pz = num_devices/ps`` and the
divisibility conditions are enumerated and the one minimizing ``sum(p_i/N_i for p_i > 1)`` is chosen (a proxy for spatial
boundary traffic for the ps you chose). Ties go to fewer split directions, then to splitting x before y before z. The
rule does not compare different ``ps`` values and does not model the current all-reduce; ``ps`` is never enabled
automatically.

Four valid layouts for ``Ns = 2``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 12 14 14 42

   * - Layout
     - ``num_devices``
     - ``species_partitions``
     - ``mesh_shape``
     - Notes
   * - Spatial only
     - 4
     - 1
     - ``[2,2,1]``
     - default style
   * - Hybrid
     - 4
     - 2
     - ``[2,1,1]``
     - 2 species groups x 2 x-slabs
   * - Species only
     - 2
     - 2
     - ``[1,1,1]``
     - whole grid replicated in each species group
   * - One device
     - 1
     - 1
     - ``[1,1,1]``
     - or omit ``mesh_shape``

Running
-------

Start each run in a **fresh Python process** (the CPU device count is fixed when the JAX backend starts).

One CPU device, several CPU devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

    $ spectrax input.toml                    # no [parallel] table: one device
    $ python Examples/OT_parallel.py         # [parallel] device="cpu", num_devices=4

For ``device = "cpu"`` (or ``"auto"`` on a CPU-only host) the requested number of logical CPU devices is created
automatically before the backend starts (public ``jax_num_cpu_devices``); you do not set ``XLA_FLAGS`` or
``JAX_NUM_CPU_DEVICES`` yourself. A larger value you set yourself is never reduced, and an ``XLA_FLAGS`` host-device count is
left alone.

Several GPUs
~~~~~~~~~~~~

Install a CUDA-enabled JAX **separately** from SPECTRAX's generic requirements, following the current
`JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_ (for example ``pip install -U "jax[cuda12]"``;
the extra name depends on your CUDA version). Then set ``device = "gpu"`` and a compatible ``num_devices`` /
``species_partitions`` / ``mesh_shape``. Scheduler visibility masks such as ``CUDA_VISIBLE_DEVICES`` are respected:
``num_devices`` counts the devices that remain visible.

Programmatic use
~~~~~~~~~~~~~~~~

The devices must be configured before any JAX computation. ``load_parameters`` (or ``spectrax.bootstrap``) does this, and
``from spectrax import ...`` is lazy and does not start the backend, so the safe order is:

.. code-block:: python

    from spectrax import load_parameters, simulation   # no JAX backend started yet
    input_parameters, solver_parameters = load_parameters("input.toml")   # devices configured here
    output = simulation(input_parameters, **solver_parameters)

Do not create JAX arrays before ``load_parameters``. If the CPU backend was already initialized with too few devices, a
clear error asks for a fresh process (private reset APIs are not used). ``simulation`` is a host-side wrapper around a
cached compiled core: it is not meant to be nested inside ``jax.jit``, and it synchronizes once, on a scalar solver status,
raising ``RuntimeError`` if the solve failed (for example ``max_steps`` exceeded). Pass ``progress_meter=diffrax.NoProgressMeter()``
for headless runs. Pass ``parallel=`` (a mapping or ``ParallelConfig``) to override the layout; ``load_parameters`` already
stores the validated layout in ``solver_parameters["parallel"]``.

Orszag-Tang example
-------------------

``Examples/OT_parallel.py`` runs the Orszag-Tang vortex of ``2D_Orszag_Tang.py`` (same physics and solver settings) with
``Examples/input_OT_parallel.toml``. The only input change is ``Nx = Ny = 32`` instead of 25, so that the grid divides
evenly over 2 and 4 devices; the original 25x25 grid can still be run species-only on two devices. It is a functionality
example, **not** a scaling benchmark.

.. code-block:: console

    $ python Examples/OT_parallel.py                          # spatial: ps=1, mesh [2,2,1], 4 CPU devices
    $ python Examples/OT_parallel.py my_hybrid.toml --plot    # edit the [parallel] table, nothing else
    $ python Examples/OT_parallel.py --no-progress --output ot.npz

Edit the ``[parallel]`` table for the other layouts: hybrid ``num_devices=4, species_partitions=2, mesh_shape=[2,1,1]``;
species-only ``num_devices=2, species_partitions=2, mesh_shape=[1,1,1]``; one device ``num_devices=1,
species_partitions=1`` with ``mesh_shape=[1,1,1]`` or removed (a leftover incompatible ``mesh_shape`` is rejected). For
GPUs set ``device = "gpu"``.

The script prints the selected backend and device ids, the mesh shape and axis names, ``ps``, global and per-device
(local) shapes of ``Ck_0``/``Fk_0`` and of the saved ``Ck``/``Fk``, synchronized timings and the number of integration
steps. ``--plot`` computes the energy series and the final-time total ``Jz`` (with the species sum) on the devices and
transfers only those compact results to the host (``OT_parallel_energy.png``, ``OT_parallel_Jz.png``). ``--output PATH``
performs one explicit gather of the saved histories to the host and writes an ``.npz``; the host memory needed is about
the printed size (``Ck`` is ``Nt*Ns*Nv*Ny*Nx*Nz*Nl`` 8-byte values), ``Fk`` is exported once, and ``Ck_0``/``Fk_0`` (which equal
the first saved sample) are skipped.

Benchmark
~~~~~~~~~

``Examples/benchmark_OT_parallel.py`` times a larger, SHORT Orszag-Tang case (set in the script's globals, not in the
32x32 input) on identical global problems at ``D = 1, 2, 4`` and, at fixed ``D``, ``ps = 1`` against ``ps = 2``. Every
layout runs in a fresh subprocess. It reports initial-state projection, compile/setup, warmed solve, diagnostics and one
host gather separately (all synchronized with ``block_until_ready``, no progress bar, plots or export), the local shapes and
the step counts, and prints the speedup ``T1/TD`` and efficiency ``T1/(D*TD)`` only when the ``D = 1`` layout was
measured. This is expected to take several minutes on a CPU.

.. code-block:: console

    $ python Examples/benchmark_OT_parallel.py                       # defaults: 64x64, t_max=0.2, 5 layouts

**Measured results (CPU logical devices only; no GPU was available).** Hardware: one AMD Ryzen 7 PRO 5850U laptop
(8 cores / 16 threads, 15 GB RAM), Windows 11. Software: Python 3.11.16, JAX/jaxlib 0.10.2, Diffrax 0.7.2, Equinox 0.13.8.
Problem: Orszag-Tang, ``Nx = Ny = 64``, ``Nn = Nm = Np = 5`` (``Nv = 125``), ``Ns = 2``, ``N_DG = 1`` (``Nl = 1``), adaptive
Dopri8 (``rtol = atol = 1e-7``), ``t_max = 0.2``, 3 saved samples, float64; every layout took 6 steps. ``solve`` is the warmed
call (it includes the diagnostics, about 5-10 ms); ``setup`` is first call minus warmed call; ``init`` is the sharded
initial projection.

.. list-table::
   :header-rows: 1

   * - D
     - ``(ps,px,py,pz)``
     - local ``Ck`` / ``Fk`` state tile
     - init s
     - setup s
     - solve s
     - ``T1/TD``
     - ``T1/(D TD)``
   * - 1
     - (1,1,1,1)
     - (2,125,64,64,1,1) / (6,64,64,1,1)
     - 4.62
     - 8.32
     - 6.52
     - 1.00
     - 1.00
   * - 2
     - (1,2,1,1)
     - (2,125,64,32,1,1) / (6,64,32,1,1)
     - 4.33
     - 9.26
     - 3.94
     - 1.66
     - 0.83
   * - 2
     - (2,1,1,1)
     - (1,125,64,64,1,1) / (6,64,64,1,1)
     - 3.79
     - 7.76
     - 4.37
     - 1.49
     - 0.75
   * - 4
     - (1,2,2,1)
     - (2,125,32,32,1,1) / (6,32,32,1,1)
     - 3.85
     - 7.55
     - 3.12
     - 2.09
     - 0.52
   * - 4
     - (2,2,1,1)
     - (1,125,64,32,1,1) / (6,64,32,1,1)
     - 3.99
     - 8.03
     - 3.90
     - 1.67
     - 0.42

Per device and RHS evaluation, the ``Ck`` face traffic is 32,000 values for ``(1,2,2,1)`` (x and y faces, both
directions) and 16,000 for ``(2,2,1,1)`` (x faces only), and the hybrid layout adds one 6,144-value current all-reduce. At
``D = 4`` both layouts hold 2.0 MB of ``Ck`` state per device. The hybrid layout holds twice the field (98 kB instead of
49 kB) because the field is replicated across species groups. On this host the spatial layouts were faster than the
species/hybrid layouts at equal ``D``. The 16 logical CPU devices share one 8-core host whose single-device run already uses
several threads, so these numbers check the machinery and are **not** evidence about GPU or multi-node scaling.

**One-device regression check** against the base commit ``21f5068``: the default ``simulation()`` (``Nx = 33``,
``Nn = 20``, 200 saved samples, ``t_max = 50``) took 16.1 s warmed (base 16.5 s) and 29.5 s for the first call, including
compilation (base 25.5 s). The extra first-call time is the separately compiled initialization. Results agree with the
base to 1.5e-10 (values up to 53).

State, sharding and ownership
-----------------------------

With ``Nv = Np*Nm*Nn``, ``H = Ns*Nv`` and ``Nl = comb(N_DG+dims-1, dims)`` the solver state is the plain tuple
``Y = (Ck, Fk)`` for every layout, including one device. The mesh has axes ``("species","x","y","z")`` and shape
``(ps,px,py,pz)`` with ``AxisType.Auto``. Axes missing from a spec are replicated.

.. list-table::
   :header-rows: 1
   :widths: 30 32 38

   * - Array
     - Shape
     - PartitionSpec
   * - ``Ck`` (solver state)
     - ``(Ns,Nv,Ny,Nx,Nz,Nl)``
     - ``P("species",None,"y","x","z",None)``
   * - ``Fk`` (solver state)
     - ``(6,Ny,Nx,Nz,Nl)``
     - ``P(None,"y","x","z",None)``
   * - projected ``Ck`` (rank 8)
     - ``(Ns,Np,Nm,Nn,Ny,Nx,Nz,Nl)``
     - ``P("species",None,None,None,"y","x","z",None)``
   * - saved ``Ck`` (public output)
     - ``(Nt,H,Ny,Nx,Nz,Nl)``
     - ``P(None,"species","y","x","z",None)``
   * - saved ``Fk`` (public output)
     - ``(Nt,6,Ny,Nx,Nz,Nl)``
     - ``P(None,None,"y","x","z",None)``
   * - ``Ck_0``, ``Fk_0``
     - given public shapes
     - species prefix / spatial sharding as above
   * - ``qs``, ``ms``; ``alpha_s``, ``u_s``
     - ``(Ns,)``; ``(Ns,3)``
     - ``P("species")``; ``P("species",None)``
   * - species-dependent ``A_pm`` matrices
     - species on the first axis
     - ``P("species",None,...)``
   * - small species-independent tables, scalars
     - various
     - ``P()``

``Fk`` has no species array axis: it is partitioned over the spatial mesh axes and **replicated across the species mesh
axis**, so each species group integrates the same field for its spatial tile. This intentional replication keeps the design
simple (no owner rank or broadcast). A species-only layout replicates the whole field grid in every group. Because
``ps`` divides ``Ns``, the output merge of the species and Hermite axes to ``H`` keeps whole contiguous species blocks on each
shard. A single species' Hermite and DG axes are always local.

Communication pattern
---------------------

*Spatial faces.* All spatial coupling (Vlasov and Maxwell fluxes) is through +/-1 element faces, implemented as global
``jnp.roll`` shifts. On current JAX these lower to boundary-sized ``collective-permute`` operations between spatial
neighbours with the same species coordinate; periodicity spans device seams and the domain edge. There is no species halo
exchange and no periodic species neighbour. Hermite shifts, DG contractions, the collision term and the mass inversion are
local.

*Species current.* Only the Maxwell current couples species. Stacking the per-species currents before one sum,
``J_species`` of shape ``(Ns,3,Ny,Nx,Nz,Nl)`` and ``J = jnp.sum(J_species, axis=0)``, lets the compiler reduce across **only
the species mesh axis at fixed spatial coordinates**, aiming at one stacked three-component all-reduce per RHS evaluation
when ``ps > 1``. The payload is three current coefficients per local cell and DG mode (a spatial-tile-sized moment payload),
never the distribution. With ``ps = 1`` the sum is local and no species collective exists. The number of collectives is
whatever the compiler emits; this is a design aim checked on compiled executables, not a guarantee.

*Controller.* One global Diffrax solve integrates ``(Ck, Fk)``. The PID controller uses a shape-preserving global tree RMS
norm (``spectrax.state_rms_norm``): sums over each leaf's global shape and the **global** element counts, so only scalar
all-reduces are needed, the replicated ``Fk`` is counted once, and every device takes the same step decisions.

Current-sum derivation
~~~~~~~~~~~~~~~~~~~~~~

In the reference right-hand side the Maxwell source is, with ``q_s`` the charge, ``(a_x,a_y,a_z)_s`` the thermal scales,
``(u_x,u_y,u_z)_s`` the drifts and ``C_{nmp}`` the Hermite-DG coefficient of species ``s``:

.. math::

   \text{source}_x = \sum_s q_s a_x a_y a_z \Bigl(u_x C_{000} + \tfrac{a_x}{\sqrt2} C_{100}\Bigr)_s ,

and likewise ``source_y`` with ``(u_y, a_y, C_{010})`` and ``source_z`` with ``(u_z, a_z, C_{001})``, entering as ``dFk/dt = ... - (1/Omega_ce) (J, 0)``. Defining

.. math::

   J^{(s)}_i = q_s a_x a_y a_z \Bigl(u_i C_{000} + \tfrac{a_i}{\sqrt2} C_{e_i}\Bigr)_s ,\qquad J = \sum_s J^{(s)},

each shard computes ``J^{(s)}`` for the species it owns from their low-order Hermite coefficients, stacks the three
components, and the species sum yields ``J``. Each species group then evaluates the same Maxwell RHS with the same ``J``.
Nothing is averaged, divided by ``ps`` or lagged, and no species is omitted.

Memory and communication trade-off
----------------------------------

Let ``V = Nx*Ny*Nz*Nl`` and ``D = ps*px*py*pz``. Ideally each device stores ``Ns*Nv*V/D`` distribution entries
regardless of the factorization, but ``6*ps*V/D`` field entries because the field is replicated over the species groups.
Species sharding reduces the number of spatial partitions, which can reduce ``Ck`` face traffic or make a decomposition
possible when spatial divisibility or local work limits scaling. It adds the current reduction at every RHS evaluation and
replicates the Maxwell work. With two species ``ps`` is at most two. Species splitting at fixed ``D`` does **not**, by itself,
reduce ``Ck`` memory relative to an already valid spatial decomposition. The layout is always the user's explicit choice;
spatial-only is the default, and neither choice is claimed to be universally better.

Compiler findings
-----------------

Checked on the compiled executables (``jit(...).lower(...).compile().as_text()``) with JAX/jaxlib 0.10.2 on 4 logical CPU
devices, float64; ``tests/test_parallel.py`` asserts the RHS and norm rules below on every run. Example RHS grid: Ns=2,
Nn=3, Nm=2, Np=1, N_DG=2, dims=2 (Nl=3), Nx=32, Ny=16, Nz=1.

.. list-table::
   :header-rows: 1
   :widths: 22 42 36

   * - Layout ``(ps,px,py,pz)``
     - RHS collectives (payload elements)
     - Meaning
   * - ``(1,2,2,1)``
     - 2 x Ck y-face (576), 2 x Ck x-face (288), 2 x Fk y-face (288), 2 x Fk x-face (144) ``collective-permute``;
       no all-reduce
     - one-cell faces in both directions; species sum local
   * - ``(2,2,1,1)``
     - 2 x Ck x-face (288), 2 x Fk x-face (288) permutes between devices with the same species coordinate;
       1 x ``all-reduce`` ``(3,16,16,1,3)`` (2304)
     - faces + stacked current reduced over the species axis only, at fixed ``(x,y,z)``
   * - ``(2,1,1,1)``
     - 1 x ``all-reduce`` ``(3,16,32,1,3)``
     - species-only: no faces cross devices
   * - ``(1,1,1,2)``, ``(2,1,1,2)`` (3D)
     - z-face permutes; for ``ps=2`` one current all-reduce
     - same pattern along z

No ``all-gather``, ``all-to-all`` or ``reduce-scatter`` appears in the RHS. The total face payload is below half of the
local state tile, and the current all-reduce carries exactly ``3 x local spatial tile x Nl`` values.

* **Tree norm:** only scalar all-reduces. In hybrid layouts the ``Ck`` sum is reduced over the spatial axes and then the
  species axis, while the ``Fk`` sum is reduced over the spatial axes only, so the replicated field is counted once.
  The norm equals ``optimistix.rms_norm`` and is identical across layouts.
* **Full adaptive Dopri8 solve** (Nx=Ny=8, Nn=Nm=Np=2): inside the time loop there are the face permutes above, one
  ``f64[3,8,4,1,1]`` current all-reduce per RHS evaluation (``ps=2``, groups over the species mesh axis only) and scalar
  controller reductions (including a boolean finiteness check). A few tiny all-reduces outside the loop, of 3 save times
  and scalars, set up the output. Saving the history and merging species and Hermite axes into ``H`` add no collective.
* **Initialization:** ``DG_project`` compiles with no collective for spatial, hybrid and species-only layouts, and its
  per-device temporary memory scales as ``1/D``. Concatenating or stacking *along the species axis* inside the integrand
  makes XLA insert ``all-to-all`` operations, so species dependence is built by broadcasting per-species coefficient
  arrays, as in the default initial condition and ``OT_parallel.py``.
* **Diagnostics:** only all-reduces of ``(Nt, Ns)``-sized moment payloads; the ``Ck`` history is never gathered.

Large outputs
-------------

The saved ``Ck`` and ``Fk`` histories stay distributed on the devices. Converting them with ``numpy.asarray`` (or
saving them) gathers the whole array into host memory once; do this outside timed numerical work, select a time slice or
reduce on the devices first (as ``OT_parallel.py --plot`` does), and avoid exporting a duplicate ``Fk`` per species group.
Arrays living on different device sets cannot be combined in one JAX operation; compare them through NumPy.

Limitations
-----------

* One process, local devices only; multi-host is neither implemented nor tested.
* A species cannot be split; ``ps`` divides ``Ns`` and is at most ``Ns``. Hermite and DG modes are never partitioned.
* The active collision term is species diagonal and the limiter and diffusion ``D`` terms of the reference branch remain
  disabled; no inter-species collisions are added.
* The deprecated implicit midpoint solver is outside this work.
* The generic ``plot`` function still expects an ``output['dCk']`` that the solver does not provide (pre-existing).
* Speedups depend on problem size, hardware and layout; none are asserted without measurement.
* ``jax.grad`` and ``jax.jit`` work through ``simulation`` on any layout with JAX >= 0.7. On Python 3.10 (JAX 0.6,
  optimistix 0.0.11) gradients work on one device only, and ``load_parameters`` must run before ``simulation`` is
  imported, because importing diffrax there starts the JAX backend.
* GPU scaling has not been measured yet.

Design and sources
------------------

The design adapts established methods; no other code's scaling figures are transferred to SPECTRAX.

* Koshkarov, Manzini, Delzanno, Pagliantini and Roytershteyn, *The multi-dimensional Hermite-discontinuous Galerkin method
  for the Vlasov-Maxwell equations*, arXiv:2004.12010 (doi:10.1016/j.cpc.2021.107866), Sections 7.1-7.4: **adopted** the
  physical-space decomposition with all velocity and DG coefficients retained in each cell, precomputed operators, an external
  time integrator, and the Orszag-Tang test case. Closest numerical precedent; species decomposition is our extension.
* Xu et al., *GSPMD*, arXiv:2105.04663, Sections 4.3 and 4.6: **adopted** compiler-managed halo exchange from sharding
  annotations as the methodological basis, and we check the actual compiled output of the installed JAX instead of assuming
  it. The rotation optimization is relied upon only as far as the compiled output shows.
* Ataei and Salehipour, *XLB*, CPC 300 (2024) 109187, Sections 3.3-3.4: **adopted** global JAX arrays with distributed
  initialization and the separation between compiler-managed code and narrowly scoped ``shard_map``/``ppermute`` streaming
  (held in reserve here). **Not adopted**: its lattice-Boltzmann physics or automatic grid enlargement.
* Bezgin, Buhendwa and Adams, *JAX-Fluids 2.0*, CPC 308 (2025) 109433, arXiv:2402.05193 (code at commit
  ``819edcd``: ``halos/inner/material.py`` face-slice ``ppermute``, ``halos/inner/halo_communication.py`` neighbour
  permutations, ``time_integration/time_step_size.py`` global ``pmin``): **adopted** the ideas of spatial blocks, face
  exchanges and globally synchronized scalar decisions. **Not adopted**: its ``pmap`` wrappers.
* Trixi.jl (``src/auxiliary/mpi.jl``, ``src/solvers/dgsem_tree/dg_2d_parallel.jl``): **adopted** the globally reduced ODE
  norm and the separation of local and remote face work. **Not adopted**: nodal indexing, the MPI layer, AMR.
* Hafner and Vicentini, *mpi4jax*, JOSS 6(65) 3419 (2021): a credible explicit-MPI alternative; MPI is not introduced
  because it is unnecessary for the local-device target. This is a simplicity decision, not a claim that MPI is slower.
* GENE (independent ``n_procs_s`` and ``n_procs_x/y/z`` with divisibility requirements, as documented in the STELLOPT
  turbulent-transport example): **adopted** only the configuration pattern of an explicit species factor independent of the
  spatial mesh; GENE solves a different (gyrokinetic) model.
* GS2 (``integrate_species_sub`` in ``le_grids.f90``): **adopted** the communication pattern of reducing compact weighted
  moments among ranks that share spatial indices instead of gathering the distribution. SPECTRAX keeps its own Hermite
  current formulas and unpartitioned velocity modes.

The species-plus-space layout is a deliberate extension of the spatial Hermite-DG precedent using per-array JAX shardings
and native Diffrax PyTrees; none of the cited codes is claimed to implement this exact SPECTRAX/JAX layout. JAX-CFD is not
cited as a spatial multi-device precedent (its README lists distributed simulation as future work). The design is a
proposal verified by tests and compiled-code inspection; performance is reported only where measured.

Further reading: the JAX documentation for `parallel programming <https://docs.jax.dev/en/latest/parallel.html>`_,
`sharding <https://docs.jax.dev/en/latest/jax.sharding.html>`_, `jax.jit <https://docs.jax.dev/en/latest/_autosummary/jax.jit.html>`_,
`jax_num_cpu_devices <https://docs.jax.dev/en/latest/config_options.html#num-cpu-devices>`_ and
`shard_map <https://docs.jax.dev/en/latest/jep/14273-shard-map.html>`_, and the Diffrax
`diffeqsolve <https://docs.kidger.site/diffrax/api/diffeqsolve/>`_, `terms <https://docs.kidger.site/diffrax/api/terms/>`_ and
`step-size controller <https://docs.kidger.site/diffrax/api/stepsize_controller/>`_ and Optimistix
`norms <https://docs.kidger.site/optimistix/api/norms/>`_ pages.
