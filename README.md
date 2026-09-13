# Particle-acceleration research evidence (outside the code PR)

This branch stores experiments and raw reports, not library changes. Code review:
https://github.com/uwplasma/SPECTRAX/pull/43 on `agent/acceleration-validation`.
The consolidated reproducible code checkpoint is
`6062da5a6ea5550d6d5100787d03b56236cef1c4`.

The Orszag–Tang excess search in `ot/results.json` used the exact two source
files under `ot/Examples`, with unchanged core solver files identified by its
SHA256 manifest. It improves a negative Gaussian-relative gain from −7.7019e−6
to −3.1731e−6 in 20 iterations; **not a nonthermal acceleration result**.
The small excess AD/FD CPU/GPU reports are under `smoke`.

The alternative electrostatic pilot `landau/pilot.py` uses the consolidated
checkpoint's helper and core solver. It follows SPECTRAX's existing Landau
initialization convention, adding eight fixed-amplitude density harmonics with
phase controls. The initial density is positive for all phase choices and E
satisfies Gauss's law. It is not a reproduction of a published optimization.
The two stored runs are early baseline probes, **both fail local negativity
at T3–4**. Positive Gaussian-relative excess is promising but not validated.

Reproduce using an authenticated clone (results are generated, not committed to
that checkout):

```sh
git clone https://github.com/uwplasma/SPECTRAX.git
cd SPECTRAX
git fetch origin agent/particle-acceleration-evidence
git checkout 6062da5a6ea5550d6d5100787d03b56236cef1c4
python -m pip install -e .
git show origin/agent/particle-acceleration-evidence:landau/pilot.py > landau_pilot.py
mkdir -p results
PYTHONPATH=. JAX_PLATFORMS=cpu python landau_pilot.py 128 32 128
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. JAX_PLATFORMS=cuda python landau_pilot.py 256 64 160
```

Future work: refine space/Hermite/time and negativity before any optimization.
A simplified example is a fallback; a resolved 2D reconnection or acceleration
showcase remains preferred. Main research context and primary sources are in
PR43's handoff comment. Keep raw experiments here or in external artifacts, not
in the small code-review diff.
