#!/bin/bash
cd /Users/rogeriojorge/local/tests/spectrax-autodiff-plan
export JAX_PLATFORMS=cpu PYTHONPATH=/Users/rogeriojorge/local/tests/spectrax-autodiff-plan
P=/Users/rogeriojorge/local/tests/spectrax-acceleration-validation/.venv/bin/python
for OBJ in conversion peak_current; do
  $P Examples/2D_Orszag_Tang_optimization.py optimize --objective $OBJ --fixed-amplitudes --modes 8 --grid 32 --hermite 4 --t-max 200 --iterations 20 --output /Users/rogeriojorge/local/spectrax-autodiff-evidence/pub_phases > /Users/rogeriojorge/local/spectrax-autodiff-evidence/pub_phases/optimize_${OBJ}_phases.log 2>&1
done
echo done > /Users/rogeriojorge/local/spectrax-autodiff-evidence/pub_phases/DONE
