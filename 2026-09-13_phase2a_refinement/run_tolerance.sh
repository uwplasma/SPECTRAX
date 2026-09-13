#!/bin/bash
# Queue behind run_validate.sh, then re-run the 64² refinement points at ODE tolerance 1e-9 with the example's validate mode.
set -u
V=$HOME/spectrax-pub/validate; T=$V/tolerance; mkdir -p $T
while ! grep -q '^end=' $V/run.info; do sleep 60; done
cd $HOME/SPECTRAX-autodiff && git fetch -q origin agent/autodiff-plan && git reset -q --hard FETCH_HEAD
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES=$G XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=$HOME/SPECTRAX-autodiff:$HOME/spectrax-extra
echo "gpu=$G commit=$(git rev-parse HEAD) start=$(date -Is)" > $T/run.info
for R in optimize_conversion optimize_peak_current; do
  $HOME/venvs/vmex-gpu/bin/python Examples/2D_Orszag_Tang_optimization.py validate $V/$R.json --resolutions 64x4 64x6 --tolerances 1e-9 --output $T > $T/$R.log 2>&1
done
echo "end=$(date -Is)" >> $T/run.info
