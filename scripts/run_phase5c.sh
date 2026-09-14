#!/bin/bash
# Phase 5c on one office GPU: forward-mode sensitivities at both 64x6 optima, serially.
set -u
OUT=$HOME/spectrax-pub/phase5c; mkdir -p $OUT
cd $HOME/SPECTRAX-autodiff && git fetch -q origin agent/autodiff-plan && git reset -q --hard FETCH_HEAD
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES=$G XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=$HOME/SPECTRAX-autodiff:$HOME/spectrax-extra
echo "gpu=$G commit=$(git rev-parse HEAD) start=$(date -Is)" > $OUT/run.info
for R in optimize_conversion optimize_peak_current; do
  $HOME/venvs/vmex-gpu/bin/python Examples/2D_Orszag_Tang_optimization.py sensitivity $HOME/spectrax-pub/gpu-64x6/$R.json --snapshots 21 --output $OUT > $OUT/sensitivity_$R.log 2>&1
  echo "$R done $(date -Is)" >> $OUT/run.info
done
echo "end=$(date -Is)" >> $OUT/run.info
