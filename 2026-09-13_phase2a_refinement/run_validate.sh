#!/bin/bash
# Office GPU: pick the least-loaded GPU, run the large refinement points.
set -u
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES=$G XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
export SPECTRAX_ROOT=$HOME/SPECTRAX-autodiff PYTHONPATH=$HOME/SPECTRAX-autodiff:$HOME/spectrax-extra
cd $HOME/spectrax-pub/validate
echo "gpu=$G commit=$(git -C $SPECTRAX_ROOT rev-parse HEAD) start=$(date -Is)" > run.info
$HOME/venvs/vmex-gpu/bin/python validate_refinement.py refinement_gpu.json optimize_conversion.json optimize_peak_current.json \
  --resolutions 64x6 64x8 128x6 --tolerances 1e-7
echo "end=$(date -Is)" >> run.info
