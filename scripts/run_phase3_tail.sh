#!/bin/bash
# Trimmed tail of run_phase3.sh gpu-64x6: wait for the running peak_current optimisation, then
# validate it, run a shorter benchmark (P <= 64, N <= 800) and the per-process peak-memory sweep.
set -u
OUT=$HOME/spectrax-pub/gpu-64x6; PID=$1
while kill -0 $PID 2>/dev/null; do sleep 30; done
echo "peak_current optimize exited $(date -Is)" >> $OUT/run.info
cd $HOME/SPECTRAX-autodiff                     # still at 8380edc; do not reset (provenance)
export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
export SPECTRAX_ROOT=$HOME/SPECTRAX-autodiff PYTHONPATH=$HOME/SPECTRAX-autodiff:$HOME/spectrax-extra
P=$HOME/venvs/vmex-gpu/bin/python; X=Examples/2D_Orszag_Tang_optimization.py
$P $X validate $OUT/optimize_peak_current.json --resolutions 64x6 64x8 128x6 --tolerances 1e-9 --output $OUT > $OUT/validate_peak_current.log 2>&1
echo "peak_current done $(date -Is)" >> $OUT/run.info
$P $X benchmark --objective conversion --grid 32 --hermite 4 --t-max 100 --modes 8 --control-counts 2 4 8 16 32 \
  --step-counts 50 100 200 400 800 --resolutions 16x4 32x4 32x6 64x6 64x8 128x6 --checkpoints 8 --output $OUT > $OUT/benchmark.log 2>&1
echo "benchmark done $(date -Is)" >> $OUT/run.info
for CFG in "32 4" "64 6" "64 8" "128 6"; do
  $P $HOME/spectrax-pub/peak_memory.py $OUT/peak_memory.jsonl $CFG 20 forward 8 >> $OUT/peak_memory.log 2>&1
  for K in 8 32; do $P $HOME/spectrax-pub/peak_memory.py $OUT/peak_memory.jsonl $CFG 20 gradient $K >> $OUT/peak_memory.log 2>&1; done
done
echo "end=$(date -Is)" >> $OUT/run.info
