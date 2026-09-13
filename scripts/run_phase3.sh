#!/bin/bash
# Phase 3 on one office GPU, strictly serial. Usage:
#   run_phase3.sh <tag> <grid> <hermite> <t_max> <modes> <iterations> <tolerance> <validate_resolution>
# 3a optimise both objectives; 3b validate one level up; 3c benchmark (compile-time workspace) and per-process peak memory.
set -u
TAG=$1 GRID=$2 HERM=$3 TMAX=$4 MODES=$5 ITERS=$6 TOL=$7 VRES=$8
OUT=$HOME/spectrax-pub/$TAG; mkdir -p $OUT
cd $HOME/SPECTRAX-autodiff && git fetch -q origin agent/autodiff-plan && git reset -q --hard FETCH_HEAD
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES=$G XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
export SPECTRAX_ROOT=$HOME/SPECTRAX-autodiff PYTHONPATH=$HOME/SPECTRAX-autodiff:$HOME/spectrax-extra
P=$HOME/venvs/vmex-gpu/bin/python; X=Examples/2D_Orszag_Tang_optimization.py
echo "gpu=$G commit=$(git rev-parse HEAD) args=$* start=$(date -Is)" > $OUT/run.info
for OBJ in conversion peak_current; do
  $P $X optimize --objective $OBJ --modes $MODES --grid $GRID --hermite $HERM --t-max $TMAX --iterations $ITERS --tolerances $TOL --output $OUT > $OUT/optimize_$OBJ.log 2>&1
  $P $X validate $OUT/optimize_$OBJ.json --resolutions ${GRID}x${HERM} $VRES --tolerances $TOL --output $OUT > $OUT/validate_$OBJ.log 2>&1
  echo "$OBJ done $(date -Is)" >> $OUT/run.info
done
$P $X benchmark --objective conversion --grid 32 --hermite 4 --t-max 100 --modes 8 --control-counts 2 4 8 16 32 64 \
  --step-counts 50 100 200 400 800 1600 --resolutions 16x4 32x4 32x6 64x6 64x8 128x6 --checkpoints 8 --output $OUT > $OUT/benchmark.log 2>&1
for CFG in "32 4" "64 6" "64 8" "128 6"; do
  $P $HOME/spectrax-pub/peak_memory.py $OUT/peak_memory.jsonl $CFG 20 forward 8 >> $OUT/peak_memory.log 2>&1
  for K in 8 32; do $P $HOME/spectrax-pub/peak_memory.py $OUT/peak_memory.jsonl $CFG 20 gradient $K >> $OUT/peak_memory.log 2>&1; done
done
echo "end=$(date -Is)" >> $OUT/run.info
