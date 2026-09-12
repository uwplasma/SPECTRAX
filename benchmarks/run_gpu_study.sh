#!/usr/bin/env bash
# Single-GPU study. Activate a CUDA-enabled environment before running.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
study_python="${STUDY_PYTHON:-python}"
study_output="${1:-results/gpu}"
mkdir -p "$study_output"
"$study_python" -m pytest tests/test_autodiff.py tests/test_gradient_benchmark.py -q
"$study_python" benchmarks/gradient_scaling.py \
  --base-grid 32 --grids 16 32 64 --steps 64 256 --controls 8 32 128 \
  --hermites 4 6 --checkpoint-counts 4 16 --output "$study_output/scaling"
# Longer rollouts without a deliberately oversized taped allocation.
"$study_python" benchmarks/gradient_scaling.py \
  --base-grid 32 --axes steps --steps 256 1024 --checkpoints 16 \
  --methods checkpointed budgeted --output "$study_output/long-rollout"
