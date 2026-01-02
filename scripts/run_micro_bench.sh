#!/usr/bin/env bash
set -euo pipefail

# Default: Colab T4
ARCH="${ARCH:-sm_75}"
OUT="${OUT:-micro_bench}"

# Include paths
INCLUDES=(
  -Iinclude
  -Iinclude/benchmark
  -Iinclude/buffers
  -Iinclude/kernels
  -Iinclude/core
  -Iinclude/utils
)

# Sources for micro-bench
SRCS=(
  src/benchmark/main_micro_bench.cu
  src/benchmark/bench_core.cu
  src/benchmark/bench_linalg.cu
  src/buffers/layer_buffers.cu
  src/buffers/vae_buffers.cu
  src/kernels/linalg.cu
  src/kernels/activations.cu
  src/kernels/loss.cu
  src/kernels/optimizers.cu
  src/kernels/reparametrization.cu
)

echo "[micro-bench] build: nvcc -arch=${ARCH} -> ${OUT}"

nvcc -O3 \
  -arch="${ARCH}" \
  "${INCLUDES[@]}" \
  "${SRCS[@]}" \
  -lcurand \
  -lineinfo \
  -o "${OUT}"

echo "[micro-bench] run: ./${OUT}"
./"${OUT}"
