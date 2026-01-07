#!/usr/bin/env bash
set -euo pipefail

# Default: Colab T4
ARCH="${ARCH:-sm_75}"
OUT="${OUT:-fused_micro_bench}"

# Include paths
INCLUDES=(
  -Iinclude
  -Iinclude/benchmark
  -Iinclude/buffers
  -Iinclude/kernels
  -Iinclude/core
  -Iinclude/utils
)

# Sources for fused micro-bench
SRCS=(
  src/main_fused_micro_bench.cu
  src/benchmark/bench_core.cu
  src/buffers/layer_buffers.cu
  src/buffers/vae_buffers.cu
  src/kernels/linalg.cu
  src/kernels/linear.cu
  src/kernels/activations.cu
  src/kernels/fused.cu
)

echo "[fused-micro-bench] build: nvcc -arch=${ARCH} -> ${OUT}"

nvcc -arch="${ARCH}" \
     "${INCLUDES[@]}" \
     "${SRCS[@]}" \
     -lcurand \
     -lineinfo \
     -o "${OUT}"

echo "[fused-micro-bench] run: ./${OUT} $*"
./"${OUT}" "$@"
