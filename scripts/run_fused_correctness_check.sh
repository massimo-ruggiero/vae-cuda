#!/usr/bin/env bash
set -euo pipefail

# Default: Colab T4
ARCH="${ARCH:-sm_75}"
OUT="${OUT:-fused_correctness_check}"

# Include paths
INCLUDES=(
  -Iinclude
  -Iinclude/buffers
  -Iinclude/kernels
  -Iinclude/core
  -Iinclude/utils
)

# Sources for fused correctness check
SRCS=(
  src/main_fused_correctness.cu
  src/kernels/fused.cu
)

echo "[fused-correctness] build: nvcc -arch=${ARCH} -> ${OUT}"

nvcc -arch="${ARCH}" \
     "${INCLUDES[@]}" \
     "${SRCS[@]}" \
     -lineinfo \
     -o "${OUT}"

echo "[fused-correctness] run: ./${OUT}"
./"${OUT}" "$@"
