#!/usr/bin/env bash
set -euo pipefail

# Default: Colab T4
ARCH="${ARCH:-sm_75}"
OUT="${OUT:-main_macro_bench}"
EPOCHS="${EPOCHS:-100}"
STRATEGIES="${STRATEGIES:-FUSED}"

# Include paths
INCLUDES=(
  -Iinclude
  -Iinclude/benchmark
  -Iinclude/buffers
  -Iinclude/kernels
  -Iinclude/core
  -Iinclude/optimizers
  -Iinclude/utils
)

# Sources for macro-benchmark
SRCS=(
  src/main_macro_bench.cu
  src/core/vae_forward.cu
  src/core/vae_backward.cu
  src/core/vae.cu
  src/core/trainer.cu
  src/buffers/layer_buffers.cu
  src/buffers/vae_buffers.cu
  src/optmizers/adam.cu
  src/kernels/linalg.cu
  src/kernels/activations.cu
  src/kernels/loss.cu
  src/kernels/optimizers.cu
  src/kernels/reparametrization.cu
  src/kernels/fused.cu
  src/kernels/linear.cu
  src/utils/mnist_loader.cpp
)

echo "[macro-bench] build: nvcc -arch=${ARCH} -> ${OUT}"

nvcc -arch="${ARCH}" \
     "${INCLUDES[@]}" \
     "${SRCS[@]}" \
     -lcurand \
     -lineinfo \
     -o "${OUT}"

echo "[macro-bench] run: ./${OUT}"
./"${OUT}" \
  --epochs "${EPOCHS}" \
  --strategies "${STRATEGIES}"
