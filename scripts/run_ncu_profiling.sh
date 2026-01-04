#!/usr/bin/env bash
set -euo pipefail

# Default: Colab T4
ARCH="${ARCH:-sm_75}"
OUT="${OUT:-micro_bench}"

RESULTS_DIR="${RESULTS_DIR:-results/micro_bench/ncu}"
KERNEL_FILES="${KERNEL_FILES:-linalg,activations,loss,reparam,optimizers}"

mkdir -p "${RESULTS_DIR}"

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
  src/main_micro_bench.cu
  src/benchmark/bench_core.cu
  src/benchmark/bench_linalg.cu
  src/benchmark/bench_activations.cu
  src/benchmark/bench_loss.cu
  src/benchmark/bench_reparam.cu
  src/benchmark/bench_optimizers.cu
  src/buffers/layer_buffers.cu
  src/buffers/vae_buffers.cu
  src/kernels/linalg.cu
  src/kernels/activations.cu
  src/kernels/loss.cu
  src/kernels/optimizers.cu
  src/kernels/reparametrization.cu
)

echo "[micro-bench] build: nvcc -arch=${ARCH} -> ${OUT}"

nvcc -arch="${ARCH}" \
     "${INCLUDES[@]}" \
     "${SRCS[@]}" \
     -lcurand \
     -lineinfo \
     -o "${OUT}"

echo "[micro-bench] run profiling on ./${OUT}"

IFS=',' read -r -a KERNEL_FILE_LIST <<< "${KERNEL_FILES}"

for kf in "${KERNEL_FILE_LIST[@]}"; do
  kf=$(echo "${kf}" | tr -d ' ')
  [ -z "${kf}" ] && continue

  report="${RESULTS_DIR}/${kf}"

  echo "[micro-bench] profiling kernel file = ${kf}"

  ncu --set full \
      --target-processes application \
      --profile-from-start yes \
      --force-overwrite true \
      --export "${report}" \
      -- ./"${OUT}" \
      --outdir "${RESULTS_DIR}" \
      --option profiling \
      --kernel-file "${kf}"

done
