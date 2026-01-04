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
  outdir="${RESULTS_DIR}/${kf}"
  mkdir -p "${outdir}"

  echo "[micro-bench] profiling kernel file = ${kf}"

  runner="${outdir}/run_${kf}.sh"
  cat > "${runner}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
"$(pwd)/${OUT}" --option profiling --kernel-file "${kf}" --outdir "${outdir}"
EOF
  chmod +x "${runner}"

  ncu --set full \
      --target-processes all \
      --profile-from-start yes \
      --import-source yes \
      --export "${report}" \
      --force-overwrite true \
      -- "${runner}"
done
