#!/usr/bin/env bash
set -euo pipefail

# Default: Colab T4
ARCH="${ARCH:-sm_75}"
OUT="${OUT:-fused_micro_bench}"
RESULTS_DIR="${RESULTS_DIR:-results/micro_bench/ncu}"

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

echo "[fused-micro-bench] profiling (ncu) -> ${RESULTS_DIR}"

runner="${RESULTS_DIR}/run_fused_micro_bench.sh"
cat > "${runner}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
"$(pwd)/${OUT}"
EOF
chmod +x "${runner}"

ncu --set detailed \
    --import-source yes \
    --kernel-name "regex:.*_kernel" \
    --export "${RESULTS_DIR}/fused_micro_bench" \
    ./"${runner}"

rm -f "${runner}"
