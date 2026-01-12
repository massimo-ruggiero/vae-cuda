#!/usr/bin/env bash
set -euo pipefail

# Default: Colab T4
ARCH="${ARCH:-sm_75}"
OUT="${OUT:-main_macro_bench}"
EPOCHS="${EPOCHS:-1}"
STRATEGIES="${STRATEGIES:-OPTIMIZED}"
RESULTS_DIR="${RESULTS_DIR:-results/macro_bench/nsys}"

mkdir -p "${RESULTS_DIR}"

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
     --extended-lambda \
     "${INCLUDES[@]}" \
     "${SRCS[@]}" \
     -lcurand \
     -lineinfo \
     -o "${OUT}"

echo "[macro-bench] run nsys profiling on ./${OUT}"

runner="${RESULTS_DIR}/run_nsys.sh"
cat > "${runner}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
"$(pwd)/${OUT}" --epochs "${EPOCHS}" --strategies "${STRATEGIES}"
EOF
chmod +x "${runner}"

nsys profile \
    --trace cuda,osrt \
    --output "${RESULTS_DIR}/macro_bench" \
    "${runner}"

rm -f "${runner}"