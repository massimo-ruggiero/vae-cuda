#include "matmul.h"
#include "utils.h"
#include <cuda_runtime.h>


__global__ void matmul_naive_kernel(const float* A,
                                    const float* B,
                                    float* C,
                                    int M,
                                    int K,
                                    int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

namespace matmul {

    void matmul_naive(const float* d_A,
                    const float* d_B,
                    float* d_C,
                    int M,
                    int K,
                    int N) {
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                    (M + blockSize.y - 1) / blockSize.y);
        DEBUG("Launching matmul_naive_kernel...")
        matmul_naive_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}