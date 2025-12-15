#include "linalg.cuh"
#include "utils.cuh"
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

__global__ void transpose_naive_kernel(const float* A,
                                       float* B,
                                       int M,
                                       int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < K) {
        B[col * M + row] = A[row * K + col];
    }
}

namespace matmul {

    void naive(const float* d_A,
               const float* d_B,
               float* d_C,
               int M,
               int K,
               int N) {
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                    (M + blockSize.y - 1) / blockSize.y);
        DEBUG("Launching matmul_naive_kernel...");
        matmul_naive_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);

        CUDA_CHECK(cudaGetLastError());
    } 
    
} // namespace matmul

namespace transpose {

    void naive(const float* d_A,
               float* d_B,
               int M,
               int K) {
        dim3 blockSize(16, 16);
        dim3 gridSize((K + blockSize.x - 1) / blockSize.x,
                      (M + blockSize.y - 1) / blockSize.y);
        DEBUG("Launching transpose_naive_kernel...");
        transpose_naive_kernel<<<gridSize, blockSize>>>(d_A, d_B, M, K);

        CUDA_CHECK(cudaGetLastError());
    } 

} // namespace transpose