#include "linalg.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>


__global__ void matmul_naive_kernel(const float* A,
                                    const float* B,
                                    float* C,
                                    int M,
                                    int K,
                                    int N,
                                    bool transpose_A,
                                    bool transpose_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        int stride_A = transpose_A ? M : K;
        int stride_B = transpose_B ? K : N;

        for (int k = 0; k < K; ++k) {
            int idx_A = transpose_A ? 
                        stride_A * k + row : 
                        stride_A * row + k;

            int idx_B = transpose_B ? 
                        stride_B * col + k : 
                        stride_B * k + col;

            sum += A[idx_A] * B[idx_B];
        }
        C[row * N + col] = sum;
    }
}

namespace matmul {

    void naive(const float* d_A,
               const float* d_B,
               float* d_C,
               int M,
               int K,
               int N,
               bool transpose_A,
               bool transpose_B) {
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                    (M + blockSize.y - 1) / blockSize.y);
        DEBUG("Launching matmul_naive_kernel...");
        matmul_naive_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N, transpose_A, transpose_B);

        CUDA_CHECK(cudaGetLastError());
    } 
    
} // namespace matmul