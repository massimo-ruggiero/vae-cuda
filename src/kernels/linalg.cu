#include "linalg.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>


const int TILE_DIM = 16;

// sgemm
__global__ void sgemm_naive_kernel(const float* A,
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

__global__ void sgemm_tiling_kernel(const float* A,
                                    const float* B,
                                    float* C,
                                    int M,
                                    int K,
                                    int N,
                                    bool transpose_A,
                                    bool transpose_B){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float A_tile[TILE_DIM][TILE_DIM];
    __shared__ float B_tile[TILE_DIM][TILE_DIM];

    int A_rows = transpose_A ? K : M;
    int A_cols = transpose_A ? M : K;
    int B_rows = transpose_B ? N : K;
    int B_cols = transpose_B ? K : N;

    int C_rows = A_rows;
    int C_cols = B_cols;

    if (row >= C_rows || col >= C_cols) return;

    int numTiles = (A_cols + TILE_DIM - 1) / TILE_DIM;
    float sum = 0.0f;

    for (int t = 0; t < numTiles; ++t) {

        // load tiles into shared memory
        int kA = t * TILE_DIM + threadIdx.x;
        if (row < A_rows && kA < A_cols) {
            int idx_A = transpose_A ?
                        kA * K + row:
                        row * K + kA;
            A_tile[threadIdx.y][threadIdx.x] = A[idx_A];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int kB = t * TILE_DIM + threadIdx.y;
        if (kB < B_rows && col < B_cols) {
            int idx_B = transpose_B ?
                        col * N + kB:
                        kB * N + col;
            B_tile[threadIdx.y][threadIdx.x] = B[idx_B];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
                                           
        __syncthreads();

        // multiply tiles accumulate result
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * C_cols + col] = sum;
}

__global__ void sgemm_padding_kernel(const float* A,
                                    const float* B,
                                    float* C,
                                    int M,
                                    int K,
                                    int N,
                                    bool transpose_A,
                                    bool transpose_B){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float A_tile[TILE_DIM][TILE_DIM + 1];
    __shared__ float B_tile[TILE_DIM][TILE_DIM + 1];

    int A_rows = transpose_A ? K : M;
    int A_cols = transpose_A ? M : K;
    int B_rows = transpose_B ? N : K;
    int B_cols = transpose_B ? K : N;

    int C_rows = A_rows;
    int C_cols = B_cols;

    if (row >= C_rows || col >= C_cols) return;

    int numTiles = (A_cols + TILE_DIM - 1) / TILE_DIM;
    float sum = 0.0f;

    for (int t = 0; t < numTiles; ++t) {

        // load tiles into shared memory
        int kA = t * TILE_DIM + threadIdx.x;
        if (row < A_rows && kA < A_cols) {
            int idx_A = transpose_A ?
                        kA * K + row:
                        row * K + kA;
            A_tile[threadIdx.y][threadIdx.x] = A[idx_A];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int kB = t * TILE_DIM + threadIdx.y;
        if (kB < B_rows && col < B_cols) {
            int idx_B = transpose_B ?
                        col * N + kB:
                        kB * N + col;
            B_tile[threadIdx.y][threadIdx.x] = B[idx_B];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
                                           
        __syncthreads();

        // multiply tiles accumulate result
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * C_cols + col] = sum;
}

// add_in_place
__global__ void add_inplace_naive_kernel(float* A, 
                                   const float* B, 
                                   int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] += B[idx];
    }
}

__global__ void add_inplace_vectorized_kernel(float* A, 
                                              const float* B, 
                                              int size) {}


namespace linalg {

    void sgemm(const float* d_A,
               const float* d_B,
               float* d_C,
               int M,
               int K,
               int N,
               bool transpose_A,
               bool transpose_B,
               const VAEStrategy& strategy) {
        dim3 blockSize(TILE_DIM, TILE_DIM);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                      (M + blockSize.y - 1) / blockSize.y);

        switch (strategy) {
            case VAEStrategy::NAIVE: 
                DEBUG("Launching sgemm_naive_kernel...");
                sgemm_naive_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N, transpose_A, transpose_B);
                break;
            case VAEStrategy::TILING: 
                DEBUG("Launching sgemm_tiling_kernel...");
                sgemm_tiling_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N, transpose_A, transpose_B);
                break;
            case VAEStrategy::PADDING: 
                DEBUG("Launching sgemm_padding_kernel...");
                sgemm_padding_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N, transpose_A, transpose_B);
                break;
            // valuta se vectorized
            default:
                DEBUG("Launching sgemm_naive_kernel...");
                sgemm_naive_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N, transpose_A, transpose_B);
                break;
        }

        CUDA_CHECK(cudaGetLastError());
    } 

    void add_in_place(float* d_A, 
                      const float* d_B, 
                      int size,
                      const VAEStrategy& strategy) {
        const int blockSize = 256;
        const int gridSize  = (size + blockSize - 1) / blockSize;
        switch(strategy) {
            case VAEStrategy::NAIVE:
                DEBUG("Launching add_inplace_naive_kernel...");
                add_inplace_naive_kernel<<<gridSize, blockSize>>>(d_A, d_B, size);
                break;
            case VAEStrategy::VECTORIZED:
                DEBUG("Launching add_inplace_vectorized_kernel...");
                add_inplace_vectorized_kernel<<<gridSize, blockSize>>>(d_A, d_B, size);
                break;
            default:
                DEBUG("Launching add_inplace_naive_kernel...");
                add_inplace_naive_kernel<<<gridSize, blockSize>>>(d_A, d_B, size);
                break;
        }

        CUDA_CHECK(cudaGetLastError());
    } 

} // namespace linalg