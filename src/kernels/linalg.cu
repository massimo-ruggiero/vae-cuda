#include "linalg.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>


static constexpr int TILE_DIM = 16;


// ==============================
// Kernels: SGEMM
// ==============================

__global__ void sgemm_naive_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
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

__global__ void sgemm_tiling_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
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

__global__ void sgemm_padding_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
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


// ==============================
// Kernels: Add inplace
// ==============================

__global__ void add_inplace_naive_kernel(float* __restrict__ A, 
                                         const float* __restrict__ B, 
                                         int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] += B[idx];
    }
}

__global__ void add_inplace_vec4_kernel(float* __restrict__ A, 
                                        const float* __restrict__ B, 
                                        int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4; 
    if (idx + 3 < size) {
        float4 A_vec = *reinterpret_cast<float4*>(&A[idx]);
        float4 B_vec = *reinterpret_cast<const float4*>(&B[idx]);

        A_vec.x += B_vec.x;
        A_vec.y += B_vec.y;
        A_vec.z += B_vec.z;
        A_vec.w += B_vec.w;

        *reinterpret_cast<float4*>(&A[idx]) = A_vec;
    } else if (idx < size) {
        for (int i = idx; i < size; ++i) {
            A[i] += B[i];
        }
    }
}


// ==============================
// Host API
// ==============================

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
        int gridSize;
        switch(strategy) {
            case VAEStrategy::NAIVE:
                gridSize  = (size + blockSize - 1) / blockSize;
                DEBUG("Launching add_inplace_naive_kernel...");
                add_inplace_naive_kernel<<<gridSize, blockSize>>>(d_A, d_B, size);
                break;
            case VAEStrategy::VECTORIZED:
                gridSize  = ((size + 3) / 4 + blockSize - 1) / blockSize;
                DEBUG("Launching add_inplace_vec4_kernel...");
                add_inplace_vec4_kernel<<<gridSize, blockSize>>>(d_A, d_B, size);
                break;
            default:
                gridSize  = (size + blockSize - 1) / blockSize;
                DEBUG("Launching add_inplace_naive_kernel...");
                add_inplace_naive_kernel<<<gridSize, blockSize>>>(d_A, d_B, size);
                break;
        }

        CUDA_CHECK(cudaGetLastError());
    } 

} // namespace linalg