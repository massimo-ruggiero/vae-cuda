#include "linalg.cuh"
#include "utils.cuh"

#include <cuda_runtime.h>

static constexpr int TILE_DIM = 32;
static constexpr int COARSE_FACTOR = 4;


// ==============================
// Kernels: SGEMM
// ==============================

__global__ void sgemm_naive_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int K, int N) {
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

__global__ void sgemm_tiling_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int K, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float A_tile[TILE_DIM][TILE_DIM];
    __shared__ float B_tile[TILE_DIM][TILE_DIM];

    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    float sum = 0.0f;

    for (int t = 0; t < numTiles; ++t) {

        // load tiles into shared memory
        int kA = t * TILE_DIM + threadIdx.x; 
        if (row < M && kA < K) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * K + kA];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        int kB = t * TILE_DIM + threadIdx.y;
        if (kB < K && col < N) {
            B_tile[threadIdx.y][threadIdx.x] = B[kB * N + col];
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
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void sgemm_padding_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int K, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float A_tile[TILE_DIM][TILE_DIM + 1];
    __shared__ float B_tile[TILE_DIM][TILE_DIM + 1];

    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    float sum = 0.0f;

    for (int t = 0; t < numTiles; ++t) {

        // load tiles into shared memory
        int kA = t * TILE_DIM + threadIdx.x; 
        if (row < M && kA < K) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * K + kA];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        int kB = t * TILE_DIM + threadIdx.y;
        if (kB < K && col < N) {
            B_tile[threadIdx.y][threadIdx.x] = B[kB * N + col];
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
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void sgemm_vec4_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int K, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE_FACTOR;

    __shared__ float A_tile[TILE_DIM][TILE_DIM];
    __shared__ float B_tile[TILE_DIM][TILE_DIM * COARSE_FACTOR];

    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    float sum[COARSE_FACTOR] = {0.0f};

    for (int t = 0; t < numTiles; ++t) {

        // load tiles into shared memory
        int kA = t * TILE_DIM + threadIdx.x;
        if (row < M && kA < K) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * K + kA];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int kB = t * TILE_DIM + threadIdx.y;
        int shared_col = threadIdx.x * COARSE_FACTOR;
        if (kB < K && col < N) {
            float4 b_vec;
            int idxB = kB * N + col;
            if (col + 3 < N) {
                b_vec = *reinterpret_cast<const float4*>(&B[idxB]);
            } else {
                b_vec.x = (col + 0 < N) ? B[idxB + 0] : 0.0f;
                b_vec.y = (col + 1 < N) ? B[idxB + 1] : 0.0f;
                b_vec.z = (col + 2 < N) ? B[idxB + 2] : 0.0f;
                b_vec.w = (col + 3 < N) ? B[idxB + 3] : 0.0f;
            }

            B_tile[threadIdx.y][shared_col + 0] = b_vec.x;
            B_tile[threadIdx.y][shared_col + 1] = b_vec.y;
            B_tile[threadIdx.y][shared_col + 2] = b_vec.z;
            B_tile[threadIdx.y][shared_col + 3] = b_vec.w;
        } else {
            B_tile[threadIdx.y][shared_col + 0] = 0.0f;
            B_tile[threadIdx.y][shared_col + 1] = 0.0f;
            B_tile[threadIdx.y][shared_col + 2] = 0.0f;
            B_tile[threadIdx.y][shared_col + 3] = 0.0f;
        }
        __syncthreads();

        // multiply tiles accumulate result
        for (int k = 0; k < TILE_DIM; ++k) {
            float a_val = A_tile[threadIdx.y][k]; 

            for (int i = 0; i < COARSE_FACTOR; ++i) {
                sum[i] += a_val * B_tile[k][threadIdx.x * COARSE_FACTOR + i];
            }
        }
        __syncthreads();
    }

    if (row < M) {
        if (col + 3 < N) {
            float4 C_vec = make_float4(sum[0], sum[1], sum[2], sum[3]);
            *reinterpret_cast<float4*>(&C[row * N + col]) = C_vec;
        } else {
            for (int i = 0; i < COARSE_FACTOR; ++i){
                int current_col = col + i;
                if (current_col < N) {
                    C[row * N + current_col] = sum[i];
                }
            }
        }
    }
}


// ==============================
// Kernels: Transpose
// ==============================

__global__ void transpose_naive_kernel(const float* __restrict__ A,
                                       float* __restrict__ AT,
                                       int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N){
        AT[col * M + row] = A[row * N + col];
    }
} 

__global__ void transpose_tiling_kernel(const float* __restrict__ A,
                                       float* __restrict__ AT,
                                       int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x  + threadIdx.x;

    __shared__ float A_tile[TILE_DIM][TILE_DIM];

    // load tile into shared memory
    if (row < M && col < N){
        A_tile[threadIdx.y][threadIdx.x] = A[row * N + col];
    }
    __syncthreads();

    int row_t = blockIdx.x * blockDim.x  + threadIdx.y;
    int col_t = blockIdx.y * blockDim.y + threadIdx.x;

    if (row_t < N && col_t < M) {
        AT[row_t * M + col_t] = A_tile[threadIdx.x][threadIdx.y];
    }
} 

__global__ void transpose_padding_kernel(const float* __restrict__ A,
                                         float* __restrict__ AT,
                                         int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x  + threadIdx.x;

    __shared__ float A_tile[TILE_DIM][TILE_DIM + 1];

    // load tile into shared memory
    if (row < M && col < N){
        A_tile[threadIdx.y][threadIdx.x] = A[row * N + col];
    }
    __syncthreads();

    int row_t = blockIdx.x * blockDim.x + threadIdx.y;
    int col_t = blockIdx.y * blockDim.y + threadIdx.x;

    if (row_t < N && col_t < M) {
        AT[row_t * M + col_t] = A_tile[threadIdx.x][threadIdx.y];
    }
} 

__global__ void transpose_vec4_kernel(const float* __restrict__ A,
                                      float* __restrict__ AT,
                                      int M, int N) {
    int idx = threadIdx.x * 4;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = (blockIdx.x * blockDim.x) * 4 + idx;

    __shared__ float A_tile[TILE_DIM][TILE_DIM + 1];

    // load tile into shared memory
    if (row < M){
        if (col + 3 < N) {
            float4 A_vec = *reinterpret_cast<const float4*>(&A[row * N + col]);
            A_tile[threadIdx.y][idx + 0] = A_vec.x;
            A_tile[threadIdx.y][idx + 1] = A_vec.y;
            A_tile[threadIdx.y][idx + 2] = A_vec.z;
            A_tile[threadIdx.y][idx + 3] = A_vec.w;
        } else if (col < N) {
            for (int i = 0; i < 4; ++i) {
                int current_col = col + i;
                if (current_col < N) A_tile[threadIdx.y][idx + i] = A[row * N + current_col];
            }
        } 
    }
    __syncthreads();

    int row_t = (blockIdx.x * blockDim.x) * 4 + threadIdx.y;
    int col_t = (blockIdx.y * blockDim.y) * 4 + idx;

    if (row_t < N) {
        if (col_t + 3 < M) {
            float4 tile_vec = make_float4(
                                A_tile[idx + 0][threadIdx.y],
                                A_tile[idx + 1][threadIdx.y],
                                A_tile[idx + 2][threadIdx.y],
                                A_tile[idx + 3][threadIdx.y]
                            );

            *reinterpret_cast<float4*>(&AT[row_t * M + col_t]) = tile_vec;
        } else if (col_t < M) {
            for (int i = 0; i < 4; ++i) {
                int current_col = col_t + i;
                if (current_col < M) AT[row_t * M + current_col] = A_tile[idx + i][threadIdx.y];
            }
        }
    }
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
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int i = idx + t;
            if (i < size) A[i] += B[i];
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
               int M, int K, int N,
               const VAEStrategy& strategy) {
        dim3 blockSize(TILE_DIM, TILE_DIM);
        dim3 gridSize;

        switch (strategy) {
            case VAEStrategy::NAIVE: 
                gridSize = dim3((N + blockSize.x - 1) / blockSize.x,
                                (M + blockSize.y - 1) / blockSize.y);
                DEBUG("Launching sgemm_naive_kernel...");
                sgemm_naive_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
                break;
            case VAEStrategy::TILING: 
                gridSize = dim3((N + blockSize.x - 1) / blockSize.x,
                                (M + blockSize.y - 1) / blockSize.y);
                DEBUG("Launching sgemm_tiling_kernel...");
                sgemm_tiling_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
                break;
            case VAEStrategy::PADDING: 
                gridSize = dim3((N + blockSize.x - 1) / blockSize.x,
                                (M + blockSize.y - 1) / blockSize.y);
                DEBUG("Launching sgemm_padding_kernel...");
                sgemm_padding_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
                break;
            case VAEStrategy::VECTORIZED: 
            case VAEStrategy::OPTIMIZED:
            default:
                gridSize = dim3((N + (blockSize.x * COARSE_FACTOR) - 1) / (blockSize.x * COARSE_FACTOR),
                                (M + blockSize.y - 1) / blockSize.y);
                DEBUG("Launching sgemm_vec4_kernel...");
                sgemm_vec4_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
                break; 
        }

        CUDA_CHECK(cudaGetLastError());
    } 

    void transpose(const float* d_A,
                   float* d_AT,
                   int M, int N,
                   const VAEStrategy& strategy) {
        dim3 blockSize(TILE_DIM, TILE_DIM);;
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                      (M + blockSize.y - 1) / blockSize.y);

        switch (strategy) {
            case VAEStrategy::NAIVE: 
                DEBUG("Launching transpose_naive_kernel...");
                transpose_naive_kernel<<<gridSize, blockSize>>>(d_A, d_AT, M, N);
                break;
            case VAEStrategy::TILING: 
                DEBUG("Launching transpose_tiling_kernel...");
                transpose_tiling_kernel<<<gridSize, blockSize>>>(d_A, d_AT, M, N);
                break;
            case VAEStrategy::PADDING:
            case VAEStrategy::OPTIMIZED: 
            default:
                DEBUG("Launching transpose_padding_kernel...");
                transpose_padding_kernel<<<gridSize, blockSize>>>(d_A, d_AT, M, N);
                break;
        }

        CUDA_CHECK(cudaGetLastError());
    }

    void add_in_place(float* d_A, 
                      const float* d_B, 
                      int size,
                      const VAEStrategy& strategy) {
        const int blockSize = 512;
        int gridSize;
        switch(strategy) {
            case VAEStrategy::NAIVE:
                gridSize  = (size + blockSize - 1) / blockSize;
                DEBUG("Launching add_inplace_naive_kernel...");
                add_inplace_naive_kernel<<<gridSize, blockSize>>>(d_A, d_B, size);
                break;
            case VAEStrategy::VECTORIZED:
            case VAEStrategy::OPTIMIZED:
            default:
                gridSize  = ((size + 3) / 4 + blockSize - 1) / blockSize;
                DEBUG("Launching add_inplace_vec4_kernel...");
                add_inplace_vec4_kernel<<<gridSize, blockSize>>>(d_A, d_B, size);
                break;
        }

        CUDA_CHECK(cudaGetLastError());
    } 

} // namespace linalg
