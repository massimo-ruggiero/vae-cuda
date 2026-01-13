#include "fused.cuh"
#include "linear.cuh"
#include "activations.cuh"
#include "utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>


using namespace nvcuda;

static constexpr int WARP_SIZE = 32;
static constexpr int TILE_DIM = 16;

static constexpr int TILE_ELEMENTS = TILE_DIM * TILE_DIM; 
static constexpr int SHMEM_HALFS_PER_WARP = 2 * TILE_ELEMENTS;


// ==============================
// Device helpers 
// ==============================
template<typename ActivationFunction>
__device__ __forceinline__ void wmma_linear_core(const float* X,
                                                 const float* W,
                                                 const float* b,
                                                 float* Z,
                                                 float* A,
                                                 int M, int K, int N,
                                                 ActivationFunction fn) {
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;                 
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;  

    const int threads_per_block = blockDim.x * blockDim.y;
    const int warps_per_block   = threads_per_block / WARP_SIZE;

    const int tile_y = blockIdx.y;
    const int tile_x = blockIdx.x * warps_per_block + warp_id;

    const int row0 = tile_y * TILE_DIM;   
    const int col0 = tile_x * TILE_DIM;

    if (row0 >= M || col0 >= N) return;

    extern __shared__ half shmem[];
    half* shmem_warp_base = shmem + warp_id * SHMEM_HALFS_PER_WARP;
    half* shmem_warp_X = shmem_warp_base;
    half* shmem_warp_W = shmem_warp_base + TILE_ELEMENTS;

    wmma::fragment<wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, half,  wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, half,  wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += TILE_DIM) {

        #pragma unroll
        for (int step = 0; step < TILE_ELEMENTS; step += WARP_SIZE) {
            const int t = step + lane_id;
            const int r = t / TILE_DIM;
            const int c = t % TILE_DIM;

            // X tile
            float x_val = 0.0f;
            const int x_r = row0 + r;
            const int x_c = k0 + c;

            if (x_r < M && x_c < K) {
                x_val = X[x_r * K + x_c];
            }
            shmem_warp_X[r * TILE_DIM + c] = __float2half(x_val);

            // W tile
            float w_val = 0.0f;
            const int w_r = k0 + r;
            const int w_c = col0 + c;

            if (w_r < K && w_c < N) {
                w_val = W[w_r * N + w_c];
            }
            shmem_warp_W[c * TILE_DIM + r] = __float2half(w_val);
        }

         __syncwarp();

        wmma::load_matrix_sync(a_frag, shmem_warp_X, TILE_DIM); // ld = 16
        wmma::load_matrix_sync(b_frag, shmem_warp_W, TILE_DIM); // ld = 16

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncwarp();
    }

    float* shmem_out = reinterpret_cast<float*>(shmem_warp_base); 
    wmma::store_matrix_sync(shmem_out, c_frag, TILE_DIM, wmma::mem_row_major);
    __syncwarp();

    #pragma unroll
    for (int step = 0; step < TILE_ELEMENTS; step += WARP_SIZE) {

        const int t = step + lane_id;
        const int r = t / TILE_DIM;
        const int c = t % TILE_DIM;

        const int out_r = row0 + r;
        const int out_c = col0 + c;

        if (out_r < M && out_c < N) {
            float val = shmem_out[r * TILE_DIM + c];
            val += b[out_c];
            if (Z) Z[out_r * N + out_c] = val;
            A[out_r * N + out_c] = fn(val);
        }
    }
}


// ===================================
// Kernels: Linear Layer + Leaky Relu
// ===================================

__global__ void linear_lrelu_wmma_kernel(const float* __restrict__ X, 
                                         const float* __restrict__ W, 
                                         const float* __restrict__ b, 
                                         float* __restrict__ A, 
                                         int M, int K, int N, 
                                         float alpha) {
    wmma_linear_core(X, W, b, nullptr, A, 
                    M, K, N, 
                    [alpha] __device__ (float v) { 
                        return (v > 0.0f) ? v : v * alpha; 
                    });
}

__global__ void linear_sigmoid_wmma_kernel(const float* __restrict__ X, 
                                           const float* __restrict__ W, 
                                           const float* __restrict__ b, 
                                           float* __restrict__ Z
                                           float* __restrict__ A, 
                                           int M, int K, int N) {
    wmma_linear_core(X, W, b, Z, A, 
                     M, K, N, 
                     [] __device__ (float v) { 
                        return 1.0f / (1.0f + __expf(-v)); 
                    });
}

__global__ void linear_wmma_kernel(const float* __restrict__ X, 
                                   const float* __restrict__ W, 
                                   const float* __restrict__ b, 
                                    float* __restrict__ Z, 
                                    int M, int K, int N) {
    wmma_linear_core(X, W, b, nullptr, Z, 
                     M, K, N, [] __device__ (float v) { 
                        return v; 
                    });
}


namespace fused {

    namespace forward {

        void linear_lrelu_tc(const float* d_X,
                            const float* d_W,
                            const float* d_b,
                            float* d_A,
                            int M, int K, int N,
                            float alpha) {
            const int warps_per_block = 4;
            dim3 blockSize(warps_per_block * WARP_SIZE, 1);

            int tiles_y = (M + TILE_DIM - 1) / TILE_DIM;
            int tiles_x = (N + TILE_DIM - 1) / TILE_DIM;

            dim3 gridSize(
                (tiles_x + warps_per_block - 1) / warps_per_block,
                tiles_y
            );

            size_t shmem_size = warps_per_block * SHMEM_HALFS_PER_WARP * sizeof(half);

            linear_lrelu_wmma_kernel<<<gridSize, blockSize, shmem_size>>>(
                d_X, d_W, d_b, d_A, M, K, N, alpha
            );

            CUDA_CHECK(cudaGetLastError());
        }

        void linear_sigmoid_tc(const float* d_X,
                               const float* d_W,
                               const float* d_b,
                               float* d_Z,
                               float* d_A,
                               int M, int K, int N) {
            const int warps_per_block = 4;
            dim3 blockSize(warps_per_block * WARP_SIZE, 1);

            int tiles_y = (M + TILE_DIM - 1) / TILE_DIM;
            int tiles_x = (N + TILE_DIM - 1) / TILE_DIM;

            dim3 gridSize(
                (tiles_x + warps_per_block - 1) / warps_per_block,
                tiles_y
            );

            size_t shmem_size = warps_per_block * SHMEM_HALFS_PER_WARP * sizeof(half);

            linear_sigmoid_wmma_kernel<<<gridSize, blockSize, shmem_size>>>(
                d_X, d_W, d_b, d_Z, d_A, M, K, N
            );

            CUDA_CHECK(cudaGetLastError());
        }

        void linear_tc(const float* d_X,
                       const float* d_W,
                       const float* d_b,
                       float* d_Z,
                       int M, int K, int N) {
            const int warps_per_block = 4;
            dim3 blockSize(warps_per_block * WARP_SIZE, 1);

            int tiles_y = (M + TILE_DIM - 1) / TILE_DIM;
            int tiles_x = (N + TILE_DIM - 1) / TILE_DIM;

            dim3 gridSize(
                (tiles_x + warps_per_block - 1) / warps_per_block,
                tiles_y
            );

            size_t shmem_size = warps_per_block * SHMEM_HALFS_PER_WARP * sizeof(half);

            linear_wmma_kernel<<<gridSize, blockSize, shmem_size>>>(
                d_X, d_W, d_b, d_Z, M, K, N
            );

            CUDA_CHECK(cudaGetLastError());
        }

    } // namespace forward

} // namespace fused
