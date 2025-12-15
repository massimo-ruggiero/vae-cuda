#include "linear.cuh"
#include "linalg.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>


__global__ void add_bias_kernel(float* Z,
                                const float* b,
                                int batch_size,
                                int output_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch_size && col < output_dim) {
        Z[row * output_dim + col] += b[col];
    }
}

__global__ void db_naive_kernel(const float* d_dZ,
                                 float* d_db,
                                 int batch_size,
                                 int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_dim) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            sum += d_dZ[b * output_dim + idx];
        }
        d_db[idx] = sum;
    }
}

namespace linear {

    namespace naive {

        void forward(const float* d_X,
                     const float* d_W,
                     const float* d_b,
                     float* d_Z,
                     int batch_size,
                     int input_dim,
                     int output_dim) {
            matmul::naive(d_X, d_W, d_Z, batch_size, input_dim, output_dim);

            dim3 blockSize(16, 16);
            dim3 gridSize((output_dim + blockSize.x - 1) / blockSize.x,
                        (batch_size + blockSize.y - 1) / blockSize.y);
            DEBUG("Launching add_bias_kernel...");
            add_bias_kernel<<<gridSize, blockSize>>>(d_Z, d_b, batch_size, output_dim);
            
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        void backward(const float* d_X,
                      const float* d_W,
                      const float* d_dZ,
                      float* d_dX,
                      float* d_dW,
                      float* d_db,
                      int batch_size,
                      int input_dim,
                      int output_dim) {
            // d_dX
            float* d_WT = nullptr;
            CUDA_CHECK(cudaMalloc((void**)&d_WT, output_dim * input_dim * sizeof(float)));
            transpose::naive(d_W, d_WT, input_dim, output_dim);
            matmul::naive(d_dZ, d_WT, d_dX, batch_size, output_dim, input_dim);
            CUDA_CHECK(cudaFree(d_WT));
            
            // d_dW
            float* d_XT = nullptr;
            CUDA_CHECK(cudaMalloc((void**)&d_XT, input_dim * batch_size * sizeof(float)));
            transpose::naive(d_X, d_XT, batch_size, input_dim);
            matmul::naive(d_XT, d_dZ, input_dim, batch_size, output_dim);
            CUDA_CHECK(cudaFree(d_XT));

            // d_db
            const int blockSize = 256;
            const int gridSize = (output_dim + blockSize - 1) / blockSize;
            DEBUG("Launching db_naive_kernel...");
            db_naive_kernel<<<gridSize, blockSize>>>(d_dZ, d_db, batch_size, output_dim);
            
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

    } // namespace naive

} // namespace linear