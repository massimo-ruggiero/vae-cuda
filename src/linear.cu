#include "linear.h"
#include "matmul.h"
#include "utils.h"
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

void linear_forward(const float* d_X,
                    const float* d_W,
                    const float* d_b,
                    float* d_Z,
                    int batch_size,
                    int input_dim,
                    int output_dim) {
    matmul_naive(d_X, d_W, d_Z, batch_size, input_dim, output_dim);

    dim3 blockSize(16, 16);
    dim3 gridSize((output_dim + blockSize.x - 1) / blockSize.x,
                  (batch_size + blockSize.y - 1) / blockSize.y);
    DEBUG("Launching add_bias_kernel...");
    add_bias_kernel<<<gridSize, blockSize>>>(d_Z, d_b, batch_size, output_dim);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}