#include "linear.cuh"
#include "linalg.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>

// add_bias
__global__ void add_bias_naive_kernel(float* Z,
                                      const float* b,
                                      int batch_size,
                                      int output_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch_size && col < output_dim) {
        Z[row * output_dim + col] += b[col];
    }
}

__global__ void add_bias_vectorized_kernel(float* Z,
                                           const float* b,
                                           int batch_size,
                                           int output_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (row < batch_size && col + 3 < output_dim) {
        int idx = row * output_dim + col;
        float4 Z_vec = *reinterpret_cast<float4*>(&Z[idx]);
        float4 b_vec = *reinterpret_cast<const float4*>(&b[col]);

        Z_vec.x += b_vec.x;
        Z_vec.y += b_vec.y;
        Z_vec.z += b_vec.z;
        Z_vec.w += b_vec.w;

        *reinterpret_cast<float4*>(&Z[idx]) = Z_vec;
    } else if (row < batch_size && col < output_dim) {
        for (int c = col; c < output_dim; ++c) {
            Z[row * output_dim + c] += b[c];
        }
    }
}

// compute db                                           
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

// TODO: vedere se fare altre versioni


namespace linear {

    void forward(const float* d_X,
                    const float* d_W,
                    const float* d_b,
                    float* d_Z,
                    int batch_size,
                    int input_dim,
                    int output_dim,
                    const VAEStrategy& strategy) {

        linalg::sgemm(d_X, d_W, d_Z, batch_size, input_dim, output_dim, strategy);

        dim3 blockSize(16, 16);
        switch(strategy) {
            case VAEStrategy::NAIVE:
                DEBUG("Launching add_bias_naive_kernel...");
                dim3 gridSize((output_dim + blockSize.x - 1) / blockSize.x,
                              (batch_size + blockSize.y - 1) / blockSize.y);
                add_bias_naive_kernel<<<gridSize, blockSize>>>(d_Z, d_b, batch_size, output_dim);
                break;
            case VAEStrategy::VECTORIZED:
                DEBUG("Launching add_bias_vectorized_kernel...");
                dim3 gridSize(((output_dim + 3) / 4 + blockSize.x - 1) / blockSize.x,
                              (batch_size + blockSize.y - 1) / blockSize.y);
                add_bias_vectorized_kernel<<<gridSize, blockSize>>>(d_Z, d_b, batch_size, output_dim);
                break;
            default:
                DEBUG("Launching add_bias_naive_kernel...");
                dim3 gridSize((output_dim + blockSize.x - 1) / blockSize.x,
                              (batch_size + blockSize.y - 1) / blockSize.y);
                add_bias_naive_kernel<<<gridSize, blockSize>>>(d_Z, d_b, batch_size, output_dim);
                break; 
        }
        
        CUDA_CHECK(cudaGetLastError());
    }

    void backward(const float* d_X,
                    const float* d_W,
                    const float* d_dZ,
                    float* d_dX,
                    float* d_dW,
                    float* d_db,
                    int batch_size,
                    int input_dim,
                    int output_dim,
                    const VAEStrategy& strategy) {
        // d_dX
        if (d_dX != nullptr) {
            linalg::sgemm(d_dZ, d_W, d_dX, batch_size, output_dim, input_dim, false, true, strategy);
        }

        // d_dW
        linalg::sgemm(d_X, d_dZ, d_dW, input_dim, batch_size, output_dim, true, false, strategy);

        // d_db
        const int blockSize = 256;
        const int gridSize = (output_dim + blockSize - 1) / blockSize;
        
        DEBUG("Launching db_naive_kernel...");
        db_naive_kernel<<<gridSize, blockSize>>>(d_dZ, d_db, batch_size, output_dim);
        
        CUDA_CHECK(cudaGetLastError());
    }

} // namespace linear