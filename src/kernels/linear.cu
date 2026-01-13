#include "linear.cuh"
#include "linalg.cuh"
#include "utils.cuh"

#include <cuda_runtime.h>


// ==============================
// Kernels: Add bias
// ==============================

__global__ void add_bias_naive_kernel(float* __restrict__ Z,
                                      const float* __restrict__ b,
                                      int batch_size,
                                      int output_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch_size && col < output_dim) {
        Z[row * output_dim + col] += b[col];
    }
}

__global__ void add_bias_vec4_kernel(float* __restrict__ Z,
                                     const float* __restrict__ b,
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

// ==============================
// Kernels: db
// ==============================     

__global__ void db_naive_kernel(const float* __restrict__ d_dZ,
                                float* __restrict__ d_db,
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

__global__ void db_vec4_kernel(const float* __restrict__ d_dZ,
                               float* __restrict__ d_db,
                               int batch_size,
                               int output_dim) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < output_dim) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int b = 0; b < batch_size; ++b) {
            const float* row = d_dZ + b * output_dim + idx;
            float4 val = *reinterpret_cast<const float4*>(row);
            sum.x += val.x;
            sum.y += val.y;
            sum.z += val.z;
            sum.w += val.w;
        }
        *reinterpret_cast<float4*>(&d_db[idx]) = sum;
    } else if (idx < output_dim) {
        for (int i = idx; i < output_dim; ++i) {
            float sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                sum += d_dZ[b * output_dim + i];
            }
            d_db[i] = sum;
        }
    }
}


// ==============================
// Host API
// ==============================

namespace linear {

    void add_bias(float* d_Z,
                  const float* d_b,
                  int batch_size,
                  int output_dim,
                  const VAEStrategy& strategy) {
        dim3 blockSize(16, 16);
        dim3 gridSize;
        switch(strategy) {
            case VAEStrategy::VECTORIZED:
                DEBUG("Launching add_bias_vec4_kernel...");
                gridSize = dim3(((output_dim + 3) / 4 + blockSize.x - 1) / blockSize.x,
                                (batch_size + blockSize.y - 1) / blockSize.y);
                add_bias_vec4_kernel<<<gridSize, blockSize>>>(d_Z, d_b, batch_size, output_dim);
                break;
            case VAEStrategy::NAIVE:
            case VAEStrategy::OPTIMIZED:
            default:
                DEBUG("Launching add_bias_naive_kernel...");
                gridSize = dim3((output_dim + blockSize.x - 1) / blockSize.x,
                                (batch_size + blockSize.y - 1) / blockSize.y);
                add_bias_naive_kernel<<<gridSize, blockSize>>>(d_Z, d_b, batch_size, output_dim);
                break;
        }

        CUDA_CHECK(cudaGetLastError());
    }

    void db(const float* d_dZ,
            float* d_db,
            int batch_size,
            int output_dim,
            const VAEStrategy& strategy) {
        const int blockSize = 512;
        int gridSize;
        switch(strategy) {
            case VAEStrategy::VECTORIZED:
            case VAEStrategy::OPTIMIZED:
                gridSize = ((output_dim + 3) / 4 + blockSize - 1) / blockSize;
                DEBUG("Launching db_vec4_kernel...");
                db_vec4_kernel<<<gridSize, blockSize>>>(d_dZ, d_db, batch_size, output_dim);
                break;
            default:
                gridSize = (output_dim + blockSize - 1) / blockSize;
                DEBUG("Launching db_naive_kernel...");
                db_naive_kernel<<<gridSize, blockSize>>>(d_dZ, d_db, batch_size, output_dim);
                break;
        }

        CUDA_CHECK(cudaGetLastError());
    }

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
        dim3 gridSize;
        switch(strategy) {
            case VAEStrategy::NAIVE:
            case VAEStrategy::TILING:
            case VAEStrategy::PADDING:
            case VAEStrategy::REDUCTION:
            case VAEStrategy::UNROLLED_REDUCTION:
            case VAEStrategy::WARP_REDUCTION:
                DEBUG("Launching add_bias_naive_kernel...");
                gridSize = dim3((output_dim + blockSize.x - 1) / blockSize.x,
                                (batch_size + blockSize.y - 1) / blockSize.y);
                add_bias_naive_kernel<<<gridSize, blockSize>>>(d_Z, d_b, batch_size, output_dim);
                break;
            case VAEStrategy::VECTORIZED:
            default:
                DEBUG("Launching add_bias_vec4_kernel...");
                gridSize = dim3(((output_dim + 3) / 4 + blockSize.x - 1) / blockSize.x,
                                (batch_size + blockSize.y - 1) / blockSize.y);
                add_bias_vec4_kernel<<<gridSize, blockSize>>>(d_Z, d_b, batch_size, output_dim);
                break;
        }
        
        CUDA_CHECK(cudaGetLastError());
    }

    void backward(const float* d_X,     
                  const float* d_W,     
                  const float* d_dZ,    
                  float* d_XT,          
                  float* d_WT,          
                  float* d_dX,          
                  float* d_dW,          
                  float* d_db,
                  int batch_size,
                  int input_dim,
                  int output_dim,
                  const VAEStrategy& strategy) {
        // d_dX
        if (d_dX != nullptr && d_WT != nullptr) {
            linalg::transpose(d_W, d_WT, input_dim, output_dim, strategy);
            linalg::sgemm(d_dZ, d_WT, d_dX, batch_size, output_dim, input_dim, strategy);
        }

        // d_dW
        linalg::transpose(d_X, d_XT, batch_size, input_dim, strategy);
        linalg::sgemm(d_XT, d_dZ, d_dW, input_dim, batch_size, output_dim, strategy);

        // d_db
        db(d_dZ, d_db, batch_size, output_dim, strategy);
    }

} // namespace linear
