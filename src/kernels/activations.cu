#include "activations.cuh"
#include "utils.cuh"

#include <cuda_runtime.h>
#include <math.h>


// ==============================
// Forward kernels: LeakyReLU
// ==============================

__global__ void leaky_relu_forward_naive_kernel(const float* Z,
                                                float* A,
                                                float alpha,
                                                int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = Z[idx];
        A[idx] = (z > 0.0f) ? z : z * alpha;
    }
}

__global__ void leaky_relu_forward_vec4_kernel(const float* Z,
                                                     float* A,
                                                     float alpha,
                                                     int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        float4 Z_vec = *reinterpret_cast<const float4*>(&Z[idx]);
        float4 A_vec;

        A_vec.x = fmaxf(Z_vec.x, alpha * Z_vec.x);
        A_vec.y = fmaxf(Z_vec.y, alpha * Z_vec.y);
        A_vec.z = fmaxf(Z_vec.z, alpha * Z_vec.z);
        A_vec.w = fmaxf(Z_vec.w, alpha * Z_vec.w);

        *reinterpret_cast<float4*>(&A[idx]) = A_vec;
    } else if (idx < size){
        for (int i = idx; i < size; ++i) {
            float z = Z[i];
            A[i] = (z > 0.0f) ? z : z * alpha;
        }
    }
}


// ==============================
// Forward kernels: Sigmoid
// ==============================

__global__ void sigmoid_forward_naive_kernel(const float* Z,
                                             float* A,
                                             int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { 
        A[idx] = 1.0f / (1.0f + expf(-Z[idx])); // TODO: ottimizzare con __expf (SFU) - Micro-Optimization
    }
}

__global__ void sigmoid_forward_vec4_kernel(const float* Z,
                                                  float* A,
                                                  int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) { 
        float4 Z_vec = *reinterpret_cast<const float4*>(&Z[idx]);
        float4 A_vec;

        A_vec.x = 1.0f / (1.0f + __expf(-Z_vec.x));
        A_vec.y = 1.0f / (1.0f + __expf(-Z_vec.y));
        A_vec.z = 1.0f / (1.0f + __expf(-Z_vec.z));
        A_vec.w = 1.0f / (1.0f + __expf(-Z_vec.w));
    
        *reinterpret_cast<float4*>(&A[idx]) = A_vec;
    } else if (idx < size) {
        for (int i = idx; i < size; ++i) {
            A[i] = 1.0f / (1.0f + expf(-Z[i])); // TODO: __frcp_rn(x) (SFU) - Micro-Optimization
        }
    }
}


// ==============================
// Backward kernels: LeakyReLU
// ==============================

__global__ void leaky_relu_backward_naive_kernel(const float* Z,
                                                 const float* dA,
                                                 float* dZ,
                                                 float alpha,
                                                 int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = Z[idx];
        float g = (z > 0.0f) ? 1.0f : alpha;
        dZ[idx] = dA[idx] * g;
    }
}

__global__ void leaky_relu_backward_vec4_kernel(const float* Z,
                                                 const float* dA,
                                                 float* dZ,
                                                 float alpha,
                                                 int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        float4 Z_vec = *reinterpret_cast<const float4*>(&Z[idx]);
        float4 dA_vec = *reinterpret_cast<const float4*>(&dA[idx]);
        float4 dZ_vec;

        dZ_vec.x = dA_vec.x * fmaf((Z_vec.x > 0.0f), 1.0f - alpha, alpha);
        dZ_vec.y = dA_vec.y * fmaf((Z_vec.y > 0.0f), 1.0f - alpha, alpha);
        dZ_vec.z = dA_vec.z * fmaf((Z_vec.z > 0.0f), 1.0f - alpha, alpha);
        dZ_vec.w = dA_vec.w * fmaf((Z_vec.w > 0.0f), 1.0f - alpha, alpha);


        *reinterpret_cast<float4*>(&dZ[idx]) = dZ_vec;
    } else if (idx < size) {
        for (int i = idx; i < size; ++i){
            float z = Z[i];
            float g = (z > 0.0f) ? 1.0f : alpha;
            dZ[i] = dA[i] * g;
        }
    }
}


// ==============================
// Backward kernels: Sigmoid
// ==============================

__global__ void sigmoid_backward_naive_kernel(const float* A,
                                              const float* dA,
                                              float* dZ,
                                              int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float a = A[idx];
        dZ[idx] = dA[idx] * a * (1.0f - a);
    }
}

__global__ void sigmoid_backward_vec4_kernel(const float* A,
                                                   const float* dA,
                                                   float* dZ,
                                                   int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        float4 A_vec = *reinterpret_cast<const float4*>(&A[idx]);
        float4 dA_vec = *reinterpret_cast<const float4*>(&dA[idx]);
        float4 dZ_vec;

        dZ_vec.x = dA_vec.x * A_vec.x * (1.0f - A_vec.x);
        dZ_vec.y = dA_vec.y * A_vec.y * (1.0f - A_vec.y);
        dZ_vec.z = dA_vec.z * A_vec.z * (1.0f - A_vec.z);
        dZ_vec.w = dA_vec.w * A_vec.w * (1.0f - A_vec.w);

        *reinterpret_cast<float4*>(&dZ[idx]) = dZ_vec;

    } else if (idx < size) {
        for (int i = idx; i < size; ++i) {
            float a = A[i];
            dZ[i] = dA[i] * a * (1.0f - a);
        }
    }
}


// ==============================
// Host API
// ==============================

namespace activations {

    namespace leaky_relu {

        void forward(const float* d_Z,
                     float* d_A,
                     float d_alpha,
                     int size,
                     const VAEStrategy& strategy) {
            const int blockSize = 256;
            int gridSize;
            switch(strategy) {
                case VAEStrategy::NAIVE:
                case VAEStrategy::TILING: 
                case VAEStrategy::PADDING: 
                case VAEStrategy::REDUCTION: 
                case VAEStrategy::UNROLLED_REDUCTION: 
                case VAEStrategy::WARP_REDUCTION: 
                    DEBUG("Launching leaky_relu_forward_naive_kernel...");
                    gridSize = (size + blockSize - 1) / blockSize;
                    leaky_relu_forward_naive_kernel<<<gridSize, blockSize>>>(d_Z, d_A, d_alpha, size);
                    break;
                case VAEStrategy::VECTORIZED:
                case VAEStrategy::OPTIMIZED:
                default:
                    DEBUG("Launching leaky_relu_forward_kernel...");
                    gridSize = ((size + 3) / 4 + blockSize - 1) / blockSize;
                    leaky_relu_forward_vec4_kernel<<<gridSize, blockSize>>>(d_Z, d_A, d_alpha, size);
                    break;
            }

            CUDA_CHECK(cudaGetLastError());
        }

        void backward(const float* d_Z,
                      const float* d_dA,
                      float* d_dZ,
                      float d_alpha,
                      int size,
                      const VAEStrategy& strategy) {
            const int blockSize = 256;
            int gridSize;
            switch(strategy) {
                case VAEStrategy::NAIVE:
                case VAEStrategy::TILING: 
                case VAEStrategy::PADDING: 
                case VAEStrategy::REDUCTION: 
                case VAEStrategy::UNROLLED_REDUCTION: 
                case VAEStrategy::WARP_REDUCTION: 
                    DEBUG("Launching leaky_relu_backward_naive_kernel...");
                    gridSize = (size + blockSize - 1) / blockSize;
                    leaky_relu_backward_naive_kernel<<<gridSize, blockSize>>>(d_Z, d_dA, d_dZ, d_alpha, size);
                    break;
                case VAEStrategy::VECTORIZED:
                case VAEStrategy::OPTIMIZED:
                default:
                    DEBUG("Launching leaky_relu_backward_vec4_kernel...");
                    gridSize = ((size + 3) / 4 + blockSize - 1) / blockSize;
                    leaky_relu_backward_vec4_kernel<<<gridSize, blockSize>>>(d_Z, d_dA, d_dZ, d_alpha, size);
                    break;
            }

            CUDA_CHECK(cudaGetLastError());
        }

    } // namespace leaky_relu

    namespace sigmoid {

        void forward(const float* d_Z, 
                     float* d_A,
                     int size, 
                     const VAEStrategy& strategy) {
            const int blockSize = 256;
            int gridSize;
            switch(strategy) {
                case VAEStrategy::VECTORIZED:
                    DEBUG("Launching sigmoid_forward_vec4_kernel...");
                    gridSize = ((size + 3) / 4 + blockSize - 1) / blockSize;
                    sigmoid_forward_vec4_kernel<<<gridSize, blockSize>>>(d_Z, d_A, size);
                    break;
                case VAEStrategy::NAIVE:
                case VAEStrategy::OPTIMIZED: 
                    DEBUG("Launching sigmoid_forward_naive_kernel...");
                    gridSize = (size + blockSize - 1) / blockSize;
                    sigmoid_forward_naive_kernel<<<gridSize, blockSize>>>(d_Z, d_A, size);
                    break;
            }
            CUDA_CHECK(cudaGetLastError());
        }

        void backward(const float* d_A, 
                      const float* d_dA, 
                      float* d_dZ, 
                      int size, 
                      const VAEStrategy& strategy) {
            const int blockSize = 256;
            int gridSize;
            switch(strategy) {
                case VAEStrategy::NAIVE:
                case VAEStrategy::TILING: 
                case VAEStrategy::PADDING: 
                case VAEStrategy::REDUCTION: 
                case VAEStrategy::UNROLLED_REDUCTION: 
                case VAEStrategy::WARP_REDUCTION: 
                    DEBUG("Launching sigmoid_backward_naive_kernel...");
                    gridSize = (size + blockSize - 1) / blockSize;
                    sigmoid_backward_naive_kernel<<<gridSize, blockSize>>>(d_A, d_dA, d_dZ, size);
                    break;
                case VAEStrategy::VECTORIZED:
                case VAEStrategy::OPTIMIZED: 
                default:
                    DEBUG("Launching sigmoid_backward_vec4_kernel...");
                    gridSize = ((size + 3) / 4 + blockSize - 1) / blockSize;
                    sigmoid_backward_vec4_kernel<<<gridSize, blockSize>>>(d_A, d_dA, d_dZ, size);
                    break;
            }
            CUDA_CHECK(cudaGetLastError());
        }

    } // namespace sigmoid

} // namespace activations