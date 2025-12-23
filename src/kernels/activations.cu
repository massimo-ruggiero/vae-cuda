#include "activations.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <math.h>


__global__ void leaky_relu_forward_kernel(const float* Z,
                                          float* A,
                                          float alpha,
                                          int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = Z[idx];
        A[idx] = (z > 0.0f) ? z : z * alpha;
    }
}

__global__ void leaky_relu_backward_kernel(const float* Z,
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

__global__ void sigmoid_forward_kernel(const float* Z,
                                       float* A,
                                       int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { 
        A[idx] = 1.0f / (1.0f + expf(-Z[idx])); // TODO: ottimizzare con __expf
    }
}

__global__ void sigmoid_backward_kernel(const float* A,
                                        const float* dA,
                                        float* dZ,
                                        int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float a = A[idx];
        dZ[idx] = dA[idx] * a * (1.0f - a);
    }
}

namespace activations {

    namespace leaky_relu {

        void forward(const float* d_Z,
                     float* d_A,
                     int size,
                     float d_alpha) {
            const int blockSize = 256;
            const int gridSize = (size + blockSize - 1) / blockSize;
            // DEBUG("Launching leaky_relu_forward_kernel...");
            leaky_relu_forward_kernel<<<gridSize, blockSize>>>(d_Z, d_A, d_alpha, size);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        void backward(const float* d_Z,
                      const float* d_dA,
                      float* d_dZ,
                      int size,
                      float d_alpha) {
            const int blockSize = 256;
            const int gridSize = (size + blockSize - 1) / blockSize;
            // DEBUG("Launching leaky_relu_backward_kernel...");
            leaky_relu_backward_kernel<<<gridSize, blockSize>>>(d_Z, d_dA, d_dZ, d_alpha, size);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

    } // namespace leaky_relu

    namespace sigmoid {

        void forward(const float* d_Z,
                    float* d_A,
                    int size) {
            const int blockSize = 256;
            const int gridSize = (size + blockSize - 1) / blockSize;
            // DEBUG("Launching sigmoid_forward_kernel...");
            sigmoid_forward_kernel<<<gridSize, blockSize>>>(d_Z, d_A, size);

            CUDA_CHECK(cudaGetLastError());
        }

        void backward(const float* d_A,
                      const float* d_dA,
                      float* d_dZ,
                      int size) {
            const int blockSize = 256;
            const int gridSize = (size + blockSize - 1) / blockSize;
            // DEBUG("Launching sigmoid_backward_kernel...");
            sigmoid_backward_kernel<<<gridSize, blockSize>>>(d_A, d_dA, d_dZ, size);

            CUDA_CHECK(cudaGetLastError());
        }

    } // namespace sigmoid

} // namespace activations