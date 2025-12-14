#include "activations.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void leaky_relu_forward_kernel(float* Z,
                                  const float alpha,
                                  int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = Z[idx];
        Z[idx] = (z > 0.0f) ? z : z * alpha;
    }
}

__global__ void leaky_relu_backward_kernel()

__global__ void sigmoid_forward_kernel(float* Z,
                                       int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float z = Z[idx]; 
        Z[idx] = 1.0f / (1.0f + expf(-z)); // TODO: ottimizzare con __expf
    }
}

__global__ void sigmoid_backward_kernel()

void leaky_relu_forward(float* d_Z,
                const float d_alpha,
                int size) {
    const int blockSize = 256;
    const int gridSize = (size + blockSize - 1) / blockSize;
    DEBUG("Launching leaky_relu_kernel...");
    leaky_relu_forward_kernel<<<gridSize, blockSize>>>(d_Z, d_alpha, size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void leaky_relu_backward() {
    
}

void sigmoid_forward(float* d_Z,
                     int size) {
    const int blockSize = 256;
    const int gridSize = (size + blockSize - 1) / blockSize;
    DEBUG("Launching sigmoid_forward_kernel...");
    sigmoid_forward_kernel<<<gridSize, blockSize>>>(d_Z, size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void sigmoid_backward() {
    
}