#include "reparametrization.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>


__global__ void init_states_kernel(curandStatePhilox4_32_10_t* states,
                                   int size,
                                   unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curand_init(seed, (unsigned long long)idx, 0ULL, &states[idx]);
    }
}

__global__ void reparametrization_forward_kernel(const float* mu,
                                                 const float* logvar,
                                                 float* z,
                                                 float* epsilon,
                                                 curandStatePhilox4_32_10_t* states,
                                                 int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandStatePhilox4_32_10_t state = states[idx];
        float sigma = expf(0.5f * logvar[idx]); // forse va fatto clamp logvar
        float eps = curand_normal(&state);

        z[idx] = mu[idx] + sigma * eps;
        epsilon[idx] = eps;

        states[idx] = state;
    }
}

__global__ void reparametrization_backward_kernel(const float* dz,
                                                  const float* logvar,
                                                  const float* epsilon,
                                                  float* dmu,
                                                  float* dlogvar,
                                                  int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dmu[idx] = dz[idx];
        float sigma_idx = expf(0.5f * logvar[idx]);
        dlogvar[idx] = dz[idx] * epsilon[idx] * 0.5f * sigma_idx;
    }
}

namespace reparametrization {

    void init(curandStatePhilox4_32_10_t* d_states, 
              int size, 
              unsigned long long seed) {
        const int blockSize = 256;
        const int gridSize = (size + blockSize - 1) / blockSize;
        DEBUG("Initializing cuRAND Philox states...");
        init_states_kernel<<<gridSize, blockSize>>>(d_states, size, seed);
        
        CUDA_CHECK(cudaGetLastError());
    }

    void forward(const float* d_mu,
                    const float* d_logvar,
                    float* d_z,
                    float* d_epsilon,
                    curandStatePhilox4_32_10_t* d_states,
                    int size) {

        const int blockSize = 256;
        const int gridSize = (size + blockSize - 1) / blockSize;
        DEBUG("Launching reparametrization_forward_kernel...");
        reparametrization_forward_kernel<<<gridSize, blockSize>>>(d_mu, d_logvar, d_z, d_epsilon, d_states, size);

        CUDA_CHECK(cudaGetLastError());
    }

    void backward(const float* d_dz,
                  const float* d_logvar,
                  const float* d_epsilon,
                  float* d_dmu,
                  float* d_dlogvar,
                  int size) {
        const int blockSize = 256;
        const int gridSize = (size + blockSize - 1) / blockSize;
        DEBUG("Launching reparametrization_backward_kernel...");
        reparametrization_backward_kernel<<<gridSize, blockSize>>>(d_dz, d_logvar, d_epsilon, d_dmu, d_dlogvar, size);

        CUDA_CHECK(cudaGetLastError());
    }
}
