#include "reparametrization.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>


// ==============================
// Kernels: State initialization
// ==============================

__global__ void init_states_kernel(curandStatePhilox4_32_10_t* __restrict__ states,
                                   int num_states,
                                   unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_states) {
        curand_init(seed, (unsigned long long)idx, 0ULL, &states[idx]);
    }
}


// ==========================================
// Forward kernels: Reparametrization trick
// ==========================================

__global__ void reparametrization_forward_naive_kernel(const float* __restrict__ mu,
                                                       const float* __restrict__ logvar,
                                                       float* __restrict__ z,
                                                       float* __restrict__ epsilon,
                                                       curandStatePhilox4_32_10_t* __restrict__ states,
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

__global__ void reparametrization_forward_vec4_kernel(const float* __restrict__ mu,
                                                      const float* __restrict__ logvar,
                                                      float* __restrict__ z,
                                                      float* __restrict__ epsilon,
                                                      curandStatePhilox4_32_10_t* __restrict__ states,
                                                      int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        float4 mu_vec = *reinterpret_cast<const float4*>(&mu[idx]);
        float4 logvar_vec = *reinterpret_cast<const float4*>(&logvar[idx]);
        
        curandStatePhilox4_32_10_t state = states[idx / 4];
        float4 eps_vec = curand_normal4(&state);

        float4 z_vec;
        z_vec.x = mu_vec.x + expf(0.5f * logvar_vec.x) * eps_vec.x;
        z_vec.y = mu_vec.y + expf(0.5f * logvar_vec.y) * eps_vec.y;
        z_vec.z = mu_vec.z + expf(0.5f * logvar_vec.z) * eps_vec.z;
        z_vec.w = mu_vec.w + expf(0.5f * logvar_vec.w) * eps_vec.w;

        states[idx / 4] = state;

        *reinterpret_cast<float4*>(&z[idx]) = z_vec;
        *reinterpret_cast<float4*>(&epsilon[idx]) = eps_vec;
    } else if (idx < size) {
        curandStatePhilox4_32_10_t state = states[idx / 4];
        for (int i = idx; i < size; ++i) {
            float sigma = expf(0.5f * logvar[i]); 
            float eps = curand_normal(&state);

            z[i] = mu[i] + sigma * eps;
            epsilon[i] = eps;
        }
        states[idx / 4] = state;
    }
}


// ==========================================
// Backward kernels: Reparametrization trick
// ==========================================

__global__ void reparametrization_backward_naive_kernel(const float* __restrict__ dz,
                                                        const float* __restrict__ logvar,
                                                        const float* __restrict__ epsilon,
                                                        float* __restrict__ dmu,
                                                        float* __restrict__ dlogvar,
                                                        int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dmu[idx] = dz[idx];
        float sigma_idx = expf(0.5f * logvar[idx]);
        dlogvar[idx] = dz[idx] * epsilon[idx] * 0.5f * sigma_idx;
    }
}

__global__ void reparametrization_backward_vec4_kernel(const float* __restrict__ dz,
                                                       const float* __restrict__ logvar,
                                                       const float* __restrict__ epsilon,
                                                       float* __restrict__ dmu,
                                                       float* __restrict__ dlogvar,
                                                       int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        float4 dz_vec = *reinterpret_cast<const float4*>(&dz[idx]);
        float4 logvar_vec = *reinterpret_cast<const float4*>(&logvar[idx]);
        float4 epsilon_vec = *reinterpret_cast<const float4*>(&epsilon[idx]);
        float4 dmu_vec;
        float4 dlogvar_vec;

        dmu_vec.x = dz_vec.x;
        dmu_vec.y = dz_vec.y;
        dmu_vec.z = dz_vec.z;
        dmu_vec.w = dz_vec.w;

        dlogvar_vec.x = dz_vec.x * epsilon_vec.x * 0.5f * expf(0.5f * logvar_vec.x);
        dlogvar_vec.y = dz_vec.y * epsilon_vec.y * 0.5f * expf(0.5f * logvar_vec.y);
        dlogvar_vec.z = dz_vec.z * epsilon_vec.z * 0.5f * expf(0.5f * logvar_vec.z);
        dlogvar_vec.w = dz_vec.w * epsilon_vec.w * 0.5f * expf(0.5f * logvar_vec.w);

        *reinterpret_cast<float4*>(&dmu[idx]) = dmu_vec;
        *reinterpret_cast<float4*>(&dlogvar[idx]) = dlogvar_vec;
    } else if (idx < size) {
        for (int i = idx; i < size; ++i) {
            dmu[i] = dz[i];
            float sigma_i = expf(0.5f * logvar[i]);
            dlogvar[i] = dz[i] * epsilon[i] * 0.5f * sigma_i;
        }
    }
}


// ==============================
// Host API
// ==============================

namespace reparametrization {

    void init(curandStatePhilox4_32_10_t* d_states, 
              int num_states, 
              unsigned long long seed,
              const VAEStrategy& strategy) {
        const int blockSize = 512;
        int gridSize;

        switch(strategy) {
            case VAEStrategy::NAIVE:
            case VAEStrategy::TILING:
            case VAEStrategy::PADDING:
            case VAEStrategy::REDUCTION:
            case VAEStrategy::UNROLLED_REDUCTION:
            case VAEStrategy::WARP_REDUCTION:
                gridSize = (num_states + blockSize - 1) / blockSize;
                DEBUG("Initializing cuRAND Philox states...");
                init_states_kernel<<<gridSize, blockSize>>>(d_states, num_states, seed);
                break;
            case VAEStrategy::VECTORIZED:
            default:
                gridSize = (num_states + blockSize - 1) / blockSize;
                DEBUG("Initializing cuRAND Philox states...");
                init_states_kernel<<<gridSize, blockSize>>>(d_states, num_states, seed);
                break;
        }
        
        CUDA_CHECK(cudaGetLastError());
    }

    void forward(const float* d_mu,
                    const float* d_logvar,
                    float* d_z,
                    float* d_epsilon,
                    curandStatePhilox4_32_10_t* d_states,
                    int size,
                    const VAEStrategy& strategy) {

        const int blockSize = 512;
        int gridSize;

        switch(strategy) {
            case VAEStrategy::NAIVE:
            case VAEStrategy::TILING:
            case VAEStrategy::PADDING:
            case VAEStrategy::REDUCTION:
            case VAEStrategy::UNROLLED_REDUCTION:
            case VAEStrategy::WARP_REDUCTION:
                gridSize = (size + blockSize - 1) / blockSize;
                DEBUG("Launching reparametrization_forward_kernel...");
                reparametrization_forward_naive_kernel<<<gridSize, blockSize>>>(d_mu, d_logvar, d_z, d_epsilon, d_states, size);
                break;
            case VAEStrategy::VECTORIZED:
            case VAEStrategy::OPTIMIZED: 
            default:
                gridSize = ((size + 3) / 4 + blockSize - 1) / blockSize;
                DEBUG("Launching reparametrization_forward_vec4_kernel...");
                reparametrization_forward_vec4_kernel<<<gridSize, blockSize>>>(d_mu, d_logvar, d_z, d_epsilon, d_states, size);
                break;
        }

        CUDA_CHECK(cudaGetLastError());
    }

    void backward(const float* d_dz,
                  const float* d_logvar,
                  const float* d_epsilon,
                  float* d_dmu,
                  float* d_dlogvar,
                  int size,
                  const VAEStrategy& strategy) {
        const int blockSize = 512;
        int gridSize;

        switch(strategy) {
            case VAEStrategy::VECTORIZED:
                gridSize = ((size + 3) / 4 + blockSize - 1) / blockSize;
                DEBUG("Launching reparametrization_backward_vec4_kernel...");
                reparametrization_backward_vec4_kernel<<<gridSize, blockSize>>>(d_dz, d_logvar, d_epsilon, d_dmu, d_dlogvar, size);
                break;
            case VAEStrategy::NAIVE:
            case VAEStrategy::OPTIMIZED: 
            default:
                gridSize = (size + blockSize - 1) / blockSize;
                DEBUG("Launching reparametrization_backward_kernel...");
                reparametrization_backward_naive_kernel<<<gridSize, blockSize>>>(d_dz, d_logvar, d_epsilon, d_dmu, d_dlogvar, size);
                break;
        }

        CUDA_CHECK(cudaGetLastError());
    }
}
