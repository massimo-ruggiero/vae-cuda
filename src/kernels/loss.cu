#include "loss.cuh"
#include "utils.cuh"

#include <cuda_runtime.h>
#include <math.h>


static constexpr int WARP_SIZE = 32;
static constexpr unsigned FULL_MASK = 0xFFFFFFFFu;


// ==============================
// Device helpers 
// ==============================

__device__ __forceinline__ float compute_bce(float x, float z) {
    return fmaxf(z, 0.0f) - z * x + logf(1.0f + expf(-fabsf(z)));
}

__device__ __forceinline__ float compute_kl(float mu, float logvar) {
    return 0.5f * (mu * mu + expf(logvar) - 1.0f - logvar);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}


// ==============================
// Forward kernels: BCE
// ==============================

__global__ void bce_forward_naive_kernel(const float* __restrict__ X,
                                         const float* __restrict__ Z,
                                         float* bce_sum,
                                         int size,
                                         float inv_batch) {                                        
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float bce = compute_bce(X[idx], Z[idx]);
        atomicAdd(bce_sum, bce * inv_batch);
    }
}

__global__ void bce_forward_reduction_kernel(const float* __restrict__ X,
                                             const float* __restrict__ Z,
                                             float* bce_sum,
                                             int size,
                                             float inv_batch) {  
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local = 0.0f;
    if (idx < size) {
        local = compute_bce(X[idx], Z[idx]);
    }
    sdata[tid] = local;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(bce_sum, sdata[0] * inv_batch);
    }
}

__global__ void bce_forward_unrolled_reduction_kernel(const float* __restrict__ X,
                                                      const float* __restrict__ Z,
                                                      float* bce_sum,
                                                      int size,
                                                      float inv_batch) {  
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local = 0.0f;
    if (idx < size) {
        local = compute_bce(X[idx], Z[idx]);
    }
    sdata[tid] = local;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid < WARP_SIZE) {
        volatile float* vshm = sdata;
        if (blockDim.x >= 64) vshm[tid] += vshm[tid + 32];
        vshm[tid] += vshm[tid + 16];
        vshm[tid] += vshm[tid + 8];
        vshm[tid] += vshm[tid + 4];
        vshm[tid] += vshm[tid + 2];
        vshm[tid] += vshm[tid + 1];
    }

    if (tid == 0) {
        atomicAdd(bce_sum, sdata[0] * inv_batch);
    }
}

__global__ void bce_forward_warp_reduction_kernel(const float* __restrict__ X,
                                                  const float* __restrict__ Z,
                                                  float* bce_sum,
                                                  int size,
                                                  float inv_batch) {  
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local = 0.0f;
    if (idx < size) {
        local = compute_bce(X[idx], Z[idx]);
    }
    sdata[tid] = local;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid < WARP_SIZE) {
        float val = sdata[tid];
        if (blockDim.x >= 64) val += sdata[tid + 32];
        val = warp_reduce_sum(val);
        if (tid == 0) {
            atomicAdd(bce_sum, val * inv_batch);
        }
    }
}

__global__ void bce_forward_vec4_kernel(const float* __restrict__ X,
                                        const float* __restrict__ Z,
                                        float* bce_sum,
                                        int size,
                                        float inv_batch) {  
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    float local = 0.0f;
    if (idx + 3 < size) {
        float4 X_vec = *reinterpret_cast<const float4*>(&X[idx]);
        float4 Z_vec = *reinterpret_cast<const float4*>(&Z[idx]);

        local += compute_bce(X_vec.x, Z_vec.x);
        local += compute_bce(X_vec.y, Z_vec.y);
        local += compute_bce(X_vec.z, Z_vec.z);
        local += compute_bce(X_vec.w, Z_vec.w);
    } else if (idx < size) {
        for (int i = idx; i < size; ++i) {
            local += compute_bce(X[i], Z[i]);
        }
    }
    sdata[tid] = local;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid < WARP_SIZE) {
        float val = sdata[tid];
        if (blockDim.x >= 64) val += sdata[tid + 32];
        val = warp_reduce_sum(val);
        if (tid == 0) {
            atomicAdd(bce_sum, val * inv_batch);
        }
    }
}


// ==============================
// Forward kernels: KL
// ==============================

__global__ void kl_forward_naive_kernel(const float* __restrict__ mu,
                                        const float* __restrict__ logvar,
                                        float* kl_sum,
                                        int size,
                                        float inv_batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float kl = compute_kl(mu[idx], logvar[idx]);
        atomicAdd(kl_sum, kl * inv_batch);
    }
}

__global__ void kl_forward_reduction_kernel(const float* __restrict__ mu,
                                            const float* __restrict__ logvar,
                                            float* kl_sum,
                                            int size,
                                            float inv_batch) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local = 0.0f;
    if (idx < size) {
        local = compute_kl(mu[idx], logvar[idx]);
    }           
    sdata[tid] = local;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(kl_sum, sdata[0] * inv_batch);
    }
}

__global__ void kl_forward_unrolled_reduction_kernel(const float* __restrict__ mu,
                                                     const float* __restrict__ logvar,
                                                     float* kl_sum,
                                                     int size,
                                                     float inv_batch) {  
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local = 0.0f;
    if (idx < size) {
        local = compute_kl(mu[idx], logvar[idx]);
    }
    sdata[tid] = local;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid < WARP_SIZE) {
        volatile float* vshm = sdata;
        if (blockDim.x >= 64) vshm[tid] += vshm[tid + 32];
        vshm[tid] += vshm[tid + 16];
        vshm[tid] += vshm[tid + 8];
        vshm[tid] += vshm[tid + 4];
        vshm[tid] += vshm[tid + 2];
        vshm[tid] += vshm[tid + 1];
    }

    if (tid == 0) {
        atomicAdd(kl_sum, sdata[0] * inv_batch);
    }
}

__global__ void kl_forward_warp_reduction_kernel(const float* __restrict__ mu,
                                                 const float* __restrict__ logvar,
                                                 float* kl_sum,
                                                 int size,
                                                 float inv_batch) { 
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local = 0.0f;
    if (idx < size) {
        local = compute_kl(mu[idx], logvar[idx]);
    }
    sdata[tid] = local;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid < WARP_SIZE) {
        float val = sdata[tid];
        if (blockDim.x >= 64) val += sdata[tid + 32];
        val = warp_reduce_sum(val);
        if (tid == 0) {
            atomicAdd(kl_sum, val * inv_batch);
        }
    }
}

__global__ void kl_forward_vec4_kernel(const float* __restrict__ mu,
                                       const float* __restrict__ logvar,
                                       float* kl_sum,
                                       int size,
                                       float inv_batch) { 
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    float local = 0.0f;
    if (idx + 3 < size) {
        float4 mu_vec = *reinterpret_cast<const float4*>(&mu[idx]);
        float4 logvar_vec = *reinterpret_cast<const float4*>(&logvar[idx]);

        local += compute_kl(mu_vec.x, logvar_vec.x);
        local += compute_kl(mu_vec.y, logvar_vec.y);
        local += compute_kl(mu_vec.z, logvar_vec.z);
        local += compute_kl(mu_vec.w, logvar_vec.w);
    } else if (idx < size) {
        for (int i = idx; i < size; ++i) {
            local += compute_kl(mu[i], logvar[i]);
        }
    }
    sdata[tid] = local;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid < WARP_SIZE) {
        float val = sdata[tid];
        if (blockDim.x >= 64) val += sdata[tid + 32];
        val = warp_reduce_sum(val);
        if (tid == 0) {
            atomicAdd(kl_sum, val * inv_batch);
        }
    }
}


// ==============================
// Backward kernels: BCE
// ==============================

__global__ void bce_backward_naive_kernel(const float* __restrict__ X,
                                          const float* __restrict__ X_hat,
                                          float* __restrict__ dA,
                                          int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dA[idx] = X_hat[idx] - X[idx];
    }
}

__global__ void bce_backward_vec4_kernel(const float* __restrict__ X,
                                               const float* __restrict__ X_hat,
                                               float* __restrict__ dA,
                                               int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        float4 X_vec = *reinterpret_cast<const float4*>(&X[idx]);
        float4 X_hat_vec = *reinterpret_cast<const float4*>(&X_hat[idx]);
        float4 dA_vec;

        dA_vec.x = X_hat_vec.x - X_vec.x;
        dA_vec.y = X_hat_vec.y - X_vec.y;
        dA_vec.z = X_hat_vec.z - X_vec.z;
        dA_vec.w = X_hat_vec.w - X_vec.w;

        *reinterpret_cast<float4*>(&dA[idx]) = dA_vec;
    } else if (idx < size) {
        for (int i = idx; i < size; ++i) {
            dA[i] = X_hat[i] - X[i];
        }
    }
}


// ==============================
// Backward kernels: KL
// ==============================

__global__ void kl_backward_naive_kernel(const float* __restrict__ mu,
                                         const float* __restrict__ logvar,
                                         float* __restrict__ dmu,
                                         float* __restrict__ dlogvar,
                                         int size,
                                         float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dmu[idx] += beta * mu[idx];
        dlogvar[idx] += beta * 0.5f * (expf(logvar[idx]) - 1.0f);
    }
}

__global__ void kl_backward_vec4_kernel(const float* __restrict__ mu,
                                        const float* __restrict__ logvar,
                                        float* __restrict__ dmu,
                                        float* __restrict__ dlogvar,
                                        int size,
                                        float beta) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < size) {
        float4 mu_vec = *reinterpret_cast<const float4*>(&mu[idx]);
        float4 logvar_vec = *reinterpret_cast<const float4*>(&logvar[idx]);
        float4 dmu_vec = *reinterpret_cast<const float4*>(&dmu[idx]);
        float4 dlogvar_vec = *reinterpret_cast<const float4*>(&dlogvar[idx]);

        dmu_vec.x += beta * mu_vec.x;
        dmu_vec.y += beta * mu_vec.y;
        dmu_vec.z += beta * mu_vec.z;
        dmu_vec.w += beta * mu_vec.w;

        dlogvar_vec.x += beta * 0.5f * (expf(logvar_vec.x) - 1.0f);
        dlogvar_vec.y += beta * 0.5f * (expf(logvar_vec.y) - 1.0f);
        dlogvar_vec.z += beta * 0.5f * (expf(logvar_vec.z) - 1.0f);
        dlogvar_vec.w += beta * 0.5f * (expf(logvar_vec.w) - 1.0f);

        *reinterpret_cast<float4*>(&dmu[idx]) = dmu_vec;
        *reinterpret_cast<float4*>(&dlogvar[idx]) = dlogvar_vec;
    } 
    else if (idx < size) {
        for (int i = idx; i < size; ++i) {
            dmu[i] += beta * mu[i];
            dlogvar[i] += beta * 0.5f * (expf(logvar[i]) - 1.0f);
        }
    }
}


// ==============================
// Host API
// ==============================

namespace loss {

    void forward(const float* d_X,
                 const float* d_Z,
                 const float* d_mu,
                 const float* d_logvar,
                 float* d_bce,
                 float* d_kl,
                 float* h_loss,
                 int batch_size,
                 int input_dim,
                 int output_dim,
                 const VAEStrategy& strategy,
                 float beta) {
        const int blockSize = 256;
        const int size_bce = batch_size * input_dim;
        const int size_kl = batch_size * output_dim;
        int gridSize_bce;
        int gridSize_kl;
        
        const float inv_batch = 1.0f / (float)batch_size;

        CUDA_CHECK(cudaMemset(d_bce, 0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_kl, 0, sizeof(float)));

        switch (strategy) {
            case VAEStrategy::NAIVE:
            case VAEStrategy::TILING:
            case VAEStrategy::PADDING:
                gridSize_bce = (size_bce + blockSize - 1) / blockSize;
                DEBUG("Launching bce_forward_naive_kernel...");
                bce_forward_naive_kernel<<<gridSize_bce, blockSize>>>(d_X, d_Z, d_bce, size_bce, inv_batch);
                break;
            case VAEStrategy::REDUCTION:
                gridSize_bce = (size_bce + blockSize - 1) / blockSize;
                DEBUG("Launching bce_forward_reduction_kernel...");
                bce_forward_reduction_kernel<<<gridSize_bce, blockSize, blockSize * sizeof(float)>>>(d_X, d_Z, d_bce, size_bce, inv_batch);
                break;
            case VAEStrategy::UNROLLED_REDUCTION:
                gridSize_bce = (size_bce + blockSize - 1) / blockSize;
                DEBUG("Launching bce_forward_unrolled_reduction_kernel...");
                bce_forward_unrolled_reduction_kernel<<<gridSize_bce, blockSize, blockSize * sizeof(float)>>>(d_X, d_Z, d_bce, size_bce, inv_batch);
                break;
            case VAEStrategy::WARP_REDUCTION:
                gridSize_bce = (size_bce + blockSize - 1) / blockSize;
                DEBUG("Launching bce_forward_warp_reduction_kernel...");
                bce_forward_warp_reduction_kernel<<<gridSize_bce, blockSize, blockSize * sizeof(float)>>>(d_X, d_Z, d_bce, size_bce, inv_batch);
                break;
            case VAEStrategy::VECTORIZED:
            case VAEStrategy::OPTIMIZED: 
            default:
                gridSize_bce = ((size_bce + 3) / 4 + blockSize - 1) / blockSize;
                DEBUG("Launching bce_forward_vec4_kernel...");
                bce_forward_vec4_kernel<<<gridSize_bce, blockSize, blockSize * sizeof(float)>>>(d_X, d_Z, d_bce, size_bce, inv_batch);
                break;
        }
        
        switch (strategy) {
            case VAEStrategy::NAIVE:
            case VAEStrategy::TILING:
            case VAEStrategy::PADDING:
                gridSize_kl = (size_kl + blockSize - 1) / blockSize;
                DEBUG("Launching kl_forward_naive_kernel...");
                kl_forward_naive_kernel<<<gridSize_kl, blockSize>>>(d_mu, d_logvar, d_kl, size_kl, inv_batch);
                break;
            case VAEStrategy::REDUCTION:
                gridSize_kl = (size_kl + blockSize - 1) / blockSize;
                DEBUG("Launching kl_forward_reduction_kernel...");
                kl_forward_reduction_kernel<<<gridSize_kl, blockSize, blockSize * sizeof(float)>>>(d_mu, d_logvar, d_kl, size_kl, inv_batch);
                break;
            case VAEStrategy::UNROLLED_REDUCTION:
                gridSize_kl = (size_kl + blockSize - 1) / blockSize;
                DEBUG("Launching kl_forward_unrolled_reduction_kernel...");
                kl_forward_unrolled_reduction_kernel<<<gridSize_kl, blockSize, blockSize * sizeof(float)>>>(d_mu, d_logvar, d_kl, size_kl, inv_batch);
                break;
            case VAEStrategy::WARP_REDUCTION:
                gridSize_kl = (size_kl + blockSize - 1) / blockSize;
                DEBUG("Launching kl_forward_warp_reduction_kernel...");
                kl_forward_warp_reduction_kernel<<<gridSize_kl, blockSize, blockSize * sizeof(float)>>>(d_mu, d_logvar, d_kl, size_kl, inv_batch);
                break;
            case VAEStrategy::VECTORIZED:
            case VAEStrategy::OPTIMIZED: 
            default:
                gridSize_kl = ((size_kl + 3) / 4 + blockSize - 1) / blockSize;
                DEBUG("Launching kl_forward_vec4_kernel...");
                kl_forward_vec4_kernel<<<gridSize_kl, blockSize, blockSize * sizeof(float)>>>(d_mu, d_logvar, d_kl, size_kl, inv_batch);
                break;
        } 

        CUDA_CHECK(cudaDeviceSynchronize()); 
        
        float h_bce, h_kl;
        
        CUDA_CHECK(cudaMemcpy(&h_bce, d_bce, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_kl, d_kl, sizeof(float), cudaMemcpyDeviceToHost));
        
        *h_loss = h_bce + beta * h_kl;

        CUDA_CHECK(cudaGetLastError());
    }

    namespace backward {

        void bce(const float* d_X,
                 const float* d_X_hat,
                 float* d_dA,
                 int size,
                 const VAEStrategy& strategy) {
            const int blockSize = 256;
            int gridSize;

            switch(strategy) {
                case VAEStrategy::NAIVE: 
                    gridSize = (size + blockSize - 1) / blockSize;
                    DEBUG("Launching bce_backward_naive_kernel...");
                    bce_backward_naive_kernel<<<gridSize, blockSize>>>(d_X, d_X_hat, d_dA, size);
                    break;   
                case VAEStrategy::VECTORIZED: 
                case VAEStrategy::OPTIMIZED: 
                default:  
                    gridSize = ((size + 3) / 4 + blockSize - 1) / blockSize;
                    DEBUG("Launching bce_backward_vec4_kernel...");
                    bce_backward_vec4_kernel<<<gridSize, blockSize>>>(d_X, d_X_hat, d_dA, size);
                    break;  
            }

            CUDA_CHECK(cudaGetLastError());
        }

        void kl(const float* d_mu, 
                const float* d_logvar,
                float* d_dmu,
                float* d_dlogvar,
                int size,
                const VAEStrategy& strategy,
                float beta) {
            const int blockSize = 256;
            int gridSize;

            switch(strategy) {
                case VAEStrategy::NAIVE: 
                case VAEStrategy::TILING:
                case VAEStrategy::PADDING:
                case VAEStrategy::REDUCTION:
                case VAEStrategy::UNROLLED_REDUCTION:
                case VAEStrategy::WARP_REDUCTION:
                    gridSize = (size + blockSize - 1) / blockSize;
                    DEBUG("Launching kl_backward_naive_kernel...");
                    kl_backward_naive_kernel<<<gridSize, blockSize>>>(d_mu, d_logvar, d_dmu, d_dlogvar, size, beta); 
                    break;   
                case VAEStrategy::VECTORIZED: 
                case VAEStrategy::OPTIMIZED: 
                default: 
                    gridSize = ((size + 3) / 4 + blockSize - 1) / blockSize;
                    DEBUG("Launching kl_backward_vec4_kernel...");
                    kl_backward_vec4_kernel<<<gridSize, blockSize>>>(d_mu, d_logvar, d_dmu, d_dlogvar, size, beta); 
                    break;     
            }      

            CUDA_CHECK(cudaGetLastError());
        }            

    } // namespace backward

} // namespace loss