#include "loss.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <math.h>

// bce forward
__global__ void bce_forward_naive_kernel(const float* X,
                                         const float* Z,
                                         float* bce_sum,
                                         int size,
                                         int batch_size) {                                        
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = X[idx];
        float z = Z[idx];
        float bce = fmaxf(z, 0.0f) - z * x + logf(1.0f + expf(-fabsf(z)));

        atomicAdd(bce_sum, bce / batch_size);
    }
}

// kl forward
__global__ void kl_forward_naive_kernel(const float* mu,
                                        const float* logvar,
                                        float* kl_sum,
                                        int size,
                                        int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float m = mu[idx];
        float lv = logvar[idx];
        float kl = 0.5f * (m * m + expf(lv) - 1.0f - lv);

        atomicAdd(kl_sum, kl / batch_size);
    }
}

// bce backward

__global__ void bce_backward_naive_kernel(const float* X,
                                          const float* X_hat,
                                          float* dA,
                                          int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dA[idx] = X_hat[idx] - X[idx];
    }
}

__global__ void bce_backward_vectorized_kernel(const float* X,
                                               const float* X_hat,
                                               float* dA,
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

// kl backward
__global__ void kl_backward_naive_kernel(const float* mu,
                                         const float* logvar,
                                         float* dmu,
                                         float* dlogvar,
                                         int size,
                                         float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dmu[idx] += beta * mu[idx];
        dlogvar[idx] += beta * 0.5f * (expf(logvar[idx]) - 1.0f);
    }
}

__global__ void kl_backward_vectorized_kernel(const float* mu,
                                              const float* logvar,
                                              float* dmu,
                                              float* dlogvar,
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
                 float beta,
                 const VAEStrategy& strategy) {
        const int blockSize = 256;
        const int size_bce = batch_size * input_dim;
        const int size_kl = batch_size * output_dim;
        const int gridSize_bce = (size_bce + blockSize - 1) / blockSize;
        const int gridSize_kl = (size_kl + blockSize - 1) / blockSize;

        CUDA_CHECK(cudaMemset(d_bce, 0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_kl, 0, sizeof(float)));

        DEBUG("Launching bce_forward_naive_kernel...");
        bce_forward_naive_kernel<<<gridSize_bce, blockSize>>>(d_X, d_Z, d_bce, size_bce, batch_size);     
        
        DEBUG("Launching kl_forward_naive_kernel...");
        kl_forward_naive_kernel<<<gridSize_kl, blockSize>>>(d_mu, d_logvar, d_kl, size_kl, batch_size);   

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
                    gridSize = ((size + 3) / 4 + blockSize - 1) / blockSize;
                    DEBUG("Launching bce_backward_vectorized_kernel...");
                    bce_backward_vectorized_kernel<<<gridSize, blockSize>>>(d_X, d_X_hat, d_dA, size);
                    break;
                default:  
                    gridSize = (size + blockSize - 1) / blockSize;
                    DEBUG("Launching bce_backward_naive_kernel...");
                    bce_backward_naive_kernel<<<gridSize, blockSize>>>(d_X, d_X_hat, d_dA, size);
                    break;      
            }

            CUDA_CHECK(cudaGetLastError());
        }

        void kl(const float* d_mu, 
                const float* d_logvar,
                float* d_dmu,
                float* d_dlogvar,
                int size,
                float beta,
                const VAEStrategy& strategy) {
            const int blockSize = 256;
            int gridSize;

            switch(strategy) {
                case VAEStrategy::NAIVE: 
                    gridSize = (size + blockSize - 1) / blockSize;
                    DEBUG("Launching kl_backward_naive_kernel...");
                    kl_backward_naive_kernel<<<gridSize, blockSize>>>(d_mu, d_logvar, d_dmu, d_dlogvar, size, beta); 
                    break;   
                case VAEStrategy::VECTORIZED: 
                    gridSize = ((size + 3) / 4 + blockSize - 1) / blockSize;
                    DEBUG("Launching kl_backward_vectorized_kernel...");
                    kl_backward_vectorized_kernel<<<gridSize, blockSize>>>(d_X, d_X_hat, d_dA, size);
                    break;
                default:  
                    gridSize = (size + blockSize - 1) / blockSize;
                    DEBUG("Launching kl_backward_naive_kernel...");
                    kl_backward_naive_kernel<<<gridSize, blockSize>>>(d_mu, d_logvar, d_dmu, d_dlogvar, size, beta); 
                    break;        
            }      

            CUDA_CHECK(cudaGetLastError());
        }            

    } // namespace backward

} // namespace loss