#include "loss.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <math.h>


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

__global__ void bce_backward_kernel(const float* X,
                                    const float* X_hat,
                                    float* dA,
                                    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dA[idx] = X_hat[idx] - X[idx];
    }
}

__global__ void kl_backward_kernel(const float* mu,
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

namespace loss {

    namespace forward {

        void naive(const float* d_X,
                   const float* d_A,
                   const float* d_mu,
                   const float* d_logvar,
                   float* d_bce,
                   float* d_kl,
                   float* h_loss,
                   int batch_size,
                   int input_dim,
                   int output_dim,
                   float beta) {
            const int blockSize = 256;
            const int size_bce = batch_size * input_dim;
            const int size_kl = batch_size * output_dim;
            const int gridSize_bce = (size_bce + blockSize - 1) / blockSize;
            const int gridSize_kl = (size_kl + blockSize - 1) / blockSize;

            CUDA_CHECK(cudaMemset(d_bce, 0, sizeof(float)));
            CUDA_CHECK(cudaMemset(d_kl, 0, sizeof(float)));

            DEBUG("Launching bce_forward_naive_kernel...");
            bce_forward_naive_kernel<<<gridSize_bce, blockSize>>>(d_X, d_A, d_bce, size_bce, batch_size);     
            
            DEBUG("Launching kl_forward_naive_kernel...");
            kl_forward_naive_kernel<<<gridSize_kl, blockSize>>>(d_mu, d_logvar, d_kl, size_kl, batch_size);   

            CUDA_CHECK(cudaDeviceSynchronize()); 
            
            float h_bce, h_kl;
            
            CUDA_CHECK(cudaMemcpy(&h_bce, d_bce, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_kl, d_kl, sizeof(float), cudaMemcpyDeviceToHost));
            
            *h_loss = h_bce + beta * h_kl;

            CUDA_CHECK(cudaGetLastError());
        }

    } // namespace forward

    namespace backward {

        void bce(const float* d_X,
                 const float* d_X_hat,
                 float* d_dA,
                 int size) {
            const int blockSize = 256;
            const int gridSize = (size + blockSize - 1) / blockSize;
            DEBUG("Launching bce_backward_kernel...");
            bce_backward_kernel<<<gridSize, blockSize>>>(d_X, d_X_hat, d_dA, size);       

            CUDA_CHECK(cudaGetLastError());
        }

        void kl(const float* d_mu, 
                const float* d_logvar,
                float* d_dmu,
                float* d_dlogvar,
                int size,
                float beta = 1.0f) {
            const int blockSize = 256;
            const int gridSize = (size + blockSize - 1) / blockSize;
            DEBUG("Launching kl_backward_kernel...");
            kl_backward_kernel<<<gridSize, blockSize>>>(d_mu, d_logvar, d_dmu, d_dlogvar, size, beta);       

            CUDA_CHECK(cudaGetLastError());
        }            

    } // namespace backward

} // namespace loss