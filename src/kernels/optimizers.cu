#include "optimizers.cuh"
#include "utils.cuh"

#include <cuda_runtime.h>
#include <math.h>


// ==============================
// Device helpers 
// ==============================

__device__ __forceinline__ void compute_adam (float g_val, 
                                              float &m_val, 
                                              float &v_val, 
                                              float &theta_val,
                                              float beta1,
                                              float beta2,
                                              float inv_bc1,
                                              float inv_bc2,
                                              float lr,
                                              float epsilon) {
    m_val = beta1 * m_val + (1.0f - beta1) * g_val;
    v_val = beta2 * v_val + (1.0f - beta2) * g_val * g_val;

    float m_hat = m_val * inv_bc1;
    float v_hat = v_val * inv_bc2;

    theta_val -= lr * m_hat / (sqrtf(v_hat) + epsilon);
};


// ==============================
// Kernles: ADAM
// ==============================

__global__ void adam_step_naive_kernel(const float* __restrict__ g,
                                       float* __restrict__ theta,
                                       float* __restrict__ m,
                                       float* __restrict__ v,
                                       int size,
                                       float lr,
                                       float beta1,
                                       float beta2,
                                       float inv_bc1,
                                       float inv_bc2,
                                       float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g_val = g[idx];
        float m_val = m[idx];
        float v_val = v[idx];

        m_val = beta1 * m_val + (1.0f - beta1) * g_val;
        v_val = beta2 * v_val + (1.0f - beta2) * g_val * g_val;

        float m_hat = m_val * inv_bc1;
        float v_hat = v_val * inv_bc2;

        theta[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);

        m[idx] = m_val;
        v[idx] = v_val;
    }
}

__global__ void adam_step_vec4_kernel(const float* __restrict__ g,
                                      float* __restrict__ theta,
                                      float* __restrict__ m,
                                      float* __restrict__ v,
                                      int size,
                                      float lr,
                                      float beta1,
                                      float beta2,
                                      float inv_bc1,
                                      float inv_bc2,
                                      float epsilon) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < size) {
        float4 g_vec = *reinterpret_cast<const float4*>(&g[idx]);
        float4 theta_vec = *reinterpret_cast<float4*>(&theta[idx]);
        float4 m_vec = *reinterpret_cast<float4*>(&m[idx]);
        float4 v_vec = *reinterpret_cast<float4*>(&v[idx]);

        compute_adam(g_vec.x, m_vec.x, v_vec.x, theta_vec.x, beta1, beta2, inv_bc1, inv_bc2, lr, epsilon);
        compute_adam(g_vec.y, m_vec.y, v_vec.y, theta_vec.y, beta1, beta2, inv_bc1, inv_bc2, lr, epsilon);
        compute_adam(g_vec.z, m_vec.z, v_vec.z, theta_vec.z, beta1, beta2, inv_bc1, inv_bc2, lr, epsilon);
        compute_adam(g_vec.w, m_vec.w, v_vec.w, theta_vec.w, beta1, beta2, inv_bc1, inv_bc2, lr, epsilon);

        *reinterpret_cast<float4*>(&theta[idx]) = theta_vec;
        *reinterpret_cast<float4*>(&m[idx]) = m_vec;
        *reinterpret_cast<float4*>(&v[idx]) = v_vec;
    } else if (idx < size) {
        for (int i = idx; i < size; ++i) {
            float g_val = g[i];
            float m_val = m[i];
            float v_val = v[i];
            float theta_val = theta[i];

            compute_adam(g_val, m_val, v_val, theta_val, beta1, beta2, inv_bc1, inv_bc2, lr, epsilon);

            m[i] = m_val;
            v[i] = v_val;
            theta[i] = theta_val;
        }
    }
}


// ==============================
// Host API
// ==============================

namespace optimizers {

    namespace adam {

        void step(const float* d_g,
                  float* d_theta,
                  float* d_m,
                  float* d_v,
                  int t,
                  int size,
                  const VAEStrategy& strategy,
                  float lr,
                  float beta1,
                  float beta2,
                  float epsilon) {
            const int blockSize = 256;
            int gridSize;

            float inv_bc1 = 1.0f / (1.0f - powf(beta1, (float)t)); 
            float inv_bc2 = 1.0f / (1.0f - powf(beta2, (float)t));

            switch(strategy) {
                case VAEStrategy::NAIVE:
                case VAEStrategy::TILING:
                case VAEStrategy::PADDING:
                case VAEStrategy::REDUCTION:
                case VAEStrategy::UNROLLED_REDUCTION:
                case VAEStrategy::WARP_REDUCTION:
                    gridSize = (size + blockSize - 1) / blockSize;
                    DEBUG("Launching adam_step_naive_kernel...");
                    adam_step_naive_kernel<<<gridSize, blockSize>>>(d_g, d_theta, d_m, d_v, size, lr, beta1, beta2, inv_bc1, inv_bc2, epsilon);
                    break;
                case VAEStrategy::VECTORIZED:
                case VAEStrategy::OPTIMIZED:
                default:
                    gridSize = ((size + 3) / 4 + blockSize - 1) / blockSize;
                    DEBUG("Launching adam_step_vec4_kernel...");
                    adam_step_vec4_kernel<<<gridSize, blockSize>>>(d_g, d_theta, d_m, d_v, size, lr, beta1, beta2, inv_bc1, inv_bc2, epsilon);
                    break;
            }

            CUDA_CHECK(cudaGetLastError());
        }

    } // namespace adam 

} // namespace optimizers 