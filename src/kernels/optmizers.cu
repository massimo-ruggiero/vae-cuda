#include "optimizers.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <math.h>


__global__ void adam_step_kernel(const float* g,
                                 float* theta,
                                 float* m,
                                 float* v,
                                 int t,
                                 int size,
                                 float lr,
                                 float beta1,
                                 float beta2,
                                 float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g_val = g[idx];
        float m_val = m[idx];
        float v_val = v[idx];

        m_val = beta1 * m_val + (1.0f - beta1) * g_val;
        v_val = beta2 * v_val + (1.0f - beta2) * g_val * g_val;

        float m_hat = m_val / (1.0f - powf(beta1, (float)t));
        float v_hat = v_val / (1.0f - powf(beta2, (float)t));

        theta[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);

        m[idx] = m_val;
        v[idx] = v_val;
    }
}

namespace adam {

    void step(const float* d_g,
              float* d_theta,
              float* d_m,
              float* d_v,
              int t,
              int size,
              float lr,
              float beta1,
              float beta2,
              float epsilon) {
        const int blockSize = 256;
        const int gridSize = (size + blockSize - 1) / blockSize;
        DEBUG("Launching adam_step_kernel...");
        adam_step_kernel<<<gridSize, blockSize>>>(d_g, d_theta, d_m, d_v, t, size, lr, beta1, beta2, epsilon);

        CUDA_CHECK(cudaGetLastError());
    }

}
