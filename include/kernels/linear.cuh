#pragma once
#include "vae_config.cuh"

namespace linear {

    void forward(const float* d_X,
                    const float* d_W,
                    const float* d_b,
                    float* d_Z,
                    int batch_size,
                    int input_dim,
                    int output_dim,
                    const VAEStrategy& strategy);

    void backward(const float* d_X,
                    const float* d_W,
                    const float* d_dZ,
                    float* d_XT,          
                    float* d_WT,
                    float* d_dX,
                    float* d_dW,
                    float* d_db,
                    int batch_size,
                    int input_dim,
                    int output_dim,
                    const VAEStrategy& strategy);

} // namespace linear