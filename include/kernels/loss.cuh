#pragma once
#include "vae_config.cuh"


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
                 float beta = 1.0f,
                 const VAEStrategy& strategy);

    namespace backward {

        void bce(const float* d_X,
                 const float* d_X_hat,
                 float* d_dA,
                 int size);

        void kl(const float* d_mu, 
                const float* d_logvar,
                float* d_dmu,
                float* d_dlogvar,
                int size,
                float beta = 1.0f,
                const VAEStrategy& strategy);

    } // namespace backward

} // namespace loss
