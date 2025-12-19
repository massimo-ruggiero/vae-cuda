#pragma once

#include "vae_buffers.cuh"
#include "vae_forward.cuh"
#include "vae_backward.cuh"


class VAE {
    private:
        VAEBuffers buf_;
        VAEGradients* grads_; // nullptr for inference
        float* d_bce_loss_;
        float* d_kl_loss_;

    public:
        VAE(const VAEConfig& config);
        ~VAE();

        float train_step(const float* h_batch);

        void encode(const float* h_input, float* h_mu, float* h_logvar);
        void decode(const float* h_z, float* h_output);
        void reconstruct(const float* h_input, float* h_output);
        void sample(float* h_output, int n_samples);

        void free_training_resources();

        void save_weights(const char* filename);
        void load_weights(const char* filename);


    private:
        void initialize_weights();
        void ensure_training_resources();
};
