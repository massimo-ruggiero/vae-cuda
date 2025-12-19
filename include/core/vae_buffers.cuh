#pragma once

#include <curand_kernel.h>

#include "layer_params.cuh"
#include "vae_config.cuh"


struct VAEBuffers {
    VAEConfig config;

    GPUBuffer d_X;     
    LinearLayer enc1;       
    LinearLayer enc2;            

    // NOTA: d_mu e d_logvar sono "viste" (puntatori raw) dentro enc2.Z
    float *d_mu = nullptr;     
    float *d_logvar = nullptr;
    GPUBuffer d_z;          
    GPUBuffer d_epsilon;
    curandStatePhilox4_32_10_t* d_states = nullptr;    

    LinearLayer dec1;       
    LinearLayer dec2;  

    GPUBuffer d_X_hat;
    GPUBuffer d_bce;
    GPUBuffer d_kl;


    VAEBuffers(const VAEConfig& cfg) : config(cfg) {
        size_t batch_size = config.batch_size;
        size_t input_dim = config.input_dim;
        size_t hidden_dim = config.hidden_dim;
        size_t latent_dim = config.latent_dim;

        d_X.allocate(batch_size * input_dim);

        // ENCODER
        enc1.allocate(batch_size, input_dim, hidden_dim, true);        
        enc2.allocate(batch_size, hidden_dim, latent_dim * 2, false); // L'output Z di questo layer contiene [mu, logvar]

        // LATENT
        // Z = [mu, logvar]
        d_mu = enc2.Z.ptr; 
        d_logvar = enc2.Z.ptr + (batch_size * latent_dim);
        d_z.allocate(batch_size * latent_dim);
        d_epsilon.allocate(batch_size * latent_dim);
        CUDA_CHECK(cudaMalloc(&d_states, batch_size * latent_dim * sizeof(curandStatePhilox4_32_10_t)));

        // DECODER
        dec1.allocate(batch_size, latent_dim, hidden_dim, true);
        dec2.allocate(batch_size, hidden_dim, input_dim, false); // La sigmoide finale verrà applicata manualmente e salvata in d_X_hat

        d_bce.allocate(batch_size);
        d_kl.allocate(batch_size);
        d_X_hat.allocate(batch_size * input_dim);
    }

    ~VAEBuffers() {
        if (d_states) {
            CUDA_CHECK(cudaFree(d_states));
            d_states = nullptr;
        }
    }
};


struct VAEGradients {
    LinearGrads enc1;
    LinearGrads enc2;

    float* d_dmu = nullptr;
    float* d_dlogvar = nullptr;
    GPUBuffer d_dz;

    LinearGrads dec1;
    LinearGrads dec2;
    GPUBuffer d_dX_hat;


    VAEGradients(const VAEConfig& cfg) {
        size_t batch_size = cfg.batch_size;
        size_t input_dim = cfg.input_dim;
        size_t hidden_dim = cfg.hidden_dim;
        size_t latent_dim = cfg.latent_dim;

        // ENCODER
        enc1.allocate(batch_size, input_dim, hidden_dim, true);
        enc2.allocate(batch_size, hidden_dim, latent_dim * 2, false);

        // LATENT
        d_dmu = enc2.dZ.ptr;
        d_dlogvar = enc2.dZ.ptr + (batch_size * latent_dim);
        d_dz.allocate(batch_size * latent_dim);

        // DECODER
        dec1.allocate(batch_size, latent_dim, hidden_dim, true);
        dec2.allocate(batch_size, hidden_dim, input_dim, false);

        d_dX_hat.allocate(batch_size * input_dim);
    }

    void zero_all_grads() {
        enc1.zero_grads();
        enc2.zero_grads();
        dec1.zero_grads();
        dec2.zero_grads();

        d_dz.zero();
        d_dX_hat.zero();
    }

    VAEGradients(const VAEGradients&) = delete;
    VAEGradients& operator=(const VAEGradients&) = delete;
};


struct VAEAdamState {
    AdamState enc1;
    AdamState enc2;
    AdamState dec1;
    AdamState dec2;


    VAEAdamState(const VAEConfig& cfg) {
        size_t input_dim = cfg.input_dim;
        size_t hidden_dim = cfg.hidden_dim;
        size_t latent_dim = cfg.latent_dim;

        enc1.allocate(input_dim, hidden_dim);
        enc2.allocate(hidden_dim, latent_dim * 2);
        
        dec1.allocate(latent_dim, hidden_dim);
        dec2.allocate(hidden_dim, input_dim);
    }


    void reset_all() {
        enc1.reset();
        enc2.reset();
        dec1.reset();
        dec2.reset();
    }
};
