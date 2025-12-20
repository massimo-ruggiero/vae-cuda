#pragma once

#include <curand_kernel.h>

#include "layer_params.cuh"
#include "vae_config.cuh"


struct VAEBuffers {
    VAEConfig config;

    GPUBuffer d_X;     
    LinearLayer enc1;       
    LinearLayer enc2_mu;            
    LinearLayer enc2_logvar;            

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
        enc2_mu.allocate(batch_size, hidden_dim, latent_dim, false); 
        enc2_logvar.allocate(batch_size, hidden_dim, latent_dim, false); 

        // LATENT
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
    GPUBuffer enc1_dA_tmp; 
    LinearGrads enc2_mu;
    LinearGrads enc2_logvar;

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
        enc1_dA_tmp.allocate(batch_size * hidden_dim);
        enc2_mu.allocate(batch_size, hidden_dim, latent_dim, false);
        enc2_logvar.allocate(batch_size, hidden_dim, latent_dim, false);

        // LATENT
        d_dz.allocate(batch_size * latent_dim);

        // DECODER
        dec1.allocate(batch_size, latent_dim, hidden_dim, true);
        dec2.allocate(batch_size, hidden_dim, input_dim, false);

        d_dX_hat.allocate(batch_size * input_dim);
    }

    void zero_all_grads() {
        enc1.zero_grads();
        enc1_dA_tmp.zero();
        enc2_mu.zero_grads();
        enc2_logvar.zero_grads();
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
    AdamState enc2_mu;
    AdamState enc2_logvar;
    AdamState dec1;
    AdamState dec2;


    VAEAdamState(const VAEConfig& cfg) {
        size_t input_dim = cfg.input_dim;
        size_t hidden_dim = cfg.hidden_dim;
        size_t latent_dim = cfg.latent_dim;

        enc1.allocate(input_dim, hidden_dim);
        enc2_mu.allocate(hidden_dim, latent_dim);
        enc2_logvar.allocate(hidden_dim, latent_dim);
        
        dec1.allocate(latent_dim, hidden_dim);
        dec2.allocate(hidden_dim, input_dim);
    }


    void reset_all() {
        enc1.reset();
        enc2_mu.reset();
        enc2_logvar.reset();
        dec1.reset();
        dec2.reset();
    }
};
