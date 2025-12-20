#pragma once

#include <curand_kernel.h>

#include "layer_buffers.cuh"
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


    explicit VAEBuffers(const VAEConfig& cfg);
    ~VAEBuffers();
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


    explicit VAEGradients(const VAEConfig& cfg);
    void zero_all_grads();
};


struct VAEAdamState {
    AdamState enc1;
    AdamState enc2_mu;
    AdamState enc2_logvar;
    AdamState dec1;
    AdamState dec2;


    explicit VAEAdamState(const VAEConfig& cfg);
    void zero_all_states();
};
