#include "vae_buffers.cuh"
#include "utils.cuh" 
#include <cuda_runtime.h>


// VAEBuffers
VAEBuffers::VAEBuffers(const VAEConfig& cfg) : config(cfg) {
    size_t batch_size = config.batch_size;
    size_t input_dim = config.input_dim;
    size_t hidden_dim = config.hidden_dim;
    size_t latent_dim = config.latent_dim;

    d_X.allocate(batch_size * input_dim);

    // encoder
    enc1.allocate(batch_size, input_dim, hidden_dim, true);        
    enc2_mu.allocate(batch_size, hidden_dim, latent_dim, false); 
    enc2_logvar.allocate(batch_size, hidden_dim, latent_dim, false); 

    // reparametrization
    d_z.allocate(batch_size * latent_dim);
    d_epsilon.allocate(batch_size * latent_dim);

    size_t num_states = 0;
    switch(config.strategy) {
        case VAEStrategy::NAIVE:
        case VAEStrategy::SHARED_MEMORY_TILING:
        case VAEStrategy::PADDING:
        case VAEStrategy::REGISTER_TILING:
        case VAEStrategy::REDUCTION:
        case VAEStrategy::UNROLLED_REDUCTION:
        case VAEStrategy::WARP_REDUCTION:
            num_states = batch_size * latent_dim;
            break;
        case VAEStrategy::VECTORIZED:
        default:
            num_states = (batch_size * latent_dim + 3) / 4;
            break;
    }

    CUDA_CHECK(cudaMalloc(&d_states, num_states * sizeof(curandStatePhilox4_32_10_t)));

    // decoder
    dec1.allocate(batch_size, latent_dim, hidden_dim, true);
    dec2.allocate(batch_size, hidden_dim, input_dim, false); 

    d_bce.allocate(batch_size);
    d_kl.allocate(batch_size);
    d_X_hat.allocate(batch_size * input_dim);
}

VAEBuffers::~VAEBuffers() {
    if (d_states) {
        CUDA_CHECK(cudaFree(d_states));
        d_states = nullptr;
    }
}

// VAEGradients
VAEGradients::VAEGradients(const VAEConfig& cfg) {
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

void VAEGradients::zero_all_grads() {
    enc1.zero_grads();
    enc1_dA_tmp.zero();
    enc2_mu.zero_grads();
    enc2_logvar.zero_grads();
    dec1.zero_grads();
    dec2.zero_grads();

    d_dz.zero();
    d_dX_hat.zero();
}

// VAEAdamState
VAEAdamState::VAEAdamState(const VAEConfig& cfg) {
    size_t input_dim = cfg.input_dim;
    size_t hidden_dim = cfg.hidden_dim;
    size_t latent_dim = cfg.latent_dim;

    enc1.allocate(input_dim, hidden_dim);
    enc2_mu.allocate(hidden_dim, latent_dim);
    enc2_logvar.allocate(hidden_dim, latent_dim);
    
    dec1.allocate(latent_dim, hidden_dim);
    dec2.allocate(hidden_dim, input_dim);
}

void  VAEAdamState::zero_all_states() {
    enc1.zero_states();
    enc2_mu.zero_states();
    enc2_logvar.zero_states();
    dec1.zero_states();
    dec2.zero_states();
}