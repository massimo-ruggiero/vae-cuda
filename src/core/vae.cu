#include "vae.cuh"
#include "loss.cuh"
#include "reparametrization.cuh" 
#include "vae_forward.cuh"       
#include "vae_backward.cuh"      
#include "utils.cuh"             

#include <cmath>
#include <vector>
#include <random>
#include <cstdio>

// guardalo bene
static void init_layer_xavier(LinearLayer& layer, int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    
    std::vector<float> h_W(fan_in * fan_out);
    std::vector<float> h_b(fan_out, 0.0f); 

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-limit, limit);

    for(auto& w : h_W) w = dis(gen);

    CUDA_CHECK(cudaMemcpy(layer.W.ptr, h_W.data(), h_W.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(layer.b.ptr, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice));
}

VAE::VAE(const VAEConfig& config)
    : buf_(config),
      grads_(nullptr),
      d_bce_loss_(nullptr),
      d_kl_loss_(nullptr) {

    int total_states = config.batch_size * config.latent_dim;
    reparametrization::init(buf_.d_states, total_states, 42);

    initialize_weights();
}

VAE::~VAE(){
    // buf_ si pulisce solo
    // grads_ no perchè abbiamo usato new
    free_training_resources(); 
}

void VAE::initialize_weights() {
    init_layer_xavier(buf_.enc1, buf_.config.input_dim, buf_.config.hidden_dim);
    init_layer_xavier(buf_.enc2, buf_.config.hidden_dim, buf_.config.latent_dim * 2);
    init_layer_xavier(buf_.dec1, buf_.config.latent_dim, buf_.config.hidden_dim);
    init_layer_xavier(buf_.dec2, buf_.config.hidden_dim, buf_.config.input_dim);
}

void VAE::ensure_training_resources() {
    if (grads_) return;

    printf("VAE: Allocating training resources...\n");
    grads_ = new VAEGradients(buf_.config);
    
    if (!d_bce_loss_) CUDA_CHECK(cudaMalloc(&d_bce_loss_, sizeof(float)));
    if (!d_kl_loss_) CUDA_CHECK(cudaMalloc(&d_kl_loss_, sizeof(float)));
}

void VAE::free_training_resources() {
    if (grads_) { 
        delete grads_; 
        grads_ = nullptr; 
    }
    if (d_bce_loss_) { 
        cudaFree(d_bce_loss_); 
        d_bce_loss_ = nullptr; 
    }
    if (d_kl_loss_) { 
        cudaFree(d_kl_loss_); 
        d_kl_loss_ = nullptr; 
    }
}

float VAE::train_step(const float* h_batch, float beta){
    ensure_training_resources();

    size_t data_size = buf_.config.batch_size * buf_.config.input_dim * sizeof(float);
    CUDA_CHECK(cudaMemcpy(buf_.d_X.ptr, h_batch, data_size, cudaMemcpyHostToDevice));

    grads_->zero_all_grads();

    switch (buf_.config.strategy) {
        case VAEStrategy::NAIVE: 
            vae::naive::forward(buf_); 
            break;
    }    

    float h_loss = 0.0f;
    loss::forward::naive(buf_.d_X.ptr, 
                         buf_.dec2.Z.ptr, 
                         buf_.d_mu, 
                         buf_.d_logvar,
                         d_bce_loss_, 
                         d_kl_loss_, 
                         &h_loss,
                         buf_.config.batch_size, 
                         buf_.config.input_dim, 
                         buf_.config.latent_dim, 
                         buf_.config.beta);
    
    switch (buf_.config.strategy) {
        case VAEStrategy::NAIVE: 
            vae::naive::backward(buf_, *grads_, beta); 
            break;
    }

    return h_loss;
}

void VAE::encode(const float* h_input, float* h_mu, float* h_logvar) {
    CUDA_CHECK(cudaMemcpy(buf_.d_X.ptr, h_input, 
                          buf_.config.batch_size * buf_.config.input_dim * sizeof(float), 
                          cudaMemcpyHostToDevice));

    switch (buf_.config.strategy) {
        case VAEStrategy::NAIVE: vae::naive::encoder_pass(buf_); break;
    }

    CUDA_CHECK(cudaMemcpy(h_mu, buf_.d_mu, 
                          buf_.config.batch_size * buf_.config.latent_dim * sizeof(float), 
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_logvar, buf_.d_logvar,
                          buf_.config.batch_size * buf_.config.latent_dim * sizeof(float), 
                          cudaMemcpyDeviceToHost));
}


void VAE::decode(const float* h_z, float* h_output) {
    CUDA_CHECK(cudaMemcpy(buf_.d_z.ptr, h_z, 
                          buf_.config.batch_size * buf_.config.latent_dim * sizeof(float), 
                          cudaMemcpyHostToDevice));

    switch (buf_.config.strategy) {
        case VAEStrategy::NAIVE: vae::naive::decoder_pass(buf_); break;
    }

    CUDA_CHECK(cudaMemcpy(h_output, buf_.d_X_hat.ptr, 
                          buf_.config.batch_size * buf_.config.input_dim * sizeof(float), 
                          cudaMemcpyDeviceToHost));
}

void VAE::reconstruct(const float* h_input, float* h_output) {
    CUDA_CHECK(cudaMemcpy(buf_.d_X.ptr, h_input, 
               buf_.config.batch_size * buf_.config.input_dim * sizeof(float), 
               cudaMemcpyHostToDevice));
    
    switch (buf_.config.strategy) {
        case VAEStrategy::NAIVE: 
            vae::naive::forward(buf_); 
            break;
    }

    CUDA_CHECK(cudaMemcpy(h_output, buf_.d_X_hat.ptr, 
               buf_.config.batch_size * buf_.config.input_dim * sizeof(float), 
               cudaMemcpyDeviceToHost));
}

// controlla
void VAE::sample(float* h_output, int n_samples) {
    // Genera Z random sulla CPU (Box-Muller)
    // Assumiamo n_samples <= batch_size per semplicità
    int z_elem = n_samples * buf_.config.latent_dim;
    std::vector<float> h_z(z_elem);

    for (int i = 0; i < z_elem; ++i) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        u1 = (u1 < 1e-9f) ? 1e-9f : u1;
        
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
        h_z[i] = z;
    }

    decode(h_z.data(), h_output);
}

void VAE::save_weights(const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Error opening %s\n", filename); return; }

    auto save = [f](LinearLayer& l) {
        int sz_w = l.input_dim * l.output_dim;
        int sz_b = l.output_dim;
        std::vector<float> hW(sz_w), hb(sz_b);
        
        CUDA_CHECK(cudaMemcpy(hW.data(), l.W.ptr, sz_w*4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hb.data(), l.b.ptr, sz_b*4, cudaMemcpyDeviceToHost));
        
        fwrite(hW.data(), 4, sz_w, f);
        fwrite(hb.data(), 4, sz_b, f);
    };

    save(buf_.enc1); save(buf_.enc2);
    save(buf_.dec1); save(buf_.dec2);
    
    fclose(f);
    printf("Saved weights to %s\n", filename);
}

void VAE::load_weights(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Error opening %s\n", filename); return; }

    auto load = [f](LinearLayer& l) {
        int sz_w = l.input_dim * l.output_dim;
        int sz_b = l.output_dim;
        std::vector<float> hW(sz_w), hb(sz_b);

        fread(hW.data(), 4, sz_w, f);
        fread(hb.data(), 4, sz_b, f);

        CUDA_CHECK(cudaMemcpy(l.W.ptr, hW.data(), sz_w*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(l.b.ptr, hb.data(), sz_b*4, cudaMemcpyHostToDevice));
    };

    load(buf_.enc1); load(buf_.enc2);
    load(buf_.dec1); load(buf_.dec2);
    
    fclose(f);
    printf("Loaded weights from %s\n", filename);
}