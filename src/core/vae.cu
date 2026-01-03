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


static void sample_normal(float* data, size_t n) {
    std::mt19937 gen(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n; ++i)
        data[i] = dist(gen);
}

static void init_uniform(LinearLayer& layer, float limit) {
    std::vector<float> h_W(layer.input_dim * layer.output_dim);
    std::vector<float> h_b(layer.output_dim, 0.0f); 

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-limit, limit);

    for(float& w : h_W) w = dis(gen);

    CUDA_CHECK(cudaMemcpy(layer.W.ptr, h_W.data(), 
                          h_W.size() * sizeof(float), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(layer.b.ptr, h_b.data(), 
                          h_b.size() * sizeof(float), 
                          cudaMemcpyHostToDevice));
}

static void init_xavier(LinearLayer& layer) {
    float limit = sqrtf(6.0f / (float)(layer.input_dim + layer.output_dim));
    init_uniform(layer, limit);
}

static void init_kaiming(LinearLayer& layer) {
    float limit = sqrtf(6.0f / (float)layer.input_dim);
    init_uniform(layer, limit);
}


VAE::VAE(const VAEConfig& config)
    : buf_(config),
      grads_(nullptr),
      d_bce_loss_(nullptr),
      d_kl_loss_(nullptr) {

    const size_t num_states = config.num_states();
    reparametrization::init(buf_.d_states, num_states, 42, config.strategy);

    initialize_weights();
    ensure_training_resources();
}

VAE::~VAE(){
    free_training_resources(); 
}

void VAE::initialize_weights() {
    init_kaiming(buf_.enc1);
    init_xavier(buf_.enc2_mu);
    init_xavier(buf_.enc2_logvar);
    init_kaiming(buf_.dec1);
    init_xavier(buf_.dec2);
}

void VAE::ensure_training_resources() {
    if (grads_) return;

    printf("VAE: Allocating training resources...\n");
    grads_ = new VAEGradients(buf_.config);
    
    if (!d_bce_loss_) CUDA_CHECK(cudaMalloc(&d_bce_loss_, sizeof(float)));
    if (!d_kl_loss_) CUDA_CHECK(cudaMalloc(&d_kl_loss_, sizeof(float)));
}

void VAE::free_training_resources() {
    printf("VAE: Freeing training resources...\n");
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

float VAE::train_step(const float* h_batch){
    ensure_training_resources();

    size_t data_size = buf_.config.batch_size * buf_.config.input_dim * sizeof(float);
    CUDA_CHECK(cudaMemcpy(buf_.d_X.ptr, h_batch, 
                          data_size, 
                          cudaMemcpyHostToDevice));

    grads_->zero_all_grads();

    vae::forward(buf_); 

    float h_loss = 0.0f;
    loss::forward(buf_.d_X.ptr, 
                  buf_.dec2.Z.ptr, 
                  buf_.enc2_mu.Z.ptr,        // mu
                  buf_.enc2_logvar.Z.ptr,    // logvar
                  d_bce_loss_, 
                  d_kl_loss_, 
                  &h_loss,
                  buf_.config.batch_size, 
                  buf_.config.input_dim, 
                  buf_.config.latent_dim, 
                  buf_.config.strategy,
                  buf_.config.beta);
    
    vae::backward(buf_, *grads_); 

    float h_bce = 0.0f, h_kl = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_bce, d_bce_loss_, 
                          sizeof(float), 
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_kl, d_kl_loss_, 
                          sizeof(float), 
                          cudaMemcpyDeviceToHost));

    DEBUG("[LOSS] total=%f bce=%f kl=%f beta=%f", h_loss, h_bce, h_kl, buf_.config.beta);

    return h_loss;
}

void VAE::encode(const float* h_input, float* h_mu, float* h_logvar) {
    CUDA_CHECK(cudaMemcpy(buf_.d_X.ptr, h_input, 
                          buf_.config.batch_size * buf_.config.input_dim * sizeof(float), 
                          cudaMemcpyHostToDevice));

    vae::encoder_pass(buf_);

    CUDA_CHECK(cudaMemcpy(h_mu, buf_.enc2_mu.Z.ptr, 
                          buf_.config.batch_size * buf_.config.latent_dim * sizeof(float), 
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_logvar, buf_.enc2_logvar.Z.ptr,
                          buf_.config.batch_size * buf_.config.latent_dim * sizeof(float), 
                          cudaMemcpyDeviceToHost));
}


void VAE::decode(const float* h_z, float* h_output) {
    CUDA_CHECK(cudaMemcpy(buf_.d_z.ptr, h_z, 
                          buf_.config.batch_size * buf_.config.latent_dim * sizeof(float), 
                          cudaMemcpyHostToDevice));

    vae::decoder_pass(buf_); 

    CUDA_CHECK(cudaMemcpy(h_output, buf_.d_X_hat.ptr, 
                          buf_.config.batch_size * buf_.config.input_dim * sizeof(float), 
                          cudaMemcpyDeviceToHost));
}

void VAE::reconstruct(const float* h_input, float* h_output) {
    CUDA_CHECK(cudaMemcpy(buf_.d_X.ptr, h_input, 
               buf_.config.batch_size * buf_.config.input_dim * sizeof(float), 
               cudaMemcpyHostToDevice));
    
    vae::forward(buf_); 

    CUDA_CHECK(cudaMemcpy(h_output, buf_.d_X_hat.ptr, 
               buf_.config.batch_size * buf_.config.input_dim * sizeof(float), 
               cudaMemcpyDeviceToHost));
}

void VAE::sample(float* h_output, int n_samples) {
    const int latent_dim = buf_.config.latent_dim;
    const int input_dim = buf_.config.input_dim;

    if (n_samples > buf_.config.batch_size) {
        std::cerr << "[VAE::sample] ERROR: n_samples > batch_size\n";
        return;
    }

    std::vector<float> h_z(n_samples * latent_dim);
    sample_normal(h_z.data(), h_z.size());

    CUDA_CHECK(cudaMemcpy(
        buf_.d_z.ptr,
        h_z.data(),
        h_z.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    vae::decoder_pass(buf_);

    CUDA_CHECK(cudaMemcpy(
        h_output,
        buf_.d_X_hat.ptr,
        n_samples * input_dim * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
}