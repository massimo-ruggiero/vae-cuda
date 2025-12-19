#pragma once
#include <cstddef>


enum class VAEStrategy {
    NAIVE
};

struct VAEConfig {
    int batch_size = 64;
    int input_dim = 784;
    int hidden_dim = 400;
    int latent_dim = 20;
    VAEStrategy strategy = VAEStrategy::NAIVE;
    float beta = 1.0f;
};
