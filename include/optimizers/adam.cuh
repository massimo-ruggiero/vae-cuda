#pragma once
#include "vae_config.cuh"
#include "vae_buffers.cuh"

class Adam {
private:
    VAEAdamState* state_;
    float lr_;
    float beta1_;
    float beta2_;
    float epsilon_;
    int timestep_;

    void apply_layer(LinearLayer& l, LinearGrads& g, AdamState& s);

public:
    Adam(const VAEConfig& config, 
         float lr = 1e-3f, 
         float beta1 = 0.9f, 
         float beta2 = 0.999f, 
         float epsilon = 1e-8f);
    
    ~Adam();

    void step(VAEBuffers& params, VAEGradients& grads);
};