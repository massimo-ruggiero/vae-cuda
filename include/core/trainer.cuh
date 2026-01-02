#pragma once

#include "vae.cuh"
#include "adam.cuh"
#include "mnist_loader.hpp"

class Trainer {
private:
    VAE& vae_;
    Adam& optim_;
    MNISTLoader& loader_;
    VAEConfig config_;

    float* h_batch_buffer_ = nullptr;

public:
    Trainer(VAE& vae, 
            Adam& optim, 
            MNISTLoader& loader, 
            const VAEConfig& cfg);

    ~Trainer();

    void fit(int epochs);
};
