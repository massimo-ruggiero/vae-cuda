#pragma once

#include <cstdio>
#include <chrono> 
#include "vae.cuh"
#include "adam.cuh"
#include "mnist_loader.h"

class Trainer {
private:
    VAE& vae_;
    Adam& optim_;
    MNISTLoader& loader_;
    VAEConfig config_;

    float* h_batch_buffer_;

public:
    Trainer(VAE& vae, Adam& optim, MNISTLoader& loader, const VAEConfig& cfg)
        : vae_(vae), optim_(optim), loader_(loader), config_(cfg) 
    {
        h_batch_buffer_ = new float[config_.batch_size * config_.input_dim];
    }

    ~Trainer() {
        if (h_batch_buffer_) {
            delete[] h_batch_buffer_;
        }
    }

    void fit(int epochs) {
        printf("----------------------------------------------------------\n");
        printf("[Trainer] Starting Training: %d Epochs, Batch Size %d\n", epochs, config_.batch_size);
        printf("----------------------------------------------------------\n");

        auto total_start = std::chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < epochs; ++epoch) {
            
            loader_.shuffle();

            auto epoch_start = std::chrono::high_resolution_clock::now();
            
            float total_loss = 0.0f;
            int batches = 0;

            while (loader_.next_batch(h_batch_buffer_, config_.batch_size)) {
                float loss = vae_.train_step(h_batch_buffer_);
                optim_.step(vae_.get_buffers(), vae_.get_gradients());
                total_loss += loss;
                batches++;
            }

            auto epoch_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = epoch_end - epoch_start;

            float avg_loss = (batches > 0) ? (total_loss / batches) : 0.0f;

            printf("Epoch [%d/%d] | Loss: %.4f | Time: %.3f sec\n", 
                   epoch + 1, epochs, avg_loss, elapsed.count());
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_elapsed = total_end - total_start;

        printf("----------------------------------------------------------\n");
        printf("[Trainer] Training Completed in %.2f seconds.\n", total_elapsed.count());
    }
};