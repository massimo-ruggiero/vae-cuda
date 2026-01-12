#include "trainer.cuh"

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>


Trainer::Trainer(VAE& vae, 
                 Adam& optim, 
                 MNISTLoader& loader, 
                 const VAEConfig& cfg)
        : vae_(vae), 
        optim_(optim), 
        loader_(loader), 
        config_(cfg) {
    
    h_batch_buffer_ = new float[config_.batch_size * config_.input_dim];
}

Trainer::~Trainer() {
    if (h_batch_buffer_) { 
        delete[] h_batch_buffer_;
        h_batch_buffer_ = nullptr;
    }
}

void Trainer::fit(int epochs) {
    std::cout << "\n🚀 Starting training\n";
    std::cout << "---------------------------------\n";
    std::cout << "[Trainer] ⚙️ Config: epochs = " << epochs
              << ", batch_size = " << config_.batch_size << "\n";

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    std::vector<double> epoch_times;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        
        loader_.shuffle();

        float total_loss = 0.0f;
        int batches = 0;

        cudaEventRecord(start);

        while (loader_.next_batch(h_batch_buffer_, config_.batch_size)) {
            float loss = vae_.train_step(h_batch_buffer_);
            optim_.step(vae_.get_buffers(), vae_.get_gradients());
            total_loss += loss;
            batches++;
        }

        cudaEventRecord(end);
        cudaEventSynchronize(end);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, end);

        float avg_loss = (batches > 0) ? (total_loss / batches) : 0.0f;
        double seconds = static_cast<double>(ms) / 1000.0;
        epoch_times.push_back(seconds);

        std::cout << "Epoch [" << std::setw(3) << epoch + 1 << "/" << epochs << "] "
                  << "| Avg Loss: " << std::fixed << std::setprecision(6) << avg_loss << " "
                  << "| Time: " << std::fixed << std::setprecision(3) << seconds << "s" 
                  << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    double summed = std::accumulate(epoch_times.begin(), epoch_times.end(), 0.0);
    std::vector<double> times_sorted = epoch_times;
    std::sort(times_sorted.begin(), times_sorted.end());
    double median = times_sorted[times_sorted.size() / 2];

    std::cout << "---------------------------------\n";
    std::cout << "[Trainer] ✅ Total time = "
              << std::fixed << std::setprecision(2) << summed
              << " s | median/epoch = " << std::setprecision(3) << median << " s\n";
}
