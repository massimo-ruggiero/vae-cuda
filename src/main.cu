#include <cstdio>
#include <iostream>

#include "vae_config.cuh"
#include "vae.cuh"
#include "adam.cuh"
#include "mnist_loader.hpp"
#include "trainer.cuh"


int main() {
    VAEConfig config = {
        .batch_size = 100,
        .input_dim  = 784,
        .hidden_dim = 400,
        .latent_dim = 200,
        .beta       = 1.0f,
        .strategy   = VAEStrategy::NAIVE
    };

    float learning_rate = 1e-3f;
    int epochs = 100;
    const char* data_path = "data/train.bin";

    MNISTLoader loader(data_path);

    VAE vae(config);

    Adam optimizer(config, learning_rate);

    Trainer trainer(vae, optimizer, loader, config);
    
    trainer.fit(epochs);

    // const char* save_path = "vae_weights.bin";
    // vae.save_weights(save_path);
    // std::cout << "[Main] ✅ Saved weights to: " << save_path << "\n";

    std::cout << "[Main] ⚙️ Generating test reconstruction...\n";

    float* h_batch_in  = new float[config.batch_size * 784];
    float* h_batch_out = new float[config.batch_size * 784];

    loader.shuffle();
    loader.next_batch(h_batch_in, config.batch_size);

    vae.reconstruct(h_batch_in, h_batch_out);

    FILE* f1 = std::fopen("original.raw", "wb");
    if (f1) {
        std::fwrite(h_batch_in, sizeof(float), 784, f1);
        std::fclose(f1);
    } else {
        std::cerr << "[Main] ERROR: cannot open original.raw for writing\n";
    }

    FILE* f2 = std::fopen("reconstructed.raw", "wb");
    if (f2) {
        std::fwrite(h_batch_out, sizeof(float), 784, f2);
        std::fclose(f2);
    } else {
        std::cerr << "[Main] ERROR: cannot open reconstructed.raw for writing\n";
    }

    std::cout << "[Main] ✅ Saved 'original.raw' and 'reconstructed.raw' (first image).\n";

    delete[] h_batch_in;
    delete[] h_batch_out;

    return 0;
}
