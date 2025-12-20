#include <cstdio>
#include <iostream>

#include "vae_config.cuh"
#include "vae.cuh"
#include "adam.cuh"
#include "mnist_loader.h" 
#include "trainer.h"      


void save_binary_image(const char* filename, float* data, int size) {
    FILE* f = fopen(filename, "wb");
    if (!f) { printf("Errore apertura %s\n", filename); return; }
    fwrite(data, sizeof(float), size, f);
    fclose(f);
    printf("[Main] Salvata immagine di test in: %s\n", filename);
}


int main() {
    VAEConfig config;
    config.batch_size = 100;
    config.input_dim  = 784;
    config.hidden_dim = 400;
    config.latent_dim = 200; 
    config.beta       = 1.0f;
    config.strategy   = VAEStrategy::NAIVE;

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
    // printf("[Main] Pesi salvati in: %s\n", save_path);

    // TEST

    printf("[Test] Generazione immagine di prova...\n");

    float* h_batch_in  = new float[config.batch_size * 784];
    float* h_batch_out = new float[config.batch_size * 784];

    loader.shuffle();
    loader.next_batch(h_batch_in, config.batch_size);

    vae.reconstruct(h_batch_in, h_batch_out);

    FILE* f1 = fopen("original.raw", "wb");
    fwrite(h_batch_in, sizeof(float), 784, f1); 
    fclose(f1);

    FILE* f2 = fopen("reconstructed.raw", "wb");
    fwrite(h_batch_out, sizeof(float), 784, f2); 
    fclose(f2);

    printf("[Test] Salvati 'original.raw' e 'reconstructed.raw' (solo 1 img).\n");

    // Pulizia
    delete[] h_batch_in;
    delete[] h_batch_out;

    return 0;
}