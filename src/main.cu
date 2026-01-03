#include <cstdio>
#include <iostream>
#include <filesystem>
#include <vector>

#include "vae_config.cuh"
#include "vae.cuh"
#include "adam.cuh"
#include "mnist_loader.hpp"
#include "trainer.cuh"


static std::string get_outdir(int argc, char** argv) {
    std::string outdir = "images";
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--outdir" && i + 1 < argc) {
            outdir = argv[i + 1];
        }
    } 
    return outdir;
}

static std::string join_path(const std::string& dir, const std::string& file) {
    namespace fs = std::filesystem;
    return (fs::path(dir) / fs::path(file)).string();
}

static bool write_raw(const std::string& path, const float* data) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fwrite(data, sizeof(float), 784, f);
    std::fclose(f);
    return true;
}

const std::vector<VAEStrategy> strategies = {
        VAEStrategy::NAIVE,
        VAEStrategy::TILING,
        VAEStrategy::PADDING,
        VAEStrategy::REDUCTION,
        VAEStrategy::UNROLLED_REDUCTION,
        VAEStrategy::WARP_REDUCTION,
        VAEStrategy::VECTORIZED,
        VAEStrategy::KERNEL_FUSION
    };

int main(int argc, char** argv) {
    const std::string outdir = get_outdir(argc, argv);

    try {
        std::filesystem::create_directories(outdir);
    } catch (const std::exception& e) {
        std::cerr << "[main] ERROR: cannot create outdir '" << outdir
                  << "': " << e.what() << "\n";
        return 1;
    }

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

    for (VAEStrategy s : strategies) {
        VAEConfig cfg = base_cfg;
        cfg.strategy = s;

        const std::string sname = s.to_string();
        const std::string outdir = join_path(base_outdir, sname);

        try {
            std::filesystem::create_directories(outdir);
        } catch (const std::exception& e) {
            std::cerr << "[" << sname << "] ERROR: cannot create dir '" << outdir
                      << "': " << e.what() << "\n";
            continue;
        }

        std::cout << "\n==============================\n";
        std::cout << "[main] Running strategy: " << sname << "\n";
        std::cout << "[main] Outdir: " << outdir << "\n";
        std::cout << "==============================\n";

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

        const std::string orig_path = join_path(outdir, "original.raw");
        const std::string reco_path = join_path(outdir, "reconstructed.raw");

        if (!write_raw(orig_path, h_batch_in))
            std::cerr << "[" << sname << "] ERROR: cannot write " << orig_path << "\n";
        if (!write_raw(reco_path, h_batch_out))
            std::cerr << "[" << sname << "] ERROR: cannot write " << reco_path << "\n";


        std::cout << "[Main] ✅ Saved 'original.raw' and 'reconstructed.raw' (first image).\n";

        delete[] h_batch_in;
        delete[] h_batch_out;
    }
    std::cout << "[main] ✅ Finished all strategies.\n";
    return 0;
}
