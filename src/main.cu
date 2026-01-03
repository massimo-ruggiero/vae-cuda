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

static std::string join_path(const std::string& dir, 
                             const std::string& file, 
                             int n) {
    namespace fs = std::filesystem;
    return (fs::path(dir) / fs::path(file)).string();
}

static bool write_raw(const std::string& path, const float* data) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fwrite(data, sizeof(float), n, f);
    std::fclose(f);
    return true;
}

static void save_samples_raw(const float* samples,
                             int n_samples,
                             int input_dim,
                             const std::string& outdir) {
    std::filesystem::create_directories(outdir);
    for (int i = 0; i < n_samples; ++i) {
        std::string path = outdir + "/sample_" + std::to_string(i) + ".raw";
        if (!write_raw(path, samples + i * input_dim, input_dim))
            std::cerr << "[save_samples] ERROR: cannot write " << path << "\n";
    }
    std::cout << "[save_samples] ✅ Saved " << n_samples
              << " samples to " << outdir << "\n";
}

const std::vector<VAEStrategy> strategies = {
        VAEStrategy::NAIVE,
        VAEStrategy::TILING,
        VAEStrategy::PADDING,
        VAEStrategy::REDUCTION,
        VAEStrategy::UNROLLED_REDUCTION,
        VAEStrategy::WARP_REDUCTION,
        VAEStrategy::VECTORIZED,
        // VAEStrategy::KERNEL_FUSION  <- TODO
    };

int main(int argc, char** argv) {
    const std::string outdir_base = get_outdir(argc, argv);

    try {
        std::filesystem::create_directories(outdir_base);
    } catch (const std::exception& e) {
        std::cerr << "[main] ERROR: cannot create outdir '" << outdir_base
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
    int epochs = 20;
    const char* data_path = "data/train.bin";

    for (VAEStrategy s : strategies) {
        config.strategy = s;

        const std::string sname = to_string(s);
        const std::string outdir = join_path(outdir_base, sname);

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

        if (!write_raw(orig_path, h_batch_in, config.input_dim))
            std::cerr << "[" << sname << "] ERROR: cannot write " << orig_path << "\n";
        if (!write_raw(reco_path, h_batch_out, config.input_dim))
            std::cerr << "[" << sname << "] ERROR: cannot write " << reco_path << "\n";


        std::cout << "[Main] ✅ Saved 'original.raw' and 'reconstructed.raw' (first image).\n";

        std::cout << "[Main] ⚙️ Generating images from sampling...\n";
        int n_samples = 16;   
        std::vector<float> h_samples(n_samples * config.input_dim);
        vae.sample(h_samples.data(), n_samples);
        save_samples_raw(
            h_samples.data(),
            n_samples,
            config.input_dim,
            outdir   
        );

        delete[] h_batch_in;
        delete[] h_batch_out;
    }

    std::cout << "[main] ✅ Finished all strategies.\n";
    return 0;
}
