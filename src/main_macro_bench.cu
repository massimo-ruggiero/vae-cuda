#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cctype>

#include "vae_config.cuh"
#include "vae.cuh"
#include "adam.cuh"
#include "trainer.cuh"
#include "mnist_loader.hpp"

struct CliArgs {
    int epochs = 100;
    std::vector<VAEStrategy> strategies;
};

static CliArgs parse_cli(int argc, char** argv) {
    CliArgs args;
    std::string strategies_string = "NAIVE,OPTIMIZED";

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--epochs" && i + 1 < argc) {
            args.epochs = std::stoi(argv[i + 1]);
        } else if (arg == "--strategies" && i + 1 < argc) {
            strategies_string = argv[i + 1];
        }
    }

    std::stringstream ss(strategies_string);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token.erase(
            std::remove_if(token.begin(), token.end(), ::isspace), 
            token.end()
        );
        if (token.empty()) continue;
        VAEStrategy s;
        if (!parse_strategy(token, s)) {
            std::cerr << "[MacroBench] ERROR: unsupported strategy '" << token << ".\n";
            std::exit(1);
        }
        args.strategies.push_back(s);
    }

    if (args.strategies.empty()) {
        std::cerr << "[MacroBench] ERROR: no strategies provided.\n";
        std::exit(1);
    }
    return args;
}

static void run_training(VAEStrategy strategy, const CliArgs& args) {
    VAEConfig config = {
        .batch_size = 128,
        .input_dim  = 784,
        .hidden_dim = 400,
        .latent_dim = 200,
        .beta       = 1.0f,
        .strategy   = strategy
    };

    const char* sname = to_string(strategy);
    const float learning_rate = 1e-3f;

    const std::string data_path = "data/train.bin";
    if (!std::filesystem::exists(data_path)) {
        std::cerr << "[MacroBench][" << sname << "] ERROR: dataset not found at '"
                  << data_path << "'.\n";
        return;
    }

    std::cout << "\n[MacroBench] ⚙️ Training strategy = " << sname
              << " for " << args.epochs << " epochs\n";

    MNISTLoader loader(data_path.c_str());
    VAE vae(config);
    Adam optimizer(config, learning_rate);
    Trainer trainer(vae, optimizer, loader, config);
    trainer.fit(args.epochs);

    std::cout << "[MacroBench] ✅ Strategy " << sname << " completed.\n";
}

int main(int argc, char** argv) {
    CliArgs args = parse_cli(argc, argv);

    std::cout << "\n🚀 Launching macro benchmark\n";
    std::cout << "---------------------------------\n";
    std::cout << "[MacroBench] Config: epochs = " << args.epochs << "\n";

    for (VAEStrategy s : args.strategies) {
        run_training(s, args);
    }

    std::cout << "\n✅ Macro benchmark completed.\n";
    std::cout << "---------------------------------\n";
    return 0;
}
