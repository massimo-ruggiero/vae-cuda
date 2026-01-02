#include <iostream>
#include <string>
#include <curand.h>
#include <filesystem>

#include "bench_core.cuh"
#include "bench_ops.cuh"
#include "layer_buffers.cuh"


static std::string get_outdir(int argc, char** argv) {
    std::string outdir = "results/micro_bench/csv";
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


int main(int argc, char** argv) {
    const std::string outdir = get_outdir(argc, argv);

    try {
        std::filesystem::create_directories(outdir);
    } catch (const std::exception& e) {
        std::cerr << "[micro-bench] ERROR: cannot create outdir '" << outdir
                  << "': " << e.what() << "\n";
        return 1;
    }

    BenchmarkConfig config = {
        .warmup_iters = 5,
        .iters = 50
    };

    curandGenerator_t gen = make_gen(1234ULL);
    Timer timer;

    DeviceSpecs specs = DeviceSpecs::detect();
    std::cout << "\n🚀 Launching micro benchmark suite\n";
    std::cout << "---------------------------------\n";

    // ====================
    // LINALG BENCHMARK
    // ====================
    {
        std::cout << "\n[Benchmark] ⚙️ Linalg...\n";
        Csv csv(join_path(outdir, "bench_linalg.csv").c_str());
        csv.header();

        run_sgemm(csv, gen, timer, config, specs);
        run_transpose(csv, gen, timer, config, specs);
        run_add_in_place(csv, gen, timer, config, specs);
        std::cout << "[Benchmark] ✅ Linalg done.\n";
    }

    // ====================
    // ACTIVATIONS BENCHMARK
    // ====================
    {
        std::cout << "\n[Benchmark] ⚙️ Activations...\n";
        Csv csv(join_path(outdir, "bench_activations.csv").c_str());
        csv.header();

        run_leaky_relu_forward(csv, gen, timer, config, specs);
        run_leaky_relu_backward(csv, gen, timer, config, specs);
        run_sigmoid_forward(csv, gen, timer, config, specs);
        run_sigmoid_backward(csv, gen, timer, config, specs);
        std::cout << "[Benchmark] ✅ Activations done.\n";
    }

    // ====================
    // LOSS BENCHMARK
    // ====================
    {
        std::cout << "\n[Benchmark] ⚙️ Loss...\n";
        Csv csv(join_path(outdir, "bench_loss.csv").c_str());
        csv.header();

        run_loss_forward(csv, gen, timer, config, specs);
        run_bce_backward(csv, gen, timer, config, specs);
        run_kl_backward(csv, gen, timer, config, specs);
        std::cout << "[Benchmark] ✅ Loss done.\n";
    }

    // ====================
    // REPARAMETRIZATION BENCHMARK
    // ====================
    {
        std::cout << "\n[Benchmark] ⚙️ Reparametrization...\n";
        Csv csv(join_path(outdir, "bench_reparam.csv").c_str());
        csv.header();

        run_reparam_forward(csv, gen, timer, config, specs);
        run_reparam_backward(csv, gen, timer, config, specs);
        std::cout << "[Benchmark] ✅ Reparametrization done.\n";
    }

    // ====================
    // OPTIMIZER BENCHMARK
    // ====================
    {
        std::cout << "\n[Benchmark] ⚙️ Optimizers...\n";
        Csv csv(join_path(outdir, "bench_optimizers.csv").c_str());
        csv.header();

        run_adam_step(csv, gen, timer, config, specs);
        std::cout << "[Benchmark] ✅ Optimizers done.\n";
    }

    curandDestroyGenerator(gen);

    std::cout << "\n✅ Benchmark suite completed successfully.\n";
    std::cout << "---------------------------------\n";
    return 0;
}
