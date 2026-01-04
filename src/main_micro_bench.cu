#include <iostream>
#include <cstdlib>
#include <string>
#include <curand.h>
#include <filesystem>

#include "bench_core.cuh"
#include "bench_ops.cuh"
#include "layer_buffers.cuh"


struct CliArgs {
    std::string outdir;
    std::string option;
};

static CliArgs parse_cli(int argc, char** argv) {
    CliArgs args{
        .outdir = "",
        .option = "benchmark"
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--outdir" && i + 1 < argc) {
            args.outdir = argv[i + 1];
        } else if (arg == "--option" && i + 1 < argc) {
            args.option = argv[i + 1];
        }
    }

    if (args.option != "benchmark" && args.option != "profiling") {
        std::cerr << "[micro-bench] ERROR: unsupported option '" << args.option
                  << "'. Use 'benchmark' or 'profiling'.\n";
        std::exit(1);
    }

    if (args.outdir.empty()) {
        args.outdir = (args.option == "profiling")
            ? "results/micro_bench/ncu"
            : "results/micro_bench/csv";
    }

    return args;
}

static std::string join_path(const std::string& dir, const std::string& file) {
    namespace fs = std::filesystem;
    return (fs::path(dir) / fs::path(file)).string();
}

static BenchmarkConfig make_config(const std::string& option) {
    BenchmarkConfig config{};
    if (option == "profiling") {
        config.warmup_iters = 0;
        config.iters = 1;
    } else {
        config.warmup_iters = 5;
        config.iters = 50;
    }
    return config;
}

int main(int argc, char** argv) {
    const CliArgs args = parse_cli(argc, argv);
    const bool write_csv = (args.option == "benchmark");

    try {
        std::filesystem::create_directories(args.outdir);
    } catch (const std::exception& e) {
        std::cerr << "[micro-bench] ERROR: cannot create outdir '" << args.outdir
                  << "': " << e.what() << "\n";
        return 1;
    }

    BenchmarkConfig config = make_config(args.option);

    curandGenerator_t gen = make_gen(1234ULL);
    Timer timer;

    DeviceSpecs specs = DeviceSpecs::detect();
    std::cout << "\n🚀 Launching micro benchmark (" << args.option << ")...\n";
    if (args.option == "profiling") {
        std::cout << "Running with no warmup and a single iteration per kernel.\n";
    }
    std::cout << "---------------------------------\n";

    // ====================
    // LINALG BENCHMARK
    // ====================
    {
        std::cout << "\n[Benchmark] ⚙️ Linalg...\n";
        Csv csv(join_path(args.outdir, "bench_linalg.csv").c_str(), write_csv);
        csv.header(specs);

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
        Csv csv(join_path(args.outdir, "bench_activations.csv").c_str(), write_csv);
        csv.header(specs);

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
        Csv csv(join_path(args.outdir, "bench_loss.csv").c_str(), write_csv);
        csv.header(specs);

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
        Csv csv(join_path(args.outdir, "bench_reparam.csv").c_str(), write_csv);
        csv.header(specs);

        run_reparam_forward(csv, gen, timer, config, specs);
        run_reparam_backward(csv, gen, timer, config, specs);
        std::cout << "[Benchmark] ✅ Reparametrization done.\n";
    }

    // ====================
    // OPTIMIZER BENCHMARK
    // ====================
    {
        std::cout << "\n[Benchmark] ⚙️ Optimizers...\n";
        Csv csv(join_path(args.outdir, "bench_optimizers.csv").c_str(), write_csv);
        csv.header(specs);

        run_adam_step(csv, gen, timer, config, specs);
        std::cout << "[Benchmark] ✅ Optimizers done.\n";
    }

    curandDestroyGenerator(gen);

    std::cout << "\n✅ Micro benchmark completed successfully.\n";
    std::cout << "---------------------------------\n";
    return 0;
}
