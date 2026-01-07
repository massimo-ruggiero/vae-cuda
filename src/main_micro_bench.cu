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
    std::string kernel_file;
};

static CliArgs parse_cli(int argc, char** argv) {
    CliArgs args{
        .outdir = "",
        .option = "benchmark",
        .kernel_file = "all"
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--outdir" && i + 1 < argc) {
            args.outdir = argv[i + 1];
        } else if (arg == "--option" && i + 1 < argc) {
            args.option = argv[i + 1];
        } else if (arg == "--kernel-file" && i + 1 < argc) {
            args.kernel_file = argv[i + 1];
        }
    }

    if (args.option != "benchmark" && args.option != "profiling") {
        std::cerr << "[micro-bench] ERROR: unsupported option '" << args.option
                  << "'. Use 'benchmark' or 'profiling'.\n";
        std::exit(1);
    }

    if (args.kernel_file != "all" &&
        args.kernel_file != "linalg" &&
        args.kernel_file != "linear" &&
        args.kernel_file != "activations" &&
        args.kernel_file != "loss" &&
        args.kernel_file != "reparam" &&
        args.kernel_file != "optimizers") {
        std::cerr << "[micro-bench] ERROR: unsupported kernel file '" << args.kernel_file
                  << "'. Use one of: all, linalg, linear, activations, loss, reparam, optimizers.\n";
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

static bool should_run(const std::string& kernel_file, 
                       const std::string& filter) {
    return filter == "all" || filter == kernel_file;
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

    std::cout << "\n🚀 Launching micro benchmark (" << args.option << ")...\n";
    if (args.kernel_file != "all") {
        std::cout << "Running kernel file: " << args.kernel_file << "\n";
    }
    if (args.option == "profiling") {
        std::cout << "Running with no warmup and a single iteration per kernel.\n";
    }
    std::cout << "---------------------------------\n";

    // ====================
    // LINALG BENCHMARK
    // ====================
    if (should_run("linalg", args.kernel_file)) {
        std::cout << "\n[Benchmark] ⚙️ Linalg...\n";
        Csv csv(join_path(args.outdir, "bench_linalg.csv").c_str(), write_csv);
        csv.header();

        run_sgemm(csv, gen, timer, config);
        run_transpose(csv, gen, timer, config);
        run_add_in_place(csv, gen, timer, config);
        std::cout << "[Benchmark] ✅ Linalg done.\n";
    }

    // ====================
    // LINEAR BENCHMARK
    // ====================
    if (should_run("linear", args.kernel_file)) {
        std::cout << "\n[Benchmark] ⚙️ Linear...\n";
        Csv csv(join_path(args.outdir, "bench_linear.csv").c_str(), write_csv);
        csv.header();

        run_add_bias(csv, gen, timer, config);
        run_db(csv, gen, timer, config);
        std::cout << "[Benchmark] ✅ Linear done.\n";
    }

    // ====================
    // ACTIVATIONS BENCHMARK
    // ====================
    if (should_run("activations", args.kernel_file)) {
        std::cout << "\n[Benchmark] ⚙️ Activations...\n";
        Csv csv(join_path(args.outdir, "bench_activations.csv").c_str(), write_csv);
        csv.header();

        run_leaky_relu_forward(csv, gen, timer, config);
        run_leaky_relu_backward(csv, gen, timer, config);
        run_sigmoid_forward(csv, gen, timer, config);
        run_sigmoid_backward(csv, gen, timer, config);
        std::cout << "[Benchmark] ✅ Activations done.\n";
    }

    // ====================
    // LOSS BENCHMARK
    // ====================
    if (should_run("loss", args.kernel_file)) {
        std::cout << "\n[Benchmark] ⚙️ Loss...\n";
        Csv csv(join_path(args.outdir, "bench_loss.csv").c_str(), write_csv);
        csv.header();

        run_loss_forward(csv, gen, timer, config);
        run_bce_backward(csv, gen, timer, config);
        run_kl_backward(csv, gen, timer, config);
        std::cout << "[Benchmark] ✅ Loss done.\n";
    }

    // ====================
    // REPARAMETRIZATION BENCHMARK
    // ====================
    if (should_run("reparam", args.kernel_file)) {
        std::cout << "\n[Benchmark] ⚙️ Reparametrization...\n";
        Csv csv(join_path(args.outdir, "bench_reparam.csv").c_str(), write_csv);
        csv.header();

        run_reparam_init(csv, gen, timer, config);
        run_reparam_forward(csv, gen, timer, config);
        run_reparam_backward(csv, gen, timer, config);
        std::cout << "[Benchmark] ✅ Reparametrization done.\n";
    }

    // ====================
    // OPTIMIZER BENCHMARK
    // ====================
    if (should_run("optimizers", args.kernel_file)) {
        std::cout << "\n[Benchmark] ⚙️ Optimizers...\n";
        Csv csv(join_path(args.outdir, "bench_optimizers.csv").c_str(), write_csv);
        csv.header();

        run_adam_step(csv, gen, timer, config);
        std::cout << "[Benchmark] ✅ Optimizers done.\n";
    }

    curandDestroyGenerator(gen);

    std::cout << "\n✅ Micro benchmark completed successfully.\n";
    std::cout << "---------------------------------\n";
    return 0;
}
