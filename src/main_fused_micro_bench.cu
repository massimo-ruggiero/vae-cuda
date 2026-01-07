#include <iomanip>
#include <iostream>
#include <string>

#include "bench_core.cuh"
#include "fused.cuh"
#include "linear.cuh"
#include "activations.cuh"
#include "layer_buffers.cuh"

struct CliArgs {
    int batch = 128;
    int input_dim = 784;
    int output_dim = 400;
    int warmup = 5;
    int iters = 50;
};

static CliArgs parse_cli(int argc, char** argv) {
    CliArgs args{};
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--batch" && i + 1 < argc) {
            args.batch = std::stoi(argv[++i]);
        } else if (arg == "--input-dim" && i + 1 < argc) {
            args.input_dim = std::stoi(argv[++i]);
        } else if (arg == "--output-dim" && i + 1 < argc) {
            args.output_dim = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            args.warmup = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            args.iters = std::stoi(argv[++i]);
        }
    }
    return args;
}

int main(int argc, char** argv) {
    const CliArgs args = parse_cli(argc, argv);

    BenchmarkConfig config{};
    config.warmup_iters = args.warmup;
    config.iters = args.iters;

    const int M = args.batch;
    const int K = args.input_dim;
    const int N = args.output_dim;
    const size_t size_X = static_cast<size_t>(M) * static_cast<size_t>(K);
    const size_t size_W = static_cast<size_t>(K) * static_cast<size_t>(N);
    const size_t size_Z = static_cast<size_t>(M) * static_cast<size_t>(N);

    GPUBuffer X, W, b, Z, A_lrelu, A_sigmoid;
    X.allocate(size_X);
    W.allocate(size_W);
    b.allocate(N);
    Z.allocate(size_Z);
    A_lrelu.allocate(size_Z);
    A_sigmoid.allocate(size_Z);

    curandGenerator_t gen = make_gen(1234ULL);
    fill_uniform(gen, X.ptr, X.size);
    fill_uniform(gen, W.ptr, W.size);
    fill_uniform(gen, b.ptr, b.size);

    Timer timer;
    std::cout << "\n[Fused Micro Bench] M=" << M
              << " K=" << K
              << " N=" << N
              << " warmup=" << config.warmup_iters
              << " iters=" << config.iters << "\n";

    auto launch_sep_lrelu = [&]() {
        linear::forward(X.ptr, W.ptr, b.ptr, Z.ptr, M, K, N, VAEStrategy::OPTIMIZED);
        activations::leaky_relu::forward(Z.ptr, A_lrelu.ptr, 0.2f, M * N, VAEStrategy::OPTIMIZED);
    };
    auto launch_fused_lrelu = [&]() {
        fused::forward::linear_lrelu_tc(X.ptr, W.ptr, b.ptr, Z.ptr, A_lrelu.ptr, M, K, N, 0.2f);
    };

    auto launch_sep_sigmoid = [&]() {
        linear::forward(X.ptr, W.ptr, b.ptr, Z.ptr, M, K, N, VAEStrategy::OPTIMIZED);
        activations::sigmoid::forward(Z.ptr, A_sigmoid.ptr, M * N, VAEStrategy::OPTIMIZED);
    };
    auto launch_fused_sigmoid = [&]() {
        fused::forward::linear_sigmoid_tc(X.ptr, W.ptr, b.ptr, Z.ptr, A_sigmoid.ptr, M, K, N);
    };

    float mad_sep = 0.0f;
    float med_sep = timer.compute_ms(launch_sep_lrelu, config, &mad_sep);
    float mad_fused = 0.0f;
    float med_fused = timer.compute_ms(launch_fused_lrelu, config, &mad_fused);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n[Linear+LeakyReLU]\n";
    std::cout << "separate: median_ms=" << med_sep << " mad_ms=" << mad_sep << "\n";
    std::cout << "fused:    median_ms=" << med_fused << " mad_ms=" << mad_fused << "\n";

    mad_sep = 0.0f;
    med_sep = timer.compute_ms(launch_sep_sigmoid, config, &mad_sep);
    mad_fused = 0.0f;
    med_fused = timer.compute_ms(launch_fused_sigmoid, config, &mad_fused);

    std::cout << "\n[Linear+Sigmoid]\n";
    std::cout << "separate: median_ms=" << med_sep << " mad_ms=" << mad_sep << "\n";
    std::cout << "fused:    median_ms=" << med_fused << " mad_ms=" << mad_fused << "\n";

    curandDestroyGenerator(gen);
    return 0;
}
