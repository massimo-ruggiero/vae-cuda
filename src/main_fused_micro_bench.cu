#include <iomanip>
#include <iostream>

#include "bench_core.cuh"
#include "bench_sizes.hpp"
#include "fused.cuh"
#include "linear.cuh"
#include "activations.cuh"
#include "layer_buffers.cuh"

int main(int argc, char** argv) {
    BenchmarkConfig config{};
    config.warmup_iters = 5;
    config.iters = 50;

    curandGenerator_t gen = make_gen(1234ULL);
    Timer timer;
    std::cout << std::fixed << std::setprecision(4);

    for (auto t : SGEMM_SIZES) {
        const int M = t.M;
        const int K = t.K;
        const int N = t.N;
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

        fill_uniform(gen, X.ptr, X.size);
        fill_uniform(gen, W.ptr, W.size);
        fill_uniform(gen, b.ptr, b.size);

        std::cout << "\n[Fused Micro Bench] M=" << M
                  << " K=" << K
                  << " N=" << N
                  << " warmup=" << config.warmup_iters
                  << " iters=" << config.iters << "\n";

        auto launch_sep_lrelu = [&]() {
            linear::forward(X.ptr, W.ptr, b.ptr, Z.ptr, M, K, N, VAEStrategy::NAIVE);
            activations::leaky_relu::forward(Z.ptr, A_lrelu.ptr, 0.2f, M * N, VAEStrategy::NAIVE);
        };
        auto launch_fused_lrelu = [&]() {
            fused::forward::linear_lrelu_tc(X.ptr, W.ptr, b.ptr, A_lrelu.ptr, M, K, N, 0.2f);
        };

        auto launch_sep_sigmoid = [&]() {
            linear::forward(X.ptr, W.ptr, b.ptr, Z.ptr, M, K, N, VAEStrategy::NAIVE);
            activations::sigmoid::forward(Z.ptr, A_sigmoid.ptr, M * N, VAEStrategy::NAIVE);
        };
        auto launch_fused_sigmoid = [&]() {
            fused::forward::linear_sigmoid_tc(X.ptr, W.ptr, b.ptr, A_sigmoid.ptr, M, K, N);
        };

        float mad_sep = 0.0f;
        float med_sep = timer.compute_ms(launch_sep_lrelu, config, &mad_sep);
        float mad_fused = 0.0f;
        float med_fused = timer.compute_ms(launch_fused_lrelu, config, &mad_fused);

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
    }

    curandDestroyGenerator(gen);
    return 0;
}
