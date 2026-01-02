#include <iostream>
#include <string>
#include <curand.h>

#include "bench_core.cuh"
#include "bench_ops.cuh"
#include "layer_buffers.cuh"

int main() {
    BenchmarkConfig config = {
        .warmup_iters = 5,
        .iters = 50
    };

    curandGenerator_t gen = make_gen(1234ULL);
    Timer timer;

    DeviceSpecs specs = DeviceSpecs::detect();

    // ====================
    // LINALG BENCHMARK
    // ====================
    {
        std::cout << "\n[Benchmark] Linalg...\n";
        Csv csv("bench_linalg.csv");
        csv.header();

        run_sgemm(csv, gen, timer, config, specs);
        run_transpose(csv, gen, timer, config, specs);
        run_add_in_place(csv, gen, timer, config, specs);
    }

    // ====================
    // ACTIVATIONS BENCHMARK
    // ====================
    {
        std::cout << "\n[Benchmark] Activations...\n";
        Csv csv("bench_activations.csv");
        csv.header();

        run_leaky_relu_forward(csv, gen, timer, config, specs);
        run_leaky_relu_backward(csv, gen, timer, config, specs);
        run_sigmoid_forward(csv, gen, timer, config, specs);
        run_sigmoid_backward(csv, gen, timer, config, specs);
    }

    // ====================
    // LOSS BENCHMARK
    // ====================
    {
        std::cout << "\n[Benchmark] Loss...\n";
        Csv csv("bench_loss.csv");
        csv.header();

        run_loss_forward(csv, gen, timer, config, specs);
        run_bce_backward(csv, gen, timer, config, specs);
        run_kl_backward(csv, gen, timer, config, specs);
    }

    // ====================
    // REPARAMETRIZATION BENCHMARK
    // ====================
    {
        std::cout << "\n[Benchmark] Reparametrization...\n";
        Csv csv("bench_reparam.csv");
        csv.header();

        run_reparam_forward(csv, gen, timer, config, specs);
        run_reparam_backward(csv, gen, timer, config, specs);
    }

    // ====================
    // OPTIMIZER BENCHMARK
    // ====================
    {
        std::cout << "\n[Benchmark] Optimizers...\n";
        Csv csv("bench_optimizers.csv");
        csv.header();

        run_adam_step(csv, gen, timer, config, specs);
    }

    curandDestroyGenerator(gen);

    std::cout << "\nBenchmark completed successfully.\n";
    return 0;
}
