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
        std::cout << "\n[Benchmark] Linalg (SGEMM)...\n";
        Csv csv("bench_linalg.csv");
        csv.header();

        run_sgemm(csv, gen, timer, config, specs);
    }

    curandDestroyGenerator(gen);

    std::cout << "\nBenchmark completed successfully.\n";
    return 0;
}