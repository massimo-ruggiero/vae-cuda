#include "linear.cuh"
#include "bench_ops.cuh"
#include "bench_sizes.hpp"
#include "layer_buffers.cuh"

#include <iostream>

void run_add_bias(Csv& csv, curandGenerator_t gen,
                  Timer& timer,
                  const BenchmarkConfig& config) {
    VAEStrategy strategies[] = {
        VAEStrategy::NAIVE,
        VAEStrategy::VECTORIZED,
    };

    for (VAEStrategy s : strategies) {
        for (auto t : ADD_BIAS_SIZES) {
            int batch = t.batch;
            int output_dim = t.output_dim;
            size_t size = static_cast<size_t>(batch) * static_cast<size_t>(output_dim);

            std::cout << "[Linear] Add bias benchmark: strategy = " << to_string(s)
                      << " batch = " << batch
                      << " output_dim = " << output_dim << std::endl;

            GPUBuffer Z, b;
            Z.allocate(size);
            b.allocate(output_dim);

            fill_uniform(gen, Z.ptr, Z.size);
            fill_uniform(gen, b.ptr, b.size);

            auto launch = [&]() {
                linear::add_bias(Z.ptr, b.ptr, batch, output_dim, s);
            };

            float std_ms = 0.0f;
            float ms = timer.compute_ms(launch, config, &std_ms);
            csv.row("add_bias", to_string(s),
                    batch, output_dim, -1,
                    ms, std_ms);
        }
    }
}
