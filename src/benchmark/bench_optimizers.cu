#include "optimizers.cuh"
#include "bench_ops.cuh"
#include "bench_sizes.hpp"
#include "layer_buffers.cuh"

#include <iostream>

void run_adam_step(Csv& csv, curandGenerator_t gen, 
                   Timer& timer, 
                   const BenchmarkConfig& config,
                   const DeviceSpecs& specs) {
    VAEStrategy strategies[] = {
        VAEStrategy::NAIVE,
        VAEStrategy::VECTORIZED,
    };

    for (VAEStrategy s : strategies) {
        for (auto t : VEC_SIZES) {
            int size = t.size;

            std::cout << "[Optimizers] Adam step benchmark: strategy = " << to_string(s)
                      << " size = " << size << std::endl;

            GPUBuffer g, theta, m, v;
            g.allocate(size);
            theta.allocate(size);
            m.allocate(size);
            v.allocate(size);

            fill_uniform(gen, g.ptr, g.size);
            fill_uniform(gen, theta.ptr, theta.size);
            fill_uniform(gen, m.ptr, m.size);
            fill_uniform(gen, v.ptr, v.size);

            auto launch = [&](){
                optimizers::adam::step(g.ptr, theta.ptr, m.ptr, v.ptr, 10, size, s);
            };

            float std_ms = 0.0f;
            float ms = timer.compute_ms(launch, config, &std_ms);
            csv.row("adam_step", to_string(s), 
                    size, -1, -1, 
                    ms, std_ms,
                    bytes_adam_step(size), flops_adam_step(size),
                    specs);
        }
    }
}
