#include "reparametrization.cuh"
#include "bench_ops.cuh"
#include "bench_sizes.hpp"
#include "layer_buffers.cuh"

#include <iostream>

void run_reparam_forward(Csv& csv, curandGenerator_t gen, 
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
            int num_states = (s == VAEStrategy::VECTORIZED) ? ((size + 3) / 4) : size;

            std::cout << "[Reparam] Forward benchmark: strategy = " << to_string(s)
                      << " size = " << size << std::endl;

            GPUBuffer mu, logvar, z, epsilon;
            mu.allocate(size);
            logvar.allocate(size);
            z.allocate(size);
            epsilon.allocate(size);

            curandStatePhilox4_32_10_t* d_states = nullptr;
            CUDA_CHECK(cudaMalloc(&d_states, num_states * sizeof(curandStatePhilox4_32_10_t)));

            fill_uniform(gen, mu.ptr, mu.size);
            fill_uniform(gen, logvar.ptr, logvar.size);

            reparametrization::init(d_states, num_states, 1234ULL, s);

            auto launch = [&](){
                reparametrization::forward(mu.ptr, logvar.ptr, z.ptr, epsilon.ptr, d_states, size, s);
            };

            float std_ms = 0.0f;
            float ms = timer.compute_ms(launch, config, &std_ms);
            csv.row("reparam_forward", to_string(s), 
                    size, -1, -1, 
                    ms, std_ms,
                    bytes_reparametrization_forward(size), flops_reparametrization_forward(size),
                    specs);

            CUDA_CHECK(cudaFree(d_states));
        }
    }
}

void run_reparam_backward(Csv& csv, curandGenerator_t gen, 
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
            int num_states = (s == VAEStrategy::VECTORIZED) ? ((size + 3) / 4) : size;

            std::cout << "[Reparam] Backward benchmark: strategy = " << to_string(s)
                      << " size = " << size << std::endl;

            GPUBuffer mu, logvar, z, epsilon, dz, dmu, dlogvar;
            mu.allocate(size);
            logvar.allocate(size);
            z.allocate(size);
            epsilon.allocate(size);
            dz.allocate(size);
            dmu.allocate(size);
            dlogvar.allocate(size);

            curandStatePhilox4_32_10_t* d_states = nullptr;
            CUDA_CHECK(cudaMalloc(&d_states, num_states * sizeof(curandStatePhilox4_32_10_t)));

            fill_uniform(gen, mu.ptr, mu.size);
            fill_uniform(gen, logvar.ptr, logvar.size);
            fill_uniform(gen, dz.ptr, dz.size);

            reparametrization::init(d_states, num_states, 1234ULL, s);
            reparametrization::forward(mu.ptr, logvar.ptr, z.ptr, epsilon.ptr, d_states, size, s);

            auto launch = [&](){
                reparametrization::backward(dz.ptr, logvar.ptr, epsilon.ptr, dmu.ptr, dlogvar.ptr, size, s);
            };

            float std_ms = 0.0f;
            float ms = timer.compute_ms(launch, config, &std_ms);
            csv.row("reparam_backward", to_string(s), 
                    size, -1, -1, 
                    ms, std_ms,
                    bytes_reparametrization_backward(size), flops_reparametrization_backward(size),
                    specs);

            CUDA_CHECK(cudaFree(d_states));
        }
    }
}
