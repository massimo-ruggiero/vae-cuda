#include "activations.cuh"
#include "bench_ops.cuh"
#include "bench_sizes.hpp"
#include "layer_buffers.cuh"

#include <iostream>

void run_leaky_relu_forward(Csv& csv, curandGenerator_t gen, 
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

            std::cout << "[Activations] LeakyReLU forward: strategy = " << to_string(s)
                      << " size = " << size << std::endl;

            GPUBuffer Z, A;
            Z.allocate(size);
            A.allocate(size);

            fill_uniform(gen, Z.ptr, Z.size);

            auto launch = [&](){
                activations::leaky_relu::forward(Z.ptr, A.ptr, 0.2f, size, s);
            };

            float std_ms = 0.0f;
            float ms = timer.compute_ms(launch, config, &std_ms);
            csv.row("leaky_relu_fwd", to_string(s), 
                    size, -1, -1, 
                    ms, std_ms,
                    bytes_leaky_relu(size), flops_leaky_relu(size),
                    specs);
        }
    }
}

void run_leaky_relu_backward(Csv& csv, curandGenerator_t gen, 
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

            std::cout << "[Activations] LeakyReLU backward: strategy = " << to_string(s)
                      << " size = " << size << std::endl;

            GPUBuffer Z, dA, dZ;
            Z.allocate(size);
            dA.allocate(size);
            dZ.allocate(size);

            fill_uniform(gen, Z.ptr, Z.size);
            fill_uniform(gen, dA.ptr, dA.size);

            auto launch = [&](){
                activations::leaky_relu::backward(Z.ptr, dA.ptr, dZ.ptr, 0.2f, size, s);
            };

            float std_ms = 0.0f;
            float ms = timer.compute_ms(launch, config, &std_ms);
            csv.row("leaky_relu_bwd", to_string(s), 
                    size, -1, -1, 
                    ms, std_ms,
                    bytes_leaky_relu(size), flops_leaky_relu(size),
                    specs);
        }
    }
}

void run_sigmoid_forward(Csv& csv, curandGenerator_t gen, 
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

            std::cout << "[Activations] Sigmoid forward: strategy = " << to_string(s)
                      << " size = " << size << std::endl;

            GPUBuffer Z, A;
            Z.allocate(size);
            A.allocate(size);

            fill_uniform(gen, Z.ptr, Z.size);

            auto launch = [&](){
                activations::sigmoid::forward(Z.ptr, A.ptr, size, s);
            };

            float std_ms = 0.0f;
            float ms = timer.compute_ms(launch, config, &std_ms);
            csv.row("sigmoid_fwd", to_string(s), 
                    size, -1, -1, 
                    ms, std_ms,
                    bytes_sigmoid(size), flops_sigmoid(size),
                    specs);
        }
    }
}

void run_sigmoid_backward(Csv& csv, curandGenerator_t gen, 
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

            std::cout << "[Activations] Sigmoid backward: strategy = " << to_string(s)
                      << " size = " << size << std::endl;

            GPUBuffer A, dA, dZ;
            A.allocate(size);
            dA.allocate(size);
            dZ.allocate(size);

            fill_uniform(gen, A.ptr, A.size);
            fill_uniform(gen, dA.ptr, dA.size);

            auto launch = [&](){
                activations::sigmoid::backward(A.ptr, dA.ptr, dZ.ptr, size, s);
            };

            float std_ms = 0.0f;
            float ms = timer.compute_ms(launch, config, &std_ms);
            csv.row("sigmoid_bwd", to_string(s), 
                    size, -1, -1, 
                    ms, std_ms,
                    bytes_sigmoid(size), flops_sigmoid(size),
                    specs);
        }
    }
}
