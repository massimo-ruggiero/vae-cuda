#include "linalg.cuh"
#include "bench_ops.cuh"
#include "layer_buffers.cuh"
#include "bench_sizes.hpp"

#include <iostream>

void run_sgemm(Csv& csv, curandGenerator_t gen, 
               Timer& timer, 
               const BenchmarkConfig& config,
               const DeviceSpecs& specs) {
    VAEStrategy strategies[] = {
        VAEStrategy::NAIVE,
        VAEStrategy::TILING,
        VAEStrategy::PADDING,
        VAEStrategy::VECTORIZED
    };

    for (VAEStrategy s : strategies) {
        for (auto t : SGEMM_SIZES) {
            int M = t.M;
            int K = t.K;
            int N = t.N;

            std::cout << "[Linalg] SGEMM benchmark: strategy = " << to_string(s)
                      << " M = " << M
                      << " K = " << K
                      << " N = " << N << std::endl;

            size_t size_A = (size_t)M * (size_t)K;
            size_t size_B = (size_t)K * (size_t)N;
            size_t size_C = (size_t)M * (size_t)N;

            GPUBuffer A, B, C;
            A.allocate(size_A);
            B.allocate(size_B);
            C.allocate(size_C);

            fill_uniform(gen, A.ptr, A.size);
            fill_uniform(gen, B.ptr, B.size);

            auto launch = [&](){
                linalg::sgemm(A.ptr, B.ptr, C.ptr, M, K, N, s);
            };
            
            float std_ms = 0.0f;
            float ms = timer.compute_ms(launch, config, &std_ms);
            csv.row("sgemm", to_string(s), 
                    M, K, N, 
                    ms, std_ms,
                    bytes_sgemm(M, K, N), flops_sgemm(M, K, N),
                    specs);
        }
    }
}

void run_transpose(Csv& csv, curandGenerator_t gen, 
                   Timer& timer, 
                   const BenchmarkConfig& config,
                   const DeviceSpecs& specs) {
    VAEStrategy strategies[] = {
        VAEStrategy::NAIVE,
        VAEStrategy::TILING,
        VAEStrategy::PADDING,
    };

    for (VAEStrategy s : strategies) {
        for (auto t : TRANSPOSE_SIZES) {
            int M = t.M;
            int N = t.N;

            std::cout << "[Linalg] Transpose benchmark: strategy = " << to_string(s)
                      << " M = " << M
                      << " N = " << N << std::endl;

            size_t size = (size_t)M * (size_t)N;
            GPUBuffer A, AT;
            A.allocate(size);
            AT.allocate(size);

            fill_uniform(gen, A.ptr, A.size);

            auto launch = [&](){
                linalg::transpose(A.ptr, AT.ptr, M, N, s);
            };

            float std_ms = 0.0f;
            float ms = timer.compute_ms(launch, config, &std_ms);
            csv.row("transpose", to_string(s), 
                    M, N, 1, 
                    ms, std_ms,
                    bytes_transpose(M, N), flops_transpose(M, N),
                    specs);
        }
    }
}

void run_add_in_place(Csv& csv, curandGenerator_t gen, 
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

            std::cout << "[Linalg] Add in place benchmark: strategy = " << to_string(s)
                      << " size = " << size << std::endl;

            GPUBuffer A, B;
            A.allocate(size);
            B.allocate(size);

            fill_uniform(gen, A.ptr, A.size);
            fill_uniform(gen, B.ptr, B.size);

            auto launch = [&](){
                linalg::add_in_place(A.ptr, B.ptr, size, s);
            };

            float std_ms = 0.0f;
            float ms = timer.compute_ms(launch, config, &std_ms);
            csv.row("add_in_place", to_string(s), 
                    size, 1, 1, 
                    ms, std_ms,
                    bytes_add_inplace(size), flops_add_inplace(size),
                    specs);
        }
    }
}
