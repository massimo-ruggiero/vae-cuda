#include "linalg.cuh"
#include "bench_ops.cuh"
#include "layer_buffers.cuh"
#include "bench_sizes.hpp"


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
                   M, N, K, 
                   ms, std_ms,
                   bytes_sgemm(M, K, N), flops_sgemm(M, K, N),
                   specs);
        }
    }
}