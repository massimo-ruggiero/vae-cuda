#include "linalg.cuh"
#include "bench_ops.cuh"
#include "layer_buffers.cuh"


void run_sgemm(Csv& csv, curandGenerator_t gen, 
               Timer& timer, 
               const BenchmarkConfig& config) {
    VAEStrategy strategies[] = {VAEStrategy::NAIVE,
                                VAEStrategy::TILING,
                                VAEStrategy::PADDING,
                                VAEStrategy::VECTORIZED};

    for (VAEStrategy s : strategies) {
        // TODO
        for (...) {
            GPUBuffer A, B, C;
            A.allocate(size_A);
            A.allocate(size_B);
            A.allocate(size_C);

            fill_uniform(gen, A.ptr, A.size);
            fill_uniform(gen, B.ptr, B.size);
            fill_uniform(gen, C.ptr, C.size);

            auto launch = [&](){
                linalg::matmul(A.ptr, B.ptr, C.ptr, M, N, K, s);
            };
            float ms = timer.compute_ms(launch, config);
            cs.row("sgemm", to_string(s), 
                   M, N, K, 
                   ms, bytes_sgemm(M, N, K), flops_sgemm());
        }
    }
}