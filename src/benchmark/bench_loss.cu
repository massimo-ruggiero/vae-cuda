#include "loss.cuh"
#include "bench_ops.cuh"
#include "bench_sizes.hpp"
#include "layer_buffers.cuh"

#include <iostream>


void run_loss_forward(Csv& csv, curandGenerator_t gen, 
                      Timer& timer, 
                      const BenchmarkConfig& config) {
    VAEStrategy strategies[] = {
        VAEStrategy::NAIVE,
        VAEStrategy::REDUCTION,
        VAEStrategy::UNROLLED_REDUCTION,
        VAEStrategy::WARP_REDUCTION,
        VAEStrategy::VECTORIZED,
    };

    for (VAEStrategy s : strategies) {
        for (auto t : LOSS_SIZES) {
            int batch = t.batch;
            int input_dim = t.input_dim;
            int latent_dim = t.latent_dim;

            int size_bce = batch * input_dim;
            int size_kl = batch * latent_dim;

            std::cout << "[Loss] Starting forward benchmark: strategy = " << to_string(s)
                      << " batch = " << batch
                      << " input_dim = " << input_dim
                      << " latent_dim = " << latent_dim << std::endl;

            GPUBuffer X, Z, mu, logvar, bce, kl;
            X.allocate(size_bce);
            Z.allocate(size_bce);
            mu.allocate(size_kl);
            logvar.allocate(size_kl);
            bce.allocate(1);
            kl.allocate(1);

            fill_uniform(gen, X.ptr, X.size);
            fill_uniform(gen, Z.ptr, Z.size);
            fill_uniform(gen, mu.ptr, mu.size);
            fill_uniform(gen, logvar.ptr, logvar.size);

            float h_loss = 0.0f;
            auto launch = [&](){
                loss::forward(X.ptr, Z.ptr, mu.ptr, logvar.ptr,
                              bce.ptr, kl.ptr, &h_loss,
                              batch, input_dim, latent_dim,
                              s, 1.0f);
            };

            float std_ms = 0.0f;
            float ms = timer.compute_ms(launch, config, &std_ms);
            csv.row("loss_forward", to_string(s), 
                    batch, input_dim, latent_dim, 
                    ms, std_ms);
        }
    }
}

void run_bce_backward(Csv& csv, curandGenerator_t gen, 
                      Timer& timer, 
                      const BenchmarkConfig& config) {
    VAEStrategy strategies[] = {
        VAEStrategy::NAIVE,
        VAEStrategy::VECTORIZED,
    };

    for (VAEStrategy s : strategies) {
        for (auto t : VEC_SIZES) {
            int size = t.size;

            std::cout << "[Loss] BCE backward benchmark: strategy = " << to_string(s)
                      << " size = " << size << std::endl;

            GPUBuffer X, X_hat, dA;
            X.allocate(size);
            X_hat.allocate(size);
            dA.allocate(size);

            fill_uniform(gen, X.ptr, X.size);
            fill_uniform(gen, X_hat.ptr, X_hat.size);

            auto launch = [&](){
                loss::backward::bce(X.ptr, X_hat.ptr, dA.ptr, size, s);
            };

            float std_ms = 0.0f;
            float ms = timer.compute_ms(launch, config, &std_ms);
            csv.row("bce_backward", to_string(s), 
                    size, -1, -1, 
                    ms, std_ms);
        }
    }
}

void run_kl_backward(Csv& csv, curandGenerator_t gen, 
                     Timer& timer, 
                     const BenchmarkConfig& config) {
    VAEStrategy strategies[] = {
        VAEStrategy::NAIVE,
        VAEStrategy::VECTORIZED,
    };

    for (VAEStrategy s : strategies) {
        for (auto t : VEC_SIZES) {
            int size = t.size;

            std::cout << "[Loss] KL backward benchmark: strategy = " << to_string(s)
                      << " size = " << size << std::endl;

            GPUBuffer mu, logvar, dmu, dlogvar;
            mu.allocate(size);
            logvar.allocate(size);
            dmu.allocate(size);
            dlogvar.allocate(size);

            fill_uniform(gen, mu.ptr, mu.size);
            fill_uniform(gen, logvar.ptr, logvar.size);

            auto launch = [&](){
                loss::backward::kl(mu.ptr, logvar.ptr, dmu.ptr, dlogvar.ptr, size, s, 1.0f);
            };

            float std_ms = 0.0f;
            float ms = timer.compute_ms(launch, config, &std_ms);
            csv.row("kl_backward", to_string(s), 
                    size, -1, -1, 
                    ms, std_ms);
        }
    }
}
