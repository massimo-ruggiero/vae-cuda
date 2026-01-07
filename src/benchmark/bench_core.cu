#include "bench_core.cuh"

#include "utils.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

Timer::Timer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
}

Timer::~Timer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

float Timer::compute_ms(const std::function<void()>& launch,
                        const BenchmarkConfig& config,
                        float* mad_out) {
    for (int i = 0; i < config.warmup_iters; ++i) {
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times;
    times.reserve(config.iters);

    for (int i = 0; i < config.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());
    float median = times[times.size() / 2];

    if (mad_out) {
        std::vector<float> deviations;
        deviations.reserve(times.size());
        for (float t : times) {
            deviations.push_back(std::fabs(t - median));
        }
        std::sort(deviations.begin(), deviations.end());
        *mad_out = deviations[deviations.size() / 2];
    }

    return median;
}

Csv::Csv(const char* path, bool enabled_) : enabled(enabled_) {
    if (!enabled) return;

    if (!path || path[0] == '\0') {
        enabled = false;
        return;
    }

    f = std::fopen(path, "w");
    if (!f) {
        std::perror("fopen");
        std::exit(1);
    }
}

Csv::~Csv() {
    if (f) {
        std::fclose(f);
        f = nullptr;
    }
}

void Csv::header() {
    if (!enabled) return;

    std::fprintf(f,
        "operation,strategy,M,K,N,median_ms,mad_ms\n"
    );
    std::fflush(f);
}


void Csv::row(const char* op,
              const char* strat,
              int M, int K, int N,
              float ms, float mad_ms)
{
    if (!enabled) return;

    std::fprintf(f,
        "%s,%s,%d,%d,%d,%.6f,%.6f\n",
        op, strat, M, K, N,
        ms, mad_ms
    );
    std::fflush(f);
}
