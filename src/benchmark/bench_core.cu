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
                        float* std_out) {
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

    if (std_out) {
        float mean = 0.0f;
        for (float t : times) mean += t;
        mean /= times.size();

        float variance = 0.0f;
        for (float t : times) {
            float diff = t - mean;
            variance += diff * diff;
        }
        variance /= times.size();
        *std_out = std::sqrt(variance);
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
        "operation,strategy,M,K,N,median_ms,std_ms\n"
    );
    std::fflush(f);
}


void Csv::row(const char* op,
              const char* strat,
              int M, int K, int N,
              float ms, float std_ms)
{
    if (!enabled) return;

    std::fprintf(f,
        "%s,%s,%d,%d,%d,%.6f,%.6f\n",
        op, strat, M, K, N,
        ms, std_ms
    );
    std::fflush(f);
}
