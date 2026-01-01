#include "bench_core.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

float DeviceSpecs::ridge_point() const {
    return peak_gflops_fp32 / peak_bandwidth_gbps;
}

DeviceSpecs DeviceSpecs::detect() {
    DeviceSpecs specs;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    specs.peak_bandwidth_gbps = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6f;

    std::cout << "Detected: " << prop.name << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Peak BW: " << specs.peak_bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "Peak GFLOPS: " << specs.peak_gflops_fp32 << std::endl;
    return specs;
}

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

        CUDA_CHECK(cudaDeviceSynchronize());

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

Csv::Csv(const char* path) {
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
    std::fprintf(f, "op,strategy,M,K,N,total,time_ms,bytes_lb,flops\n");
    std::fflush(f);
}

void Csv::row(const char* op,
              const char* strat,
              int M, int K, int N, int total,
              float ms, float std_ms,
              long long bytes, long long flops,
              const DeviceSpecs& specs) {
    float bandwidth_gbps = (bytes / 1e9f) / (ms / 1e3f);
    float bandwidth_eff = (bandwidth_gbps / specs.peak_bandwidth_gbps) * 100.0f;

    float gflops = (flops / 1e9f) / (ms / 1e3f);
    float gflops_eff = (gflops / specs.peak_gflops_fp32) * 100.0f;

    float ai = (flops > 0 && bytes > 0) ? static_cast<float>(flops) / static_cast<float>(bytes) : 0.0f;

    std::fprintf(f, 
                 "%s,%s,%d,%d,%d,%d,%.6f,%lld,%lld\n",
                 op,strat,M,K,N,total,ms,bytes,flops);
    std::fflush(f);
}

