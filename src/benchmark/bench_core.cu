#include "bench_core.cuh"

#include "utils.cuh"
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

void Csv::header(const DeviceSpecs& specs) {
    std::fprintf(f, "# peak_gflops_fp32=%.6f\n", specs.peak_gflops_fp32);
    std::fprintf(f, "# peak_bandwidth_gbps=%.6f\n", specs.peak_bandwidth_gbps);
    std::fprintf(f, "# ridge_point=%.6f\n", specs.ridge_point());
    
    std::fprintf(f,
        "op,strategy,M,K,N,time_ms,time_ms_std,bytes_lb,flops,"
        "bandwidth_gbps,bandwidth_eff_pct,gflops,gflops_eff_pct,ai\n"
    );
    std::fflush(f);
}


void Csv::row(const char* op,
              const char* strat,
              int M, int K, int N,
              float ms, float std_ms,
              long long bytes, long long flops,
              const DeviceSpecs& specs)
{
    // avoid div-by-zero
    const float seconds = (ms > 0.0f) ? (ms / 1e3f) : 0.0f;

    float bandwidth_gbps = 0.0f;
    float bandwidth_eff  = 0.0f;
    if (seconds > 0.0f && bytes > 0) {
        bandwidth_gbps = (bytes / 1e9f) / seconds;
        if (specs.peak_bandwidth_gbps > 0.0f)
            bandwidth_eff = (bandwidth_gbps / specs.peak_bandwidth_gbps) * 100.0f;
    }

    float gflops = 0.0f;
    float gflops_eff = 0.0f;
    if (seconds > 0.0f && flops > 0) {
        gflops = (flops / 1e9f) / seconds;
        if (specs.peak_gflops_fp32 > 0.0f)
            gflops_eff = (gflops / specs.peak_gflops_fp32) * 100.0f;
    }

    float ai = 0.0f;
    if (bytes > 0 && flops > 0) {
        ai = static_cast<float>(flops) / static_cast<float>(bytes);
    }

    std::fprintf(f,
        "%s,%s,%d,%d,%d,%.6f,%.6f,%lld,%lld,%.6f,%.2f,%.6f,%.2f,%.6f\n",
        op, strat, M, K, N,
        ms, std_ms,
        bytes, flops,
        bandwidth_gbps, bandwidth_eff,
        gflops, gflops_eff,
        ai
    );
    std::fflush(f);
}
