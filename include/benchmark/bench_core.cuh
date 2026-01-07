#pragma once

#include "utils.cuh"
#include <cstdio>
#include <functional>
#include <cuda_runtime.h>
#include <curand.h>
#include <cstddef>


struct BenchmarkConfig {
    int warmup_iters = 10;
    int iters = 50;
};

class Timer {
public:
    Timer();
    ~Timer();

    float compute_ms(const std::function<void()>& launch,
                     const BenchmarkConfig& config,
                     float* mad_out = nullptr);

private:
    cudaEvent_t start{};
    cudaEvent_t stop{};
};

class Csv {
public:
    explicit Csv(const char* path, bool enabled = true);
    ~Csv();

    void header();
    void row(const char* op,
             const char* strat,
             int M, int K, int N,
             float ms, float mad_ms);

private:
    FILE* f = nullptr;
    bool enabled = false;
};

inline curandGenerator_t make_gen(unsigned long long seed){
  curandGenerator_t g{};
  curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(g, seed);
  return g;
}

inline void fill_uniform(curandGenerator_t g, float* d, size_t n){
  curandGenerateUniform(g, d, n);
}
