#pragma once

#include "utils.cuh"
#include <cstdio>
#include <functional>
#include <cuda_runtime.h>


struct DeviceSpecs {
    float peak_bandwidth_gbps = 320.0f;
    float peak_gflops_fp32 = 8100.0f;

    float ridge_point() const;
    static DeviceSpecs detect();
};

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
                     float* std_out = nullptr);

private:
    cudaEvent_t start{};
    cudaEvent_t stop{};
};

class Csv {
public:
    explicit Csv(const char* path);
    ~Csv();

    void header();
    void row(const char* op,
             const char* strat,
             int M, int K, int N, int total,
             float ms, float std_ms,
             long long bytes, long long flops,
             const DeviceSpecs& specs);

private:
    FILE* f = nullptr;
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


// --- Matrix Multiplication ---
inline long long bytes_sgemm(int M, int K, int N) {
  // Read A (M×K), B (K×N), Write C (M×N)
  return (long long)M * K * sizeof(float) +
         (long long)K * N * sizeof(float) +
         (long long)M * N * sizeof(float);
}

inline long long flops_matmul(int M, int K, int N) {
  // 2*M*N*K (mul + add per element)
  return 2LL * (long long)M * (long long)N * (long long)K;
}

// --- Transpose ---
inline long long bytes_transpose_lb(int M, int N) {
  // Read input + Write output
  return 2LL * (long long)M * (long long)N * sizeof(float);
}

inline long long flops_transpose(int M, int N) {
  // No FLOPs, pure memory operation
  return 0LL;
}

// --- Element-wise operations (2 operands) ---
inline long long bytes_elementwise_2op(int total) {
  // Read 2 inputs + Write 1 output
  return 3LL * (long long)total * sizeof(float);
}

inline long long bytes_elementwise_1op(int total) {
  // Read 1 input + Write 1 output
  return 2LL * (long long)total * sizeof(float);
}

// --- LeakyReLU ---
inline long long bytes_leaky_relu(int total) {
  return bytes_elementwise_1op(total);
}

inline long long flops_leaky_relu(int total) {
  // 1 comparison + 1 conditional mul ≈ 2 ops
  return 2LL * (long long)total;
}

// --- Sigmoid ---
inline long long bytes_sigmoid(int total) {
  return bytes_elementwise_1op(total);
}

inline long long flops_sigmoid(int total) {
  // exp(-x) + 1 add + 1 div ≈ 4 ops (exp simplified)
  return 4LL * (long long)total;
}

// --- BCE Loss (forward) ---
inline long long bytes_bce_forward(int total) {
  // Read X, Z + Write 1 scalar result
  return 2LL * (long long)total * sizeof(float) + sizeof(float);
}

inline long long flops_bce_forward(int total) {
  // Per element: max(z,0) - z*x + log(1 + exp(-|z|))
  // Approximation: 8 ops per element + log(N) for reduction
  return 8LL * (long long)total + (long long)std::ceil(std::log2((double)total));
}

// --- KL Divergence (forward) ---
inline long long bytes_kl_forward(int total) {
  // Read mu, logvar + Write 1 scalar result
  return 2LL * (long long)total * sizeof(float) + sizeof(float);
}

inline long long flops_kl_forward(int total) {
  // Per element: 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
  // Approximation: 6 ops per element + log(N) for reduction
  return 6LL * (long long)total + (long long)std::ceil(std::log2((double)total));
}

// --- BCE Loss (backward) ---
inline long long bytes_bce_backward(int total) {
  // Read X, X_hat + Write dA
  return bytes_elementwise_2op(total);
}

inline long long flops_bce_backward(int total) {
  // dA = X_hat - X (1 sub)
  return (long long)total;
}

// --- KL Divergence (backward) ---
inline long long bytes_kl_backward(int total) {
  // Read mu, logvar, dmu, dlogvar + Write dmu, dlogvar
  return 6LL * (long long)total * sizeof(float);
}

inline long long flops_kl_backward(int total) {
  // dmu += beta * mu (1 mul, 1 add)
  // dlogvar += beta * 0.5 * (exp(logvar) - 1) (exp + mul + sub + add)
  // ≈ 6 ops per element
  return 6LL * (long long)total;
}

// --- Reparametrization Trick (forward) ---
inline long long bytes_reparametrization_forward(int total) {
  // Read mu, logvar + Write z, epsilon
  return 4LL * (long long)total * sizeof(float);
}

inline long long flops_reparametrization_forward(int total) {
  // sigma = exp(0.5 * logvar)
  // z = mu + sigma * epsilon (curand_normal)
  // exp + mul + mul + add + random ≈ 15 ops (random generation is complex)
  return 15LL * (long long)total;
}

// --- Reparametrization Trick (backward) ---
inline long long bytes_reparametrization_backward(int total) {
  // Read dz, logvar, epsilon + Write dmu, dlogvar
  return 5LL * (long long)total * sizeof(float);
}

inline long long flops_reparametrization_backward(int total) {
  // dmu = dz
  // dlogvar = dz * epsilon * 0.5 * sigma
  // ≈ 4 ops per element
  return 4LL * (long long)total;
}

// --- Adam Optimizer ---
inline long long bytes_adam_step(int total) {
  // Read: g, theta, m, v (4 reads)
  // Write: theta, m, v (3 writes)
  return 7LL * (long long)total * sizeof(float);
}

inline long long flops_adam_step(int total) {
  // m = beta1*m + (1-beta1)*g
  // v = beta2*v + (1-beta2)*g^2
  // m_hat = m / (1 - beta1^t)
  // v_hat = v / (1 - beta2^t)
  // theta -= lr * m_hat / (sqrt(v_hat) + eps)
  // ≈ 10-12 ops per element
  return 10LL * (long long)total;
}

// --- Add in-place ---
inline long long bytes_add_inplace(int total) {
  // Read A, B + Write A
  return bytes_elementwise_2op(total);
}

inline long long flops_add_inplace(int total) {
  // A += B (1 add)
  return (long long)total;
}