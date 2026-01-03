#pragma once

#include "utils.cuh"
#include <cstdio>
#include <functional>
#include <cuda_runtime.h>
#include <curand.h>
#include <cstddef>


constexpr long long EXP = 8LL;
constexpr long long LOG = 8LL;
constexpr long long DIV = 4LL;
constexpr long long SQRT = 4LL;
constexpr long long REC = 3LL;
constexpr long long ADD = 1LL;
constexpr long long MUL = 1LL;


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
             int M, int K, int N,
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
  const long long m = (long long)M;
  const long long k = (long long)K;
  const long long n = (long long)N;
  return m * k * sizeof(float) +
         k * n * sizeof(float) +
         m * n * sizeof(float);
}

inline long long flops_sgemm(int M, int K, int N) {
  // 2*M*N*K (mul + add per element)
  const long long m = (long long)M;
  const long long k = (long long)K;
  const long long n = (long long)N;
  const long long per_product = MUL + ADD;
  return per_product * m * n * k;
}

// --- Transpose ---
inline long long bytes_transpose(int M, int N) {
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
  const long long N = (long long)total;
  return 3LL * N * sizeof(float);
}

inline long long bytes_elementwise_1op(int total) {
  // Read 1 input + Write 1 output
  const long long N = (long long)total;
  return 2LL * N * sizeof(float);
}

// --- LeakyReLU ---
inline long long bytes_leaky_relu(int total) {
  return bytes_elementwise_1op(total);
}

inline long long flops_leaky_relu(int total) {
  // 1 comparison (no flop) + 1 conditional mul ≈ 1 op
  const long long N = (long long)total;
  return MUL * N;
}

// --- Sigmoid ---
inline long long bytes_sigmoid(int total) {
  return bytes_elementwise_1op(total);
}

inline long long flops_sigmoid(int total) {
  // exp(-x) + 1 add + 1 div 
  const long long N = (long long)total;
  const long long per_elem =
      EXP           // exp(-x)
    + ADD           // +1
    + DIV;          // 1.0f / ((1.0f + expf(-x))
  return per_elem * N;
}

// --- BCE Loss (forward) ---
inline long long bytes_bce_forward(int total) {
  // Read X, Z + Write 1 scalar result
  return 2LL * (long long)total * sizeof(float) + sizeof(float);
}

inline long long flops_bce_forward(int total) {
  const long long N = (long long)total;
  // Per element: max(z,0) - z*x + log(1 + exp(-|z|))
  // Approximation: 8 ops per element + log(N) for reduction
  const long long per_elem =
      EXP           // expf
    + LOG           // logf
    + MUL           // z*x
    + 4LL * ADD;    // -fabs (neg), 1+exp, subtract, add
  const long long reduction = (N > 0) ? (N - 1) * ADD : 0;
  return per_elem * N + reduction;
}

// --- KL Divergence (forward) ---
inline long long bytes_kl_forward(int total) {
  // Read mu, logvar + Write 1 scalar result
  return 2LL * (long long)total * sizeof(float) + sizeof(float);
}

inline long long flops_kl_forward(int total) {
  // Per element: 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
  const long long N = (long long)total;
  const long long per_elem =
      EXP           // exp(logvar)
    + 2LL * MUL     // mu^2, *0.5
    + 3LL * ADD;    // +exp, -1, -logvar
  const long long reduction = (N > 0) ? (N - 1) * ADD : 0;
  return per_elem * N + reduction;
}

// --- BCE Loss (backward) ---
inline long long bytes_bce_backward(int total) {
  // Read X, X_hat + Write dA
  return bytes_elementwise_2op(total);
}

inline long long flops_bce_backward(int total) {
  // dA = X_hat - X (1 sub)
  const long long N = (long long)total;
  return ADD * N;
}

// --- KL Divergence (backward) ---
inline long long bytes_kl_backward(int total) {
  // Read mu, logvar, dmu, dlogvar + Write dmu, dlogvar
  return 6LL * (long long)total * sizeof(float);
}

inline long long flops_kl_backward(int total) {
  // dmu += beta * mu (1 mul, 1 add)
  // dlogvar += beta * 0.5 * (exp(logvar) - 1) (exp + mul + sub + add)
  // exp + 3 mul + 3 add per element
  const long long N = (long long)total;
  const long long per_elem =
      EXP            // exp(logvar)
    + 3LL * MUL      // beta*mu, *0.5, *beta
    + 3LL * ADD;     // dmu add, -1, dlogvar add
  return per_elem * N;
}

// --- Reparametrization Trick (forward) ---
inline long long bytes_reparametrization_forward(int total) {
  // Read mu, logvar + Write z, epsilon
  return 4LL * (long long)total * sizeof(float);
}

inline long long flops_reparametrization_forward(int total) {
  // sigma = exp(0.5 * logvar)
  // z = mu + sigma * epsilon (curand_normal)
  // exp + mul + mul + add (random generation cost omitted)
  const long long N = (long long)total;
  const long long per_elem =
      EXP           // exp
    + 2LL * MUL     // 0.5 * logvar, sigma * epsilon
    + ADD;          // mu + ...
  return per_elem * N;
}

// --- Reparametrization Trick (backward) ---
inline long long bytes_reparametrization_backward(int total) {
  // Read dz, logvar, epsilon + Write dmu, dlogvar
  return 5LL * (long long)total * sizeof(float);
}

inline long long flops_reparametrization_backward(int total) {
  // dmu = dz
  // dlogvar = dz * epsilon * 0.5 * sigma
  const long long N = (long long)total;
  const long long per_elem =
      EXP           // expf(0.5 * logvar)
    + 3LL * MUL;    // dz * epsilon * 0.5 * sigma
  return per_elem * N;
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
  // m_hat = m * inv_bc1;
  // v_hat = v * inv_bc2;
  // theta -= lr * m_hat / (sqrt(v_hat) + eps)
  // Matches kernel: 8 mul, 6 add, 1 div, 1 sqrt
  const long long N = (long long)total;
  const long long per_elem =
      8LL * MUL   // m/v updates, bias correction, lr * m_hat
    + 6LL * ADD   // (1 - beta) terms, accumulations, eps, subtraction
    + SQRT        // sqrt(v_hat)
    + DIV;        // final division
  return per_elem * N;
}

// --- Add in-place ---
inline long long bytes_add_inplace(int total) {
  // Read A, B + Write A
  return bytes_elementwise_2op(total);
}

inline long long flops_add_inplace(int total) {
  // A += B (1 add)
  const long long N = (long long)total;
  return ADD * N;
}
