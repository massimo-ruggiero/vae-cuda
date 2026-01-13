#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "fused.cuh"
#include "utils.cuh"

struct TestCase {
    int M;
    int K;
    int N;
};

static void fill_random(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : v) {
        x = dist(rng);
    }
}

static void linear_cpu(const std::vector<float>& X,
                       const std::vector<float>& W,
                       const std::vector<float>& b,
                       std::vector<float>& Z,
                       int M, int K, int N) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = b[n];
            const int x_row = m * K;
            const int w_col = n;
            for (int k = 0; k < K; ++k) {
                sum += X[x_row + k] * W[k * N + w_col];
            }
            Z[m * N + n] = sum;
        }
    }
}

static void leaky_relu_cpu(const std::vector<float>& Z,
                           std::vector<float>& A,
                           float alpha) {
    const size_t size = Z.size();
    for (size_t i = 0; i < size; ++i) {
        const float z = Z[i];
        A[i] = (z > 0.0f) ? z : z * alpha;
    }
}

static void sigmoid_cpu(const std::vector<float>& Z,
                        std::vector<float>& A) {
    const size_t size = Z.size();
    for (size_t i = 0; i < size; ++i) {
        const float z = Z[i];
        A[i] = 1.0f / (1.0f + std::exp(-z));
    }
}

static bool compare_vectors(const std::string& label,
                            const std::vector<float>& ref,
                            const std::vector<float>& got,
                            float max_abs_tol,
                            float mean_abs_tol) {
    float max_abs = 0.0f;
    double sum_abs = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float diff = std::fabs(ref[i] - got[i]);
        if (diff > max_abs) {
            max_abs = diff;
        }
        sum_abs += diff;
    }
    const float mean_abs = static_cast<float>(sum_abs / ref.size());
    const bool ok = (max_abs <= max_abs_tol) && (mean_abs <= mean_abs_tol);
    std::cout << "[" << label << "] max_abs=" << max_abs
              << " mean_abs=" << mean_abs
              << " tol_max=" << max_abs_tol
              << " tol_mean=" << mean_abs_tol
              << (ok ? " OK" : " FAIL") << "\n";
    return ok;
}

static void parse_arg_value(int& i, int argc, char** argv, float& out) {
    if (i + 1 >= argc) {
        std::cerr << "Missing value for " << argv[i] << "\n";
        std::exit(1);
    }
    out = std::stof(argv[++i]);
}

static void parse_arg_value(int& i, int argc, char** argv, unsigned int& out) {
    if (i + 1 >= argc) {
        std::cerr << "Missing value for " << argv[i] << "\n";
        std::exit(1);
    }
    out = static_cast<unsigned int>(std::stoul(argv[++i]));
}

int main(int argc, char** argv) {
    float max_abs_tol = 2.0e-1f;
    float mean_abs_tol = 2.0e-2f;
    unsigned int seed = 1234;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--max-abs") {
            parse_arg_value(i, argc, argv, max_abs_tol);
        } else if (arg == "--mean-abs") {
            parse_arg_value(i, argc, argv, mean_abs_tol);
        } else if (arg == "--seed") {
            parse_arg_value(i, argc, argv, seed);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    const std::vector<TestCase> cases = {
        {17, 19, 23},
        {33, 35, 37},
        {64, 64, 64}
    };

    std::mt19937 rng(seed);
    bool all_ok = true;

    for (const auto& t : cases) {
        const int M = t.M;
        const int K = t.K;
        const int N = t.N;
        const size_t size_X = static_cast<size_t>(M) * static_cast<size_t>(K);
        const size_t size_W = static_cast<size_t>(K) * static_cast<size_t>(N);
        const size_t size_Z = static_cast<size_t>(M) * static_cast<size_t>(N);

        std::vector<float> X(size_X);
        std::vector<float> W(size_W);
        std::vector<float> b(static_cast<size_t>(N));
        std::vector<float> ref_Z(size_Z);
        std::vector<float> ref_lrelu(size_Z);
        std::vector<float> ref_sigmoid(size_Z);
        std::vector<float> out_Z(size_Z);
        std::vector<float> out_lrelu(size_Z);
        std::vector<float> out_sigmoid(size_Z);

        fill_random(X, rng);
        fill_random(W, rng);
        fill_random(b, rng);

        linear_cpu(X, W, b, ref_Z, M, K, N);
        leaky_relu_cpu(ref_Z, ref_lrelu, 0.2f);
        sigmoid_cpu(ref_Z, ref_sigmoid);

        float* d_X = nullptr;
        float* d_W = nullptr;
        float* d_b = nullptr;
        float* d_Z = nullptr;
        float* d_A = nullptr;
        CUDA_CHECK(cudaMalloc(&d_X, size_X * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_W, size_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b, static_cast<size_t>(N) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_Z, size_Z * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_A, size_Z * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_X, X.data(), size_X * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W, W.data(), size_W * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b.data(), static_cast<size_t>(N) * sizeof(float), cudaMemcpyHostToDevice));

        std::cout << "\n[TestCase] M=" << M << " K=" << K << " N=" << N << "\n";

        fused::forward::linear_tc(d_X, d_W, d_b, d_Z, M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out_Z.data(), d_Z, size_Z * sizeof(float), cudaMemcpyDeviceToHost));
        all_ok &= compare_vectors("linear_tc", ref_Z, out_Z, max_abs_tol, mean_abs_tol);

        fused::forward::linear_lrelu_tc(d_X, d_W, d_b, d_A, M, K, N, 0.2f);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out_lrelu.data(), d_A, size_Z * sizeof(float), cudaMemcpyDeviceToHost));
        all_ok &= compare_vectors("linear_lrelu_tc", ref_lrelu, out_lrelu, max_abs_tol, mean_abs_tol);

        fused::forward::linear_sigmoid_tc(d_X, d_W, d_b, d_A, M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out_sigmoid.data(), d_A, size_Z * sizeof(float), cudaMemcpyDeviceToHost));
        all_ok &= compare_vectors("linear_sigmoid_tc", ref_sigmoid, out_sigmoid, max_abs_tol, mean_abs_tol);

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_Z));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_W));
        CUDA_CHECK(cudaFree(d_X));
    }

    if (!all_ok) {
        std::cerr << "\nFused correctness check FAILED.\n";
        return 1;
    }

    std::cout << "\nFused correctness check PASSED.\n";
    return 0;
}
