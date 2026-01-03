#pragma once
#include <cstddef>


enum class VAEStrategy {
    NAIVE,
    TILING,
    PADDING,
    REDUCTION,
    UNROLLED_REDUCTION,
    WARP_REDUCTION,
    VECTORIZED,
    OPTIMIZED,
    KERNEL_FUSION
};

inline const char* to_string(VAEStrategy s) {
    switch (s) {
        case VAEStrategy::NAIVE:              return "NAIVE";
        case VAEStrategy::TILING:             return "TILING";
        case VAEStrategy::PADDING:            return "PADDING";
        case VAEStrategy::REDUCTION:          return "REDUCTION";
        case VAEStrategy::UNROLLED_REDUCTION: return "UNROLLED_REDUCTION";
        case VAEStrategy::WARP_REDUCTION:     return "WARP_REDUCTION";
        case VAEStrategy::VECTORIZED:         return "VECTORIZED";
        case VAEStrategy::OPTIMIZED:          return "OPTIMIZED";
        case VAEStrategy::KERNEL_FUSION:      return "KERNEL_FUSION";
        default:                              return "Unknown";
    }
}

struct VAEConfig {
    size_t batch_size = 128;
    size_t input_dim = 784;
    size_t hidden_dim = 400;
    size_t latent_dim = 200;
    float beta = 1.0f;
    VAEStrategy strategy = VAEStrategy::NAIVE;

    size_t num_states() const {
        const size_t n = batch_size * latent_dim;
        if (strategy == VAEStrategy::VECTORIZED) {
            return (n + 3) / 4;  // one Philox4 state per 4 floats
        }
        return n;
    }
};