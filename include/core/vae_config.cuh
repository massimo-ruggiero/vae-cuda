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
    KERNEL_FUSION
};

inline const char* to_string(VAEStrategy s) {
    switch (s) {
        case VAEStrategy::NAIVE:              return "Naive";
        case VAEStrategy::TILING:             return "Tiling";
        case VAEStrategy::PADDING:            return "Padding";
        case VAEStrategy::REDUCTION:          return "Reduction";
        case VAEStrategy::UNROLLED_REDUCTION: return "Unrolled Reduction";
        case VAEStrategy::WARP_REDUCTION:     return "Warp Reduction";
        case VAEStrategy::VECTORIZED:         return "Vectorized";
        case VAEStrategy::KERNEL_FUSION:      return "Kernel Fusion";
        default:                              return "Unknown";
    }
}

struct VAEConfig {
    size_t batch_size = 100;
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