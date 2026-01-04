#pragma once
#include <cstddef>
#include <string>
#include <algorithm>
#include <cctype>


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


inline bool parse_strategy(const std::string& name, 
                           VAEStrategy& out) {
    std::string s = name;

    for (char& c : s) {
        c = static_cast<char>(std::toupper(c));
    }
    if (s == "NAIVE") {
        out = VAEStrategy::NAIVE;
    } else if (s == "TILING") {
        out = VAEStrategy::TILING;
    } else if (s == "PADDING") {
        out = VAEStrategy::PADDING;
    } else if (s == "REDUCTION") {
        out = VAEStrategy::REDUCTION;
    } else if (s == "UNROLLED_REDUCTION") {
        out = VAEStrategy::UNROLLED_REDUCTION;
    } else if (s == "WARP_REDUCTION") {
        out = VAEStrategy::WARP_REDUCTION;
    } else if (s == "VECTORIZED") {
        out = VAEStrategy::VECTORIZED;
    } else if (s == "OPTIMIZED") {
        out = VAEStrategy::OPTIMIZED;
    } else if (s == "KERNEL_FUSION") {
        out = VAEStrategy::KERNEL_FUSION;
    } else {
        return false;
    }
    return true;
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
