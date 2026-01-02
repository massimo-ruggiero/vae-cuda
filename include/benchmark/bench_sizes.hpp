#pragma once

#include <vector>
#include <string>

struct SgemmSizes {int M, K, N; }
struct TransposeSizes {int M, N; }
struct VecSizes {int size; }

static constexpr SgemmSizes SGEMM_SIZES[] = {
    {128, 256, 64},
    {128, 784, 512},
    {512, 1024, 512},
};
static constexpr TransposeSizes TRANSPOSE_SIZES[] = {
    {256, 128},
    {784, 512},
    {4096, 2048},
};

static constexpr VecSizes VEC_SIZES[] = {
    {1 << 16},        // ~65k
    {128 * 784},      // ~100k
    {1 << 24},        // ~16M
};