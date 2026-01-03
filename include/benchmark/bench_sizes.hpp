#pragma once

#include <vector>
#include <string>

struct SgemmSizes { int M, K, N; };
struct TransposeSizes { int M, N; };
struct VecSizes { int size; };
struct LossSizes { int batch, input_dim, latent_dim; };

static constexpr SgemmSizes SGEMM_SIZES[] = {
    {128, 256, 64},        
    {128, 784, 400},       
    {4096, 4096, 4096},    
};

static constexpr TransposeSizes TRANSPOSE_SIZES[] = {
    {256, 128},       
    {784, 400},       
    {8192, 8192},     
};

static constexpr VecSizes VEC_SIZES[] = {
    {1 << 16},        // ~65k
    {128 * 784},      // ~100k
    {1 << 26},        // ~67M
};

static constexpr LossSizes LOSS_SIZES[] = {
    {128, 784, 200},      
    {1024, 784, 200},    
    {64, 65536, 256},
};