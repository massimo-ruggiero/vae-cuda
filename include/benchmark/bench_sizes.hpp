#pragma once

#include <vector>
#include <string>

struct SgemmSizes { int M, K, N; };
struct TransposeSizes { int M, N; };
struct AddBiasSizes { int batch, output_dim; };
struct VecSizes { int size; };
struct LossSizes { int batch, input_dim, latent_dim; };

static constexpr SgemmSizes SGEMM_SIZES[] = { 
    {4096, 4096, 4096},    
};

static constexpr TransposeSizes TRANSPOSE_SIZES[] = {      
    {65536, 65536},     
};

static constexpr AddBiasSizes ADD_BIAS_SIZES[] = {
    {65536, 65536},
};

static constexpr VecSizes VEC_SIZES[] = {
    {1 << 27},        // ~134M
};

static constexpr LossSizes LOSS_SIZES[] = {
    {8192, 8192, 8192}
};
