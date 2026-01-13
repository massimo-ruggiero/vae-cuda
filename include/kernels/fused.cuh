#pragma once
#include "vae_config.cuh"

namespace fused {

    namespace forward {

        void linear_lrelu_tc(const float* d_X,
                            const float* d_W,
                            const float* d_b,
                            float* d_A,
                            int M, int K, int N,
                            float alpha);

        void linear_sigmoid_tc(const float* d_X,
                            const float* d_W,
                            const float* d_b,
                            float* d_Z,
                            float* d_A,
                            int M, int K, int N);

        void linear_tc(const float* d_X,
                       const float* d_W,
                       const float* d_b,
                       float* Z,
                       int M, int K, int N);

    } // namespace forward

} // namespace fused