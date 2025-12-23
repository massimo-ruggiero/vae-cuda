#pragma once
#include "vae_config.h"


namespace activations {

    namespace leaky_relu {

        void forward(const float* d_Z,
                    float* d_A,
                    int size,
                    float d_alpha = 0.2f
                    const VAEStrategy& strategy = VAEStrategy::NAIVE);

        void backward(const float* d_Z,
                    const float* d_dA,
                    float* d_dZ,
                    int size,
                    float alpha = 0.2f,
                    const VAEStrategy& strategy = VAEStrategy::NAIVE);

    } // namespace leaky_relu

    namespace sigmoid {

        void forward(const float* d_Z,
                    float* d_A,
                    int size
                    const VAEStrategy& strategy = VAEStrategy::NAIVE);

        void backward(const float* d_A,
                    const float* d_dA,
                    float* d_dZ,
                    int size
                    const VAEStrategy& strategy = VAEStrategy::NAIVE);

    } // namespace sigmoid

} // namespace activations