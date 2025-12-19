#pragma once


namespace activations {

    namespace leaky_relu {

        void forward(const float* d_Z,
                    float* d_A,
                    int size,
                    float d_alpha = 0.2f);

        void backward(const float* d_Z,
                    const float* d_dA,
                    float* d_dZ,
                    int size,
                    float alpha = 0.2f);

    } // namespace leaky_relu

    namespace sigmoid {

        void forward(const float* d_Z,
                    float* d_A,
                    int size);

        void backward(const float* d_A,
                    const float* d_dA,
                    float* d_dZ,
                    int size);

    } // namespace sigmoid

} // namespace activations