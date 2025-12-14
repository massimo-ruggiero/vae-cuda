#pragma once

void leaky_relu_forward(const float* d_Z,
                        float* d_A,
                        float d_alpha,
                        int size);

void leaky_relu_backward(const float* d_Z,
                         const float* d_dA,
                         float* d_dZ,
                         float alpha,
                         int size);

void sigmoid_forward(const float* d_Z,
                     float* d_A,
                     int size);

void sigmoid_backward(const float* d_A,
                      const float* d_dA,
                      float* d_dZ,
                      int size);