#pragma once

void leaky_relu_forward(float* d_Z,
                        const float d_alpha,
                        int size);

void sigmoid_forward(float* d_Z,
                     int size);