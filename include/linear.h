#pragma once

void linear_forward(const float* d_X,
                    const float* d_W,
                    const float* d_b,
                    float* d_Z,
                    int batch_size,
                    int input_dim,
                    int output_dim);