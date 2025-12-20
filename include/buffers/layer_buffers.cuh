#pragma once

#include <cstddef>


struct GPUBuffer {
    float *ptr = nullptr;
    size_t size = 0;

    GPUBuffer() = default;
    ~GPUBuffer() {free();}

    GPUBuffer(const GPUBuffer&) = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;

    void allocate(size_t num_elements);
    void free();
    void zero();
};


struct LinearLayer {
    GPUBuffer W;
    GPUBuffer b;
    GPUBuffer Z;
    GPUBuffer A;
    size_t input_dim = 0;
    size_t output_dim = 0;
    size_t batch_size = 0;

    void allocate(size_t batch_size, 
                  size_t input_dim, 
                  size_t output_dim, 
                  bool use_activation);
};


struct LinearGrads {
    GPUBuffer dW;
    GPUBuffer db;
    GPUBuffer dZ;
    GPUBuffer dA;
    size_t input_dim = 0;
    size_t output_dim = 0;
    size_t batch_size = 0;

    void allocate(size_t batch_size, 
                  size_t input_dim, 
                  size_t output_dim, 
                  bool use_activation);
    void zero_grads();
};


struct AdamState {
    GPUBuffer m_W;
    GPUBuffer v_W;
    GPUBuffer m_b;
    GPUBuffer v_b;
    size_t input_dim = 0;
    size_t output_dim = 0;

    void allocate(size_t input_dim, 
                  size_t output_dim);
    void zero_states();
};