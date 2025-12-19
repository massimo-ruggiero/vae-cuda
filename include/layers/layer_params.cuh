#pragma once

#include <cuda_runtime.h>
#include "utils.cuh" 


struct GPUBuffer {
    float *ptr = nullptr;
    size_t size = 0;

    GPUBuffer() = default;
    ~GPUBuffer() {free();}

    GPUBuffer(const GPUBuffer&) = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;

    void allocate(size_t num_elements) {
        if (ptr) CUDA_CHECK(cudaFree(ptr)); 
        size = num_elements;
        CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(float)));
    }

    void free() {
        if (ptr) {
            CUDA_CHECK(cudaFree(ptr));
            ptr = nullptr;
            size = 0;
        }
    }
    
    void zero() {
        if (ptr) {
            CUDA_CHECK(cudaMemset(ptr, 0, size * sizeof(float)));
        }
    }
};


struct LinearLayer {
    GPUBuffer W;
    GPUBuffer b;
    GPUBuffer Z;
    GPUBuffer A;
    size_t input_dim;
    size_t output_dim;
    size_t batch_size;

    LinearLayer() = default;

    void allocate(size_t batch_size, size_t input_dim, size_t output_dim, bool use_activation) {
        this->batch_size = batch_size;
        this->input_dim = input_dim;
        this->output_dim = output_dim;

        W.allocate(input_dim * output_dim);
        b.allocate(output_dim);
        Z.allocate(batch_size * output_dim);

        if (use_activation) {
            A.allocate(batch_size * output_dim);
        } else {
            A.free();
        }
    }
};


struct LinearGrads {
    GPUBuffer dW;
    GPUBuffer db;
    GPUBuffer dZ;
    GPUBuffer dA;
    size_t input_dim;
    size_t output_dim;
    size_t batch_size;

    LinearGrads() = default;

    void allocate(size_t batch_size, size_t input_dim, size_t output_dim, bool use_activation) {
        this->batch_size = batch_size;
        this->input_dim = input_dim;
        this->output_dim = output_dim;

        dW.allocate(input_dim * output_dim);
        db.allocate(output_dim);
        dZ.allocate(batch_size * output_dim);

        if (use_activation) {
            dA.allocate(batch_size * output_dim);
        } else {
            dA.free();
        }
    }

    void zero_grads() {
        dW.zero();
        db.zero();
        dZ.zero();
        dA.zero(); 
    }
};


struct AdamState {
    GPUBuffer m_W;
    GPUBuffer v_W;
    GPUBuffer m_b;
    GPUBuffer v_b;
    size_t input_dim;
    size_t output_dim;

    AdamState() = default;

    void allocate(size_t input_dim, size_t output_dim) {
        this->input_dim = input_dim;
        this->output_dim = output_dim;

        m_W.allocate(input_dim * output_dim);
        v_W.allocate(input_dim * output_dim);
        m_b.allocate(output_dim);
        v_b.allocate(output_dim);

        reset();
    }

    void reset() {
        m_W.zero(); v_W.zero();
        m_b.zero(); v_b.zero();
    }
};