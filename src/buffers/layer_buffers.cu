#include "layer_buffers.cuh"
#include "utils.cuh" 
#include <cuda_runtime.h>


// GPUBuffer
void GPUBuffer::allocate(size_t num_elements) {
    free(); 
    size = num_elements;

    if (size == 0){
        ptr = nullptr;
        return;
    }

    CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(float)));
}

void GPUBuffer::free() {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
        ptr = nullptr;
    }
    size = 0;
}

void GPUBuffer::zero() {
    if (ptr) {
        CUDA_CHECK(cudaMemset(ptr, 0, size * sizeof(float)));
    }
}

// LinearLayer
void LinearLayer::allocate(size_t b_size, 
                           size_t in_dim, 
                           size_t out_dim, 
                           bool use_activation) {
    batch_size = b_size;
    input_dim = in_dim;
    output_dim = out_dim;

    W.allocate(input_dim * output_dim);
    b.allocate(output_dim);
    Z.allocate(batch_size * output_dim);

    if (use_activation) {
        A.allocate(batch_size * output_dim);
    } else {
        A.free();
    }
}

// LinearGrads
void LinearGrads::allocate(size_t b_size, 
                           size_t in_dim, 
                           size_t out_dim, 
                           bool use_activation) {
    batch_size = b_size;
    input_dim = in_dim;
    output_dim = out_dim;

    dW.allocate(input_dim * output_dim);
    db.allocate(output_dim);
    dZ.allocate(batch_size * output_dim);

    if (use_activation) {
        dA.allocate(batch_size * output_dim);
    } else {
        dA.free();
    }
}

void LinearGrads::zero_grads() {
    dW.zero();
    db.zero();
    dZ.zero();
    if (dA.ptr) dA.zero(); 
}

// AdamState
void AdamState::allocate(size_t in_dim, 
                         size_t out_dim) {
    input_dim = in_dim;
    output_dim = out_dim;

    m_W.allocate(input_dim * output_dim);
    v_W.allocate(input_dim * output_dim);
    m_b.allocate(output_dim);
    v_b.allocate(output_dim);

    zero_states();
}

void AdamState::zero_states() {
    m_W.zero(); 
    v_W.zero();
    m_b.zero(); 
    v_b.zero();
}