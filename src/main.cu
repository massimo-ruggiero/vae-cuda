#include <cstdio>
#include <cuda_runtime.h>

#include "utils.cuh"
#include "linear.cuh" 
#include "activations.cuh"   

static void print_matrix(const char* name,
                        const float* mat, 
                        int rows, 
                        int cols) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%0.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    const int B = 2;
    const int D = 3;
    const int H = 4;

    DEBUG("Matrix sizes: B=%d, D=%d, H=%d\n", B, D, H);

    // Host matrices 
    float h_X[B * D] = {
        1, 2, 3,
        4, 5, 6
    };

    float h_W[D * H] = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12
    };

    float h_b[H] = {1, 1, 1, 1};
    float h_Z[B * H];

    float h_A_leaky[B * H];
    float h_A_sigmoid[B * H];


    print_matrix("X", h_X, B, D);
    print_matrix("W", h_W, D, H);
    printf("b:\n");
    for (int j = 0; j < H; ++j)
        printf("%8.1f ", h_b[j]);
    printf("\n");

    // Allocate device memory
    float* d_X = nullptr;
    float* d_W = nullptr;
    float* d_b = nullptr;
    float* d_Z = nullptr;
    float* d_A = nullptr;

    // Allocate device memory
    DEBUG("Allocating device memory...\n");
    CUDA_CHECK(cudaMalloc((void**)&d_X, B * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_W, D * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, H * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Z, B * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_A, B * H * sizeof(float)));
    
    // Copy matrices to device
    CUDA_CHECK(cudaMemcpy(d_X, h_X, B * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W, D * H * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, H * sizeof(float), cudaMemcpyHostToDevice));

    // Perform matrix multiplication
    DEBUG("Launching linear forward...\n");
    linear::naive::forward(d_X, d_W, d_b, d_Z, B, D, H);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_Z, d_Z, B * H * sizeof(float), cudaMemcpyDeviceToHost));

    print_matrix("Z", h_Z, B, H);
    // --- LeakyReLU forward: A = leaky_relu(Z) ---
    const float alpha = 0.1f;
    DEBUG("Launching LeakyReLU forward...\n");
    activations::leaky_relu::forward(d_Z, d_A, alpha, B * H);

    // Copia A (LeakyReLU) su host e stampa
    CUDA_CHECK(cudaMemcpy(h_A_leaky, d_A, B * H * sizeof(float), cudaMemcpyDeviceToHost));
    print_matrix("A_leaky", h_A_leaky, B, H);

    // --- Sigmoid forward: A = sigmoid(Z) ---
    DEBUG("Launching Sigmoid forward...\n");
    activations::sigmoid::forward(d_Z, d_A, B * H);

    // Copia A (Sigmoid) su host e stampa
    CUDA_CHECK(cudaMemcpy(h_A_sigmoid, d_A, B * H * sizeof(float), cudaMemcpyDeviceToHost));
    print_matrix("A_sigmoid", h_A_sigmoid, B, H);


    // Free device memory
    DEBUG("Freeing device memory...\n");
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_Z));
    CUDA_CHECK(cudaFree(d_A));

    return 0;
}