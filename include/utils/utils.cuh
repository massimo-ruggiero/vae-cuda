#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


#ifdef ENABLE_DEBUG
    #define DEBUG(...) do { printf(__VA_ARGS__); printf("\n"); } while(0)
#else
    #define DEBUG(...)
#endif


#define CUDA_CHECK(err) do {                                      \
    cudaError_t err_ = (err);                                     \
    if (err_ != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(err_), __FILE__, __LINE__);    \
        exit(EXIT_FAILURE);                                       \
    }                                                             \
} while(0)