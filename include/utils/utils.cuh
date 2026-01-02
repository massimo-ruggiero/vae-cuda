#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


#ifdef ENABLE_DEBUG
    #define DEBUG(...) do { printf(__VA_ARGS__); printf("\n"); } while(0)
#else
    #define DEBUG(...)
#endif


#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while(0)