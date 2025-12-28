#pragma once
#include "vae_config.cuh"


namespace linalg {

    void sgemm(const float* A, 
               const float* B, 
               float* C, 
               int M, int K, int N,
               const VAEStrategy& strategy);

    void transpose(const float* A, 
                   float* AT,  
                   int M, int K,
                   const VAEStrategy& strategy);

    void add_in_place(float* d_A, 
                      const float* d_B, 
                      int size,
                      const VAEStrategy& strategy);

} //namespace linalg