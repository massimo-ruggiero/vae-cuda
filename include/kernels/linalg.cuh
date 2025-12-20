#pragma once


namespace matmul {

    void naive(const float* A, 
               const float* B, 
               float* C, 
               int M, 
               int K,
               int N,
               bool transpose_A = false,
               bool transpose_B = false);

} // namespace matmul

void add_in_place(float* d_A, 
                  const float* d_B, 
                  int size);