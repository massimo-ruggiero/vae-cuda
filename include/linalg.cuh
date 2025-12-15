#pragma once


namespace matmul {

    void naive(const float* A, 
               const float* B, 
               float* C, 
               int M, 
               int K,
               int N);

} // namespace matmul

namespace transpose {

    void naive(const float* A,
               float* B,
               int M,
               int K);

}