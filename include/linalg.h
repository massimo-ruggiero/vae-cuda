#pragma once


namespace matmul {

    void naive(const float* A, 
               const float* B, 
               float* C, 
               int M, 
               int K,
               int N);

}
