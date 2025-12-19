#pragma once
#include <curand_kernel.h>


namespace reparametrization {

    void init(curandStatePhilox4_32_10_t* d_states,
              int size, 
              unsigned long long seed);

    void forward(const float* d_mu,
                 const float* d_logvar,
                 float* z,
                 float* d_epsilon,
                 curandStatePhilox4_32_10_t* d_states,
                 int size);

    void backward(const float* d_dz,
                  const float* d_logvar,
                  const float* d_epsilon,
                  float* d_dmu,
                  float* d_dlogvar,
                  int size);

}