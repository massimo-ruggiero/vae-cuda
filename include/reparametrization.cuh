#pragma once


namespace reparametrization {

    void init(int size, unsigned long long seed);

    void destroy();

    void forward(const float* d_mu,
                 const float* d_logvar,
                 float* z,
                 int size);

    void backward(const float* d_dz,
                  const float* d_logvar,
                  const float* d_epsilon,
                  float* d_dmu,
                  float* d_dlogvar,
                  int size);

}