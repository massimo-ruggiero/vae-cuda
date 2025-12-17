#pragma once

namespace adam {

    void step(const float* d_g,
              float* d_theta,
              float* d_m,
              float* d_v,
              int t,
              int size,
              float alpha = 1e-3f, 
              float beta1 = 0.9f,
              float beta2 = 0.999f,
              float epsilon = 1e-8f);

}