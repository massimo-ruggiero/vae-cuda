#pragma once


namespace linear {

    namespace naive{

        void forward(const float* d_X,
                     const float* d_W,
                     const float* d_b,
                     float* d_Z,
                     int batch_size,
                     int input_dim,
                     int output_dim);

        void backward(const float* d_X,
                      const float* d_W,
                      const float* d_dZ,
                      float* d_dX,
                      float* d_dW,
                      float* d_db,
                      int batch_size,
                      int input_dim,
                      int output_dim);
                     
    } // namespace naive

} // namespace linear